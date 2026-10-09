"""Isolated, chronological draft/comfort diagnostic; never writes production or DB.

Uses the existing draft feature vocabulary, antisymmetrized so reversing sides
negates the isolated score. Comfort adapts wpx_playerchamp's historical win-rate,
experience and unseen-pick signals, with role-specific keys and whole-day lag.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import draft

ROLES = ('top', 'jng', 'mid', 'bot', 'sup')


def draft_features(g, reverse=False):
    b, r = ('red', 'blue') if reverse else ('blue', 'red')
    row = dict(patch=g['patch'] or '', own_picks=sorted(g[b]['picks'].values()),
               enemy_picks=sorted(g[r]['picks'].values()),
               own_roles={c: p for p, c in g[b]['picks'].items()},
               enemy_roles={c: p for p, c in g[r]['picks'].items()},
               own_bans=sorted(set(g[b]['bans'])), enemy_bans=sorted(set(g[r]['bans'])))
    return Counter(draft._features(row))


def signed_draft(g):
    a, b = draft_features(g), draft_features(g, True)
    return {k: (a[k] - b[k]) / 2 for k in a.keys() | b.keys() if a[k] != b[k]}


def history_features(games):
    """Only earlier UTC dates update comfort and Elo; no same-day outcomes."""
    pc, player = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    elo = defaultdict(lambda: 1500.)
    result = {}
    for day, batch in itertools.groupby(sorted(games, key=lambda g: (g['day'], g['id'])),
                                         key=lambda g: g['day']):
        batch = list(batch)
        for g in batch:
            values, coverage, controls = [], [], []
            for side in ('blue', 'red'):
                vals, known, overall = np.zeros(3), [], np.zeros(3)
                for role in ROLES:
                    p = (g[side]['players'][role], role)
                    key = p + (g[side]['picks'][role],)
                    n, w = player[p]
                    nc, wc = pc[key]
                    base = (w + 5.) / (n + 10.)
                    overall += [base - .5, np.log1p(n), float(n == 0)]
                    vals += [(wc + 5. * base) / (nc + 5.) - base,
                             np.log1p(nc), float(nc == 0)]
                    known.append(nc)
                values.append(vals)
                controls.append(overall)
                coverage.extend(known)
            result[g['id']] = ((elo[g['blue_team']] - elo[g['red_team']]) / 400.,
                                values[0] - values[1], coverage, controls[0] - controls[1])
        # Elo updates use the day's opening ratings, avoiding game-ID ordering.
        updates = defaultdict(float)
        for g in batch:
            expected = 1 / (1 + 10 ** ((elo[g['red_team']] - elo[g['blue_team']]) / 400.))
            delta = 30 * (g['y'] - expected)
            updates[g['blue_team']] += delta
            updates[g['red_team']] -= delta
            for side in ('blue', 'red'):
                won = g['y'] if side == 'blue' else 1 - g['y']
                for role in ROLES:
                    p = (g[side]['players'][role], role)
                    key = p + (g[side]['picks'][role],)
                    for store, k in ((player, p), (pc, key)):
                        store[k][0] += 1
                        store[k][1] += won
        for team, delta in updates.items():
            elo[team] += delta
    return result


def fit(X, y):
    penalty = np.full(X.shape[1], 300.)
    penalty[:2] = 0  # intercept and team-Elo control
    def objective(beta):
        z = X @ beta
        return (np.logaddexp(0, z).sum() - np.sum(y*z) + .5 * np.sum(penalty*beta*beta),
                np.asarray(X.T @ (expit(z) - y)).ravel() + penalty * beta)
    result = minimize(objective, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1500, 'ftol': 1e-11, 'gtol': 1e-6})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def metrics(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return dict(games=len(y), brier=float(np.mean((p-y)**2)),
                logloss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))))


def interval(games, deltas):
    keys = [(g['day'], tuple(sorted((g['blue_team'], g['red_team'])))) for g in games]
    unique = sorted(set(keys)); lookup = {k:i for i,k in enumerate(unique)}
    idx = np.array([lookup[k] for k in keys])
    sums = np.bincount(idx, weights=deltas); counts = np.bincount(idx)
    rng = np.random.default_rng(916)
    draws = rng.integers(0, len(unique), (2000, len(unique)))
    vals = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return dict(delta=float(np.mean(deltas)), ci95=np.quantile(vals, [.025,.975]).tolist(),
                clusters=len(unique))


def read_games(conn, before):
    raw = conn.execute('SELECT * FROM oe_games WHERE date_utc IS NOT NULL AND date_utc < %s AND winner IS NOT NULL ORDER BY date_utc, game_id', (before,)).fetchall()
    players, picks, bans = defaultdict(dict), defaultdict(dict), defaultdict(list)
    for r in conn.execute('SELECT * FROM oe_players'):
        players[(r['game_id'], r['team'])][r['position']] = r['player_id'] or r['player']
    for r in conn.execute('SELECT * FROM oe_picks'):
        picks[(r['game_id'], r['team'])][r['position']] = r['champion']
    for r in conn.execute('SELECT * FROM oe_bans'):
        bans[(r['game_id'], r['team'])].append(r['champion'])
    games = []; rejected = 0
    for r in raw:
        g = dict(id=r['game_id'], day=datetime.fromtimestamp(r['date_utc'], timezone.utc).date().isoformat(),
                 patch=r['patch'], league=r['league'], blue_team=r['blue_team'], red_team=r['red_team'],
                 y=int(r['winner'] == r['blue_team']))
        valid = r['winner'] in (r['blue_team'], r['red_team'])
        for side in ('blue', 'red'):
            key = (r['game_id'], r[side+'_team'])
            g[side] = dict(players=players[key], picks=picks[key], bans=bans[key])
            valid &= all(players[key].get(p) and picks[key].get(p) for p in ROLES)
        valid &= len(set(g['blue']['picks'].values()) | set(g['red']['picks'].values())) == 10
        if valid: games.append(g)
        else: rejected += 1
    return games, rejected


def run(games, out):
    features = [signed_draft(g) for g in games]
    tr = np.array([g['day'] < '2026-01-16' for g in games])
    va = np.array(['2026-01-16' <= g['day'] < '2026-05-01' for g in games])
    te = np.array(['2026-05-01' <= g['day'] < '2026-09-03' for g in games])
    counts = Counter(k for f, keep in zip(features, tr) if keep for k in f)
    vocabulary = sorted(k for k,n in counts.items() if n >= 60 or not draft._is_interaction(k))
    lookup = {k:i for i,k in enumerate(vocabulary)}
    ii,jj,vv = [],[],[]
    for i,f in enumerate(features):
        for k,v in f.items():
            if k in lookup: ii.append(i); jj.append(lookup[k]); vv.append(v)
    D = sparse.csr_matrix((vv,(ii,jj)), shape=(len(games),len(vocabulary)))
    hist = history_features(games)
    E = np.array([hist[g['id']][0] for g in games])
    C = np.array([hist[g['id']][1] for g in games])
    scale = np.maximum(C[tr].std(axis=0), 1e-8)
    P = np.array([hist[g['id']][3] for g in games])
    player_scale = np.maximum(P[tr].std(axis=0), 1e-8)
    base = sparse.csr_matrix(np.column_stack((np.ones(len(games)), E, P/player_scale)))
    nbase = base.shape[1]
    matrices = {'elo_player':base, 'draft':sparse.hstack((base,D)).tocsr(),
                'draft_comfort':sparse.hstack((base,D,sparse.csr_matrix(C/scale))).tocsr()}
    y = np.array([g['y'] for g in games])
    report = dict(protocol='Exploratory chronological diagnostic; no fresh promotion claim',
                  train_before='2026-01-16', validation_before='2026-05-01', diagnostic_before='2026-09-03',
                  training_games=int(tr.sum()), draft_features=len(vocabulary), models={}, comparisons={},
                  controls=['intercept','team_elo','player_overall_win_history','player_overall_experience','new_players'])
    predictions = {}; coefficients = {}
    for name,X in matrices.items():
        beta = fit(X[tr], y[tr]); p = expit(X@beta)
        coefficients[name] = beta; predictions[name] = p
        report['models'][name] = {label: metrics(y[mask],p[mask]) for label,mask in [('validation',va),('diagnostic',te)]}
    for label,mask in [('validation',va),('diagnostic',te)]:
        report['comparisons'][label] = interval([g for g,m in zip(games,mask) if m],
            ((predictions['draft_comfort']-y)**2-(predictions['draft']-y)**2)[mask])
    beta = coefficients['draft_comfort']
    dc = np.asarray(D @ beta[nbase:nbase+len(vocabulary)]).ravel()
    cc = np.sum((C/scale)*beta[-3:],axis=1)
    for values in [dc,cc,*predictions.values()]:
        if not np.all(np.isfinite(values)): raise ValueError('Nonfinite predictions')
    report['comfort_coefficients_raw'] = dict(zip(['relative_win_history','log_experience','unseen_picks'],(beta[-3:]/scale).tolist()))
    report['diagnostic_comfort_effect'] = dict(median_abs_logit=float(np.median(abs(cc[te]))),
        median_abs_neutral_pp=float(np.median(abs(expit(dc[te]+cc[te])-expit(dc[te])))*100),
        ten_players_with_prior_champion_games=int(sum(min(hist[g['id']][2])>0 for g,m in zip(games,te) if m)))
    # Scores with team strength and intercept removed, not full match forecasts.
    rows = [dict(game_id=g['id'],date=g['day'],league=g['league'],patch=g['patch'],blue=g['blue_team'],red=g['red_team'],
                 draft_logit=float(dc[i]),comfort_logit=float(cc[i]),neutral_blue=float(expit(dc[i]+cc[i])),
                 full_blue=float(predictions['draft_comfort'][i]),prior_champion_games=hist[g['id']][2])
            for i,g in enumerate(games) if te[i]]
    (out/'scores.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(out/'model.npz', vocabulary=np.array(vocabulary), comfort_scale=scale,
                        player_scale=player_scale, nbase=nbase, **coefficients)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True)
    parser.add_argument('--input', help='Replay a saved inputs.json without database access')
    args = parser.parse_args(); out = Path(args.out); out.mkdir(parents=True,exist_ok=False)
    if args.input:
        games = json.loads(Path(args.input).read_text()); rejected = None; existing = None
        if any(g['day'] >= '2026-09-03' for g in games):
            raise ValueError('Inputs extend beyond the diagnostic cutoff')
    else:
        import psycopg
        from psycopg.rows import dict_row
        from lol_ticker.config import PG_DSN
        cutoff = int(datetime(2026,9,3,tzinfo=timezone.utc).timestamp())
        with psycopg.connect(PG_DSN, row_factory=dict_row,
                             options='-c default_transaction_read_only=on') as conn:
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            games,rejected = read_games(conn, cutoff)
            existing = conn.execute("SELECT key,value FROM draft_outcome_meta").fetchall()
    source = json.dumps(games,sort_keys=True).encode()
    (out/'inputs.json').write_bytes(source)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    (out/'manifest.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(source).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        games=len(games),rejected=rejected,existing_draft_meta=existing,
        fit='fixed ridge 300; no calibration/tuning; train-only feature support and scaling',
        history='earlier UTC days only; role-specific player/champion histories; cold start zero deviation',
        scope='offline refit of signed existing draft vocabulary plus historical comfort; not deployed'),indent=2))
    print(json.dumps(run(games,out),indent=2))


if __name__ == '__main__': main()
