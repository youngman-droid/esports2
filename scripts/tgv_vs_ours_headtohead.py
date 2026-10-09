"""Head-to-head: our outcome draft model vs the recoverable TGV champion model.

Ours is refit walk-forward (half-month cutoffs, strictly earlier days) so every
evaluated map is out of sample for it. TGV is the fixed 2026-09-15 release
(unary + matchup + synergy [+ recovered comfort curve]); its unrecovered
composition term is absent and its training saw every map before 2026-09-14.
Both are added to the SAME controls-only offset. Read-only; nothing deployed.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draft_comfort_compare as dcc
from lol_ticker import draft
from lol_ticker.tgv_reconstruction import PublicDraftModel
from tgv_fidelity_gaps import A, K, OE2TGV, norm

ROOT = Path('data/tgv/20260916'); OUT = Path('data/tgv/head-to-head')
CUTS = ['2026-06-16', '2026-07-01', '2026-07-16', '2026-08-01', '2026-08-16', '2026-09-01', '2026-09-16']
TGV_ASOF = '2026-09-14'
MAJOR = {'LCK', 'LPL', 'LEC', 'LTA N', 'LTA S', 'LCS', 'LCP', 'MSI', 'EWC', 'WLDs', 'FST'}


def category(f):
    if f.startswith(('syn:', 'vs:')): return 'pair'
    if f.startswith(('own_ban:', 'enemy_ban:')): return 'ban'
    return 'pick'


def ll(y, p):
    p = np.clip(p, 1e-9, 1-1e-9); return -(y*np.log(p)+(1-y)*np.log1p(-p))


def fit_scales(y, c, S):
    """Unpenalised logistic slopes on scores S with fixed offset c (Newton)."""
    b = np.zeros(S.shape[1])
    for _ in range(30):
        p = expit(c + S@b); g = S.T@(p-y); H = (S*(p*(1-p))[:, None]).T@S
        step = np.linalg.solve(H+1e-9*np.eye(len(b)), g); b -= step
        if abs(step).max() < 1e-10: break
    return b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import psycopg
    from psycopg.rows import dict_row
    from lol_ticker.config import PG_DSN
    cutoff = int(datetime(2026, 9, 16, tzinfo=timezone.utc).timestamp())
    with psycopg.connect(PG_DSN, row_factory=dict_row, options='-c default_transaction_read_only=on') as conn:
        games, rejected = dcc.read_games(conn, cutoff)
    games.sort(key=lambda g: (g['day'], g['id']))
    y = np.array([g['y'] for g in games]); days = np.array([g['day'] for g in games])
    feats = [dcc.signed_draft(g) for g in games]
    hist = dcc.history_features(games)
    E = np.array([hist[g['id']][0] for g in games]); P = np.array([hist[g['id']][3] for g in games])

    # --- TGV scores (fixed release) with strictly-earlier-day comfort counts
    model = PublicDraftModel(ROOT)
    cid = {norm(c['name']): c['cid'] for c in model.catalog['champions']}
    tgv = {}; counts = defaultdict(int); pending = []; day = None
    for i, g in enumerate(games):
        if g['day'] != day:
            for k in pending: counts[k] += 1
            pending = []; day = g['day']
        keys = {s: [(g[s]['players'][r], r, g[s]['picks'][r]) for r in OE2TGV] for s in ('blue', 'red')}
        if g['patch'] in model.strength and all(norm(k[2]) in cid for s in keys for k in keys[s]):
            try:
                parts = model.components(g['patch'], *[[cid[norm(k[2])] for k in keys[s]] for s in ('blue', 'red')])
                com = [sum(A*counts[k]/(counts[k]+K) for k in keys[s]) for s in ('blue', 'red')]
                tgv[i] = dict(unary=parts['strength'], matchup=parts['matchup'], synergy=parts['synergy'], comfort=com[0]-com[1])
            except KeyError:
                pass
        pending += keys['blue']+keys['red']
    ev = np.array(sorted(i for i in tgv if days[i] >= CUTS[0]))

    # --- ours, walk-forward
    ours = {}
    for lo, hi in zip(CUTS[:-1], CUTS[1:]):
        tr = days < lo; te = [i for i in ev if lo <= days[i] < hi]
        if not te: continue
        cnt = Counter(k for f, keep in zip(feats, tr) if keep for k in f)
        vocab = sorted(k for k, n in cnt.items() if n >= 60 or not draft._is_interaction(k))
        look = {k: j for j, k in enumerate(vocab)}
        ii, jj, vv = [], [], []
        for i, f in enumerate(feats):
            for k, v in f.items():
                if k in look: ii.append(i); jj.append(look[k]); vv.append(v)
        D = sparse.csr_matrix((vv, (ii, jj)), shape=(len(games), len(vocab)))
        base = sparse.csr_matrix(np.column_stack((np.ones(len(games)), E, P/np.maximum(P[tr].std(axis=0), 1e-8))))
        b0 = dcc.fit(base[tr], y[tr]); X = sparse.hstack((base, D)).tocsr(); b1 = dcc.fit(X[tr], y[tr])
        nb = base.shape[1]; cats = np.array([category(k) for k in vocab])
        for i in te:
            row = D[i]; contrib = row.multiply(b1[nb:]).toarray().ravel()
            ours[i] = dict(c=float(base[i]@b0), native=float(X[i]@b1),
                           **{k: float(contrib[cats == k].sum()) for k in ('pick', 'pair', 'ban')})
        print(lo, 'train', int(tr.sum()), 'eval', len(te), 'vocab', len(vocab), flush=True)

    ev = np.array([i for i in ev if i in ours]); Y = y[ev].astype(float)
    col = lambda d, k: np.array([d[i][k] for i in ev])
    c = col(ours, 'c'); o_pick, o_pair, o_ban = col(ours, 'pick'), col(ours, 'pair'), col(ours, 'ban')
    o_champ = o_pick+o_pair; o_all = o_champ+o_ban
    t_un = col(tgv, 'unary'); t_base = t_un+col(tgv, 'matchup')+col(tgv, 'synergy'); t_full = t_base+col(tgv, 'comfort')
    scores = dict(ours_unary=o_pick, ours_champion=o_champ, ours_with_bans=o_all,
                  tgv_unary=t_un, tgv_champion=t_base, tgv_champion_comfort=t_full)
    preds = dict(controls=expit(c), ours_native_joint=expit(col(ours, 'native')),
                 **{k: expit(c+v) for k, v in scores.items()},
                 blend_half=expit(c+.5*(o_champ+t_base)))

    keys = [(games[i]['day'], tuple(sorted((games[i]['blue_team'], games[i]['red_team'])))) for i in ev]
    uniq = sorted(set(keys)); lk = {k: j for j, k in enumerate(uniq)}; cl = np.array([lk[k] for k in keys])
    rng = np.random.default_rng(917)

    def paired(delta, mask):
        idx = np.unique(cl[mask], return_inverse=True)[1]
        sums = np.bincount(idx, weights=delta[mask]); n = np.bincount(idx)
        dr = rng.integers(0, len(sums), (2000, len(sums)))
        v = sums[dr].sum(1)/n[dr].sum(1)
        return dict(delta=float(delta[mask].mean()), ci95=[float(x) for x in np.quantile(v, [.025, .975])])

    league = np.array([games[i]['league'] for i in ev]); d = days[ev]
    slices = dict(all=np.ones(len(ev), bool), major=np.isin(league, list(MAJOR)), minor=~np.isin(league, list(MAJOR)),
                  tgv_out_of_sample=d >= TGV_ASOF, **{'month_'+m: np.char.startswith(d, m) for m in sorted({x[:7] for x in d})})
    report = dict(maps=len(ev), date_range=[min(d.tolist()), max(d.tolist())], rejected_inputs=rejected, clusters=len(uniq),
                  leagues_major=sorted(set(league[slices['major']])), slices={})
    for name, m in slices.items():
        if m.sum() < 20: continue
        r = dict(maps=int(m.sum()), metrics={}, vs_controls={}, tgv_minus_ours={})
        for k, p in preds.items():
            r['metrics'][k] = dict(brier=float(np.mean((p[m]-Y[m])**2)), logloss=float(ll(Y[m], p[m]).mean()),
                                   accuracy=float(np.mean((p[m] >= .5) == Y[m])))
            if k != 'controls':
                r['vs_controls'][k] = dict(brier=paired((p-Y)**2-(preds['controls']-Y)**2, m),
                                           logloss=paired(ll(Y, p)-ll(Y, preds['controls']), m))
        for a, b in (('tgv_unary', 'ours_unary'), ('tgv_champion', 'ours_champion'), ('tgv_champion_comfort', 'ours_champion')):
            r['tgv_minus_ours'][a+' - '+b] = dict(brier=paired((preds[a]-Y)**2-(preds[b]-Y)**2, m),
                                                  logloss=paired(ll(Y, preds[a])-ll(Y, preds[b]), m))
        report['slices'][name] = r

    members = [np.flatnonzero(cl == j) for j in range(len(uniq))]
    resample = lambda: np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])

    # --- signal diagnostics: slope each score deserves on top of controls (1.0 = perfectly scaled)
    diag = {}
    for k, v in scores.items():
        b = fit_scales(Y, c, v[:, None])[0]
        boots = []
        for _ in range(400):
            take = resample()
            boots.append(fit_scales(Y[take], c[take], v[take, None])[0])
        diag[k] = dict(sd=float(v.std()), slope=float(b), slope_ci95=[float(x) for x in np.quantile(boots, [.025, .975])],
                       logloss_at_fitted_slope=float(ll(Y, expit(c+b*v)).mean()))
    S = np.column_stack((o_champ, t_base)); jb = fit_scales(Y, c, S); jboots = []
    for _ in range(400):
        take = resample()
        jboots.append(fit_scales(Y[take], c[take], S[take]))
    jboots = np.array(jboots)
    report['signal'] = dict(per_score=diag, controls_logloss=float(ll(Y, expit(c)).mean()),
                            joint_ours_champion_and_tgv_champion=dict(
                                slopes=dict(ours=float(jb[0]), tgv=float(jb[1])),
                                ci95=dict(ours=[float(x) for x in np.quantile(jboots[:, 0], [.025, .975])],
                                          tgv=[float(x) for x in np.quantile(jboots[:, 1], [.025, .975])]),
                                logloss=float(ll(Y, expit(c+S@jb)).mean())),
                            correlations=dict(champion=float(np.corrcoef(o_champ, t_base)[0, 1]),
                                              unary=float(np.corrcoef(o_pick, t_un)[0, 1]),
                                              pair=float(np.corrcoef(o_pair, t_base-t_un)[0, 1])))
    (OUT/'report.json').write_text(json.dumps(report, indent=1))
    (OUT/'predictions.json').write_text(json.dumps([dict(game_id=games[i]['id'], day=games[i]['day'], league=games[i]['league'],
        patch=games[i]['patch'], y=int(y[i]), c=ours[i]['c'], **{k: float(v[j]) for k, v in scores.items()}) for j, i in enumerate(ev)]))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
