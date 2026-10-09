"""Does the solo-queue pair score survive inside the live GAM stack?

Fits the production GAM (wpgam.fit_arrays, temporal calibration) on gol.gg games
before CUTOFF, then asks of the untouched newer games whether sq_pair still
carries signal (a) on top of the pregame prior -- slope learned on TRAINING games
against out-of-fold pregame logits, applied forward -- and (b) on top of the full
in-game GAM logit by game-time bucket (in-block slope + fixed forward slope).
Read-only; nothing saved but the report.
"""
import json
from pathlib import Path
import sys

import numpy as np
import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lol_ticker import sqpairs, wpgam
from lol_ticker.config import PG_DSN
from tgv_vs_ours_headtohead import fit_scales, ll

CUTOFF = '2026-08-01'; OUT = Path('data/sq/eval/live-check.json')
ROLE = {'TOP': 0, 'JUNGLE': 1, 'MID': 2, 'ADC': 3, 'SUPPORT': 4}
sig = lambda z: 1/(1+np.exp(-z)); lg = lambda p: np.log(np.clip(p, 1e-9, 1-1e-9)/np.clip(1-p, 1e-9, 1))


def main():
    d = np.load('data/wpx/states.npz', allow_pickle=True)
    X, y, gid, t_s, seq, C = d['X'], d['y'], d['gid'], d['t'], d['seq'], d['C']
    names, champ_names = list(d['names']), list(d['champ_names'])
    first = wpgam._game_rows(gid); g_gid = gid[first]; g_date = d['date'][first].astype(str); g_patch = d['patch'][first].astype(str); g_y = y[first]
    picks = {}
    with psycopg.connect(PG_DSN, row_factory=dict_row, options='-c default_transaction_read_only=on') as conn:
        for r in conn.execute("SELECT game_id, side, role, champion FROM golgg_players WHERE game_id = ANY(%s)", (g_gid.tolist(),)):
            picks.setdefault(r['game_id'], {}).setdefault(r['side'], {})[ROLE.get(r['role'])] = r['champion']
    sc = sqpairs.scorer(); sq = np.zeros(len(g_gid))
    for i, (g, p) in enumerate(zip(g_gid.tolist(), g_patch)):
        s = picks.get(g, {})
        if all(len(s.get(k, {})) == 5 and None not in s[k] for k in ('blue', 'red')):
            r = sc.score(p, *[[s[k][j] for j in range(5)] for k in ('blue', 'red')])
            if r: sq[i] = r['score']
    train = g_date < CUTOFF; cov = sq != 0
    print('games', len(g_gid), 'train', int(train.sum()), 'test', int((~train).sum()), 'covered train/test', int((cov & train).sum()), int((cov & ~train).sum()), flush=True)

    model = wpgam.fit_arrays(X, y, gid, t_s, seq, C, names, champ_names, train_games=train, dates=d['date'])
    print('GAM fit done', flush=True)
    rng = np.random.default_rng(922)

    def boot(delta, groups):
        u, idx = np.unique(groups, return_inverse=True); s = np.bincount(idx, weights=delta); n = np.bincount(idx)
        dr = rng.integers(0, len(u), (2000, len(u))); v = s[dr].sum(1)/n[dr].sum(1)
        return dict(delta=float(delta.mean()), ci95=[float(x) for x in np.quantile(v, [.025, .975])])

    def slope(Y, off, s, groups):
        b = fit_scales(Y, off, s[:, None])[0]; u = np.unique(groups); mem = {g: np.flatnonzero(groups == g) for g in u}; bs = []
        for _ in range(300):
            t = np.concatenate([mem[g] for g in rng.choice(u, len(u))]); bs.append(fit_scales(Y[t], off[t], s[t, None])[0])
        return dict(slope=float(b), ci95=[float(x) for x in np.quantile(bs, [.025, .975])], z=float(b/np.std(bs)))

    # (a) pregame prior: forward slope from OOF training logits
    pre = wpgam.pregame_values_from_matrix(X[first], names); gC = C[first]
    team_oof, champ_oof = wpgam.oof_pregame(pre[train], gC[train], g_y[train], g_gid[train], champ_names, k=5, dates=g_date[train])
    off_tr = team_oof+champ_oof; m = cov[train]
    b_pre = fit_scales(g_y[train][m].astype(float), off_tr[m], sq[train][m, None])[0]
    eta, _, _ = wpgam.predict_pregame(model['pregame'], pre[~train], gC[~train]); te = cov[~train]
    Yt = g_y[~train][te].astype(float); off = eta[te]; s = sq[~train][te]; days = g_date[~train][te]
    report = dict(cutoff=CUTOFF, train_games=int(train.sum()), test_games=int((~train).sum()), covered_train=int((cov & train).sum()), covered_test=int(te.sum()),
                  pregame=dict(train_slope_on_oof_prior=float(b_pre), baseline_logloss=float(ll(Yt, sig(off)).mean()),
                               forward_slope=boot(ll(Yt, sig(off+b_pre*s))-ll(Yt, sig(off)), days),
                               forward_slope_brier=boot((sig(off+b_pre*s)-Yt)**2-(sig(off)-Yt)**2, days),
                               in_block=slope(Yt, off, s, days),
                               corr_sq_with_prior_champ_logit=float(np.corrcoef(s, wpgam.predict_pregame(model['pregame'], pre[~train], gC[~train])[2][te])[0, 1])))

    # (b) in-game: full GAM logit on fixed-minute test rows
    test_g = set(g_gid[~train][te].tolist()); sq_by = dict(zip(g_gid.tolist(), sq)); date_by = dict(zip(g_gid.tolist(), g_date))
    rows = (seq < 0) & np.isin(gid, list(test_g))
    prior = wpgam.prior_values_from_matrix(model, X[rows], names, C[rows])
    raw = np.column_stack([prior, wpgam.state_values_from_matrix(X[rows], names)])
    p = wpgam.predict_state(model['state'], raw, t_s[rows]/60.0); z = lg(p)
    Yr = y[rows].astype(float); sr = np.array([sq_by[g] for g in gid[rows]]); dr_ = np.array([date_by[g] for g in gid[rows]]); tm = t_s[rows]/60.0
    report['in_game'] = {}
    for name, lo, hi in (('0-5', 0, 5), ('5-10', 5, 10), ('10-15', 10, 15), ('15-20', 15, 20), ('20-25', 20, 25), ('25-30', 25, 30), ('30+', 30, 1e9), ('all', 0, 1e9)):
        k = (tm >= lo) & (tm < hi)
        report['in_game'][name] = dict(states=int(k.sum()), gam_logloss=float(ll(Yr[k], p[k]).mean()), in_block=slope(Yr[k], z[k], sr[k], dr_[k]),
                                       forward_pregame_slope=boot(ll(Yr[k], sig(z[k]+b_pre*sr[k]))-ll(Yr[k], p[k]), dr_[k]))
    OUT.write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
