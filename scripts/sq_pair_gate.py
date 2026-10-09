"""Newest-date holdout gate for the sq_pair input of draft.fit_outcome_model.

Trains the production trainer (same lambda/iters/support) on games before the
cutoff, with and without the solo-queue pair score, and compares on the games
after it. Cutoffs sit inside the score's coverage window (pro patches 16.9+,
May 2026 on) so training sees covered maps. No DB writes.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import psycopg
from psycopg.rows import dict_row

from lol_ticker import draft
from lol_ticker.config import PG_DSN

OUT = Path('data/sq/eval/gate.json')


def ll(y, p):
    p = np.clip(p, 1e-6, 1-1e-6); return -(y*np.log(p)+(1-y)*np.log(1-p))


def main():
    rng = np.random.default_rng(921); report = {}
    with psycopg.connect(PG_DSN, row_factory=dict_row, options='-c default_transaction_read_only=on') as conn:
        for cutoff in ('2026-08-01', '2026-07-16'):
            ts = int(datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp())
            res = {flag: draft.fit_outcome_model(conn, sq_pair=flag, holdout_since=ts, write=False) for flag in (False, True)}
            a, b = res[False]['holdout'], res[True]['holdout']
            assert [r['game_id'] for r in a] == [r['game_id'] for r in b]
            y = np.array([r['y'] for r in a]); p0 = np.array([r['p'] for r in a]); p1 = np.array([r['p'] for r in b])
            covered = np.array([r['sq'] != 0 for r in b])
            keys = [(datetime.fromtimestamp(r['date_utc'], timezone.utc).date().isoformat(), tuple(sorted((r['blue_team'], r['red_team'])))) for r in a]
            uk = {k: i for i, k in enumerate(sorted(set(keys)))}; cl = np.array([uk[k] for k in keys])
            out = dict(train_games=res[True]['games']-len(a), holdout_games=len(a), covered=int(covered.sum()),
                       sq_coef_scaled=res[True]['sq_coef'], implied_slope_on_raw_score=res[True]['sq_coef']/0.25,
                       elo_only_logloss=res[True]['ll_elo'])
            for name, d in (('logloss', ll(y, p1)-ll(y, p0)), ('brier', (p1-y)**2-(p0-y)**2)):
                s = np.bincount(cl, weights=d); n = np.bincount(cl); dr = rng.integers(0, len(s), (4000, len(s)))
                v = s[dr].sum(1)/n[dr].sum(1)
                out[name] = dict(without=float((ll(y, p0) if name == 'logloss' else (p0-y)**2).mean()),
                                 with_sq=float((ll(y, p1) if name == 'logloss' else (p1-y)**2).mean()),
                                 delta=float(d.mean()), ci95=[float(x) for x in np.quantile(v, [.025, .975])],
                                 p_improves=float((v < 0).mean()))
            report[cutoff] = out
    OUT.write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
