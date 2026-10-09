"""Does TGV's historical edge survive out of sample?

The 2026-09-03 legacy release (standardized probit) cannot have seen maps played
after it was generated; the 2026-09-15 release has seen all of them. Same maps,
same controls offset, slope refit per score so link/scale differences cancel.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draft_comfort_compare as dcc
from tgv_fidelity_gaps import OE2TGV, norm
from tgv_vs_ours_headtohead import fit_scales, ll

ROOT = Path('data/tgv/20260916'); OUT = Path('data/tgv/head-to-head')
ROLES = list(OE2TGV.values())


class Release:
    def __init__(self, catalog, relroot):
        self.cat = json.loads(Path(catalog).read_text()); self.relroot = Path(relroot); self.cache = {}
        self.u = {p: {r: {v['champion']['cid']: v['strength'] for v in rows} for r, rows in d['roles'].items()}
                  for p, d in self.cat['insightsByPatch'].items()}

    def rel(self, role, c):
        if (role, c) not in self.cache:
            j = json.loads((self.relroot/f'{role}-{c}.json').read_text())
            self.cache[role, c] = {k: {(r, x): v for x, r, v in vals} for k, vals in j['relations'][f'{role}:{c}'].items()}
        return self.cache[role, c]

    def score(self, patch, blue, red):
        un = sum(self.u[patch][r][b]-self.u[patch][r][d] for r, b, d in zip(ROLES, blue, red))
        m = sum(self.rel(r, b)['matchups'][(s, d)] for r, b in zip(ROLES, blue) for s, d in zip(ROLES, red))
        sy = sum(self.rel(ROLES[i], blue[i])['synergies'][(ROLES[j], blue[j])]-self.rel(ROLES[i], red[i])['synergies'][(ROLES[j], red[j])]
                 for i in range(5) for j in range(i+1, 5))
        return un, un+m+sy


def main():
    import psycopg
    from psycopg.rows import dict_row
    from lol_ticker.config import PG_DSN
    with psycopg.connect(PG_DSN, row_factory=dict_row, options='-c default_transaction_read_only=on') as conn:
        games, _ = dcc.read_games(conn, int(datetime(2026, 9, 16, tzinfo=timezone.utc).timestamp()))
    games = {g['id']: g for g in games}
    rows = json.loads((OUT/'predictions.json').read_text())
    new = Release(ROOT/'current-model-catalog.json', ROOT/'model-relations/full')
    old = Release(ROOT/'current-surface/fallback-catalog.json', ROOT/'legacy-static/model-relations/full')
    cid = {norm(c['name']): c['cid'] for c in new.cat['champions']}
    data = []
    for r in rows:
        g = games[r['game_id']]
        if g['patch'] not in old.u or g['patch'] not in new.u: continue
        b, d = ([cid[norm(g[s]['picks'][p])] for p in OE2TGV] for s in ('blue', 'red'))
        try:
            data.append((r['day'], r['y'], r['c'], r['ours_champion'], *new.score(g['patch'], b, d), *old.score(g['patch'], b, d),
                         tuple(sorted((g['blue_team'], g['red_team'])))))
        except KeyError:
            continue
    day = np.array([x[0] for x in data]); Y = np.array([x[1] for x in data], float); c = np.array([x[2] for x in data])
    S = dict(ours_champion=np.array([x[3] for x in data]), new_unary=np.array([x[4] for x in data]), new_champion=np.array([x[5] for x in data]),
             old_unary=np.array([x[6] for x in data]), old_champion=np.array([x[7] for x in data]))
    keys = [(x[0], x[8]) for x in data]; rng = np.random.default_rng(918)
    report = dict(legacy_generated=old.cat['generatedAt'], current_generated=new.cat['generatedAt'], blocks={})
    for label, m in (('before_legacy_release (both TGV releases in-sample)', day < '2026-09-03'),
                     ('after_legacy_release (legacy out-of-sample, current in-sample)', day >= '2026-09-04')):
        idx = np.flatnonzero(m); uk = sorted({keys[i] for i in idx}); members = [[i for i in idx if keys[i] == k] for k in uk]
        base = ll(Y[idx], expit(c[idx])).mean()
        blk = dict(maps=int(m.sum()), controls_logloss=float(base), corr_old_new_champion=float(np.corrcoef(S['old_champion'][idx], S['new_champion'][idx])[0, 1]), scores={})
        for k, v in S.items():
            b = fit_scales(Y[idx], c[idx], v[idx, None])[0]
            gain = ll(Y[idx], expit(c[idx]+b*v[idx])).mean()-base
            boots_b, boots_g1 = [], []
            for _ in range(600):
                t = np.concatenate([members[j] for j in rng.integers(0, len(uk), len(uk))])
                boots_b.append(fit_scales(Y[t], c[t], v[t, None])[0])
            fixed = None
            if not k.startswith('old'):   # logit-scale scores can also be tested as published (slope 1)
                dl = ll(Y, expit(c+v))-ll(Y, expit(c))
                for _ in range(2000):
                    t = np.concatenate([members[j] for j in rng.integers(0, len(uk), len(uk))]); boots_g1.append(dl[t].mean())
                fixed = dict(delta_logloss=float(dl[idx].mean()), ci95=[float(x) for x in np.quantile(boots_g1, [.025, .975])])
            blk['scores'][k] = dict(sd=float(v[idx].std()), fitted_slope=float(b), slope_ci95=[float(x) for x in np.quantile(boots_b, [.025, .975])],
                                    slope_z=float(b/np.std(boots_b)), insample_slope_logloss_gain=float(gain), as_published_slope1=fixed)
        report['blocks'][label] = blk
    (OUT/'leak-test.json').write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()


def clean_block():
    """Legacy release as published (probit*scoreScale -> logit via 1.702) vs ours, maps it never saw."""
    import psycopg
    from psycopg.rows import dict_row
    from lol_ticker.config import PG_DSN
    with psycopg.connect(PG_DSN, row_factory=dict_row, options='-c default_transaction_read_only=on') as conn:
        games, _ = dcc.read_games(conn, int(datetime(2026, 9, 16, tzinfo=timezone.utc).timestamp()))
    games = {g['id']: g for g in games}
    old = Release(ROOT/'current-surface/fallback-catalog.json', ROOT/'legacy-static/model-relations/full')
    cid = {norm(c['name']): c['cid'] for c in old.cat['champions']}
    k = old.cat['scoreScale']*1.702
    rows = [r for r in json.loads((OUT/'predictions.json').read_text()) if r['day'] >= '2026-09-04']
    Y, c, o, t, keys = [], [], [], [], []
    for r in rows:
        g = games[r['game_id']]
        try:
            s = old.score(g['patch'], *([cid[norm(g[x]['picks'][p])] for p in OE2TGV] for x in ('blue', 'red')))[1]
        except KeyError:
            continue
        Y.append(r['y']); c.append(r['c']); o.append(r['ours_champion']); t.append(k*s); keys.append((r['day'], tuple(sorted((g['blue_team'], g['red_team'])))))
    Y, c, o, t = map(np.array, (Y, c, o, t)); uk = sorted(set(keys)); members = [np.array([i for i, x in enumerate(keys) if x == u]) for u in uk]
    rng = np.random.default_rng(919); out = dict(maps=len(Y), clusters=len(uk), probit_to_logit=k)
    L = dict(controls=ll(Y, expit(c)), ours=ll(Y, expit(c+o)), tgv_legacy=ll(Y, expit(c+t)), half_blend=ll(Y, expit(c+.5*(o+t))))
    out['logloss'] = {a: float(v.mean()) for a, v in L.items()}
    for a, b in (('ours', 'controls'), ('tgv_legacy', 'controls'), ('half_blend', 'controls'), ('tgv_legacy', 'ours')):
        d = L[a]-L[b]; bs = [d[np.concatenate([members[j] for j in rng.integers(0, len(uk), len(uk))])].mean() for _ in range(2000)]
        out[a+' - '+b] = dict(delta=float(d.mean()), ci95=[float(x) for x in np.quantile(bs, [.025, .975])])
    rep = json.loads((OUT/'leak-test.json').read_text()); rep['clean_block_as_published'] = out
    (OUT/'leak-test.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(out, indent=1))
