"""Test a solo-queue-derived matchup/synergy score as ONE input to our draft model.

Pair effects come only from Lolalytics solo-queue counts (research/sq_pairs_scrape.py),
so no professional outcome enters them: every covered pro map is a clean test.
  effect = 0.04 * d2, where Lolalytics' d2 is the pair win rate in pp beyond both
  champions' own baselines (verified: d2 = vsWr-(100-wr_b)-(wr_a-50)-(avgWr-50)); 0.04 = pp -> logit
each shrunk by n/(n+k), k = 4/tau^2 with tau^2 by method of moments on SQ data alone.
Patch conventions: 'prior' = n-weighted pool of patches strictly before the pro
map's patch (no look-ahead); 'same' = that patch only; 'upto' = pool incl. same.
Ours + controls are refit walk-forward exactly as in tgv_vs_ours_headtohead.py; the
SQ slope is also walk-forward (fit on earlier evaluated maps only). Read-only.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draft_comfort_compare as dcc
from lol_ticker import draft
from sq_pairs_scrape import LANES, OE2L, key
from tgv_vs_ours_headtohead import category, fit_scales, ll

SQ = Path('data/sq/lolalytics'); OUT = Path('data/sq/eval')
CUTS = ['2026-05-01', '2026-05-16', '2026-06-01', '2026-06-16', '2026-07-01', '2026-07-16', '2026-08-01', '2026-08-16', '2026-09-01', '2026-09-16']
MAJOR = {'LCK', 'LPL', 'LEC', 'LTA N', 'LTA S', 'LCS', 'LCP', 'MSI', 'EWC', 'WLDs', 'FST'}
pnum = lambda p: tuple(int(x) for x in p.split('.'))


def load_tables():
    """-> {patch: (match, syn)}, entries keyed by ((lane_a,cid_a),(lane_b,cid_b)) -> (raw logit effect, n)."""
    tables = {}
    for pdir in sorted((p for p in SQ.iterdir() if p.is_dir()), key=lambda p: pnum(p.name)):
        match, syn, cid_of = {}, {}, {}
        for f in pdir.glob('*-vs-*.json'):
            j = json.loads(f.read_text()); s = j.get('stats')
            if not s or not j.get('counters'): continue
            a = (s['lane'], s['cid'])
            for r in j['counters']:
                if r['n'] <= 0: continue
                b = (s['vsLane'], r['cid']); v = .04*r['d2']
                for k, val in (((a, b), v), ((b, a), -v)):
                    if k in match:      # same-lane pairs are seen from both pages: average
                        match[k] = ((match[k][0]+val)/2, max(match[k][1], r['n']))
                    else:
                        match[k] = (val, r['n'])
        for f in pdir.glob('*-team.json'):
            j = json.loads(f.read_text()); lane, k = f.name.split('-')[:2]
            # own cid: read from any vs file of the same champion
            vs = next(iter(pdir.glob(f'{lane}-{k}-vs-*.json')), None)
            if vs is None or 'team' not in j: continue
            a = (lane, json.loads(vs.read_text())['stats']['cid']); h = j['team_h']
            for mate_lane, rows in j['team'].items():
                for r in rows:
                    r = dict(zip(h, r))
                    if r['n'] <= 0: continue
                    b = (mate_lane, r['id']); kk = tuple(sorted((a, b)))
                    val = .04*r['d2']
                    syn[kk] = ((syn[kk][0]+val)/2, max(syn[kk][1], r['n'])) if kk in syn else (val, r['n'])
        tables[pdir.name] = (match, syn)
    return tables


def pooled(tables, patches):
    out = []
    for kind in (0, 1):
        acc = defaultdict(lambda: [0., 0.])
        for p in patches:
            for k, (v, n) in tables[p][kind].items():
                acc[k][0] += v*n; acc[k][1] += n
        out.append({k: (s/n, n) for k, (s, n) in acc.items()})
    return out


def shrink_k(table):
    v = np.array([x[0] for x in table.values()]); n = np.array([x[1] for x in table.values()], float)
    w = n/n.sum(); tau2 = max(np.sum(w*v*v)-np.sum(w*4/n), 1e-5)     # weighted MoM; sampling var of a logit ~ 4/n
    return 4/tau2, tau2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import psycopg
    from psycopg.rows import dict_row
    from lol_ticker.config import PG_DSN
    with psycopg.connect(PG_DSN, row_factory=dict_row, options='-c default_transaction_read_only=on') as conn:
        games, _ = dcc.read_games(conn, int(datetime(2026, 9, 16, tzinfo=timezone.utc).timestamp()))
        names = {key(r['champion']) for r in conn.execute('SELECT DISTINCT champion FROM oe_picks')}
    games.sort(key=lambda g: (g['day'], g['id']))
    tables = load_tables(); have = sorted(tables, key=pnum)
    complete = [p for p in have if len(list((SQ/p).glob('*.json'))) >= 0.97*max(len(list((SQ/q).glob('*.json'))) for q in have)]
    print('patches scraped', have, 'complete', complete, flush=True)
    tables = {p: tables[p] for p in complete}; have = complete
    # champion key -> cid from the scraped pages
    cid = {}
    for p in have:
        for f in (SQ/p).glob('*-vs-*.json'):
            s = json.loads(f.read_text()).get('stats')
            if s: cid[f.name.split('-')[1]] = s['cid']

    conv = {}; ks = {}
    for p in have:
        prior = [q for q in have if pnum(q) < pnum(p)]
        conv[p] = dict(same=pooled(tables, [p]), upto=pooled(tables, prior+[p]), **({'prior': pooled(tables, prior)} if prior else {}))
    for name in ('same', 'upto', 'prior'):   # one k per convention/kind, from the largest pool
        ref = next((conv[p][name] for p in reversed(have) if name in conv[p]), None)
        if ref is None: continue
        ks[name] = [shrink_k(ref[0]), shrink_k(ref[1])]
    print('shrinkage k (matchup, synergy) and tau:', {n: [(round(k), round(t**.5, 4)) for k, t in v] for n, v in ks.items()}, flush=True)

    def sq_scores(g):
        p = '.'.join(str(int(x)) for x in g['patch'].split('.'))
        if p not in conv: return None
        try:
            sides = {s: [(OE2L[r], cid[key(g[s]['picks'][r])]) for r in OE2L] for s in ('blue', 'red')}
        except KeyError:
            return None
        out = {}
        for name, (match, syn) in conv[p].items():
            (km, _), (ksy, _) = ks[name]
            m = cov = 0.
            for a in sides['blue']:
                for b in sides['red']:
                    e = match.get((a, b))
                    if e: m += e[0]*e[1]/(e[1]+km); cov += 1
            s = 0.
            for sign, side in ((1, 'blue'), (-1, 'red')):
                for i in range(5):
                    for j in range(i+1, 5):
                        e = syn.get(tuple(sorted((sides[side][i], sides[side][j]))))
                        if e: s += sign*e[0]*e[1]/(e[1]+ksy); cov += 1
            out[name] = dict(matchup=m, synergy=s, coverage=cov/45)
        return out

    y = np.array([g['y'] for g in games]); days = np.array([g['day'] for g in games])
    sq = {i: v for i, g in enumerate(games) if days[i] >= CUTS[0] for v in [sq_scores(g)] if v}
    ev = sorted(sq)
    feats = [dcc.signed_draft(g) for g in games]; hist = dcc.history_features(games)
    E = np.array([hist[g['id']][0] for g in games]); P = np.array([hist[g['id']][3] for g in games])
    ours = {}
    for lo, hi in zip(CUTS[:-1], CUTS[1:]):
        tr = days < lo; te = [i for i in ev if lo <= days[i] < hi]
        if not te: continue
        cnt = Counter(k for f, keep in zip(feats, tr) if keep for k in f)
        vocab = sorted(k for k, n in cnt.items() if n >= 60 or not draft._is_interaction(k)); look = {k: j for j, k in enumerate(vocab)}
        ii, jj, vv = [], [], []
        for i in te:
            for k, v in feats[i].items():
                if k in look: ii.append(i); jj.append(look[k]); vv.append(v)
        rows_tr = np.flatnonzero(tr); pos = {i: n for n, i in enumerate(rows_tr)}
        ti, tj, tv = [], [], []
        for i in rows_tr:
            for k, v in feats[i].items():
                if k in look: ti.append(pos[i]); tj.append(look[k]); tv.append(v)
        Dtr = sparse.csr_matrix((tv, (ti, tj)), shape=(len(rows_tr), len(vocab)))
        scale = np.maximum(P[tr].std(axis=0), 1e-8)
        B = np.column_stack((np.ones(len(games)), E, P/scale))
        b0 = dcc.fit(sparse.csr_matrix(B[tr]), y[tr]); b1 = dcc.fit(sparse.hstack((sparse.csr_matrix(B[tr]), Dtr)).tocsr(), y[tr])
        nb = B.shape[1]; cats = np.array([category(k) for k in vocab])
        for i in te:
            contrib = defaultdict(float)
            for k, v in feats[i].items():
                if k in look: contrib[cats[look[k]]] += v*b1[nb+look[k]]
            ours[i] = dict(c=float(B[i]@b0), native=float(B[i]@b1[:nb]+sum(contrib.values())), pair=contrib['pair'], pick=contrib['pick'], fold=lo)
        print(lo, 'train', int(tr.sum()), 'eval', len(te), flush=True)

    ev = np.array([i for i in ev if i in ours]); Y = y[ev].astype(float); d = days[ev]; fold = np.array([ours[i]['fold'] for i in ev])
    c = np.array([ours[i]['c'] for i in ev]); nat = np.array([ours[i]['native'] for i in ev]); opair = np.array([ours[i]['pair'] for i in ev])
    league = np.array([games[i]['league'] for i in ev])
    keys = [(games[i]['day'], tuple(sorted((games[i]['blue_team'], games[i]['red_team'])))) for i in ev]
    uk = sorted(set(keys)); lk = {k: j for j, k in enumerate(uk)}; cl = np.array([lk[k] for k in keys]); rng = np.random.default_rng(920)

    def paired(delta, m):
        idx = np.unique(cl[m], return_inverse=True)[1]; s = np.bincount(idx, weights=delta[m]); n = np.bincount(idx)
        dr = rng.integers(0, len(s), (2000, len(s))); v = s[dr].sum(1)/n[dr].sum(1)
        return dict(delta=float(delta[m].mean()), ci95=[float(x) for x in np.quantile(v, [.025, .975])])

    def slope_stats(off, S, m):
        idx = np.flatnonzero(m); b = fit_scales(Y[idx], off[idx], S[idx]); members = defaultdict(list)
        for i in idx: members[cl[i]].append(i)
        mem = [np.array(v) for v in members.values()]; bs = []
        for _ in range(500):
            t = np.concatenate([mem[j] for j in rng.integers(0, len(mem), len(mem))]); bs.append(fit_scales(Y[t], off[t], S[t]))
        bs = np.array(bs)
        return [dict(slope=float(b[j]), ci95=[float(x) for x in np.quantile(bs[:, j], [.025, .975])], z=float(b[j]/bs[:, j].std())) for j in range(len(b))]

    report = dict(patches=have, shrinkage={n: [dict(k=float(k), tau=float(t**.5)) for k, t in v] for n, v in ks.items()}, conventions={})
    folds = sorted(set(fold))
    for name in ('prior', 'same', 'upto'):
        m = np.array([name in sq[i] for i in ev])
        if m.sum() < 200: continue
        g = lambda f: np.array([sq[i][name][f] if name in sq[i] else 0. for i in ev])
        sm, ss = g('matchup'), g('synergy'); s = sm+ss
        # walk-forward slope on top of our native model: fit on earlier folds, apply to this fold
        wf = np.full(len(ev), np.nan); used = {}
        for f in folds:
            past = m & (fold < f); now = m & (fold == f)
            if past.sum() < 400 or not now.any(): continue
            b = fit_scales(Y[past], nat[past], s[past, None])[0]; used[f] = float(b); wf[now] = b
        t = m & ~np.isnan(wf)
        r = dict(maps=int(m.sum()), coverage_mean=float(g('coverage')[m].mean()), score_sd=dict(total=float(s[m].std()), matchup=float(sm[m].std()), synergy=float(ss[m].std())),
                 corr_with_our_pair_terms=float(np.corrcoef(s[m], opair[m])[0, 1]),
                 logloss=dict(controls=float(ll(Y[m], expit(c[m])).mean()), ours_native=float(ll(Y[m], expit(nat[m])).mean())),
                 on_controls=dict(slope=slope_stats(c, s[:, None], m)[0], matchup_and_synergy=slope_stats(c, np.column_stack((sm, ss)), m),
                                  as_defined_slope1=paired(ll(Y, expit(c+s))-ll(Y, expit(c)), m)),
                 on_our_native_model=dict(slope=slope_stats(nat, s[:, None], m)[0],
                                          as_defined_slope1=paired(ll(Y, expit(nat+s))-ll(Y, expit(nat)), m),
                                          half_slope=paired(ll(Y, expit(nat+.5*s))-ll(Y, expit(nat)), m),
                                          walk_forward_slope=dict(maps=int(t.sum()), slopes_by_fold=used,
                                                                  **paired(ll(Y, expit(nat+np.nan_to_num(wf)*s))-ll(Y, expit(nat)), t)) if t.any() else None),
                 slices={})
        for sl, mm in (('major', np.isin(league, list(MAJOR))), ('minor', ~np.isin(league, list(MAJOR))), ('patch>=16.12', d >= '2026-06-17'), ('sep_4_on', d >= '2026-09-04')):
            mm = mm & m
            if mm.sum() >= 150:
                r['slices'][sl] = dict(maps=int(mm.sum()), slope_on_native=slope_stats(nat, s[:, None], mm)[0], slope1_on_native=paired(ll(Y, expit(nat+s))-ll(Y, expit(nat)), mm))
        report['conventions'][name] = r
    (OUT/'report.json').write_text(json.dumps(report, indent=1))
    (OUT/'scores.json').write_text(json.dumps([dict(game_id=games[i]['id'], day=games[i]['day'], league=games[i]['league'], patch=games[i]['patch'], y=int(y[i]),
                                                    c=ours[i]['c'], native=ours[i]['native'], our_pair=ours[i]['pair'], sq=sq[i]) for i in ev]))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
