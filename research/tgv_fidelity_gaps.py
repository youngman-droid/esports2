"""End-to-end fidelity check of the TGV reconstruction on real professional drafts.

TGV publishes, per team, the median side-neutral draft score of its last ten maps
(original patch, actual player comfort). We rebuild that median from local OE drafts
using only recovered factors (unary + matchup + synergy [+ comfort curve]); whatever
is left is the unrecovered composition term plus source mismatches. Read-only.
"""
from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

from lol_ticker.tgv_reconstruction import PublicDraftModel

ROOT = Path('data/tgv/20260916')
OE2TGV = dict(top='top', jng='jungle', mid='middle', bot='bottom', sup='support')
A, K = 0.07896877604172117, 1.036248079221949


def norm(s):
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower())


def load_games():
    import psycopg
    from psycopg.rows import dict_row
    from lol_ticker.config import PG_DSN
    with psycopg.connect(PG_DSN, row_factory=dict_row, options='-c default_transaction_read_only=on') as c:
        games = c.execute('SELECT * FROM oe_games WHERE date_utc IS NOT NULL ORDER BY date_utc, game_id').fetchall()
        players, picks = defaultdict(dict), defaultdict(dict)
        for r in c.execute('SELECT * FROM oe_players'):
            players[(r['game_id'], r['team'])][r['position']] = r['player_id'] or r['player']
        for r in c.execute('SELECT * FROM oe_picks'):
            picks[(r['game_id'], r['team'])][r['position']] = r['champion']
    return games, players, picks


def main():
    model = PublicDraftModel(ROOT)
    cid = {norm(c['name']): c['cid'] for c in model.catalog['champions']}
    gaps = json.loads((ROOT/'team-draft-gaps.v2.json').read_text())
    ranks = json.loads((ROOT/'team-rankings.v3.json').read_text())
    asof = np.datetime64(gaps['asOf'][:10]).astype('datetime64[s]').astype(int)
    games, players, picks = load_games()
    final = defaultdict(int)       # export-time counts
    for g in games:
        if g['date_utc'] >= asof: continue
        for t in (g['blue_team'], g['red_team']):
            for pos, p in players[(g['game_id'], t)].items():
                ch = picks[(g['game_id'], t)].get(pos)
                if ch: final[(p, pos, ch)] += 1
    running = defaultdict(int)
    per_team = defaultdict(list)
    for g in games:
        if g['date_utc'] >= asof: break
        sides = []
        ok = True
        for t in (g['blue_team'], g['red_team']):
            pk, pl = picks[(g['game_id'], t)], players[(g['game_id'], t)]
            ok &= all(pk.get(r) and norm(pk[r]) in cid for r in OE2TGV)
            sides.append((t, pk, pl))
        row = None
        if ok and g['patch'] in model.strength:
            blue = [cid[norm(sides[0][1][r])] for r in OE2TGV]
            red = [cid[norm(sides[1][1][r])] for r in OE2TGV]
            try:
                parts = model.components(g['patch'], blue, red)
                hist = [sum(A*n/(n+K) for n in (running[(pl.get(r), r, pk[r])] for r in OE2TGV)) for _, pk, pl in sides]
                # export-time count includes this game itself
                exp = [sum(A*n/(n+K) for n in (final[(pl.get(r), r, pk[r])] for r in OE2TGV)) for _, pk, pl in sides]
                base = parts['strength'] + parts['matchup'] + parts['synergy']
                row = dict(base=base, unary=parts['strength'], hist=base+hist[0]-hist[1], exp=base+exp[0]-exp[1])
            except KeyError:
                row = None
        for sign, (t, pk, pl) in zip((1, -1), sides):
            per_team[t].append(None if row is None else {k: sign*v for k, v in row.items()})
        for t, pk, pl in sides:
            for r in OE2TGV:
                if pk.get(r) and pl.get(r): running[(pl[r], r, pk[r])] += 1
    # team id -> name via rankings order (gaps list is keyed by id; rankings by name). Try id field first.
    names = {}
    for t in ranks['teams']:
        tid = t.get('teamId') or t.get('id')
        if tid: names[tid] = t['name']
    out = []
    for t in gaps['teams']:
        name = names.get(t['teamId'])
        if name is None or name not in per_team: continue
        last = per_team[name][-gaps['maxGames']:]
        scored = [x for x in last if x is not None]
        if not scored: continue
        out.append(dict(team=name, published=t['median'], considered=t['gamesConsidered'], scored=t['gamesScored'],
                        local_considered=len(last), local_scored=len(scored),
                        **{k: float(np.median([x[k] for x in scored])) for k in ('unary', 'base', 'hist', 'exp')}))
    aligned = [r for r in out if r['scored'] == r['local_scored'] and r['considered'] == r['local_considered']]
    rep = dict(teams_in_gaps=len(gaps['teams']), id_mapped=len(names), matched=len(out), count_aligned=len(aligned))
    for label, rows in (('all_matched', out), ('count_aligned', aligned), ('aligned_10of10', [r for r in aligned if r['scored'] == 10])):
        pub = np.array([r['published'] for r in rows])
        rep[label] = dict(n=len(rows), published_sd=float(pub.std()))
        for k in ('unary', 'base', 'hist', 'exp'):
            v = np.array([r[k] for r in rows]); e = v - pub
            rep[label][k] = dict(pearson=float(np.corrcoef(v, pub)[0, 1]), rmse=float(np.sqrt(np.mean(e**2))),
                                 median_abs=float(np.median(abs(e))), p90_abs=float(np.quantile(abs(e), .9)),
                                 r2=float(1-np.sum(e**2)/np.sum((pub-pub.mean())**2)))
    (ROOT/'fidelity-draft-gaps.json').write_text(json.dumps(dict(report=rep, rows=out), indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
