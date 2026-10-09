"""Build the last-eight-action inverse problem, including contested Viktor.

Only offline saved observations are used. Bans and picks remove champions
globally, even when a champion belongs to both sides' pools.
"""
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from lol_ticker.tgv_reconstruction import PublicDraftModel


def main():
    root = Path('data/tgv/20260916')
    c = json.loads((root / 'challenge.json').read_text())['challenge']
    model = PublicDraftModel(root)
    sequence = c['puzzle']['sequence']
    used = set(sequence[:12])
    groups = [sorted(set(c['pools'][side][role]) - used)
              for side, role in [('blue', 'mid'), ('blue', 'bot'),
                                 ('red', 'top'), ('red', 'bot')]]
    keys = [k for k in itertools.product(*groups) if len(set(k)) == 4]
    index = {k: i for i, k in enumerate(keys)}
    roles = ['top', 'jgl', 'mid', 'bot', 'sup']
    mods = {s: {(v['role'], v['champion_id']): v['value']
                for v in c['modifiers'][s]} for s in ['blue', 'red']}
    known = []
    for mid, bot, top, rbot in keys:
        blue, red = [2, 904, mid, bot, 902], [top, 78, 126, rbot, 53]
        for side, picks in [('blue', blue), ('red', red)]:
            assignments = sum(all(cid in c['pools'][side][role]
                                  for role, cid in zip(roles, perm))
                              for perm in itertools.permutations(picks))
            assert assignments == 1
        parts = model.components(c['patch'], blue, red,
                                 [mods['blue'][r, cid] for r, cid in zip(roles, blue)],
                                 [mods['red'][r, cid] for r, cid in zip(roles, red)])
        known.append(c['puzzle']['finalEvaluation']['sideBiasScore'] + sum(parts.values()))

    champions = sorted(set(itertools.chain.from_iterable(groups)))
    bits = {cid: 1 << i for i, cid in enumerate(champions)}
    nodes, cache = [], {}

    def node(maximize, children):
        children = tuple(sorted(set(children)))
        assert children
        if len(children) == 1:
            return children[0]
        key = maximize, children
        if key not in cache:
            cache[key] = len(keys) + len(nodes)
            nodes.append(dict(maximize=maximize, children=children))
        return cache[key]

    def slots(turn):
        # Bans target the opposing pools; picks use the acting side's pools.
        return (0, 1) if turn in (12, 14, 17, 18) else (2, 3)

    def legal(turn, unavailable, picked):
        choices = sorted({cid for slot in slots(turn)
                          if not picked[slot] for cid in groups[slot]
                          if not unavailable & bits[cid]})
        return [-1] + choices if turn < 16 else choices

    def apply(turn, cid, unavailable, picked):
        if cid == -1:
            assert turn < 16
            return unavailable, picked
        assert not unavailable & bits[cid]
        unavailable |= bits[cid]
        if turn >= 16:
            candidates = [s for s in slots(turn) if cid in groups[s] and not picked[s]]
            assert len(candidates) == 1
            pp = list(picked)
            pp[candidates[0]] = cid
            picked = tuple(pp)
        return unavailable, picked

    @lru_cache(None)
    def build(turn, unavailable, picked):
        if turn == 20:
            return index[picked]
        return node(turn in (13, 15, 17, 18),
                    [build(turn + 1, *apply(turn, cid, unavailable, picked))
                     for cid in legal(turn, unavailable, picked)])

    observations = []
    for decision in c['puzzle']['decisions']:
        turn = decision['index']
        if turn < 12:
            continue
        unavailable, picked = 0, (0, 0, 0, 0)
        for earlier in range(12, turn):
            unavailable, picked = apply(earlier, sequence[earlier], unavailable, picked)
        assert {v['championId'] for v in decision['choices']} == set(legal(turn, unavailable, picked))
        for choice in decision['choices']:
            assert not choice['forcedWinner']
            expr = build(turn + 1, *apply(turn, choice['championId'], unavailable, picked))
            observations.append(dict(index=turn, championId=choice['championId'],
                                     target=choice['blueValue'], node=expr))
        print('Built action', turn, 'states', build.cache_info().currsize,
              'nodes', len(nodes), flush=True)

    # Validate overlap handling and the old 320-leaf restriction independently.
    old = np.load(root / 'ban-design.npz')
    known_lookup = dict(zip(keys, known))
    assert max(abs(known_lookup[tuple(k)] - v)
               for k, v in zip(old['keys'], old['known'])) < 1e-12
    assert sum(k[0] == 112 and k[3] == 112 for k in keys) == 0
    np.savez_compressed(root / 'earlier-design.npz', keys=np.array(keys), known=known)
    (root / 'earlier-design-graph.json').write_text(json.dumps(
        dict(groups=groups, nodes=nodes, observations=observations,
             firstAction=12, cachedStates=build.cache_info().currsize,
             contestedChampions=sorted(set(groups[0]) & set(groups[3]))), indent=2))
    print('Exported', len(keys), 'terminal drafts and', len(observations), 'observations')


if __name__ == '__main__':
    main()
