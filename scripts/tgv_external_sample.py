"""Verify independent raw composition measurements in public GPTilt samples.

This produces historical measurement targets, not pre-draft predictions or a fit.
"""
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path('data/tgv/independent-sources')
    matches = {x['row']['matchId']: x['row'] for x in json.loads((root/'matches-first-rows.json').read_text())['rows']}
    grouped = defaultdict(list)
    for item in json.loads((root/'participants-first-rows.json').read_text())['rows']:
        grouped[item['row']['matchId']].append(item['row'])
    rows, rejected, differences = [], [], []
    required = ['goldEarned', 'physicalDamageDealtToChampions', 'magicDamageDealtToChampions',
                'trueDamageDealtToChampions', 'totalDamageDealtToChampions',
                'totalDamageTaken', 'damageSelfMitigated', 'deaths']
    roles = {'TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY'}
    for mid, participants in grouped.items():
        if mid not in matches or len(participants) != 10:
            rejected.append(dict(matchId=mid, reason='Incomplete joined match'))
            continue
        match = matches[mid]
        if any({p['teamPosition'] for p in participants if p['teamId']==side} != roles
               or sum(p['teamId']==side for p in participants) != 5 for side in [100,200]):
            rejected.append(dict(matchId=mid, reason='Incomplete role assignment'))
            continue
        if not all(all(isinstance(p.get(k),(int,float)) and p[k]>=0 for k in required) for p in participants):
            rejected.append(dict(matchId=mid, reason='Missing or invalid measurement'))
            continue
        minutes = match['gameDuration']/60
        assert minutes > 0
        gold = {side:sum(p['goldEarned'] for p in participants if p['teamId']==side) for side in [100,200]}
        assert all(gold.values())
        for p in participants:
            physical, magic, true = [p[k]/minutes for k in required[1:4]]
            differences.append(p['physicalDamageDealtToChampions']+p['magicDamageDealtToChampions']+p['trueDamageDealtToChampions']-p['totalDamageDealtToChampions'])
            rows.append(dict(matchId=mid, championId=p['championId'], role=p['teamPosition'], teamId=p['teamId'],
                             patch=match.get('patch') or '.'.join(match['gameVersion'].split('.')[:2]),
                             startTime=datetime.fromtimestamp(match['gameStartTimestamp']/1000,tz=timezone.utc).isoformat(),
                             minutes=minutes, goldShare=p['goldEarned']/gold[p['teamId']],
                             physicalDpm=physical, magicDpm=magic, trueDpm=true,
                             totalDpm=physical+magic+true,
                             tankinessCandidate=(p['totalDamageTaken']+p['damageSelfMitigated'])/minutes/(p['deaths']+1)))
        for side in [100,200]:
            assert abs(sum(r['goldShare'] for r in rows if r['matchId']==mid and r['teamId']==side)-1)<1e-12
    output = dict(source='https://huggingface.co/datasets/gptilt/lol-basic-matches-challenger-10k',
                  sourceScope='Public viewer sample, region_americas; not the full dataset',
                  verifiedGames=len({r['matchId'] for r in rows}), participantRows=len(rows),
                  patches=sorted({r['patch'] for r in rows}), rejected=rejected,
                  maxDamageTypeSumDiscrepancy=max(map(abs,differences),default=None),
                  conventions=['Gold share uses total goldEarned, not Oracle earnedgoldshare.',
                               'Tankiness uses deaths+1 as a candidate life denominator; exact TGV convention remains unverified.',
                               'Damage rates use damage to champions, not damage to all targets.',
                               'These end-game measurements must be aggregated from earlier games to produce any pre-draft feature.'],
                  rows=rows)
    (root/'composition-sample.json').write_text(json.dumps(output,indent=2))
    print({k:v for k,v in output.items() if k not in ['rows','conventions','rejected']})


if __name__ == '__main__':
    main()
