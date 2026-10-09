"""Recover central order statistics from TGV's paired public medians.

For even n, m=(a+b)/2 and lift=(sigmoid(a)+sigmoid(b))/2-.5.
Writing a=m-d,b=m+d gives 2*lift=sinh(m)/(cosh(m)+cosh(d)).
This recovers two scores but does not identify which matches produced them.
"""
import json
import math
from pathlib import Path


def recover(m, lift):
    if abs(m) < 1e-12 or abs(lift) < 1e-12:
        return None  # At zero the spread is unidentifiable.
    cosh_d = math.sinh(m) / (2 * lift) - math.cosh(m)
    if cosh_d < 1 - 1e-10:
        raise ValueError('Inconsistent median pair')
    d = math.acosh(max(1, cosh_d))
    return [m-d, m+d]


def main():
    root = Path('data/tgv/20260916')
    source = json.loads((root/'team-draft-gaps.v2.json').read_text())
    records = []
    for t in source['teams']:
        n = t['gamesScored']
        if not n:
            continue
        m, lift = t['median'], t['winRateLift']
        scores = recover(m, lift) if n % 2 == 0 else [m]
        if scores is None:
            continue
        rebuilt = sum(1/(1+math.exp(-v))-.5 for v in scores)/len(scores)
        records.append(dict(teamId=t['teamId'], gamesScored=n,
            centralScores=scores, probabilityReconstructionError=rebuilt-lift))
    result = dict(modelVersion=source['modelVersion'], method=source['method'],
        interpretation='Central order statistics only; game identities and remaining scores unknown. Numeric inversion becomes ill-conditioned near zero.',
        teamsRecovered=len(records), scoresRecovered=sum(len(r['centralScores']) for r in records),
        maxProbabilityReconstructionError=max(abs(r['probabilityReconstructionError']) for r in records), teams=records)
    (root/'gap-central-score-constraints.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='teams'},indent=2))


if __name__ == '__main__':
    main()
