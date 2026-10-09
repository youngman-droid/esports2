"""Compare independently versioned public coefficient exports; never merge them."""
import json
import math
from pathlib import Path


def regression(pairs):
    n=len(pairs)
    sx=sum(x for x,y in pairs); sy=sum(y for x,y in pairs)
    xx=sum(x*x for x,y in pairs)-sx*sx/n
    yy=sum(y*y for x,y in pairs)-sy*sy/n
    xy=sum(x*y for x,y in pairs)-sx*sy/n
    slope=xy/xx
    intercept=sy/n-slope*sx/n
    errors=[y-slope*x-intercept for x,y in pairs]
    return dict(cells=n,oldToNewSlope=slope,intercept=intercept,
        correlation=xy/math.sqrt(xx*yy),rmse=math.sqrt(sum(e*e for e in errors)/n),
        maxAbsoluteResidual=max(abs(e) for e in errors))


def main():
    root=Path('data/tgv/20260916')
    old=json.loads((root/'current-surface/fallback-catalog.json').read_text())
    new=json.loads((root/'current-model-catalog.json').read_text())
    inventory=json.loads((root/'legacy-static/relation-inventory.json').read_text())
    if inventory['errors'] or len(inventory['assets'])!=len(old['roles'])*len(old['champions']):
        raise ValueError('Legacy relation inventory incomplete')
    pairs={'matchups':[],'synergies':[]}
    schemas=set()
    for entry in inventory['assets']:
        a=json.loads((root/'legacy-static'/entry['path']).read_text())
        b=json.loads((root/entry['path']).read_text())
        schemas.add(tuple(sorted(a)))
        for role,values in a['relations'].items():
            for kind,rows in values.items():
                lookup={(cid,r):v for cid,r,v in b['relations'][role][kind]}
                for cid,r,v in rows:
                    if (cid,r) in lookup:pairs[kind].append((v,lookup[cid,r]))
    unary=[]
    for patch in old['patches']:
        if patch not in new['patches']:continue
        for role,rows in old['insightsByPatch'][patch]['roles'].items():
            lookup={x['champion']['cid']:x['strength'] for x in new['insightsByPatch'][patch]['roles'][role]}
            unary.extend((x['strength'],lookup[x['champion']['cid']]) for x in rows)
    result=dict(oldVersion=old['modelVersion'],newVersion=new['modelVersion'],
        oldScoreSpace=old['scoreSpace'],newScoreSpace=new['scoreSpace'],
        oldOnlyPatches=sorted(set(old['patches'])-set(new['patches'])),
        relationSchemas=[list(x) for x in schemas],
        regression={**{k:regression(v) for k,v in pairs.items()},'unary':regression(unary)},
        limitation='Descriptive in-sample affine fits, not training recovery or predictive validation. Correlated symmetric/antisymmetric cells are counted as exported cells, not independent parameters.')
    (root/'legacy-static/comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
