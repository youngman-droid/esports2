"""Recover terminal composition values and minimax inequalities from public choices."""
import itertools
import json
from pathlib import Path
from lol_ticker.tgv_reconstruction import PublicDraftModel


def main():
    root=Path('data/tgv/20260916');model=PublicDraftModel(root)
    c=json.loads((root/'challenge.json').read_text())['challenge']
    roles=['top','jgl','mid','bot','sup'];seq=c['puzzle']['sequence'];actions=c['rules']['action_sequence']
    modifiers={side:{(x['role'],x['champion_id']):x['value'] for x in c['modifiers'][side]} for side in ['blue','red']}
    side_bias=c['puzzle']['finalEvaluation']['sideBiasScore']
    def assignments(sequence,side):
        picked=[cid for cid,act in zip(sequence,actions) if act==side+'_pick']
        return [list(p) for p in itertools.permutations(picked) if all(cid in c['pools'][side][role] for role,cid in zip(roles,p))]
    def known(sequence):
        b=assignments(sequence,'blue');r=assignments(sequence,'red')
        if len(b)>1 or len(r)>1:raise ValueError('Mixed role strategies require a separate game-value solver')
        if not b or not r:return None
        bc=[modifiers['blue'][role,cid] for role,cid in zip(roles,b[0])]
        rc=[modifiers['red'][role,cid] for role,cid in zip(roles,r[0])]
        parts=model.components(c['patch'],b[0],r[0],bc,rc)
        return dict(blue=b[0],red=r[0],knownLogit=side_bias+sum(parts.values()),components=parts)
    last=c['puzzle']['decisions'][-1];penultimate=c['puzzle']['decisions'][-2]
    assert last['index']==19 and last['side']=='red' and last['type']=='pick'
    exact=[]
    for choice in last['choices']:
        terminal=seq[:19]+[choice['championId']];k=known(terminal)
        if not k or choice['forcedWinner']:continue
        exact.append(dict(finalPick=choice['championId'],targetLogit=choice['blueValue'],
            inferredComposition=choice['blueValue']-k['knownLogit'],**k))
    bounds=[]
    assert penultimate['index']==18 and penultimate['side']=='blue'
    for choice in penultimate['choices']:
        if choice['forcedWinner']:continue
        prefix=seq[:18]+[choice['championId']]
        available=sorted(set(cid for pool in c['pools']['red'].values() for cid in pool)-set(prefix))
        leaves=[]
        for cid in available:
            k=known(prefix+[cid])
            if k:leaves.append(dict(redFinalPick=cid,compositionLowerBound=choice['blueValue']-k['knownLogit'],**k))
        if leaves:
            bounds.append(dict(bluePenultimatePick=choice['championId'],minimaxValue=choice['blueValue'],
                relation='Every legal leaf has composition >= lower bound; at least one is equal, up to engine numeric error.',leaves=leaves))
    result=dict(patch=c['patch'],modelVersion=model.version,terminalCompositions=exact,penultimateConstraints=bounds,
        numericCaveat='Targets are search-engine float outputs. Reference engine/review difference is about 1.12e-8; recovered composition is approximate at that precision.',
        inference='Full pick sets and unique legal role assignments were enumerated, not assumed. All non-composition factors and current-patch side bias are independently available.')
    (root/'terminal-composition-constraints.json').write_text(json.dumps(result,indent=2))
    print('Recovered terminal compositions',[(x['finalPick'],x['inferredComposition']) for x in exact])
    print('Minimax rows',len(bounds),'leaf inequalities',sum(len(x['leaves']) for x in bounds))


if __name__=='__main__':main()
