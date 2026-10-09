"""Bound-free algebraic test of independent champion composition effects.

Terminal observations eliminate red response effects; row maxima identify blue
mid effects and column maxima are then held-out checks, without optimization.
"""
import json,itertools
from pathlib import Path
from research.tgv_reconstruction import PublicDraftModel


def main():
 root=Path('data/tgv/20260916');model=PublicDraftModel(root);c=json.loads((root/'challenge.json').read_text())['challenge'];roles=['top','jgl','mid','bot','sup']
 modifiers={s:{(v['role'],v['champion_id']):v['value'] for v in c['modifiers'][s]} for s in ['blue','red']}
 obs={d['index']:{v['championId']:v['blueValue'] for v in d['choices']} for d in c['puzzle']['decisions']}
 mids=[1,3,34,134];bots=[81,498,523];reds=[8,163,804]
 def known(m,b,r):
  blue=[2,904,m,b,902];red=[799,78,126,r,53]
  parts=model.components(c['patch'],blue,red,[modifiers['blue'][role,cid] for role,cid in zip(roles,blue)],[modifiers['red'][role,cid] for role,cid in zip(roles,red)])
  return c['puzzle']['finalEvaluation']['sideBiasScore']+sum(parts.values())
 K={(m,b,r):known(m,b,r) for m,b,r in itertools.product(mids,bots,reds)}
 C={r:obs[19][r]-K[1,498,r] for r in reds}
 Q={(m,b):min(K[m,b,r]+C[r] for r in reds) for m,b in itertools.product(mids,bots)}
 bot_effect={b:obs[18][b]-Q[1,b] for b in bots}
 mid_effect={m:obs[17][m]-max(Q[m,b]+bot_effect[b] for b in bots) for m in mids}
 checks=[]
 for b in bots:
  pred=max(Q[m,b]+bot_effect[b]+mid_effect[m] for m in mids)
  checks.append(dict(blueBot=b,predicted=pred,published=obs[17][b],error=pred-obs[17][b]))
 result=dict(hypothesis='Composition difference from canonical blue team is independent mid-champion effect plus independent bot-champion effect, invariant to red final pick.',
  terminalComposition=C,blueBotCompositionDifferences=bot_effect,blueMidCompositionDifferences=mid_effect,heldOutColumnMaxima=checks,
  maxAbsoluteError=max(abs(v['error']) for v in checks),method='Exact algebraic elimination; no coefficient bounds, optimizer, or fit penalties.',
  conclusion='A nonzero discrepancy beyond numerical error refutes this additive hypothesis on the observed subgame; it does not identify a unique replacement.')
 (root/'additivity-identity-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
