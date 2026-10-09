"""Independent ordered-move replay of a fitted terminal table."""
import argparse,json
from functools import lru_cache
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('file');args=p.parse_args();path=Path(args.file);fit=json.loads(path.read_text());root=path.parent;c=json.loads((root/'challenge.json').read_text())['challenge']
 table={tuple(r['picks']):r['known']+r['composition'] for r in fit['terminalValues']}
 groups=tuple(tuple(sorted({k[i] for k in table})) for i in range(4));first_turn=12 if fit.get('designPrefix')=='earlier' else 14
 def slots(turn):return (0,1) if turn in (12,14,17,18) else (2,3)
 def apply(turn,cid,banned,picked):
  if turn<16:return tuple(sorted(set(banned)|({cid} if cid!=-1 else set()))),picked
  pp=list(picked);candidates=[i for i in slots(turn) if cid in groups[i] and not picked[i]];assert len(candidates)==1;slot=candidates[0];assert cid not in picked and cid not in banned;pp[slot]=cid;return banned,tuple(pp)
 @lru_cache(None)
 def solve(turn,banned,picked):
  if turn==20:return table[picked]
  legal=sorted({x for i in slots(turn) if not picked[i] for x in groups[i] if x not in banned and x not in picked})
  if turn<16:legal=[-1]+legal
  assert legal
  values=[solve(turn+1,*apply(turn,x,banned,picked)) for x in legal]
  return (min if turn in (12,14,16,19) else max)(values)
 results=[]
 for row in fit['observations']:
  idx=row['index'];banned=();picked=(0,0,0,0)
  for turn in range(first_turn,idx):banned,picked=apply(turn,c['puzzle']['sequence'][turn],banned,picked)
  banned,picked=apply(idx,row['championId'],banned,picked)
  value=solve(idx+1,banned,picked)
  results.append(dict(index=idx,championId=row['championId'],replayed=value,graphPrediction=row['prediction'],difference=value-row['prediction']))
 out=dict(file=str(path),observations=len(results),maxReplayDifference=max(abs(x['difference']) for x in results),memoizedStates=solve.cache_info().currsize,results=results)
 assert out['maxReplayDifference']<1e-10
 path.with_name(path.stem+'-ordered-replay.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k!='results'})


if __name__=='__main__':main()
