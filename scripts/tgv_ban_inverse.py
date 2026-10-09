"""Fit an offline structured composition surrogate using bans and late picks.

A cached min/max expression graph preserves the move order. Analytic piecewise
Jacobians use the selected terminal leaf. Multi-start least squares is not a
certificate of global optimality.
"""
import itertools,json,time,argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,linprog
from lol_ticker.tgv_reconstruction import PublicDraftModel


def main():
 p=argparse.ArgumentParser();p.add_argument('--starts',type=int,default=8);p.add_argument('--holdout',default='');p.add_argument('--cross',choices=['bottop','all'],default='bottop');p.add_argument('--polish',action='store_true');p.add_argument('--polish-from');p.add_argument('--seed',type=int,default=2416);p.add_argument('--initial-from');p.add_argument('--export-design',action='store_true');args=p.parse_args()
 root=Path('data/tgv/20260916');model=PublicDraftModel(root);c=json.loads((root/'challenge.json').read_text())['challenge'];roles=['top','jgl','mid','bot','sup'];used=set(c['puzzle']['sequence'][:14])
 groups=[sorted(set(c['pools'][side][role])-used) for side,role in [('blue','mid'),('blue','bot'),('red','top'),('red','bot')]]
 assert groups==[[1,3,34,134],[81,110,498,523],[67,82,516,799,897],[8,115,163,804]]
 mods={s:{(v['role'],v['champion_id']):v['value'] for v in c['modifiers'][s]} for s in ['blue','red']}
 bluepairs=list(itertools.product(*groups[:2]));redpairs=list(itertools.product(*groups[2:]));keys=list(itertools.product(*groups));index={key:i for i,key in enumerate(keys)}
 cross_axes=[(1,2)] if args.cross=='bottop' else [(0,2),(0,3),(1,2),(1,3)]
 names=['constant']+[f'blue:{a}:{b}' for a,b in bluepairs[1:]]+[f'red:{a}:{b}' for a,b in redpairs[1:]]+[f'cross:{a}:{b}:{u}:{v}' for a,b in cross_axes for u,v in itertools.product(groups[a][1:],groups[b][1:])];ni={v:i for i,v in enumerate(names)}
 X=np.zeros((len(keys),len(names)));K=[]
 for i,(mid,bot,top,rbot) in enumerate(keys):
  blue=[2,904,mid,bot,902];red=[top,78,126,rbot,53]
  for side,picks in [('blue',blue),('red',red)]:
   assert sum(all(cid in c['pools'][side][role] for role,cid in zip(roles,perm)) for perm in itertools.permutations(picks))==1
  parts=model.components(c['patch'],blue,red,[mods['blue'][r,cid] for r,cid in zip(roles,blue)],[mods['red'][r,cid] for r,cid in zip(roles,red)])
  K.append(c['puzzle']['finalEvaluation']['sideBiasScore']+sum(parts.values()));X[i,0]=1
  for name,sign in [(f'blue:{mid}:{bot}',1),(f'red:{top}:{rbot}',-1)]:
   if name in ni:X[i,ni[name]]=sign
  key=(mid,bot,top,rbot)
  for a,b in cross_axes:
   name=f'cross:{a}:{b}:{key[a]}:{key[b]}'
   if name in ni:X[i,ni[name]]=1
 K=np.array(K);nodes=[];cache={}
 def node(maximize,children):
  children=tuple(sorted(set(children)))
  assert children
  if len(children)==1:return children[0]
  key=(maximize,children)
  if key not in cache:cache[key]=len(keys)+len(nodes);nodes.append((maximize,np.array(children)))
  return cache[key]
 def redlast(first,mid,bot,redban):
  return node(False,[index[mid,bot,t,r] for t,r in redpairs if redban not in (t,r) and first in (t,r)])
 def bluepair(first,blueban,redban,bluefirst=None):
  return node(True,[redlast(first,m,b,redban) for m,b in bluepairs if blueban not in (m,b) and (bluefirst is None or bluefirst in (m,b))])
 def redfirst(blueban,redban):
  return node(False,[bluepair(f,blueban,redban) for f in groups[2]+groups[3] if f!=redban])
 def blueban(removed):
  return node(True,[redfirst(removed,r) for r in [-1]+groups[2]+groups[3]])
 obs=[]
 for d in c['puzzle']['decisions']:
  idx=d['index']
  if idx<14:continue
  for v in d['choices']:
   cid=v['championId'];assert not v['forcedWinner']
   if idx==14:expr=blueban(cid)
   elif idx==15:expr=redfirst(110,cid)
   elif idx==16:expr=bluepair(cid,110,115)
   elif idx==17:expr=bluepair(799,110,115,cid)
   elif idx==18:expr=redlast(799,1,cid,115)
   else:expr=index[1,498,799,cid]
   obs.append(dict(index=idx,championId=cid,target=v['blueValue'],node=expr))
 held={tuple(map(int,v.split(':'))) for v in args.holdout.split(',') if v};train=np.array([(o['index'],o['championId']) not in held for o in obs]);target=np.array([o['target'] for o in obs]);roots=np.array([o['node'] for o in obs]);N=len(keys)+len(nodes)
 if args.export_design:
  np.savez_compressed(root/'ban-design.npz',known=K,keys=np.array(keys))
  (root/'ban-design-graph.json').write_text(json.dumps(dict(groups=groups,nodes=[dict(maximize=maximum,children=children.tolist()) for maximum,children in nodes],observations=obs),indent=2))
  print('Exported',len(keys),'terminal drafts and',len(nodes),'minimax nodes');return
 def evaluate(theta):
  values=np.empty(N);active=np.empty(N,dtype=int);values[:len(keys)]=K+np.einsum('ni,i->n',X,theta);active[:len(keys)]=np.arange(len(keys))
  for j,(maximum,children) in enumerate(nodes,len(keys)):
   chosen=children[np.argmax(values[children]) if maximum else np.argmin(values[children])];values[j]=values[chosen];active[j]=active[chosen]
  return values[roots],X[active[roots]],active[roots]
 last=[None,None]
 def cached(theta):
  if last[0] is None or not np.array_equal(last[0],theta):last[:]=[theta.copy(),evaluate(theta)]
  return last[1]
 rng=np.random.default_rng(args.seed);runs=[];best=None;start=time.time()
 if args.polish_from:
  saved=json.loads(Path(args.polish_from).read_text());theta=np.array([saved['coefficients'][name] for name in names]);pred,_,active=evaluate(theta);best=(float(np.max(np.abs((pred-target)[train]))),theta,pred,active)
 for k in range(0 if args.polish_from else args.starts):
  initial=np.zeros(len(names)) if k==0 else rng.normal(0,.08,len(names))
  if k==0 and args.initial_from:
   saved=json.loads(Path(args.initial_from).read_text())['coefficients'];initial=np.array([saved.get(name,0) for name in names])
  result=least_squares(lambda t:(cached(t)[0]-target)[train],initial,jac=lambda t:cached(t)[1][train],bounds=(-1,1),max_nfev=1500,ftol=1e-12,xtol=1e-12,gtol=1e-12)
  pred,_,active=evaluate(result.x);error=pred-target;score=float(np.max(np.abs(error[train])));runs.append(dict(start=k,maxTrainError=score,nfev=result.nfev))
  if best is None or score<best[0]:best=(score,result.x.copy(),pred.copy(),active.copy())
  print('Start',k,'training max error',score,flush=True)
 score,theta,pred,active=best
 polish=None
 if args.polish:
  before=float(np.sum(np.abs(theta)));values=np.empty(N);selected=np.empty(N,dtype=int);values[:len(keys)]=K+np.einsum('ni,i->n',X,theta);selected[:len(keys)]=np.arange(len(keys));A=[];rhs=[];P=len(names)
  for j,(maximum,children) in enumerate(nodes,len(keys)):
   winner=children[np.argmax(values[children]) if maximum else np.argmin(values[children])];values[j]=values[winner];selected[j]=selected[winner];w=selected[winner]
   for child in children:
    other=selected[child]
    if w==other:continue
    sign=-1 if maximum else 1;A.append(np.r_[sign*(X[w]-X[other]),np.zeros(P)]);rhs.append(sign*(K[other]-K[w])+1e-10)
  tolerance=max(score,1e-8)
  for i,node_id in enumerate(roots):
   if not train[i]:continue
   leaf=selected[node_id]
   A.extend([np.r_[X[leaf],np.zeros(P)],np.r_[-X[leaf],np.zeros(P)]]);rhs.extend([target[i]-K[leaf]+tolerance,-target[i]+K[leaf]+tolerance])
  for i in range(P):
   row=np.zeros(2*P);row[i]=1;row[P+i]=-1;A.append(row);rhs.append(0);row=row.copy();row[i]=-1;A.append(row);rhs.append(0)
  lp=linprog(np.r_[np.zeros(P),np.ones(P)],A_ub=np.array(A),b_ub=rhs,bounds=[(-1,1)]*P+[(0,1)]*P,method='highs')
  polish=dict(status=int(lp.status),message=lp.message,beforeL1=before,scope='Minimum L1 within the current min/max selection region, not across all strategies.')
  if lp.success:
   theta=lp.x[:P];pred,_,active=evaluate(theta);score=float(np.max(np.abs((pred-target)[train])));polish.update(afterL1=float(np.sum(np.abs(theta))),nonzeroCoefficients=int(np.sum(np.abs(theta)>1e-7)),replayTrainMaxError=score)
 for i,o in enumerate(obs):o.update(prediction=float(pred[i]),error=float(pred[i]-target[i]),heldout=not bool(train[i]),activeTerminal=list(keys[active[i]]));o.pop('node')
 out=dict(seed=args.seed,initialFrom=args.initial_from,polish=polish,method='Multi-start local piecewise-linear least squares; no global optimality claim.',hypothesis='Blue-team minus red-team composition plus cross-role interaction tables',crossAxes=cross_axes,terminalDrafts=len(keys),compositionParameters=len(names),minimaxNodes=len(nodes),observations=obs,runs=runs,seconds=time.time()-start,coefficients=dict(zip(names,theta.tolist())),trainMaxError=score,heldoutMaxError=float(np.max(np.abs((pred-target)[~train]))) if (~train).any() else None,
  terminalValues=[dict(picks=key,known=float(K[i]),composition=float(np.dot(X[i],theta))) for i,key in enumerate(keys)])
 suffix=('-holdout'+args.holdout.replace(':','_').replace(',','-') if held else '')+('-polished' if args.polish else '')+(f'-seed{args.seed}' if args.seed!=2416 else '')
 (root/f'ban-inverse-{args.cross}{suffix}.json').write_text(json.dumps(out,indent=2));print('Finished',len(keys),'drafts',len(obs),'observations',score,flush=True)


if __name__=='__main__':main()
