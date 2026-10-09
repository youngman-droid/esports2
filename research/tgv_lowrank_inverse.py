"""Offline low-rank opposing-team composition fit to a fixed minimax graph."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from tgv_minimax_graph import MinimaxGraph


def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=1);p.add_argument('--starts',type=int,default=12);p.add_argument('--holdout',default='');p.add_argument('--initial-from');p.add_argument('--seed',type=int,default=3501);p.add_argument('--refine-from');p.add_argument('--factor-structure',choices=['free','red-additive','blue-additive','both'],default='free');p.add_argument('--base-structure',choices=['free','additive'],default='free');p.add_argument('--jump-all',action='store_true');p.add_argument('--jump-size',type=float,default=.03);p.add_argument('--design-prefix',default='ban');p.add_argument('--temperatures',default='0');args=p.parse_args()
 root=Path('data/tgv/20260916');data=np.load(root/f'{args.design_prefix}-design.npz');K=data['known'];keys=data['keys'];g=json.loads((root/f'{args.design_prefix}-design-graph.json').read_text());groups=g['groups'];nb=len(groups[0])*len(groups[1]);nr=len(groups[2])*len(groups[3]);rank=args.rank
 bi=np.array([groups[0].index(int(k[0]))*len(groups[1])+groups[1].index(int(k[1])) for k in keys]);ri=np.array([groups[2].index(int(k[2]))*len(groups[3])+groups[3].index(int(k[3])) for k in keys])
 nodes=[(v['maximize'],np.array(v['children'])) for v in g['nodes']];obs=g['observations'];roots=np.array([v['node'] for v in obs]);y=np.array([v['target'] for v in obs]);held={tuple(map(int,x.split(':'))) for x in args.holdout.split(',') if x};train=np.array([(v['index'],v['championId']) not in held for v in obs]);u0=nb+nr
 def additive_basis(n1,n2):
  Z=np.zeros((n1*n2,n1+n2-1));Z[:,0]=1
  for i in range(n1):
   for j in range(n2):
    if i:Z[i*n2+j,i]=1
    if j:Z[i*n2+j,n1+j-1]=1
  return Z
 ZB=additive_basis(len(groups[0]),len(groups[1])) if args.factor_structure in ['blue-additive','both'] else np.eye(nb)
 ZR=additive_basis(len(groups[2]),len(groups[3])) if args.factor_structure in ['red-additive','both'] else np.eye(nr)
 FB=additive_basis(len(groups[0]),len(groups[1])) if args.base_structure=='additive' else np.eye(nb)
 GR=additive_basis(len(groups[2]),len(groups[3])) if args.base_structure=='additive' else np.eye(nr)
 nf,ng=FB.shape[1],GR.shape[1];u0=nf+ng;db,dr=ZB.shape[1],ZR.shape[1];v0=u0+db*rank;P=v0+dr*rank
 def terminal(theta):
  F=np.einsum('ni,i->n',FB,theta[:nf]);G=np.einsum('ni,i->n',GR,theta[nf:nf+ng]);Uc=theta[u0:v0].reshape(db,rank);Vc=theta[v0:].reshape(dr,rank);U=np.einsum('ni,ik->nk',ZB,Uc);V=np.einsum('ni,ik->nk',ZR,Vc)
  C=F[bi]-G[ri]+np.einsum('ij,ij->i',U[bi],V[ri]);J=np.zeros((len(K),P));ar=np.arange(len(K));J[:,:nf]=FB[bi];J[:,nf:nf+ng]=-GR[ri]
  for k in range(rank):
   J[:,u0+k:v0:rank]=ZB[bi]*V[ri,k,None];J[:,v0+k::rank]=ZR[ri]*U[bi,k,None]
  return C,J
 graph=MinimaxGraph(len(K),nodes,roots)
 def evaluate(theta):
  C,J=terminal(theta);values,active=graph.evaluate(K+C)
  return values,J[active],active
 last=[None,None]
 temperature=0.
 def cached(theta):
  if last[0] is None or not np.array_equal(theta,last[0]):
   if temperature>0:
    C,J=terminal(theta);result=graph.smooth(K+C,J,temperature)
   else:result=evaluate(theta)
   last[:]=[theta.copy(),result]
  return last[1]
 rng=np.random.default_rng(args.seed);initializer=None
 if args.initial_from:
  saved=json.loads(Path(args.initial_from).read_text())
  if held:
   excluded={(v['index'],v['championId']) for v in saved['observations'] if v.get('heldout')}
   assert held<=excluded,'Initializer used withheld values'
  lookup={tuple(v['picks']):v['composition'] for v in saved['terminalValues']};C=np.array([lookup[tuple(k)] for k in keys]).reshape(nb,nr);F=C.mean(axis=1);G=-C.mean(axis=0)+C.mean();center=C-F[:,None]+G[None,:];u,s,vh=np.linalg.svd(center,full_matrices=False);initializer=np.r_[np.linalg.lstsq(FB,F,rcond=None)[0],np.linalg.lstsq(GR,G,rcond=None)[0],np.linalg.lstsq(ZB,u[:,:rank]*np.sqrt(s[:rank]),rcond=None)[0].ravel(),np.linalg.lstsq(ZR,vh[:rank].T*np.sqrt(s[:rank]),rcond=None)[0].ravel()]
 if args.refine_from:
  saved=json.loads(Path(args.refine_from).read_text());assert saved.get('designPrefix','ban')==args.design_prefix;assert saved['rank']==rank;assert saved.get('factorStructure','free')==args.factor_structure;assert saved.get('baseStructure','free')==args.base_structure;assert args.starts<=2*(P if args.jump_all else nf)+1;initializer=np.array(saved['latentParameters'])
  if held:assert held<={(v['index'],v['championId']) for v in saved['observations'] if v.get('heldout')},'Refinement used withheld values'
 # Verify analytic derivatives at a generic point away from strategy ties.
 check=rng.normal(0,.1,P);pred,J,_=evaluate(check);h=1e-6;cols=[0,nf,u0,v0,P-1];jacerr=0
 for j in cols:
  plus=check.copy();minus=check.copy();plus[j]+=h;minus[j]-=h;numeric=(evaluate(plus)[0]-evaluate(minus)[0])/(2*h);jacerr=max(jacerr,float(np.max(np.abs(numeric-J[:,j]))))
 assert jacerr<1e-5
 best=None;runs=[];start=time.time()
 for i in range(args.starts):
  theta=initializer.copy() if i==0 and initializer is not None else rng.normal(0,.15,P)
  if args.refine_from:
   theta=initializer.copy()
   if i>0:
    row=(i-1)//2;theta[row]+=(args.jump_size if i%2 else -args.jump_size);theta=np.clip(theta,-1.999999,1.999999)
  for temperature in map(float,args.temperatures.split(',')):
   assert temperature>=0;last[:]=[None,None]
   if temperature>0:
    C,J=terminal(check);smooth_values,smooth_jac=graph.smooth(K+C,J,temperature)
    for col in cols:
     plus=check.copy();minus=check.copy();plus[col]+=h;minus[col]-=h
     cp,jp=terminal(plus);cm,jm=terminal(minus)
     numeric=(graph.smooth(K+cp,jp,temperature)[0]-graph.smooth(K+cm,jm,temperature)[0])/(2*h)
     jacerr=max(jacerr,float(np.max(np.abs(numeric-smooth_jac[:,col]))))
    assert jacerr<1e-5
   res=least_squares(lambda t:(cached(t)[0]-y)[train],theta,jac=lambda t:cached(t)[1][train],bounds=(-2,2),max_nfev=1000,ftol=1e-11,xtol=1e-11,gtol=1e-11)
   theta=res.x.copy();pred,_,active=evaluate(theta);err=float(np.max(np.abs((pred-y)[train])));runs.append(dict(start=i,temperature=temperature,trainMaxError=err,nfev=res.nfev))
   if best is None or err<best[0]:best=(err,theta.copy(),pred.copy(),active.copy())
   print('Rank',rank,'start',i,'temperature',temperature,'training error',err,flush=True)
 err,theta,pred,active=best;C,_=terminal(theta)
 for i,o in enumerate(obs):o.update(prediction=float(pred[i]),error=float(pred[i]-y[i]),heldout=not bool(train[i]),activeTerminal=keys[active[i]].tolist());o.pop('node')
 out=dict(baseStructure=args.base_structure,factorStructure=args.factor_structure,hypothesis=f'C(B,R)=F(B)-G(R)+sum of {rank} products U_k(B)*V_k(R)',rank=rank,parameters=P,terminalDrafts=len(K),jacobianCheckMaxError=jacerr,trainMaxError=err,heldoutMaxError=float(np.max(np.abs((pred-y)[~train]))) if (~train).any() else None,initialFrom=args.initial_from,runs=runs,seconds=time.time()-start,observations=obs,latentParameters=theta.tolist(),
  terminalValues=[dict(picks=k.tolist(),known=float(K[i]),composition=float(C[i])) for i,k in enumerate(keys)],
  caveat='Low-rank hypothesis and local optimization only. Factors have gauge freedoms and are not identified as physical tankiness or balance inputs.')
 out.update(designPrefix=args.design_prefix,seed=args.seed,refineFrom=args.refine_from,temperatures=args.temperatures)
 suffix=('' if args.design_prefix=='ban' else '-'+args.design_prefix)+('-holdout'+args.holdout.replace(':','_').replace(',','-') if held else '')+('-refined' if args.refine_from else '')+('' if args.factor_structure=='free' else '-'+args.factor_structure)+('' if args.base_structure=='free' else '-additivebase')+('-alljumps' if args.jump_all else '')
 suffix+=('-annealed' if args.temperatures!='0' else '')
 (root/f'lowrank-inverse-rank{rank}{suffix}.json').write_text(json.dumps(out,indent=2));print('Finished rank',rank,'train',err,'heldout',out['heldoutMaxError'],flush=True)


if __name__=='__main__':main()
