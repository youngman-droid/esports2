"""Offline inverse minimax: test additive composition against late puzzle values."""
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from lol_ticker.tgv_reconstruction import PublicDraftModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bound',type=float,default=1);parser.add_argument('--seconds',type=float,default=45);parser.add_argument('--structure',choices=['champion','team'],default='champion');parser.add_argument('--cross',choices=['none','botbot','midbot','bottop','midtop','all'],default='none');parser.add_argument('--holdout',default='');parser.add_argument('--regularize',action='store_true');args=parser.parse_args()
    root=Path('data/tgv/20260916');model=PublicDraftModel(root);c=json.loads((root/'challenge.json').read_text())['challenge'];seq=c['puzzle']['sequence'];short=['top','jgl','mid','bot','sup']
    pools=c['pools'];used=set(seq[:16]);mods={side:{(x['role'],x['champion_id']):x['value'] for x in c['modifiers'][side]} for side in ['blue','red']}
    groups=[sorted(set(pools[side][role])-used) for side,role in [('blue','mid'),('blue','bot'),('red','top'),('red','bot')]]
    assert groups==[[1,3,34,134],[81,498,523],[67,82,516,799,897],[8,163,804]]
    known={}
    for mid,bot,top,rbot in itertools.product(*groups):
        blue=[2,904,mid,bot,902];red=[top,78,126,rbot,53]
        # Check uniqueness of role assignments against pool membership.
        for side,picks in [('blue',blue),('red',red)]:
            count=sum(all(cid in pools[side][role] for role,cid in zip(short,p)) for p in itertools.permutations(picks))
            assert count==1
        components=model.components(c['patch'],blue,red,
            [mods['blue'][r,i] for r,i in zip(short,blue)], [mods['red'][r,i] for r,i in zip(short,red)])
        known[mid,bot,top,rbot]=c['puzzle']['finalEvaluation']['sideBiasScore']+sum(components.values())
    lower=[];upper=[];integer=[];objective=[];names=[];eq=[]
    def var(name,lo,hi,integral=0,cost=0):
        i=len(names);names.append(name);lower.append(lo);upper.append(hi);integer.append(integral);objective.append(cost);return i
    def row(coeff,lo=-np.inf,hi=np.inf):eq.append((coeff,lo,hi))
    cross_map={'botbot':(1,3),'midbot':(0,3),'bottop':(1,2),'midtop':(0,2)}
    cross_families=list(cross_map) if args.cross=='all' else ([] if args.cross=='none' else [args.cross])
    B=args.bound;limit=max(abs(x) for x in known.values())+(5+len(cross_families))*B;big=2*limit
    const=var('composition_intercept',-B,B);params={}
    categories=groups if args.structure=='champion' else [list(itertools.product(*groups[:2])),list(itertools.product(*groups[2:]))]
    for g,values in enumerate(categories):
        for cid in values[1:]:params[g,cid]=var(f'group{g}:{cid}',-B,B)
    cross_params={}
    for family in cross_families:
        a,b=cross_map[family]
        for ca,cb in itertools.product(groups[a][1:],groups[b][1:]):
            cross_params[family,ca,cb]=var(f'cross:{family}:{ca}:{cb}',-B,B)
    leaves={}
    for key,k in known.items():
        i=var('leaf:'+str(key),-limit,limit);leaves[key]=i;co={i:1,const:-1}
        terms=list(key) if args.structure=='champion' else [key[:2],key[2:]]
        for g,cid in enumerate(terms):
            if (g,cid) in params:co[params[g,cid]]=(-1 if g<(2 if args.structure=='champion' else 1) else 1)
        for family in cross_families:
            a,b=cross_map[family];term=cross_params.get((family,key[a],key[b]))
            if term is not None:co[term]=-1
        row(co,k,k)
    def extreme(name,children,maximize):
        v=var(name,-limit,limit);zs=[]
        for child in children:
            z=var(name+':select:'+str(child),0,1,1);zs.append(z)
            if maximize:
                row({v:1,child:-1},lo=0)
                row({v:1,child:-1,z:big},hi=big)
            else:
                row({v:1,child:-1},hi=0)
                row({v:1,child:-1,z:-big},lo=-big)
        row({z:1 for z in zs},1,1);return v
    redlast={}
    for first in groups[2]+groups[3]:
        for mid,bot in itertools.product(*groups[:2]):
            keys=[key for key in leaves if key[:2]==(mid,bot) and first in key[2:]]
            redlast[first,mid,bot]=extreme('redlast:'+str((first,mid,bot)),[leaves[k] for k in keys],False)
    observations=[]
    for d in c['puzzle']['decisions']:
        if d['index']<16:continue
        for ch in d['choices']:
            cid=ch['championId'];idx=d['index']
            if idx==16:v=extreme('bluepair:'+str(cid),[redlast[cid,m,b] for m,b in itertools.product(*groups[:2])],True)
            elif idx==17:v=extreme('blueother:'+str(cid),[redlast[799,m,b] for m,b in itertools.product(*groups[:2]) if cid in (m,b)],True)
            elif idx==18:v=redlast[799,1,cid]
            else:v=leaves[1,498,799,cid]
            observations.append((idx,cid,v,ch['blueValue']))
    error=var('maximum_observation_error',0,1e-7 if args.regularize else np.inf,cost=0 if args.regularize else 1)
    if args.regularize:
        for parameter in [const,*params.values(),*cross_params.values()]:
            magnitude=var('absolute:'+names[parameter],0,B,cost=1)
            row({parameter:1,magnitude:-1},hi=0);row({parameter:-1,magnitude:-1},hi=0)
    heldout={tuple(map(int,x.split(':'))) for x in args.holdout.split(',') if x}
    for idx,cid,v,target in observations:
        if (idx,cid) in heldout:continue
        row({v:1,error:-1},hi=target);row({v:1,error:1},lo=target)
    rr=[];cc=[];vv=[]
    for i,(co,_,_) in enumerate(eq):
        for j,x in co.items():rr.append(i);cc.append(j);vv.append(x)
    A=coo_matrix((vv,(rr,cc)),shape=(len(eq),len(names))).tocsc()
    start=time.time();result=milp(np.array(objective),integrality=np.array(integer),bounds=Bounds(lower,upper),constraints=LinearConstraint(A,[x[1] for x in eq],[x[2] for x in eq]),options={'time_limit':args.seconds,'mip_rel_gap':1e-7})
    out=dict(bound=B,seconds=time.time()-start,status=int(result.status),message=result.message,
        regularized=args.regularize,certificateObjective='coefficient L1 norm' if args.regularize else 'maximum observation error',crossFamilies=cross_families,heldout=list(sorted(heldout)),variables=len(names),binaryVariables=sum(integer),terminalDrafts=len(leaves),observations=len(observations),
        hypothesis=('Independent late-role champion effects' if args.structure=='champion' else 'Arbitrary blue-team effect minus arbitrary red-team effect')+' plus constant and cross families '+str(cross_families)+'; gauge-fixed coefficients bounded as reported.',
        objectiveValue=float(result.fun) if result.fun is not None else None,bestMaximumError=float(result.x[error]) if result.x is not None else None,
        certifiedLowerBound=float(result.mip_dual_bound) if getattr(result,'mip_dual_bound',None) is not None else None,
        mipGap=float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None)
    if result.x is not None:
        x=result.x;out['coefficients']={names[i]:float(x[i]) for i in [const,*params.values(),*cross_params.values()]}
        # Independently replay min/max from terminal values, not solver node values.
        values={k:float(x[v]) for k,v in leaves.items()}
        R={(f,m,b):min(y for k,y in values.items() if k[:2]==(m,b) and f in k[2:]) for f,m,b in redlast}
        replay=[]
        for idx,cid,v,target in observations:
            if idx==16:pred=max(y for (f,m,b),y in R.items() if f==cid)
            elif idx==17:pred=max(y for (f,m,b),y in R.items() if f==799 and cid in (m,b))
            elif idx==18:pred=R[799,1,cid]
            else:pred=values[1,498,799,cid]
            replay.append(dict(heldout=(idx,cid) in heldout,index=idx,championId=cid,target=target,prediction=pred,error=pred-target))
        out['replayedObservations']=replay;out['replayMaxError']=max(abs(z['error']) for z in replay)
        out['trainingReplayMaxError']=max(abs(z['error']) for z in replay if not z['heldout'])
        out['heldoutReplayMaxError']=max((abs(z['error']) for z in replay if z['heldout']),default=None)
        out['terminalValues']=[dict(picks=list(k),known=known[k],value=v,composition=v-known[k]) for k,v in values.items()]
    suffix=('-regularized' if args.regularize else '')+('' if args.cross=='none' else '-cross'+args.cross)+('' if not args.holdout else '-holdout'+args.holdout.replace(':','_').replace(',','-'))
    path=root/f'inverse-minimax-{args.structure}-bound{B:g}{suffix}.json';path.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['coefficients','replayedObservations','terminalValues']},indent=2),flush=True)


if __name__=='__main__':main()
