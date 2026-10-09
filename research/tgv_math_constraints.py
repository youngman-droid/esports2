"""Test low-dimensional explanations of published checksum scores offline.

Leave-one-patch-out predictions prevent patch-specific fitted offsets from leaking
into their evaluation. These are reconstruction diagnostics, not outcome tests.
"""
import json
from pathlib import Path
import numpy as np
from research.tgv_reconstruction import PublicDraftModel,ROLES


def metrics(target,pred):
    error=pred-target
    return dict(rmse=float(np.sqrt(np.mean(error**2))),maxAbsoluteError=float(np.max(np.abs(error))),
                meanAbsoluteError=float(np.mean(np.abs(error))))


def main():
    root=Path('data/tgv/20260916');model=PublicDraftModel(root)
    rows=json.loads((root/'checksum-reconstruction.json').read_text())
    patches=sorted(set(x['patch'] for x in rows));y=np.array([x['target'] for x in rows])
    base=np.array([x['sq_unary']+x['matchup']+x['synergy'] for x in rows]);residual=y-base
    simple=[];pairwise=[]
    for row in rows:
        b,d=row['blue'],row['red']
        M=[model.pair('matchups',r,b[i],s,d[j]) for i,r in enumerate(ROLES) for j,s in enumerate(ROLES)]
        S=[model.pair('synergies',ROLES[i],b[i],ROLES[j],b[j])-model.pair('synergies',ROLES[i],d[i],ROLES[j],d[j]) for i in range(5) for j in range(i+1,5)]
        diag=sum(M[i*5+i] for i in range(5))
        simple.append([1,row['sq_unary'],row['full_unary']-row['sq_unary'],diag,sum(M)-diag,sum(S)])
        pairwise.append([1,row['sq_unary'],row['full_unary']-row['sq_unary'],*M,*S])
    simple=np.array(simple);pairwise=np.array(pairwise);results={}
    hypotheses={'constant':(simple[:,:1],[0]),'global_components':(simple,[0]),
                'role_pair_shrinkage':(pairwise,[.1,1,10,100,1000])}
    for name,(X,penalties) in hypotheses.items():
        for penalty in penalties:
            pred=np.zeros(len(y))
            for patch in patches:
                test=np.array([r['patch']==patch for r in rows]);train=~test
                scale=np.std(X[train],axis=0);scale[scale<1e-12]=1
                A=X[train]/scale;B=X[test]/scale
                if penalty:
                    coef=np.linalg.solve(np.einsum('ni,nj->ij',A,A)+penalty*np.eye(A.shape[1]),np.einsum('ni,n->i',A,residual[train]))
                else:coef=np.linalg.lstsq(A,residual[train],rcond=None)[0]
                pred[test]=base[test]+np.einsum('ni,i->n',B,coef)
            results[name+':'+str(penalty)]=dict(leavePatchOut=metrics(y,pred),predictions=pred.tolist())
    # Exact least-norm interpolation is shown explicitly to distinguish fitting
    # observed constraints from recovering unique underlying factors.
    cids=[c['cid'] for c in model.catalog['champions']];ci={c:i for i,c in enumerate(cids)}
    Z=np.zeros((len(rows),len(patches)*5*len(cids)))
    for k,row in enumerate(rows):
        offset=patches.index(row['patch'])*5*len(cids)
        for role,(b,d) in enumerate(zip(row['blue'],row['red'])):
            Z[k,offset+role*len(cids)+ci[b]]+=1;Z[k,offset+role*len(cids)+ci[d]]-=1
    correction=np.linalg.lstsq(Z,residual,rcond=None)[0]
    rank=int(np.linalg.matrix_rank(Z))
    output=dict(baseline=metrics(y,base),hypotheses=results,
        additiveInterpolation=dict(rank=rank,unknowns=Z.shape[1],nullity=Z.shape[1]-rank,
            fit=metrics(y,base+np.einsum('ni,i->n',Z,correction)),interpretation='One minimum-norm solution to observations, not unique composition recovery.'),
        scope='56 published checksum drafts; leave-one-patch-out is a diagnostic. All hypotheses remain estimates unless residuals reach numerical precision on independent data.')
    (root/'math-constraint-diagnostics.json').write_text(json.dumps(output,indent=2))
    np.savez_compressed(root/'minimum-norm-checksum-correction.npz',patches=patches,championIds=cids,correction=correction.reshape(len(patches),5,len(cids)))
    print(json.dumps({**output,'hypotheses':{k:v['leavePatchOut'] for k,v in results.items()}},indent=2))


if __name__=='__main__':main()
