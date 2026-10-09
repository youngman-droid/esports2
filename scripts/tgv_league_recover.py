"""Test TGV's league-mixture formula against scraped rankings and local counts.

This is parameter reconstruction, not win-prediction training. Player identities
are split deterministically so coefficient fitting cannot absorb held-out rows.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import lstsq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lol_ticker.tgv_reconstruction import recover_two_point_curve


def recover(root):
    root=Path(root)
    raw=json.loads((root/'local-player-league-counts.json').read_text())
    by_role=defaultdict(Counter);by_player=defaultdict(Counter)
    for row in raw:
        key=(row['player_id'],{'jng':'jgl'}.get(row['position'],row['position']))
        by_role[key][row['league']]+=row['n']
        by_player[row['player_id']][row['league']]+=row['n']
    roster=[p for t in json.loads((root/'team-rankings.v3.json').read_text())['teams'] for p in t['roster']]
    eligible={}
    for p in roster:
        if sum(by_role[(p['playerId'],p['role'])].values())==p['pickCount']:
            eligible[p['playerId']]=p
    pure=defaultdict(lambda:defaultdict(list))
    for pid,p in eligible.items():
        counts=by_player[pid]
        if len(counts)==1:
            league=next(iter(counts));n=sum(counts.values())
            pure[league][n].append((pid,p['components']['leagueMix']))
    candidates=[l for l,counts in pure.items() if 1 in counts and 2 in counts]
    anchor=max(candidates,key=lambda l:sum(len(v) for v in pure[l].values()))
    v1=Counter(v for _,v in pure[anchor][1]).most_common(1)[0][0]
    v2=Counter(v for _,v in pure[anchor][2]).most_common(1)[0][0]
    _,k=recover_two_point_curve(v1,v2)
    anchor_ids={pid for n in [1,2] for pid,_ in pure[anchor][n]}
    ids=sorted(eligible);leagues=sorted({l for pid in ids for l in by_player[pid]})
    index={l:i for i,l in enumerate(leagues)}
    X=np.zeros((len(ids),len(leagues)))
    for i,pid in enumerate(ids):
        for league,n in by_player[pid].items():X[i,index[league]]=n
    y=np.array([eligible[pid]['components']['leagueMix'] for pid in ids])
    X=X/(X.sum(axis=1)+k)[:,None]
    train=np.array([int(hashlib.sha256(pid.encode()).hexdigest(),16)%5!=0 or pid in anchor_ids for pid in ids])
    A=X[train];target=y[train];weights=np.ones(len(target))
    for _ in range(30):
        beta,_,rank,_=lstsq(A*np.sqrt(weights)[:,None],target*np.sqrt(weights))
        errors=np.einsum('ij,j->i',A,beta)-target
        weights=1/np.maximum(abs(errors),1e-10)
    errors=np.einsum('ij,j->i',X,beta)-y
    # Report whether held-out predictor rows are identified by training data.
    _,singular,Vt=np.linalg.svd(A,full_matrices=False)
    tol=max(A.shape)*np.finfo(float).eps*singular[0]
    basis=Vt[singular>tol]
    projections=np.einsum('ij,kj->ik',X,basis)
    residual=X-np.einsum('ik,kj->ij',projections,basis)
    identifiable=np.linalg.norm(residual,axis=1)<1e-10
    result=dict(formula='sum_league count_all_roles[player,league] * league_coefficient / (total_all_role_count + K)',
        K=k,anchor=dict(league=anchor,value_n1=v1,value_n2=v2,player_ids=sorted(anchor_ids)),
        counts='Local cumulative counts through 2026-09-14; matching role totals do not certify every all-role exposure',
        coefficients=dict(zip(leagues,beta.tolist())),train_rank=int(rank),league_columns=len(leagues),splits={})
    for label,mask in [('fit',train),('heldout',~train),('heldout_identifiable',~train & identifiable)]:
        e=errors[mask]
        result['splits'][label]=dict(players=int(mask.sum()),within_1e_6=int(sum(abs(e)<1e-6)),
            median_abs_error=float(np.median(abs(e))),max_abs_error=float(max(abs(e))),
            rmse=float(np.sqrt(np.mean(e*e))))
    result['exceptions']=[dict(player_id=pid,name=eligible[pid]['name'],error=float(errors[i]),
        heldout=bool(not train[i]),identifiable=bool(identifiable[i]))
        for i,pid in enumerate(ids) if abs(errors[i])>=1e-6]
    result['limitation']='Supported approximation with explicit exceptions, not exact recovery of all league effects or their fitting procedure.'
    result['provenance']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ['team-rankings.v3.json','local-player-league-counts.json']}
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root/'league-reconstruction.json').write_text(json.dumps(result,indent=2))
    (root/'league-recovery-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ['coefficients','exceptions']},indent=2))
    return result


if __name__=='__main__':recover('data/tgv/20260916')
