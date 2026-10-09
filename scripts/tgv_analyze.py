"""Audit the scraped public release and recover identifiable model parameters."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.linalg import svdvals

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lol_ticker.tgv_reconstruction import (PublicDraftModel, ROLES, recover_two_point_curve,
                                          saturating_count, verify_daily)


def norm(name):
    key=re.sub('[^a-z0-9]','',name.lower())
    return {'monkeyking':'wukong','nunu':'nunuwillump','renata':'renataglasc'}.get(key,key)


def analyze(root):
    root=Path(root)
    scrape=json.loads((root/'scrape-manifest.json').read_text())
    for entry in scrape['assets']:
        raw=(root/entry['path']).read_bytes()
        if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('Asset audit failed: '+entry['path'])
    model=PublicDraftModel(root)
    sq=json.loads((root/'social-betting-strengths.v1.json').read_text())
    ids=sq['championIds']; ci={c:i for i,c in enumerate(ids)}; ri={r:i for i,r in enumerate(ROLES)}
    M=np.full((5,5,len(ids),len(ids)),np.nan); S=np.full_like(M,np.nan)
    for role in ROLES:
        for champion in ids:
            relation=model.relations(role,champion)
            for kind,array in [('matchups',M),('synergies',S)]:
                for (other,cid),value in relation[kind].items():
                    array[ri[role],ri[other],ci[champion],ci[cid]]=value
    full=np.array([[[model.strength[p][r][c] for c in ids] for r in ROLES] for p in sq['patches']])
    sq_unary=np.array(sq['unary']).reshape(full.shape)
    np.savez_compressed(root/'recovered-factors.npz',roles=np.array(ROLES),champion_ids=ids,
                        patches=np.array(sq['patches']),full_unary=full,sq_unary=sq_unary,
                        matchup=M,synergy=S)
    ranks=json.loads((root/'team-rankings.v3.json').read_text())['teams']
    players=[p for t in ranks for p in t['roster']]
    e1=next(p['components']['experience'] for p in players if p['pickCount']==1)
    e2=next(p['components']['experience'] for p in players if p['pickCount']==2)
    ea,ek=recover_two_point_curve(e1,e2)
    c1=max(p['components']['comfort'] for p in players if p['pickCount']==1)
    c2=max(p['components']['comfort'] for p in players if p['pickCount']==2)
    ca,ck=recover_two_point_curve(c1,c2)
    counts=defaultdict(lambda:defaultdict(int))
    for row in json.loads((root/'local-player-champion-counts.json').read_text()):
        role={'jng':'jgl'}.get(row['position'],row['position'])
        counts[(row['player_id'],role)][norm(row['champion'])]+=row['n']
    name_idx={norm(v['name']):ci[v['cid']] for v in model.catalog['champions']}
    short_roles=['top','jgl','mid','bot','sup']
    matched=[]; champion_offsets=defaultdict(list)
    for p in players:
        cc=counts[(p['playerId'],p['role'])];total=sum(cc.values())
        if total!=p['pickCount']:continue
        predicted=sum(n/total*saturating_count(n,ca,ck) for n in cc.values())
        matched.append(dict(player_id=p['playerId'],role=p['role'],name=p['name'],
                            games=total,error=predicted-p['components']['comfort']))
        if all(name in name_idx for name in cc):
            role=short_roles.index(p['role'])
            strength=sum(n/total*sq_unary[0,role,name_idx[name]] for name,n in cc.items())
            champion_offsets[p['role']].append(dict(player_id=p['playerId'],name=p['name'],
                value=p['components']['championStrength']-strength))
    offsets={}
    for role,rows in champion_offsets.items():
        # A robust estimate exposes, rather than absorbs, count-source mismatches.
        offset=float(np.median([row['value'] for row in rows]))
        offsets[role]=dict(offset=offset,rows=len(rows),
            rows_within_1e_10=sum(abs(row['value']-offset)<1e-10 for row in rows),
            exceptions=[row for row in rows if abs(row['value']-offset)>=1e-10])
    checks=[]
    coverage_rows=[]
    for q in sq['checksumDrafts']:
        pi=sq['patchIndices'].index(q['patchIdx'])
        parts=model.components(sq['patches'][pi],q['blue'],q['red'])
        raw_sq=sum(sq_unary[pi,r,ci[b]]-sq_unary[pi,r,ci[d]] for r,(b,d) in enumerate(zip(q['blue'],q['red'])))
        checks.append(dict(patch=sq['patches'][pi],blue=q['blue'],red=q['red'],target=q['value'],
            sq_unary=raw_sq,full_unary=parts['strength'],matchup=parts['matchup'],synergy=parts['synergy'],
            residual=q['value']-raw_sq-parts['matchup']-parts['synergy']))
        row=np.zeros(full.size)
        for role,(b,d) in enumerate(zip(q['blue'],q['red'])):
            row[(pi*5+role)*len(ids)+ci[b]]=1
            row[(pi*5+role)*len(ids)+ci[d]]=-1
        coverage_rows.append(row)
    challenge=json.loads((root/'challenge.json').read_text())['challenge']
    terminal=challenge['puzzle']['finalEvaluation']
    row=np.zeros(full.size);pi=sq['patches'].index(challenge['patch'])
    for sign,side in [(1,'bluTeam'),(-1,'redTeam')]:
        for p in terminal[side]:
            row[(pi*5+ri[p['role']])*len(ids)+ci[p['champion']['cid']]]=sign
    coverage_rows.append(row)
    observation=np.array(coverage_rows);singular=svdvals(observation)
    rank=int(sum(singular>max(observation.shape)*np.finfo(float).eps*singular[0]))
    seen=int(np.any(observation!=0,axis=0).sum())
    coverage=dict(observed_terminal_drafts=len(coverage_rows),patch_role_champion_cells=full.size,
        cells_appearing=seen,cells_not_appearing=int(full.size-seen),additive_observation_rank=rank,
        unresolved_additive_dimensions=int(full.size-rank),
        scope='Terminal numerical constraints only, for one hypothetical hidden additive scalar per patch/role/champion. Hashes pin but do not reveal values. Puzzle minimax scores are not terminal assignments.')
    (root/'composition-identifiability-audit.json').write_text(json.dumps(coverage,indent=2))
    report=dict(model_version=model.version,verified_assets=len(scrape['assets']),
        verified_bytes=scrape['total_bytes'],roles=5,champions=len(ids),patches=sq['patches'],
        sq_games_by_patch=model.catalog['sqGamesByPatch'],provenance=model.catalog['modelProvenance'],
        factor_coverage=dict(full_unary=full.size,sq_unary=sq_unary.size,
            matchup=int(np.isfinite(M).sum()),synergy=int(np.isfinite(S).sum()),
            matchup_antisymmetry_max=float(np.nanmax(abs(M+M.transpose(1,0,3,2)))),
            synergy_symmetry_max=float(np.nanmax(abs(S-S.transpose(1,0,3,2))))),
        daily=verify_daily(root),experience=dict(A=ea,K=ek,fit_counts=[1,2],rows=len(players),
            max_error=max(abs(saturating_count(p['pickCount'],ea,ek)-p['components']['experience']) for p in players)),
        comfort=dict(A=ca,K=ck,fit_counts=[1,2],formula='sum_c n_pc/N * A*n_pc/(n_pc+K)',
            matched_rows=len(matched),unique_player_roles=len({(p['player_id'],p['role']) for p in matched}),
            rows_within_1e_10=sum(abs(p['error'])<1e-10 for p in matched),
            heldout_n_gt_2_rows=sum(p['games']>2 for p in matched),
            heldout_n_gt_2_rows_within_1e_10=sum(p['games']>2 and abs(p['error'])<1e-10 for p in matched),
            exceptions=[p for p in matched if abs(p['error'])>=1e-10]),
        roster_champion_strength_offsets=offsets,
        ranking=dict(teams=len(ranks),roster_rows=len(players),
            player_component_max_error=max(abs(p['skill']-sum(p['components'].values())) for p in players),
            team_component_max_error=max(abs(t['skill']-t['teamStrength']-sum(p['skill'] for p in t['roster'])) for t in ranks),
            elo_max_error=max(abs(t['elo']-(1200+400/np.log(10)*t['skill'])) for t in ranks)),
        independent_checksum_drafts=dict(count=len(checks),
            recovered_terms_only_max_residual=max(abs(x['residual']) for x in checks),
            warning='Published checksum scores are not reproduced by the recovered terms alone; omitted composition/contract details remain unresolved.'),
        composition_coverage=coverage,
        unknowns=['arbitrary-draft per-champion composition inputs','damage-balance center and interaction/normalization parameters',
                  'side bias for six historical patches','exact training objective penalties, optimizer and data splits',
                  'professional player/team and league parameter update rules'])
    report=json.loads(json.dumps(report,default=lambda x:x.item()))
    (root/'reconstruction-report.json').write_text(json.dumps(report,indent=2))
    (root/'checksum-reconstruction.json').write_text(json.dumps(checks,indent=2))
    (root/'analysis-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k not in ['roster_champion_strength_offsets','daily','sq_games_by_patch','provenance']},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='data/tgv/20260916')
    analyze(parser.parse_args().root)
