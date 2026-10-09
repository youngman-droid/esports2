"""Retrospective fixed-coefficient substitution; no fitting or deployment.

TGV was released after these outcomes. Results cannot establish future benefit.
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from lol_ticker import draft


def norm(s):
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower())


def metrics(y,p):
    p=np.clip(p,1e-12,1-1e-12)
    return dict(maps=len(y),brier=float(np.mean((p-y)**2)),
                logloss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
                accuracy=float(np.mean((p>=.5)==y)))


def main():
    root=Path('data/tgv/20260916');out=Path('data/tgv/champ-swap-test')
    source=json.loads((out/'production-snapshot.json').read_text())
    coefs={r['feature']:r['coef'] for r in source['coefficients']}
    stored={(r['oe_game_id'],r['team']):r for r in source['scores']}
    games=json.loads(Path('data/wpx/draft_comfort_comparison_20260916/inputs.json').read_text())
    catalog=json.loads((root/'current-model-catalog.json').read_text())
    sq=json.loads((root/'social-betting-strengths.v1.json').read_text())
    sq_u=np.asarray(sq['unary']).reshape(len(sq['patches']),5,len(sq['championIds']))
    champions={norm(c['name']):c for c in catalog['champions']}
    strengths={p:{role:{c['champion']['cid']:c['strength'] for c in rows}
                  for role,rows in d['roles'].items()} for p,d in catalog['insightsByPatch'].items()}
    roles=dict(top='top',jng='jungle',mid='middle',bot='bottom',sup='support')
    froot=Path('data/wpx/draft_comfort_comparison_20260916/player_controlled')
    frozen=np.load(froot/'model.npz')
    fc=dict(zip(frozen['vocabulary'].tolist(),frozen['draft_comfort'][int(frozen['nbase']):]))
    fs={r['game_id']:r for r in json.loads((froot/'scores.json').read_text())}
    rows=[];excluded=[];reconstruction_error=0.;eligibility_mismatches=0
    for g in games:
        patch=g['patch']
        if patch not in strengths:continue
        ps=stored.get((g['id'],g['blue_team']))
        if ps is None or g['id'] not in fs:
            excluded.append(dict(game=g['id'],reason='Missing baseline prediction'));continue
        if any(norm(c) not in champions for side in ['blue','red'] for c in g[side]['picks'].values()):
            excluded.append(dict(game=g['id'],reason='Unmapped champion'));continue
        assert ps['won']==g['y']
        frow=dict(patch=patch,own_picks=sorted(g['blue']['picks'].values()),enemy_picks=sorted(g['red']['picks'].values()),
                  own_roles={c:r for r,c in g['blue']['picks'].items()},enemy_roles={c:r for r,c in g['red']['picks'].items()},
                  own_bans=sorted(set(g['blue']['bans'])),enemy_bans=sorted(set(g['red']['bans'])))
        features=draft._features(frow)
        original=logit(ps['p_full'])
        reconstructed=expit(logit(ps['p_elo'])+sum(coefs.get(f,0) for f in features))
        reconstruction_error=max(reconstruction_error,abs(reconstructed-ps['p_full']))
        ours=sum(coefs.get(f,0) for f in features if f.startswith(('own_pick:','enemy_pick:')))
        tgv=0.;tgv_sq=0.;frozen_ours=0.;eligible=True
        for side,sign in [('blue',1),('red',-1)]:
            for role,name in g[side]['picks'].items():
                champion=champions[norm(name)];cid=champion['cid'];tr=roles[role]
                tgv+=sign*strengths[patch][tr][cid]
                tgv_sq+=sign*sq_u[sq['patches'].index(patch),sq['roles'].index(tr),sq['championIds'].index(cid)]
                eligible &= tr in champion['roles']
                frozen_ours+=sign*sum(.5*(fc.get('own_pick:'+name+s,0)-fc.get('enemy_pick:'+name+s,0)) for s in ['', '@'+role,'#'+patch])
        eligibility_mismatches+=not eligible
        frozen_z=logit(fs[g['id']]['full_blue'])
        rows.append(dict(game_id=g['id'],date=g['day'],patch=patch,league=g['league'],blue=g['blue_team'],red=g['red_team'],y=g['y'],tgv_roles_eligible=eligible,
                         ours_unary=ours,tgv_unary=tgv,tgv_sq_unary=tgv_sq,frozen_ours_unary=frozen_ours,
                         stored_ours=float(expit(original)),stored_drop_unary=float(expit(original-ours)),
                         stored_tgv_full=float(expit(original-ours+tgv)),stored_tgv_sq=float(expit(original-ours+tgv_sq)),
                         stored_half_blend=float(expit(original+.5*(tgv-ours))),
                         frozen_ours=float(expit(frozen_z)),frozen_drop_unary=float(expit(frozen_z-frozen_ours)),
                         frozen_tgv_full=float(expit(frozen_z-frozen_ours+tgv)),frozen_tgv_sq=float(expit(frozen_z-frozen_ours+tgv_sq))))
    assert reconstruction_error<1e-6
    y=np.array([r['y'] for r in rows])
    names=['stored_ours','stored_drop_unary','stored_tgv_full','stored_tgv_sq','stored_half_blend','frozen_ours','frozen_drop_unary','frozen_tgv_full','frozen_tgv_sq']
    predictions={n:np.array([r[n] for r in rows]) for n in names}
    results={n:metrics(y,p) for n,p in predictions.items()}
    clusters=[(r['date'],tuple(sorted([r['blue'],r['red']]))) for r in rows]
    unique=sorted(set(clusters));indices={k:i for i,k in enumerate(unique)}
    ix=np.array([indices[k] for k in clusters]);sizes=np.bincount(ix)
    draws=np.random.default_rng(91626).integers(0,len(unique),(2000,len(unique)))
    denominators=sizes[draws].sum(axis=1)
    deltas={}
    for name,p in predictions.items():
        base='stored_ours' if name.startswith('stored_') else 'frozen_ours'
        if name==base:continue
        p0=predictions[base]
        changes=dict(brier=(p-y)**2-(p0-y)**2,
                     logloss=(-y*np.log(p)-(1-y)*np.log1p(-p))-(-y*np.log(p0)-(1-y)*np.log1p(-p0)))
        deltas[name]={}
        for metric,change in changes.items():
            sums=np.bincount(ix,weights=change)
            ci=np.quantile(sums[draws].sum(axis=1)/denominators,[.025,.975])
            deltas[name][metric]=dict(delta=float(np.mean(change)),ci95=ci.tolist())
    per_patch={}
    for patch in sorted({r['patch'] for r in rows}):
        mask=np.array([r['patch']==patch for r in rows])
        per_patch[patch]={name:metrics(y[mask],p[mask]) for name,p in predictions.items() if name in ['stored_ours','stored_tgv_full','frozen_ours','frozen_tgv_full']}
    eligible=np.array([r['tgv_roles_eligible'] for r in rows])
    eligible_metrics={name:metrics(y[eligible],p[eligible]) for name,p in predictions.items() if name in ['stored_ours','stored_tgv_full','frozen_ours','frozen_tgv_full']}
    result=dict(protocol='Retrospective, no refitting, no tuning, full champion/role/patch pick contribution replaced; every other term unchanged.',
                formula='z_swap=z_original-our_pick_terms+sum_blue_TGV_unary-sum_red_TGV_unary',
                maps=len(rows),date_range=[min(r['date'] for r in rows),max(r['date'] for r in rows)],
                tgv_generated=catalog['generatedAt'],our_metadata=source['metadata'],clusters=len(unique),bootstrap_draws=2000,
                baseline_reconstruction_max_probability_error=float(reconstruction_error),excluded=excluded,
                tgv_role_ineligible_maps=eligibility_mismatches,metrics=results,paired_changes=deltas,by_patch=per_patch,eligible_only=eligible_metrics,
                median_absolute_probability_change=float(np.median(np.abs(predictions['stored_tgv_full']-predictions['stored_ours']))),
                limitations=['TGV September release postdates every evaluated outcome; its training overlap is unknown, so this is not a clean holdout for TGV.',
                             'The stored outcome draft model was also fitted after these games. Its metrics are a retrospective substitution diagnostic.',
                             'The frozen January draft-plus-comfort model predates the evaluated outcomes, but the TGV replacement does not.',
                             'Bootstrap intervals capture map/series sampling variation, not look-ahead bias, coefficient uncertainty or future model performance.',
                             'This tests the outcome-based draft model and frozen draft/comfort experiment, not replacing live GAM champion-state coefficients.',
                             'Standalone TGV weights may not be calibrated to our remaining interactions and controls; no scale or intercept was refitted.'])
    (out/'report.json').write_text(json.dumps(result,indent=2))
    (out/'predictions.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({k:result[k] for k in ['maps','date_range','metrics','paired_changes','tgv_role_ineligible_maps','median_absolute_probability_change']},indent=2))


if __name__=='__main__':main()
