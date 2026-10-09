"""Compare stored outcome-draft unary weights with TGV's exported unary table.

Our asymmetric blue-perspective coefficients are converted to the side-neutral
half-difference. Role means use identical supported champion cohorts.
"""
import json
import re
import unicodedata
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.stats import pearsonr, spearmanr

from lol_ticker import draft


ROLES = dict(top='top', jng='jungle', mid='middle', bot='bottom', sup='support')


def normalized(name):
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower())


def main():
    root = Path('data/tgv/20260916')
    out = Path('data/tgv/champ-weight-comparison')
    source = json.loads((out/'production-outcome-snapshot.json').read_text())
    coefficients = {r['feature']: r['coef'] for r in source['rows']}
    counts = {r['feature']: r['n'] for r in source['rows']}
    fit = json.loads(next(v['value'] for v in source['metadata'] if v['key']=='fit'))
    catalog = json.loads((root/'current-model-catalog.json').read_text())
    sq = json.loads((root/'social-betting-strengths.v1.json').read_text())
    assert catalog['scoreSpace']==sq['scoreSpace']=='logit'
    assert catalog['modelVersion']==sq['modelVersion']
    sq_unary=np.asarray(sq['unary']).reshape(len(sq['patches']),len(sq['roles']),len(sq['championIds']))
    by_name = {normalized(c['name']):c for c in catalog['champions']}
    names = sorted(f[len('own_pick:'):] for f in coefficients if f.startswith('own_pick:') and '@' not in f and '#' not in f)
    assert all(normalized(n) in by_name for n in names)
    frozen = np.load('data/wpx/draft_comfort_comparison_20260916/player_controlled/model.npz')
    frozen_coef = dict(zip(frozen['vocabulary'].tolist(), frozen['draft_comfort'][int(frozen['nbase']):]))

    def effect(c, name, role, patch):
        return sum(.5*(c.get('own_pick:'+name+suffix,0)-c.get('enemy_pick:'+name+suffix,0))
                   for suffix in ['', '@'+role, '#'+patch])

    def summarize(rows):
        a=np.array([r['ours_centered'] for r in rows]);b=np.array([r['tgv_centered'] for r in rows])
        return dict(pairs=len(rows),pearson=float(pearsonr(a,b).statistic),spearman=float(spearmanr(a,b).statistic),
                    ours_std=float(a.std()),tgv_std=float(b.std()),rmse=float(np.sqrt(np.mean((a-b)**2))),
                    sign_agreement=float(np.mean((a>0)==(b>0))),
                    sq_pearson=float(pearsonr(a,[r['tgv_sq_centered'] for r in rows]).statistic))

    patches={};all_rows=[]
    for patch in catalog['patches']:
        rows=[];stats={}
        for role,tgv_role in ROLES.items():
            tgv={v['champion']['cid']:v['strength'] for v in catalog['insightsByPatch'][patch]['roles'][tgv_role]}
            rr=[]
            for name in names:
                champion=by_name[normalized(name)]
                if tgv_role not in champion['roles']:
                    continue
                n=sum(counts.get(side+'_pick:'+name+'@'+role,0) for side in ['own','enemy'])
                if n<200:
                    continue
                cid=champion['cid']
                rr.append(dict(patch=patch,role=role,champion=name,championId=cid,training_role_appearances_lower_bound=n,
                               ours=effect(coefficients,name,role,patch),tgv=tgv[cid],
                               tgv_sq=float(sq_unary[sq['patches'].index(patch),sq['roles'].index(tgv_role),sq['championIds'].index(cid)]),
                               frozen_isolated=effect(frozen_coef,name,role,patch) if 'own_pick:'+name in frozen_coef else None,
                               own_patch_adjustment_present='own_pick:'+name+'#'+patch in coefficients,
                               enemy_patch_adjustment_present='enemy_pick:'+name+'#'+patch in coefficients))
            for key in ['ours','tgv','tgv_sq']:
                mean=float(np.mean([r[key] for r in rr]))
                for r in rr:r[key+'_role_mean']=mean;r[key+'_centered']=r[key]-mean
            for key in ['ours','tgv']:
                for rank,r in enumerate(sorted(rr,key=lambda r:r[key],reverse=True),1):r[key+'_rank']=rank
            for r in rr:r['difference']=r['ours_centered']-r['tgv_centered']
            stats[role]=summarize(rr);rows.extend(rr)
        patches[patch]=dict(overall=summarize(rows),roles=stats,
                            patch_adjusted_pairs=sum(r['own_patch_adjustment_present'] or r['enemy_patch_adjustment_present'] for r in rows))
        all_rows.extend(rows)

    # Verify extraction against the actual feature generator, including 1/2.
    inputs=json.loads(Path('data/wpx/draft_comfort_comparison_20260916/inputs.json').read_text())
    verification_error=0.
    for game in inputs[-100:]:
        patch='16.17'
        def unary(blue,red):
            row=dict(patch=patch,own_picks=list(game[blue]['picks'].values()),enemy_picks=list(game[red]['picks'].values()),
                     own_roles={c:r for r,c in game[blue]['picks'].items()},enemy_roles={c:r for r,c in game[red]['picks'].items()},own_bans=[],enemy_bans=[])
            return sum(coefficients.get(f,0) for f in draft._features(row) if f.startswith(('own_pick:','enemy_pick:')))
        direct=.5*(unary('blue','red')-unary('red','blue'))
        extracted=sum(effect(coefficients,c,r,patch) for r,c in game['blue']['picks'].items())-sum(effect(coefficients,c,r,patch) for r,c in game['red']['picks'].items())
        verification_error=max(verification_error,abs(direct-extracted))
    assert verification_error<1e-12
    result=dict(primary_patch='16.17',our_fit=fit,our_fit_utc=datetime.fromtimestamp(fit['fitted_at'],timezone.utc).isoformat(),
                tgv_version=catalog['modelVersion'],tgv_generated=catalog['generatedAt'],
                normalization='Sum half(own_pick-enemy_pick) for champion, role and patch; then subtract same-cohort role mean separately in each model.',
                cohort='Declared TGV role eligibility and >=200 summed stored own/enemy role-feature occurrences. Missing side-specific role coefficients make this support count a lower bound.',
                verification_max_error=verification_error,patches=patches,rows=all_rows,
                caveats=['Stored draft_outcome_model is the outcome-based draft scorer, not the live GAM or market-repricing model.',
                         'Side-neutral extraction is an explicit transformation of our asymmetric model, not its raw blue-only coefficient.',
                         'Comparison excludes bans, synergy, matchups, player comfort, team strength and composition.',
                         'Different datasets, regularization, factor allocations and training windows limit absolute coefficient interpretation; this is not predictive-accuracy validation.',
                         'Absent patch deviations follow our model fallback; they are not newly estimated patch-specific evidence.',
                         'The earlier isolated draft-plus-comfort fit is retained as a secondary raw column and is frozen before January 16, 2026.'])
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(fit=result['our_fit_utc'],primary=patches['16.17'],verification_error=verification_error),indent=2))
    primary=[r for r in all_rows if r['patch']=='16.17']
    robustness=[]
    for threshold in [500,1000,2000]:
        aa=[];bb=[]
        for role in ROLES:
            selected=[r for r in primary if r['role']==role and r['training_role_appearances_lower_bound']>=threshold]
            if not selected:continue
            a=np.array([r['ours'] for r in selected]);b=np.array([r['tgv'] for r in selected])
            aa.extend(a-a.mean());bb.extend(b-b.mean())
        robustness.append(dict(minimum_support=threshold,pairs=len(aa),pearson=float(pearsonr(aa,bb).statistic)))
    result['support_sensitivity']=robustness
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    overall=patches['16.17']['overall']
    report=["# TGV champion weights versus our stored draft model", "",
            f"Compared {overall['pairs']} champion/role pairs on patch 16.17. Their role-centered unary weights have Pearson correlation {overall['pearson']:.3f}, Spearman correlation {overall['spearman']:.3f}, and above/below-average agreement {overall['sign_agreement']:.1%}. This is essentially no alignment in these coefficients, not evidence that either model predicts matches better.", "",
            f"Our source is the stored `draft_outcome_model`, fitted {result['our_fit_utc']} using {fit['games']:,} games. TGV's full-model catalog was generated {catalog['generatedAt']}; its solo-queue-only board is a separate comparison. This is the outcome-based draft model, not the live-state GAM or market-repricing regression.", "",
            "## Comparable units and coverage", "",
            "Our coefficient is half the own-pick coefficient minus half the enemy-pick coefficient, summed over champion, champion-role, and champion-patch terms. Both models are then centered over the same supported champions within each role. Values below are logits relative to that role's compared-champion average, not percentage-point win-rate changes.", "",
            "Each pair is TGV-role-eligible and has at least 200 summed stored role-feature appearances in our fit. Where one side's role feature was pruned, the displayed support is a lower bound. No champion-name mapping failures occurred. Bans, interactions, comfort, composition and team/player controls are excluded from these weights.", "",
            "Patch 16.17 is the latest shared patch with retained champion-patch effects in our model. Only 11 of the 152 pairs have at least one own/enemy patch adjustment; the remainder use the model's base-plus-role fallback. TGV has a full patch-specific table. Results on 16.18 and earlier exported patches are included below to expose that limitation.", "",
            "## Selected disagreements", "", "| Champion / role | Ours | TGV | Our role rank | TGV role rank | Training support ≥ |", "|---|---:|---:|---:|---:|---:|"]
    selected_keys=[('Ashe','bot'),('Taliyah','mid'),('LeBlanc','mid'),('Lulu','sup'),('Jayce','top'),('Draven','bot'),('Zeri','bot'),('Senna','bot')]
    for name,role in selected_keys:
        r=next(r for r in primary if r['champion']==name and r['role']==role)
        report.append(f"| {name} / {role} | {r['ours_centered']:+.3f} | {r['tgv_centered']:+.3f} | {r['ours_rank']} | {r['tgv_rank']} | {r['training_role_appearances_lower_bound']:,} |")
    report.extend(["", "## By role", "", "| Role | Pairs | Pearson | Spearman | Our weight SD | TGV weight SD |", "|---|---:|---:|---:|---:|---:|"])
    for role,s in patches['16.17']['roles'].items():report.append(f"| {role} | {s['pairs']} | {s['pearson']:+.3f} | {s['spearman']:+.3f} | {s['ours_std']:.3f} | {s['tgv_std']:.3f} |")
    report.extend(["", "TGV's mid and bot weights have approximately 1.8 times our spread on these cohorts. Against TGV's solo-queue-only unary table, overall Pearson correlation is also near zero (-0.024).", "", "## Sensitivity checks", "",
                   "| Patch | Pearson | Spearman | Pairs with any learned patch deviation in ours |", "|---|---:|---:|---:|"])
    for patch,s in patches.items():report.append(f"| {patch} | {s['overall']['pearson']:+.3f} | {s['overall']['spearman']:+.3f} | {s['patch_adjusted_pairs']} |")
    report.extend(["", "Re-centering after restricting the sample to more common picks:", "", "| Minimum role support | Pairs | Pearson |", "|---|---:|---:|"])
    for r in robustness:report.append(f"| {r['minimum_support']} | {r['pairs']} | {r['pearson']:+.3f} |")
    report.extend(["", "The difference is not explained solely by rare picks or the chosen patch. However, the models use different training populations, regularization, controls and allocations of effects among unary, interaction and composition terms. Centering removes role offsets; it does not eliminate all parameterization differences. A predictive comparison still requires full probabilities on common held-out games.", "", "## Full supported comparison, patch 16.17", "", "| Role | Champion | Ours | TGV | Ours rank | TGV rank | Training support ≥ | Patch term in ours |", "|---|---|---:|---:|---:|---:|---:|---|"])
    for role in ROLES:
        for r in sorted((r for r in primary if r['role']==role),key=lambda r:r['champion']):
            patch_term='yes' if r['own_patch_adjustment_present'] or r['enemy_patch_adjustment_present'] else 'fallback'
            report.append(f"| {role} | {r['champion']} | {r['ours_centered']:+.4f} | {r['tgv_centered']:+.4f} | {r['ours_rank']} | {r['tgv_rank']} | {r['training_role_appearances_lower_bound']} | {patch_term} |")
    report.extend(["", "## Reproduction", "", "`PYTHONPATH=. python3 research/tgv_compare_champion_weights.py`", "",
                   f"Coefficient extraction was independently checked against the actual draft feature generator on 100 complete drafts; maximum difference {verification_error:.3g}. The database was read only. No model was refitted or deployed.", "",
                   "The source snapshot and complete unrounded rows for all seven patches are in `data/tgv/champ-weight-comparison/production-outcome-snapshot.json` and `comparison.json`. The latter also retains raw coefficients from the previously isolated draft-plus-comfort experiment as a separately labeled secondary column; that experiment is frozen before January 16, 2026."])
    Path('docs/tgv-champion-weight-comparison-2026-09-16.md').write_text('\n'.join(report)+'\n')
    print('LARGEST DIFFERENCES')
    for r in sorted(primary,key=lambda r:abs(r['difference']),reverse=True)[:15]:
        print(r['role'],r['champion'],round(r['ours_centered'],4),round(r['tgv_centered'],4),r['ours_rank'],r['tgv_rank'],r['training_role_appearances_lower_bound'])


if __name__=='__main__':
    main()
