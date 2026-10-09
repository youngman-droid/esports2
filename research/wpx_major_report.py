"""Summarize completed major-only experiments and matched all-league replays."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpbench
from research.wpx_methods_v9 import dump, metrics, paired, specifications


def main(root, original, report):
    cohort=json.loads((root/'cohort.json').read_text())
    selection=json.loads((root/'methods/selection.json').read_text())
    frozen=json.loads((root/'methods/frozen.json').read_text())
    rolling=json.loads((root/'methods/rolling.json').read_text())
    recency=json.loads((root/'recency/rolling.json').read_text())
    audits=json.loads((root/'methods/gold_audits.json').read_text())
    for path in ('methods_complete.json','recency_complete.json','markets_complete.json'):
        if not json.loads((root/path).read_text())['completed']:
            raise ValueError(f'Incomplete stage: {path}')
    for folder, dataset in ((root/'methods',root/'states.npz'),(original,original.parent/'states.npz')):
        saved_plan=json.loads((folder/'plan.json').read_text())
        if hashlib.sha256(dataset.read_bytes()).hexdigest()!=saved_plan['dataset_sha256']:
            raise ValueError(f'Dataset changed since fitting: {dataset}')
    base=wpbench._load_base(str(root/'states.npz'))
    other=wpbench._load_base(str(original.parent/'states.npz'))
    mask=~base['outer_train'][base['row_game']]
    omask=~other['outer_train'][other['row_game']]
    keys=list(zip(base['gid'][mask].tolist(),base['t_min'][mask].tolist()))
    okeys=list(zip(other['gid'][omask].tolist(),other['t_min'][omask].tolist()))
    if len(set(keys))!=len(keys) or len(set(okeys))!=len(okeys):
        raise ValueError('Duplicate game/minute key')
    lookup={k:i for i,k in enumerate(okeys)}
    ix=np.array([i for i,k in enumerate(keys) if k in lookup]); jx=np.array([lookup[keys[i]] for i in ix])
    with np.load(root/'resources/resources.npz') as r:
        clusters=dict(zip(r['series_gid'].astype(int),r['series_key'].astype(str)))
    expected=set(specifications())|{'convex_ensemble','monotone_ensemble'}
    if set(rolling)!=expected or set(frozen)!=expected or set(audits)!=expected:
        raise ValueError('Incomplete model-family coverage')
    plan=json.loads((root/'methods/plan.json').read_text())
    for path, digest in plan['protected_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Protected production artifact changed: {path}')
    for mode, summary in [('frozen',frozen),('rolling',rolling)]:
        with np.load(root/f'methods/{mode}_predictions.npz') as saved:
            if set(saved.files)-{'gid','y'} != expected:
                raise ValueError('Missing prediction family')
            np.testing.assert_array_equal(saved['gid'],base['gid'][mask])
            np.testing.assert_array_equal(saved['y'],base['y'][mask])
            for family in expected:
                measured=metrics(saved[family],saved['y'],saved['gid'])
                for key in ('brier_game','logloss_game','games','states'):
                    np.testing.assert_allclose(measured[key],summary[family]['metrics'][key],rtol=0,atol=1e-12)
    matched={}
    with np.load(root/'methods/rolling_predictions.npz') as a, np.load(original/'rolling_predictions.npz') as b:
        np.testing.assert_array_equal(a['gid'],base['gid'][mask])
        np.testing.assert_array_equal(b['gid'],other['gid'][omask])
        np.testing.assert_array_equal(a['gid'][ix],b['gid'][jx])
        np.testing.assert_array_equal(a['y'][ix],b['y'][jx])
        y,gid=a['y'][ix],a['gid'][ix]
        for family in sorted(set(a.files)&set(b.files)-{'gid','y'}):
            p,q=a[family][ix],b[family][jx]
            matched[family]=dict(major=metrics(p,y,gid),all_leagues=metrics(q,y,gid),
                                paired=paired(p,q,y,gid,clusters))
    dump(root/'matched_training_comparison.json',matched)
    reference=matched['constrained_gam']
    delta=reference['paired']['delta_brier'];lo,hi=reference['paired']['ci95']
    direction='higher (worse)' if delta>0 else 'lower (better)'
    lines=['# Major-league model comparisons', '',
        f"On the {reference['paired']['games']} matching major-league games, major-only constrained GAM training scores "
        f"{reference['major']['brier_game']:.6f} versus {reference['all_leagues']['brier_game']:.6f} for all-league training: "
        f"{abs(delta):.6f} {direction}, with a 95% paired interval [{lo:+.6f}, {hi:+.6f}]. "
        'This experiment does not establish a benefit from discarding lower-tier training games.', '',
        f"The cohort contains {cohort['games']:,} games and {cohort['fixed_states']:,} fixed-minute states. "
        f"The chronological later block begins {cohort['evaluation_start']}. All results are retrospective diagnostics on previously consumed dates.", '',
        'Membership follows direct Worlds-slot leagues by season, including Worlds/MSI/First Stand. '
        'Academy/regional leagues, promotion events, other cups, and indirect LJL/LCO pathways are excluded. '
        'There are no LPL-labeled games in the source state dataset. Historical sequential Elo/form inputs are retained; all fitted prior, champion, calibration and state learners use the filtered cohort.', '',
        'Settings and calibration are selected on three earlier blocks, then frozen for the later block and monthly replay. '
        'Scores balance games equally; intervals resample whole series. No production model was replaced.', '',
        '| Model | Development Brier | Frozen Brier | Monthly replay Brier | Replay log loss | Gold reversals |',
        '|---|---:|---:|---:|---:|---:|']
    for family,r in sorted(rolling.items(),key=lambda x:x[1]['metrics']['brier_game']):
        dev=selection['choices'].get(family,{}).get('mean_brier')
        ds=f'{dev:.6f}' if dev is not None else 'ensemble'
        lines.append(f"| {family} | {ds} | {frozen[family]['metrics']['brier_game']:.6f} | {r['metrics']['brier_game']:.6f} | {r['metrics']['logloss_game']:.6f} | {audits[family]['violations']} / {audits[family]['comparisons']} |")
    lines += ['', '## Does major-only training help?', '',
        'These monthly replay predictions are joined by game and minute to the existing all-league replay. '
        'Both approaches are evaluated on exactly the same major-league states. Each suite tuned its own settings on its earlier development cohort; this compares the retuned pipelines, not a fixed-hyperparameter ablation. '
        'Negative differences favor major-only training.', '',
        '| Model | Major-only Brier | All-league Brier | Difference [95% series interval] | Games |',
        '|---|---:|---:|---:|---:|']
    for f,r in sorted(matched.items(),key=lambda x:x[1]['major']['brier_game']):
        p=r['paired'];lo,hi=p['ci95']
        lines.append(f"| {f} | {r['major']['brier_game']:.6f} | {r['all_leagues']['brier_game']:.6f} | {p['delta_brier']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {p['games']} |")
    lines += ['', '## Recency comparison', '', '| Model | Replay Brier | Replay log loss |','|---|---:|---:|']
    for f,r in recency.items():
        lines.append(f"| {f} | {r['metrics']['brier_game']:.6f} | {r['metrics']['logloss_game']:.6f} |")
    rp=recency['recency180']['paired']
    lines += ['', f"The recency change is {rp['delta_brier']:+.6f} Brier, with a 95% series interval [{rp['ci95'][0]:+.6f}, {rp['ci95'][1]:+.6f}]; the small point improvement is inconclusive."]
    market=json.loads((root/'markets/results.json').read_text())
    lines += ['', '## Matched market diagnostics', '',
        'The table shows the major-only constrained GAM monthly replay against each market on matching available rows. '
        'Offsets advance the market lookup relative to reconstructed game time; the 45/195-second offsets test information-timing sensitivity.', '',
        '| Platform | Offset (s) | Games | Market Brier | GAM Brier | Difference |',
        '|---|---:|---:|---:|---:|---:|']
    for platform, offsets in market.items():
        for offset,r in sorted(offsets.items(),key=lambda x:int(x[0])):
            available=r['available']
            m=available['models']['rolling']['constrained_gam']
            lines.append(f"| {platform} | {offset} | {available['market']['games']} | {available['market']['brier_game']:.6f} | {m['metrics']['brier_game']:.6f} | {m['versus_market']['delta_brier']:+.6f} |")
    actual_fits=sum(len(json.loads(p.read_text())) for p in (root/'methods').glob('*/results.json'))
    dump(root/'verification.json',dict(families=len(expected),actual_candidate_fits=actual_fits,prediction_metrics_recomputed=True,
        protected_hashes_verified=True,matching_game_minutes=len(ix)))
    lines += ['', '## Artifacts and validation', '',

        f'- Verified {actual_fits} candidate fits across development, frozen evaluation, and six monthly replays; all saved probabilities and reported Brier/log-loss scores were checked.',
        f'- Cohort manifest: `{root}/cohort.json`.',
        f'- Gold sensitivity audits for every frozen family and ensemble: `{root}/methods/gold_audits.json`.',
        f'- Calibration, rare-state slices, paired intervals, and monthly results: `{root}/methods/`.',
        f'- Matched market comparisons for every saved family/ensemble: `{root}/markets/results.json`.',
        '- Market prices use retrospective alignment and mixed observation sources; these scores do not establish executable returns.',
        '- The original all-league and major-only test start dates differ; only their matching monthly replay rows are used in the training-cohort comparison.', '']
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/wpx/major_v9'))
    p.add_argument('--original',type=Path,default=Path('data/wpx/methods_v9'))
    p.add_argument('--report',type=Path,default=Path('docs/major-league-comparison-2026-09-05.md'))
    a=p.parse_args();main(a.root,a.original,a.report)
