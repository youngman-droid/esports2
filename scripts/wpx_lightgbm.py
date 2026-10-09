"""Compare LightGBM with the live-contract GAM on consumed historical data.

Optional experiment dependency: lightgbm==4.6.0. Reuses wpbench's nested
chronological split, stacked priors, equal-game weights and calibration.
Writes diagnostic results only; never changes production or the registry.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpbench, wpgam


def run(dataset, output):
    import lightgbm as lgb
    registry = json.loads((Path(wpgam.OUT_DIR) / 'evaluation_registry.json').read_text())
    with np.load(dataset, allow_pickle=True) as d:
        dates = d['date'].astype(str)
        if any(dates == '') or max(dates) > registry['consumed_through']:
            raise ValueError('Use only dated, consumed data; preserve fresh evaluation games.')

    original_fit, original_predict = wpbench._fit_method, wpbench._predict_method
    original_blend, original_paired = wpbench._fit_blend, wpbench._paired_differences
    captures = {}

    def specs(quick=False):
        tree_specs = [
            dict(num_leaves=7, min_child_samples=400, reg_lambda=30., n_estimators=400),
            dict(num_leaves=15, min_child_samples=200, reg_lambda=15., n_estimators=400),
            dict(num_leaves=31, min_child_samples=400, reg_lambda=30., n_estimators=400),
        ]
        return {
            'constrained_gam': [dict(l2=24., smooth=70.)],
            'monotone_hist_boost': [dict(max_leaf_nodes=31, min_samples_leaf=200, l2=15.)],
            'lightgbm': tree_specs,
            'monotone_lightgbm': tree_specs,
        }

    def fit(family, spec, raw, y, gids, t_min):
        if family not in ('lightgbm', 'monotone_lightgbm'):
            return original_fit(family, spec, raw, y, gids, t_min)
        X, scale = wpbench._tree_design_fit(raw, t_min)
        params = dict(objective='binary', learning_rate=0.03,
                      n_jobs=4, random_state=43, verbosity=-1,
                      deterministic=True, force_col_wise=True,
                      feature_fraction=1., bagging_fraction=1., **spec)
        if family == 'monotone_lightgbm':
            params['monotone_constraints'] = [0] + [
                int(n in wpgam.MONOTONE_FEATURES)
                for n in wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES]
            params['monotone_constraints_method'] = 'advanced'
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y, sample_weight=wpgam._game_balanced_weights(gids))
        if model.n_estimators_ < 1:
            raise RuntimeError('LightGBM did not fit any trees')
        return dict(family=family, estimator=model, scale=scale)

    def predict(model, raw, t_min):
        if model['family'] not in ('lightgbm', 'monotone_lightgbm'):
            return original_predict(model, raw, t_min)
        X = wpbench._tree_design_apply(raw, t_min, model['scale'])
        # Native Booster avoids sklearn's spurious feature-name warning for
        # numpy inputs; inference is otherwise identical to predict_proba.
        p = model['estimator'].booster_.predict(X)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise RuntimeError('Invalid LightGBM probabilities')
        return p

    def blend(predictions, y, gids, members):
        if members == ['constrained_gam', 'monotone_hist_boost']:
            members = members + ['monotone_lightgbm']
        return original_blend(predictions, y, gids, members)

    def paired(predictions, y, gids, best, **kwargs):
        captures.update(predictions=predictions, y=y, gid=gids)
        return original_paired(predictions, y, gids, best, **kwargs)

    # Restrict monkeypatching to this run; importing the script has no effect.
    with patch.multiple(wpbench,
                        FAMILIES=('constrained_gam', 'monotone_hist_boost',
                                  'lightgbm', 'monotone_lightgbm'),
                        _specs=specs, _fit_method=fit, _predict_method=predict,
                        _fit_blend=blend, _paired_differences=paired):
        result = wpbench.run(str(dataset), output_path=str(output), bootstrap=1000)
    result['paired_vs_gam'] = original_paired(
        captures['predictions'], captures['y'], captures['gid'], 'constrained_gam')
    result['experiment'] = {
        'lightgbm_version': lgb.__version__,
        'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'registry_consumed_through': registry['consumed_through'],
        'production_changed': False,
        'scope': 'Existing live feature contract; no individual champion categorical columns. Retrospective diagnostic only.',
        'monotonicity': 'Partial feature constraints only; end-to-end live sweeps not run.',
        'fixed_parameters': dict(learning_rate=0.03, seed=43, num_threads=4,
                                 monotone_method='advanced', n_estimators=400),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    np.savez_compressed(output.with_suffix('.npz'), gid=captures['gid'], y=captures['y'],
                        **captures['predictions'])
    wpbench.report(result)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(wpgam.OUT_DIR) / 'states.npz')
    parser.add_argument('--output', type=Path, default=Path(wpgam.OUT_DIR) / 'lightgbm_benchmark.json')
    args = parser.parse_args()
    run(args.dataset, args.output)
