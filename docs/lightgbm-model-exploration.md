# LightGBM benchmark — 2026-09-04

LightGBM had not previously been tested in this repository. Earlier boosting
experiments used scikit-learn's `HistGradientBoostingClassifier`.

The new experiment does not support replacing the production GAM. Neither
standalone LightGBM nor validation-weighted blends improved the later diagnostic
block on the existing live feature contract.

## Comparison

`scripts/wpx_lightgbm.py` uses LightGBM 4.6.0 and reuses `wpbench`'s chronological
split: 9,845 inner-training games, 2,466 validation games, and 3,097 later games
starting 2026-05-01. The later block contains 99,408 fixed-minute states. Each
game receives equal total weight. All models share the same stacked priors,
live feature contract, training-only transformations and calibration procedure.
Calibration slopes come from validation; final calibration intercepts are
refit on training predictions, following the existing benchmark.

Each LightGBM family tests three fixed specifications: 7/15/31 leaves,
400/200/400 minimum leaf observations, and L2 penalties 30/15/30. All use 400
trees, learning rate 0.03, seed 43 and four threads. No early stopping or broad
hyperparameter search was performed. Hyperparameters and blend weights use
validation only.

The monotone family uses the GAM's oriented feature list with LightGBM's
`advanced` constraint method. LightGBM documents this as less restrictive than
its basic method. These are partial feature constraints, not a substitute for
the repository's end-to-end live-input sweeps.
[LightGBM parameter documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#monotone_constraints_method)

| Model | Validation Brier, selected setting before calibration | Later Brier after calibration |
|---|---:|---:|
| Constrained GAM | **0.145086** | **0.143164** |
| LightGBM | 0.146580 | 0.14553 |
| Monotone LightGBM | 0.147396 | 0.14552 |
| Monotone sklearn histogram boosting | 0.147318 | 0.14497 |

Lower is better. LightGBM's paired game-weighted Brier difference versus GAM is
approximately +0.00237, with a game-block bootstrap 95% interval
[+0.00136, +0.00340]. The monotone variant's difference is approximately +0.00236,
interval [+0.00153, +0.00323]. Both also worsen log loss. These intervals resample
games, not whole series; within-series dependence remains a limitation.

The validation-fit unconstrained blend gives about 76.7% weight to GAM and
23.3% to LightGBM. Its later Brier is 0.14327, versus 0.14316 for GAM. The monotone
blend gives about 94.6% to GAM and 5.4% to monotone LightGBM, scoring 0.14318.
Neither establishes an improvement; both paired intervals include zero.

## Interpretation and remaining scope

This tests a model-family substitution under the existing feature contract.
It does not test native per-role champion categories, individual gold levels,
or the SIDO-inspired performance priors. Champion information reaches these
models through the existing compressed champion channels.

The next distinct LightGBM hypothesis would expose ten champion categories
alongside individual resource levels, allowing champion×gold×time interactions.
That needs a richer historical/live feature contract and another prespecified
chronological comparison. The present result should not be generalized to
that experiment, but it gives no evidence for changing the deployed model now.

This is a retrospective diagnostic on dates already consumed by the evaluation
registry. The script rejects data beyond that date. Production models, the
registry and database were not changed. LightGBM was installed in a temporary
virtual environment, leaving project runtime dependencies unchanged.

## Artifacts

- `scripts/wpx_lightgbm.py`: reproducible adapter; optional dependency
  `lightgbm==4.6.0` and an available OpenMP runtime on macOS.
- `data/wpx/lightgbm_benchmark.json`: full metrics, selected settings, version,
  dataset SHA-256, calibration and blend weights.
- `data/wpx/lightgbm_benchmark.npz`: later-block predictions and labels.

Run `python scripts/wpx_lightgbm.py` in an environment with LightGBM available.
The macOS run used the existing sklearn-bundled `libomp.dylib` through
`DYLD_LIBRARY_PATH`. The benchmark completed, predictions passed finite/range
checks, and the adapter passed Python compilation and whitespace checks.
No production monotonicity audit was run because this was an offline comparison.
