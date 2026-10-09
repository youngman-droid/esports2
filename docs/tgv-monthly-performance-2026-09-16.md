# Last-three-month champion substitution comparison

Requested window: June 16–September 16, 2026. Saved comparable predictions cover June 17–September 2 only. No scores are available in this snapshot for September 3–16; this is an incomplete window.

TGV means replacement of champion weights in our model, not TGV’s complete standalone model. No refitting. Lower Brier and log loss are better.

| Month | Maps | Our Brier | TGV Brier | Our log loss | TGV log loss | Our accuracy | TGV accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-06 | 51 | 0.24267 | 0.24498 | 0.68156 | 0.68317 | 50.98% | 60.78% |
| 2026-07 | 862 | 0.20728 | 0.20972 | 0.60412 | 0.61002 | 67.40% | 67.75% |
| 2026-08 | 1358 | 0.20669 | 0.21091 | 0.59924 | 0.60894 | 69.29% | 66.57% |
| 2026-09 | 96 | 0.20740 | 0.19479 | 0.60340 | 0.57698 | 65.62% | 71.88% |
| overall | 2367 | 0.20771 | 0.21056 | 0.60296 | 0.60963 | 68.06% | 67.09% |

## Frozen draft-plus-comfort cross-check

| Month | Maps | Our Brier | TGV Brier | Our log loss | TGV log loss |
|---|---:|---:|---:|---:|---:|
| 2026-06 | 51 | 0.25871 | 0.25492 | 0.71643 | 0.70438 |
| 2026-07 | 862 | 0.20734 | 0.20610 | 0.59935 | 0.59774 |
| 2026-08 | 1358 | 0.21072 | 0.21149 | 0.60871 | 0.61054 |
| 2026-09 | 96 | 0.21444 | 0.20295 | 0.61762 | 0.59297 |
| overall | 2367 | 0.21067 | 0.21012 | 0.60798 | 0.60719 |

The current model fit and TGV release postdate the evaluated games. This is retrospective, not a clean prospective holdout. The frozen baseline predates the games, but the replacement TGV weights do not. The earlier paired bootstrap found the frozen aggregate gain inconclusive. Partial June and September samples should not be treated as whole-month results.

Method and aggregate uncertainty: [substitution test](tgv-champion-swap-test-2026-09-16.md).
