# TGV champion-weight substitution test — 2026-09-16

A direct substitution did not improve the current stored outcome draft model on the available historical maps. A frozen January draft-plus-comfort baseline showed a small, inconclusive gain. Neither establishes prospective performance.

## Method

Evaluated 2,367 maps, June 17–September 2, 2026, patches 16.12–16.17. Replaced all our champion pick contributions (base, role and patch; own and enemy) with TGV full-model role/patch champion strengths. Other contributions and original baseline probabilities were retained, including comfort in the frozen experiment. No fitting, scale tuning or deployment occurred.

`new logit = original logit − our champion contribution + TGV blue champion sum − TGV red champion sum`

This tests the stored outcome draft model and the frozen draft-plus-comfort experiment, not the live GAM. Full-model TGV strengths and the soloqueue-only table were tested separately.

## Results

Brier and log loss: lower is better. Accuracy uses a 50% threshold.

| Baseline / substitution | Brier | Log loss | Accuracy |
|---|---:|---:|---:|
| Current: ours | 0.20771 | 0.60296 | 68.06% |
| Current: TGV full | 0.21056 | 0.60963 | 67.09% |
| Current: TGV soloqueue | 0.21024 | 0.60903 | 67.22% |
| Current: 50/50 blend | 0.20843 | 0.60476 | 67.55% |
| Current: remove champion terms | 0.21135 | 0.61146 | 67.13% |
| Frozen draft + comfort: ours | 0.21067 | 0.60798 | 65.95% |
| Frozen draft + comfort: TGV full | 0.21012 | 0.60719 | 66.71% |
| Frozen draft + comfort: TGV soloqueue | 0.20994 | 0.60674 | 66.79% |
| Frozen draft + comfort: remove champion terms | 0.21099 | 0.60903 | 66.54% |

Current-model TGV substitution increased Brier by 0.00285 (95% paired cluster-bootstrap interval +0.00083 to +0.00478) and log loss by 0.00667 (+0.00218 to +0.01102). Median absolute prediction movement was 3.39 percentage points.

Frozen-model TGV substitution decreased Brier by 0.00056 (interval −0.00239 to +0.00122) and log loss by 0.00080 (−0.00504 to +0.00334). Both intervals include no improvement. Bootstrap used 2,000 draws over 1,111 team-pair/date clusters.

Current-model Brier worsened on four of six patches; it improved on 16.13 and 16.17. Removing the two maps with champions outside TGV’s current role eligibility does not change the aggregate conclusion. No maps with supported patches were excluded for missing baseline predictions or champion mappings.

## Limits and decision

The stored current model was fitted September 16, after the evaluated games. TGV’s release is dated September 15 and also postdates all evaluated outcomes; its training overlap is unknown. The frozen baseline predates these games, but its TGV replacement does not. Bootstrap uncertainty does not correct this date leakage.

Champion weights are conditional on the rest of each model. A direct swap may require recalibration of scale or interactions; this experiment does not test a retrained hybrid. The results provide no basis to replace our champion values now. A fair next experiment would freeze both sources before new outcomes, tune any hybrid only on earlier data, and compare paired Brier and log loss on untouched future maps.

## Reproduction and verification

Run from the repository root: `PYTHONPATH=. python3 research/tgv_champion_swap_test.py`.

The stored baseline was reconstructed from its coefficients and Elo baseline for every evaluated map; maximum probability discrepancy was 9.48e−8. Inputs are the saved production snapshot, the September TGV catalog and soloqueue table, and the prior frozen draft/comfort experiment. Detailed patch results, confidence intervals and limitations are in `data/tgv/champ-swap-test/report.json`; per-map outputs are in `data/tgv/champ-swap-test/predictions.json`.
