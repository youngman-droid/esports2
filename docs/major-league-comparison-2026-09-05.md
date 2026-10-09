# Major-league model comparisons

On the 861 matching major-league games, major-only constrained GAM training scores 0.154188 versus 0.152380 for all-league training: 0.001807 higher (worse), with a 95% paired interval [-0.000675, +0.004223]. This experiment does not establish a benefit from discarding lower-tier training games.

The cohort contains 4,922 games and 162,238 fixed-minute states. The chronological later block begins 2026-04-18. All results are retrospective diagnostics on previously consumed dates.

Membership follows direct Worlds-slot leagues by season, including Worlds/MSI/First Stand. Academy/regional leagues, promotion events, other cups, and indirect LJL/LCO pathways are excluded. There are no LPL-labeled games in the source state dataset. Historical sequential Elo/form inputs are retained; all fitted prior, champion, calibration and state learners use the filtered cohort.

Settings and calibration are selected on three earlier blocks, then frozen for the later block and monthly replay. Scores balance games equally; intervals resample whole series. No production model was replaced.

| Model | Development Brier | Frozen Brier | Monthly replay Brier | Replay log loss | Gold reversals |
|---|---:|---:|---:|---:|---:|
| constrained_gam | 0.154649 | 0.149924 | 0.148936 | 0.454315 | 0 / 100000 |
| monotone_ensemble | ensemble | 0.150636 | 0.149108 | 0.454525 | 0 / 100000 |
| convex_ensemble | ensemble | 0.150608 | 0.149161 | 0.454418 | 15 / 100000 |
| unconstrained_gam | 0.154627 | 0.150132 | 0.149217 | 0.454981 | 2 / 100000 |
| spline_gam | 0.154770 | 0.150500 | 0.149529 | 0.456307 | 0 / 100000 |
| rich_gam | 0.155472 | 0.153931 | 0.149756 | 0.456516 | 0 / 100000 |
| ridge_logit | 0.155716 | 0.151162 | 0.150178 | 0.457492 | 0 / 100000 |
| rff_logit | 0.156916 | 0.151251 | 0.150217 | 0.457909 | 1574 / 100000 |
| gam_boost_stack | 0.155756 | 0.151238 | 0.150418 | 0.457860 | 0 / 100000 |
| mlp | 0.156257 | 0.151512 | 0.150462 | 0.457932 | 203 / 100000 |
| monotone_hist_boost | 0.157529 | 0.152868 | 0.151575 | 0.461101 | 0 / 100000 |
| monotone_lightgbm | 0.157732 | 0.153333 | 0.152289 | 0.462806 | 0 / 100000 |
| lightgbm | 0.158304 | 0.154412 | 0.153459 | 0.464981 | 1280 / 100000 |
| hist_boost | 0.162277 | 0.156218 | 0.153985 | 0.466308 | 2939 / 100000 |
| rich_lightgbm | 0.167258 | 0.159290 | 0.157433 | 0.477700 | 45 / 100000 |
| monotone_rich_lightgbm | 0.169775 | 0.160535 | 0.158144 | 0.479618 | 0 / 100000 |

## Does major-only training help?

These monthly replay predictions are joined by game and minute to the existing all-league replay. Both approaches are evaluated on exactly the same major-league states. Each suite tuned its own settings on its earlier development cohort; this compares the retuned pipelines, not a fixed-hyperparameter ablation. Negative differences favor major-only training.

| Model | Major-only Brier | All-league Brier | Difference [95% series interval] | Games |
|---|---:|---:|---:|---:|
| constrained_gam | 0.154188 | 0.152380 | +0.001807 [-0.000675, +0.004223] | 861 |
| unconstrained_gam | 0.154424 | 0.152453 | +0.001971 [-0.000598, +0.004439] | 861 |
| convex_ensemble | 0.154464 | 0.152546 | +0.001919 [-0.000577, +0.004171] | 861 |
| monotone_ensemble | 0.154482 | 0.152489 | +0.001992 [-0.000365, +0.004173] | 861 |
| spline_gam | 0.154918 | 0.152776 | +0.002143 [-0.000197, +0.004308] | 861 |
| rff_logit | 0.155259 | 0.153796 | +0.001463 [-0.001387, +0.004023] | 861 |
| rich_gam | 0.155291 | 0.152505 | +0.002786 [-0.000533, +0.006002] | 861 |
| ridge_logit | 0.155468 | 0.154049 | +0.001419 [-0.001191, +0.003844] | 861 |
| mlp | 0.155702 | 0.153193 | +0.002508 [-0.000639, +0.005360] | 861 |
| gam_boost_stack | 0.155708 | 0.153729 | +0.001979 [-0.000668, +0.004359] | 861 |
| monotone_hist_boost | 0.157393 | 0.154526 | +0.002867 [-0.000013, +0.005456] | 861 |
| monotone_lightgbm | 0.158064 | 0.154573 | +0.003491 [+0.000205, +0.006490] | 861 |
| lightgbm | 0.158701 | 0.155013 | +0.003688 [+0.000270, +0.006907] | 861 |
| hist_boost | 0.158945 | 0.155893 | +0.003052 [-0.001489, +0.007519] | 861 |
| monotone_rich_lightgbm | 0.163175 | 0.162380 | +0.000795 [-0.004008, +0.005710] | 861 |
| rich_lightgbm | 0.163189 | 0.161882 | +0.001308 [-0.003829, +0.006424] | 861 |

## Recency comparison

| Model | Replay Brier | Replay log loss |
|---|---:|---:|
| recency180 | 0.148806 | 0.453352 |
| repaired | 0.148937 | 0.454317 |

The recency change is -0.000131 Brier, with a 95% series interval [-0.001058, +0.000852]; the small point improvement is inconclusive.

## Matched market diagnostics

The table shows the major-only constrained GAM monthly replay against each market on matching available rows. Offsets advance the market lookup relative to reconstructed game time; the 45/195-second offsets test information-timing sensitivity.

| Platform | Offset (s) | Games | Market Brier | GAM Brier | Difference |
|---|---:|---:|---:|---:|---:|
| kalshi | 0 | 432 | 0.156428 | 0.156982 | +0.000554 |
| kalshi | 45 | 432 | 0.154818 | 0.159919 | +0.005101 |
| kalshi | 195 | 432 | 0.149698 | 0.168787 | +0.019088 |
| polymarket | 0 | 743 | 0.156195 | 0.155451 | -0.000744 |
| polymarket | 45 | 743 | 0.154204 | 0.158079 | +0.003875 |
| polymarket | 195 | 743 | 0.148574 | 0.167182 | +0.018608 |

## Artifacts and validation

- Verified 188 candidate fits across development, frozen evaluation, and six monthly replays; all saved probabilities and reported Brier/log-loss scores were checked.
- Cohort manifest: `data/wpx/major_v9/cohort.json`.
- Gold sensitivity audits for every frozen family and ensemble: `data/wpx/major_v9/methods/gold_audits.json`.
- Calibration, rare-state slices, paired intervals, and monthly results: `data/wpx/major_v9/methods/`.
- Matched market comparisons for every saved family/ensemble: `data/wpx/major_v9/markets/results.json`.
- Market prices use retrospective alignment and mixed observation sources; these scores do not establish executable returns.
- The original all-league and major-only test start dates differ; only their matching monthly replay rows are used in the training-cohort comparison.
