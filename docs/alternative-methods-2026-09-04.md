# Alternative methods after the v9 repairs — 2026-09-04

The **champion/patch-aware GAM is the strongest alternative to investigate
further**, but no replacement is established. It has the best monthly replay
Brier and passes the sampled gold audit. Its improvement is uncertain, and it
loses to the repaired core GAM in development and in the frozen comparison.
Keep the constrained architecture as the reference; no model was promoted.

## Results and decisions

All scores balance games equally; lower Brier is better. The table is sorted
by monthly replay results for readability. Settings and ensemble weights were
already frozen before these later results were inspected.

| Method | Development Brier | Frozen Brier | Monthly replay Brier | Gold reversals / 100,000 |
|---|---:|---:|---:|---:|
| Champion/patch-aware GAM | 0.149526 | 0.145042 | 0.143160 | 0 |
| Unrestricted ensemble | weights fitted on development | 0.144392 | 0.143377 | 18 |
| Monotone ensemble | weights fitted on development | 0.144496 | 0.143461 | 0 |
| Unconstrained GAM | 0.148725 | 0.144336 | 0.143565 | 28 |
| Repaired constrained GAM | 0.148738 | 0.144386 | 0.143596 | 0 |
| Monotone spline GAM | 0.148887 | 0.144563 | 0.143777 | 0 |
| GAM + monotone boosting | 0.149076 | 0.145305 | 0.144566 | 0 |
| Neural network ensemble | 0.150398 | 0.145483 | 0.144661 | 41 |
| Random Fourier kernel features | 0.151098 | 0.146003 | 0.145032 | 537 |
| Ridge logistic regression | 0.150490 | 0.146458 | 0.145917 | 6 |
| Monotone histogram boosting | 0.150489 | 0.146654 | 0.145969 | 0 |
| Monotone LightGBM | 0.150567 | 0.146873 | 0.146105 | 0 |
| LightGBM | 0.151388 | 0.146911 | 0.146443 | 387 |
| Histogram boosting | 0.151743 | 0.147648 | 0.147033 | 1,941 |
| Champion/patch-aware LightGBM | 0.156119 | 0.150668 | 0.150320 | 10 |
| Champion/patch-aware monotone LightGBM | 0.156581 | 0.151217 | 0.150756 | 0 |

The richer GAM improves monthly Brier by **−0.000436**, with a 95% paired
series/date-match interval **[−0.001647, +0.000804]**. Resampling whole dates gives
**[−0.001496, +0.000711]**. Monthly log loss improves from 0.436009 to 0.434891.
However, its development Brier is 0.149526 versus 0.148738 for the reference,
and its frozen Brier is 0.145042 versus 0.144386. It was not the single-model
winner selected in development. Its favorable replay result is a follow-up
hypothesis, not confirmation of a selected winner.

The monotone ensemble is the other reasonable shortlist entry: monthly Brier
improves by **−0.000135**, interval **[−0.000552, +0.000287]**, with no sampled
gold reversals. Its frozen result is also worse than the reference. The fixed
ensemble weights are approximately 54.9% constrained GAM, 34.7% GAM-plus-boosting,
6.6% rich GAM, 2.2% spline GAM and 1.7% rich monotone LightGBM.

The development-selected **unconstrained GAM is unsuitable for promotion**.
Its monthly gain is only −0.000031, interval [−0.000174, +0.000121], and it
reintroduces 28 gold reversals. In the worst sampled case, giving blue support
1,000 gold lowers P(blue) by **2.107 percentage points**. The unrestricted
ensemble also fails, with 18 reversals. Better aggregate scores do not repair
these input-behavior failures.

Spline GAMs, standalone boosting, neural networks, random Fourier features and
ridge logistic regression do not improve overall replay Brier in this search.
Native champion/patch categories make the tested LightGBM configurations worse.
This conclusion applies to the recorded configurations and training protocol;
it is not a claim that every possible neural or tree architecture has been
ruled out.

![Paired alternative-method comparisons](/Users/itch/Documents/Github/esports2/data/wpx/methods_v9/comparison.png)

## Stability and rare states

The rich GAM wins May and August, loses June and September, and nearly ties
July. September contains only 19 games and contributes its actual game weight
to the aggregate. Its point improvements concentrate in midgame (−0.001070
Brier) and late game (−0.000830), with both slice intervals spanning zero.
Active-Elder states (+0.001617) and the first three observed days of a patch
(+0.000458) are worse on point estimates, also with intervals spanning zero.
The experiment therefore does not establish a special Elder or new-patch
advantage from the richer inputs.

| Replay month | Games | Repaired GAM | Rich GAM | Monotone ensemble |
|---|---:|---:|---:|---:|
| 2026-05 | 1,132 | 0.141275 | 0.139697 | 0.140924 |
| 2026-06 | 328 | 0.157017 | 0.158311 | 0.158274 |
| 2026-07 | 619 | 0.139710 | 0.139854 | 0.139549 |
| 2026-08 | 999 | 0.143680 | 0.143394 | 0.143380 |
| 2026-09 | 19 | 0.172350 | 0.183316 | 0.170608 |

## Verification

- **160 candidate fits completed**, plus the nested prior and GAM-stack fits.
- **118 regression tests pass**, including new checks for chronological state
  stacking, rare-feature support, equal-game scoring, coherent gold changes
  and import isolation.
- **1,600,000 gold comparisons** across the 14 frozen families and two ensembles
  found **2,968 reversals**. Eight methods/ensembles passed all 100,000 checks;
  eight failed at least one. These are sensitivity checks, not independent
  accuracy samples. They do not certify every monthly refit or future artifact.
- All scored probabilities are finite. All 14 serialized frozen models reproduce
  their saved predictions exactly on 100 sampled states. Small-batch NumPy BLAS
  warnings during the extra round-trip check were investigated: independent
  neural/kernel contractions agree within 4.5×10⁻¹⁶ and are finite.
- The deployed model and evaluation registry retain their original SHA-256
  hashes. Production remains v8; the repaired v9 model from the earlier task
  remains staged. No fresh outcome block was consumed.

## Protocol

This reevaluation compares fourteen model families and two ensembles with the
repaired feature contract. Earlier saved alternative-model scores used different
preprocessing and calibration, so they are not substituted for new results.

Every family uses the same fixed-minute states, equal total training weight per
game, and strictly earlier stacked team/champion priors. The core features
preserve rare objectives and use five direct role-gold differences. Rich models
add individual signed gold, champion slots, season, patch, competition, and
pooled champion/resource interactions. Current-game final statistics and market
prices are excluded.

Three development blocks begin May 2, 2025, August 5, 2025, and February 6, 2026.
Each reserves the preceding 28 calendar days for calibration, with model fitting
strictly earlier. Mean development game-balanced Brier selects each family's
specification and calibration method (none, temperature, or Platt). Both fitted
calibration coefficients remain intact; no training-prediction intercept
recentering occurs. Tree counts are fixed within each specification, without
using calibration labels for early stopping. GAM-plus-boosting uses chronological
state stacking as well as chronological upstream priors.

The later diagnostic contains 3,097 games / 99,408 states from May 1 through
September 2, 2026. Frozen settings are also replayed with monthly refits and
earlier 28-day calibration blocks. The same May predictions are reused exactly.
Two nonnegative ensembles select their weights on the earlier development
predictions: one admits every family, while the other admits only methods
designed to preserve the direction of gold effects. Ensemble weights also
balance the three development blocks equally and remain fixed during replay.

The 95% paired intervals resample whole date/match clusters. Separate whole-date
intervals test sensitivity to broader shared conditions. They are exploratory,
not adjusted for searching multiple families. These dates had already been
consumed by the evaluation registry; this is not a fresh promotion test.

Each selected frozen model receives 100,000 coherent gold checks: 100 and 1,000
gold added separately to all ten players across the same 5,000 sampled states.
Total gold, role difference, momentum, relative share, and the affected player's
absolute gold move together. These sampled checks do not constitute a complete
proof of behavior outside the tested states. Calibration, early/mid/late phases,
Elder states, inhibitor advantages and newly observed patch dates are reported.
The later block contains 492 active-Elder states across 222 games, 772 states
with an Elder-count advantage, and 6,598 states with an inhibitor advantage.
Individual gold snapshots are complete for all 500,122 fixed-minute states,
including all 99,408 later states, so enriched models receive no advantage from
dropping difficult games with missing gold.

The calibration separation follows the requirement for independent fitting and
calibration data in the [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).
The constrained tree variants use LightGBM's `advanced` monotonicity method on
the oriented continuous inputs; champion and patch categories remain
unconstrained, consistent with the [LightGBM parameter documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#monotone_constraints_method).

## Reproduction and provenance

The isolated runner is `research/wpx_methods_v9.py`. Optional dependencies are
LightGBM 4.6.0 and the existing project Python libraries. On this Mac it runs in
`/tmp/esports2-methods-v9-env`, with the existing sklearn OpenMP runtime:

```bash
DYLD_LIBRARY_PATH=/Users/itch/Library/Python/3.9/lib/python/site-packages/sklearn/.dylibs \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/tmp/esports2-methods-v9-env/bin/python research/wpx_methods_v9.py \
  --output data/wpx/methods_v9_rerun \
  --resource-cache data/wpx/adapt_cache/408b0e7b37ff020354920067
python3 -m unittest discover -s tests
```

`data/wpx/methods_v9/plan.json` records the dataset/resource hashes, every tested
specification, the source hashes, package versions, and protected production
artifact hashes. The directory also retains source snapshots, fitted models,
stage results, selection, predictions, sensitivity checks and comparison
intervals. A changed source or plan requires a new output directory. Importing
the alternative-model helpers no longer changes shared benchmark dispatch.
