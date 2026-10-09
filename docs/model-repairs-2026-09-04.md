# Implementation and validation — 2026-09-04

All five recommendations from the model evaluation are implemented. Live
collection has been updated and restarted. The repaired GAM and the recency
challenger are staged experimental artifacts; the deployed v8 model is retained.

## Changes

1. **Preserve rare objectives.** The v9 scaler uses declared support bounds for
   discrete objectives, buffs, death counts and flags. A feature with observed
   variation cannot silently become constant through percentile clipping.
   Elder count and active-buff inputs now survive training and live inference.

2. **Make physical gold changes monotone.** Five direct role-gold differences
   replace four allocation deviations. Each has a nonnegative time curve,
   regularized alongside the shared total-gold effect. Adding gold to one
   player moves only nondecreasing inputs, including momentum and relative
   share. The deployment audit now changes individual players' gold, tests
   asymmetric states and clipping boundaries, covers 0–75 minutes, and uses
   unrounded GAM probabilities. `wpaudit` provides repeatable artifact and
   dataset-based sensitivity checks.

3. **Nest calibration correctly.** Champion-state fits are rebuilt inside the
   earlier calibration-training block. Pregame and champion stacking use
   expanding chronological folds whenever dates are available, with neutral
   scores for a small initial warmup block. Both calibration coefficients are
   retained from held-out predictions; the final intercept is no longer
   recentered on fitted training predictions. A regression test flips all
   calibration outcomes and verifies that the input logits to the calibrator
   remain unchanged. Insufficient or missing date history leaves predictions
   uncalibrated instead of silently reverting to random-fold calibration.

4. **Decouple collection and measure coverage.** The shadow recorder uses a
   bounded background quote cache, with at most four workers, a shared
   12-second lookup budget, and a 15-second maximum quote age. Cold, expired,
   future-dated, differently oriented, or different-attempt cache entries are
   missing rather than backfilled. Resolution and scoring use a separate
   background connection. Reports include eligible-row coverage, remaining
   games, upstream feed age and processing time after receipt. The process's
   source revision stays fixed until restart. Optional blend failures cannot
   prevent the independent model from recording.

5. **Retest 180-day decay.** `scripts/wpx_recency.py` compares only the repaired
   expanding-history GAM and its 180-day state-weight-decay challenger. Three
   development blocks select calibration and the candidate. A later frozen
   diagnostic and five monthly replay periods use those frozen choices, with
   28 earlier calendar days reserved for calibration. All inspected dates were
   already consumed by the registry.

The artifact contract is `wpgam_v9_physical_gold`. Inference also supports v8
using the feature names stored in that artifact. Existing historical and
market blends validate against the actual base artifact's kind, rather than
assuming the newest source-code constant describes the deployed model.

An architectural refresh now stages a `.candidate.npz` file and preserves the
incumbent and evaluation registry. The existing blend-versus-new-GAM gate is
not sufficient evidence to replace the incumbent GAM itself.

## Results

The recency experiment's later block contains **3,097 games / 99,408 states**.
All Brier scores below balance games equally.

| Repaired-contract model | Mean development Brier | Frozen later block | Monthly replay |
|---|---:|---:|---:|
| Expanding history, selected calibration: none | 0.148737 | **0.144388** | **0.143597** |
| 180-day decay, selected calibration: Platt | **0.148503** | 0.144431 | 0.143742 |

Recency-minus-expanding replay Brier is **+0.000145**, with a paired date/match
cluster 95% interval **[-0.000695, +0.001013]**. The frozen-block difference is
+0.000044, interval [-0.000258, +0.000353]. Recency won development selection
but did not improve either later comparison. It is not adopted.

The new 28-day calibration protocol withholds the latest month from the state
fit; its scores must not be compared directly with the older production
protocol as if only the architecture changed. A separate comparison therefore
fits v9 on the **same outer training games as v8**, using the corrected nested
calibration and no recency weighting:

| Production training protocol | Later Brier |
|---|---:|
| v8 reference | 0.143168 |
| Repaired v9 | 0.143540 |

V9-minus-v8 is **+0.000373**, interval **[-0.000128, +0.000882]**. This does not
demonstrate an overall accuracy gain. It also does not establish equivalence;
future evidence is needed before promotion. The recorded v8 implementation was
refitted for this comparison; the all-data deployed artifact was not scored
in-sample and presented as a holdout result.

## Verification

- **113 tests pass**, including the original suite and new feature, calibration,
  compatibility, deployment-isolation and asynchronous-collection regressions.
- **2,131,100 physical-gold comparisons pass with zero reversals** across the
  frozen repaired and recency models, current recency candidate, production-
  protocol reference, and current repaired candidate. These are sensitivity
  tests on sampled/coherent states, not independent accuracy observations.
- Both current candidates pass **1,204 live shape comparisons** and the artifact
  support/constraint checks. Probabilities are finite and artifacts round-trip.
- On the recency candidate, the previously recorded red-support reversal now
  moves P(blue) from **47.49% to 47.25%** when red support receives 1,000 gold.
  The same 35-minute control gives P(blue) **50.54%, 65.84%, 78.44%** for red
  Elder, no Elder, and blue Elder respectively. These illustrate responsiveness;
  they are not claims that those particular probabilities are perfectly calibrated.
- The restarted recorder has captured actual live frames under
  `shadow_v12_nonblocking_quotes-00b9f6bfa179`, including cached quotes. Observed
  market-stage time was about **1 ms** on a completed capture. The feed was
  still roughly **190–200 seconds old**; local concurrency cannot remove that
  upstream delay. The **90-second primary lead limit and 100-game target** are
  preserved, and earlier ledgers remain queryable.
- The deployed model SHA-256 remains
  `fed34df9a7831a1d11c83d81a4b53f63661f108338890af737dbcda085c46589`.

## Artifacts and reproduction

`data/wpx/repairs_v9/` contains the frozen plan and selection, per-block results,
predictions, full comparison intervals, input audits and source snapshots.

- `repaired_candidate.npz`: current v9 candidate without recency, fitted through
  September 2 with the repaired production pipeline; **not deployed**.
- `repaired_candidate_status.json`: its artifact, live and sampled gold audits.
- `candidate.npz`: development-selected recency challenger; retained as an
  experimental artifact, **not deployed**.
- `frozen.json`, `rolling.json`, `production_protocol.json`: accuracy diagnostics.
- `incumbent_source.py`, `source_snapshot/`: implementation provenance.

```bash
python3 -m unittest discover -s tests
python3 scripts/wpx_recency.py \
  --output data/wpx/repairs_v9_rerun \
  --resource-cache data/wpx/adapt_cache/408b0e7b37ff020354920067 \
  --incumbent-source data/wpx/repairs_v9/incumbent_source.py
python3 -m lol_ticker shadow status
python3 -m lol_ticker shadow score
```

The standalone recency runner needs the project Python dependencies, not
LightGBM. Use a new output directory after source or plan changes. A cache miss
for historical player gold uses a consistent read-only database transaction.
Fresh outcomes beyond the registry's consumed date are rejected by this
diagnostic runner.

The code and collector fixes are complete. Model promotion and the prospective
100-game result remain pending; no new holdout outcomes were consumed here.
