# Model evaluation — 2026-09-04

The constrained GAM is the strongest established model in this repository, but
two verified feature-processing defects should be fixed before expanding the
model search. Its historical accuracy is credible as a retrospective diagnostic;
the current live ledger is too small to establish performance against markets.

This review inspected the current source, deployed artifact, saved experiments,
and shadow results; ran all 99 tests; and performed additional inference audits.
It did not retrain or deploy a model, change database contents, or consume a new
evaluation block. Existing uncommitted work was preserved.

## What the existing evidence supports

The deployed artifact is `wpgam_v8_recency_series`, fitted on 15,408 games and
500,122 fixed-minute states. Its SHA-256 is
`fed34df9a7831a1d11c83d81a4b53f63661f108338890af737dbcda085c46589`.
The manifest uses GAM alone.

The latest chronological method benchmark uses 3,097 later games / 99,408
states. Lower game-balanced Brier is better:

| Model | Brier | Game-balanced log loss |
|---|---:|---:|
| Pregame baseline | 0.210971 | 0.608304 |
| Constrained GAM | **0.143164** | **0.434756** |
| Unconstrained GAM | 0.143241 | 0.434882 |
| Monotone histogram boosting | 0.144965 | 0.439733 |
| Ridge logistic regression | 0.145650 | 0.441321 |
| Histogram boosting | 0.145877 | 0.442662 |

The separate matched LightGBM experiment scores approximately 0.14553; its
paired difference from GAM is about +0.00237, with a 95% game-bootstrap interval
of [+0.00136, +0.00340]. Tested ensembles also fail to improve the later block.
The GAM benchmark's calibration slope is 1.007 and intercept 0.088: aggregate
calibration looks reasonable, with some residual side/base-rate offset.

These are refitted historical reference models, not out-of-sample scores of
the all-data deployed artifact. The diagnostic dates have already been used
for development. They should not be described as a new confirmatory test.

Sources: `data/wpx/method_benchmark.json`, `walk_forward_diagnostics.json`,
`lightgbm_benchmark.json`, and `model_live_gam.npz`.

The latest registered forward blend test is newer than parts of the README:
131 games from August 25 through September 2, GAM Brier 0.167414 versus candidate
blend 0.167504, delta +0.000090, interval [-0.000517, +0.000742]. The candidate
also fails the shape gate. The registry is now consumed through September 2.
The higher absolute Brier in this small, different cohort does not by itself
establish model deterioration.

## 1. Restore the Elder features erased by clipping

**Verified defect.** `_scale_fit` applies the same 0.5th/99.5th percentile bounds
to every feature. In the deployed artifact, both `d_elder` and `elder_active`
have lower bound zero, upper bound zero, and zero coefficients at all six knots.
Every nonzero live Elder input is therefore erased.

This is not absence of historical data: there are 4,471 fixed-minute states
with a nonzero Elder-count difference and 2,803 with a nonzero active-buff
difference, each covering 1,250 games. Each signed tail is too rare to survive
the global clipping rule.

With otherwise identical neutral inputs at 35 minutes and four dragons per
side, P(blue) is **57.94% with no Elder, blue Elder, or red Elder**. Live inference
reports the Elder inputs as clipped and their contribution as zero.

Use feature-specific preprocessing: preserve the legal values of discrete
objectives/buffs and reserve percentile clipping for suitable continuous
measurements. Keep shrinkage for rare effects. Add an artifact audit that
rejects accidentally constant transforms of features with observed variation,
plus late-game sensitivity and calibration checks. The existing monotonicity
gate permits a completely flat response, so it cannot detect this defect.

Relevant code: `lol_ticker/wpgam.py:258`, `lol_ticker/wpgam.py:269`.

## 2. Enforce monotonicity for an actual player's gold

**Verified defect in the deployed model, not just an experimental candidate.**
The ordinary shape audit passes all 372 comparisons. However, its gold sweep
distributes gold evenly across roles, leaving the allocation features zero.
The four allocation coefficients are unconstrained, and independent clipping
can saturate the compensating total/relative-gold effects.

A deterministic sample of 20,000 historical fixed-minute states, seed 904,
was perturbed by adding 100 and 1,000 gold to each of ten player slots. Total
gold, momentum, relative share, and role allocation were updated coherently;
all other inputs stayed fixed. There were **541 direction reversals across
400,000 comparisons**, affecting 168 sampled states. All predictions were
finite. These are synthetic sensitivity checks, not new accuracy measurements
or an estimate of the frequency of real gameplay errors.

The largest reversal was reproduced through `wpgam.predict_live`: in game
59769 at minute 2, adding 1,000 gold to **red support** changed **P(blue) from
46.02% to 49.15%**. The relative-gold input was already clipped. Even the smaller
100-gold perturbations produced reversals of about 0.31 percentage points.

Reparameterize gold using five monotone role-gold effects, pooled toward a
shared time curve, or constrain the complete derivative of every physical
gold change. Bounds on isolated engineered columns are insufficient. Audit
each side and role over real states, clipping boundaries, and the whole time
range, using unrounded probabilities. Keep exact live-inference regression
cases for observed failures.

Relevant code: `lol_ticker/wpgam.py:157`, `lol_ticker/wpgam.py:187`,
`lol_ticker/wpdeploy.py:130`, `lol_ticker/wpdeploy.py:189`.

The audit results, artifact/dataset hashes, full before/after live inputs and
champions are in `data/wpx/model_audit_gold_20260904.json`. That file allows the
largest failure to be replayed directly:

```python
import json
from lol_ticker import wpgam

with open("data/wpx/model_audit_gold_20260904.json") as f:
    case = json.load(f)["live_reproduction"]
for key in ("before_state", "after_state"):
    print(wpgam.predict_live(case[key], case["blue_champions"],
                            case["red_champions"])["p_blue"])
```

## 3. Make the complete calibration stack chronological

**Verified protocol issue; the accuracy impact has not been measured.**
`_temporal_state_calibration` splits the state and pregame fits by date, but
reuses `cs_by_gid` from champion fits made with random folds across the entire
outer training block, including its later calibration dates. Thus calibration
outcomes can influence the champion features used to fit the earlier state
learner. Excluding a game's own outcome from its own champion score does not
remove this indirect dependence through the other training games.

Rebuild the champion learner inside the earlier calibration-training block:
use earlier-block OOF scores for state training and an earlier-only full fit
for calibration predictions. Prefer rolling chronological folds for stacked
training features when testing a deployable temporal workflow. Also compare
the current in-sample intercept recentering with calibration learned solely
from held-out predictions.

The code comment claiming the bias is necessarily conservative is unsupported.
Its direction and size require an ablation. The outer future test labels are
still excluded, so this finding does not invalidate every reported holdout
score. The newer adaptive experiment already separates the shared prior fit
from its calibration block and provides a useful implementation pattern.

Relevant code: `lol_ticker/wpgam.py:585`, `lol_ticker/wpgam.py:657`,
`scripts/wpx_adapt.py:125`. The requirement for disjoint model/calibration data
is also explained in the [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

## 4. Improve live timing and accumulate usable evidence

The saved shadow score at **2026-09-04 17:53:33 UTC** uses protocol
`shadow_v11_market_event_match-b1056ad0841a`. Only **four games / 58 states per
platform** qualify under its registered 90-second market-lead limit. The
paired intervals are wide and include material benefit and harm.

The all-leads data contains 21 Kalshi games / 216 states and 18 Polymarket
games / 140 states, with median market leads of approximately 195 and 194
seconds. The primary filter excludes 158 and 82 rows respectively. These
aggregates mix information timing and cannot establish a model edge.

Decouple quote collection from model-frame capture, use bounded network
deadlines and timestamped quote caches, and report eligible-row coverage
alongside forecast scores. Measure upstream feed age separately from local
processing latency: the latest log includes a roughly 194-second-old frame
despite only 5.6 seconds of local capture work. Fixing local blocking alone
will not eliminate upstream delay. Preserve the registered cutoff and collect
the planned 100 eligible resolved games per platform before interpreting the
confirmatory result; do not relax the filter based on outcomes.

Sources: `data/wpx/shadow_score.json`, `data/shadow.log`,
`lol_ticker/shadow.py:358`.

## 5. Resume bounded accuracy experiments after the fixes

The best remaining established candidate is **180-day recency weighting** of
the existing GAM. In the newer monthly-replay protocol it moves Brier from
0.143982 to 0.143780: delta -0.000202, with a series/date-cluster interval
[-0.000497, +0.000100]. This is promising but inconclusive, and its frozen
candidate fails its own gold audit.

The newer rich champion/patch/resource configurations have already been tried:
the best rich GAM loses on mean development Brier, as does the selected tree.
Generic additional player-performance priors also failed their pregame screen.
Do not repeat those proposals as if they were untested. Retest recency only
after fixing gold behavior; prioritize preserved Elder signal and measured
historical/live telemetry gaps over another broad model-family search.

For any subsequent candidate, freeze features, calibration and selection rules;
use matched causal minute rows and equal game weights; report log loss,
calibration and rare-state performance; and use paired whole-series intervals
with date-block sensitivity checks. Extend promotion comparisons to the
incumbent GAM when changing the GAM itself: the current outcome gate tests a
blend against the newly fitted GAM, while `refresh` replaces the GAM even when
that blend is rejected. Architectural changes need their own prospective
comparison, beyond the existing optimizer and shape checks.

Sources: `docs/meta-adaptive-benchmark.md`, `docs/sido-model-exploration.md`,
`data/wpx/adapt_v2/rolling_replay.json`, `lol_ticker/wpdeploy.py:528`.

## Verification and limits

- `python3 -m unittest discover -s tests`: **99 passed**.
- Current deployed artifact: ordinary shape audit **372 comparisons passed**.
- Additional physical-gold audit: **541 reversals / 400,000 comparisons**;
  repeated using an independent champion-score contraction, with matching
  counts and numerical agreement; all probabilities finite.
- Largest gold failure reproduced through the actual live prediction method.
- Elder clipping verified in the saved artifact and through live inference.
- Existing accuracy results were inspected, not regenerated. No forecast
  improvements from the proposed fixes are claimed until they are measured.
