# Season, patch, champion and recency benchmark — 2026-09-04

The implemented search favors **180-day recency weighting of the existing GAM**.
Explicit champion/patch categories and resource interactions did not improve
development accuracy in this search. LightGBM remained behind the GAM.
The recency gain is small and statistically inconclusive, and its frozen
candidate fails a sampled gold-monotonicity check. Production remains unchanged.

## Implemented inputs and model families

Both families receive the existing live contract and, in the rich configurations:

- Calendar year, numeric patch major/minor, and days since that patch first
  appeared in the dataset. This is **observed patch age**, not an official release date.
- Competition names with calendar/stage labels removed; LCK and LCK CL remain distinct.
- Ten role/side-specific champion identities, champion×year and champion×patch categories.
- Ten individual gold levels, signed by side, plus a coverage flag. All 500,122
  fixed-minute snapshots across 15,408 games matched the stored player gold;
  reconstructed role differences agree with the original training matrix.

LightGBM uses native categorical features and tree interactions. The rich GAM
uses pooled sparse categorical terms and explicit champion×gold×time terms.
Category dictionaries are fitted on training games only; repeated states do
not inflate category support. Unseen/rare categories fall back to missing in
LightGBM and zero additional effect in the GAM. Gold never comes from a future
snapshot or end-of-game value.

Training windows and decay apply to the **state learner**. Shared pregame and
champion-score priors retain all earlier fit-block history as long-term context.
This experiment does not retune Elo update constants or every upstream prior.
Season here means calendar year; competitive splits are represented indirectly
through patch/time, not a separate Spring/Summer identifier. No external buff,
nerf, item-change or champion-release descriptions were ingested.

## Parameter coverage

This was a bounded, reproducible search, not an exhaustive Cartesian product
of every LightGBM option. It screened **36 configurations**: 10 GAM and 26
LightGBM configurations. Nine finalists, including matched controls, were
evaluated across all three development blocks. Each fit tested uncalibrated,
temperature-scaled, and Platt-calibrated probabilities.

| Parameter group | Search choices |
|---|---|
| History | Expanding; 180, 365, or 730-day windows |
| Recency | None; 90, 180, or 365-day half-life |
| Feature set | Existing contract; rich calendar/champion/resource inputs |
| GAM regularization | State L2 24/60; time smoothness 70/150; context L2 300/800/2000 |
| Tree complexity | 7/15/31/63 leaves; depth 4/6/8/unlimited, with compatible leaf limits |
| Leaf support | 100/300/700/1500 rows; minimum Hessian 1/10/30; controls also use 200 rows |
| Boosting | Learning rate 0.015/0.03/0.06; up to 1,200 rounds; 80-round early stopping |
| Regularization | L1 0/1/5; L2 5/20/60/150; minimum split gain 0/0.05/0.2 |
| Histograms | 63/127/255 bins |
| Sampling | Feature fraction 0.7/0.85/1; bagging fraction 0.7/0.85/1; extra-random split variants |
| Categories | 5/10/25-game support for interactions; smoothing/L2 10/50/150; group support 100/300/700; split threshold 16/32 |
| Path smoothing | 0/10/50 |
| Shape restrictions | Unconstrained or advanced monotonic constraints; all features retained with advanced constraints |
| Calibration | None, temperature, Platt; a strictly earlier 28-calendar-day block |
| Randomness | Seed 43 for selection; seed 73 sensitivity check for the selected tree |

Exact combinations are in `data/wpx/adapt_v2/search_plan.json`. Binary log-loss
training, CPU execution, and no class rebalancing were fixed. DART/GOSS, GPU
settings, alternative losses, alternative GAM time knots, and a broad upstream
prior search were not run. DART was deferred because it would require a
different stopping protocol. These choices are recorded rather than implied
to have been exhaustively tuned. Parameter definitions and tuning considerations
were checked against the [LightGBM documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters-Tuning.html).

## Evaluation protocol

All inputs and nuisance fits are restricted to earlier dates. The shared
training prior stack uses game-group out-of-fold scores inside the earlier
training block; these internal folds are not themselves chronological.
Calibration and validation dates remain outside that block. Complete dates
stay together at every boundary, and each game has equal total loss weight
before optional calendar decay.

Development blocks are May 2–August 4, 2025; August 5, 2025–February 5, 2026;
and February 6–April 30, 2026. Settings and one calibration method are selected
by mean Brier across those blocks. Early stopping uses the earlier calibration
block, not the development validation outcomes. Calibration uses 28 calendar
days so sparse offseasons cannot turn it into a many-month exclusion window.

The later diagnostic block has **3,097 games / 99,408 states**, May 1–September 2,
2026. A frozen-model evaluation and a separate monthly replay use the same
selected settings. Replay refits at month starts with the preceding 28 days
reserved for stopping/calibration. Earlier replay outcomes may inform later
months; current/future-month outcomes cannot. No settings are reselected from
the replay. September contains only 19 games.

These are refitted reference models under this new protocol, not measurements
of the saved deployed artifact. Their absolute Brier scores should not be
compared directly with the earlier benchmark's 0.14316.

## Results

Lower Brier is better.

| Model | Mean development Brier | Frozen later block | Monthly replay |
|---|---:|---:|---:|
| Reference GAM | 0.148854 | 0.144863 | 0.143982 |
| GAM, 180-day decay | **0.148499** | **0.144571** | **0.143780** |
| Selected LightGBM | 0.150786 | 0.147261 | 0.146651 |

The selected GAM keeps the original feature set and uses 180-day exponential
decay with Platt calibration. It improves two of three development blocks.
The strongest rich GAM averages 0.149382, still worse than the reference.

The selected tree uses the original feature set, a 365-day window and half-life,
advanced monotonic constraints, 31 leaves, learning rate 0.015, minimum leaf
support 1,500, L1 1, L2 60 and 127 bins. Its selected calibration is none.
On the last development block, changing seed 43 to 73 changes Brier from
0.146469 to 0.146535; this check did not alter selection.

For the recency GAM, frozen-block Brier changes by **−0.000292**, with a
date/match-cluster bootstrap 95% interval **[−0.000588, +0.000016]**.
Monthly replay changes by **−0.000202**, interval **[−0.000497, +0.000100]**.
Replay log loss changes from 0.436849 to 0.436282. The tree's replay Brier is
worse by **+0.002669**, interval **[+0.001373, +0.003962]**.
Clusters combine games with the same date and match ID; cross-date series are
not joined into a single cluster. All intervals are retrospective diagnostics.

| Replay month | Reference GAM | 180-day GAM | Selected LightGBM |
|---|---:|---:|---:|
| May | 0.140794 | 0.140744 | 0.143822 |
| June | 0.157900 | 0.157560 | 0.158770 |
| July | 0.141567 | 0.141316 | 0.143443 |
| August | 0.143836 | 0.143527 | 0.147383 |
| September, 19 games | 0.180018 | 0.180335 | 0.172069 |

The season-transition audit includes 336 games in the first 30 days of 2026:
reference Brier 0.147806, recency GAM 0.146798, selected tree 0.151453.
This is a subgroup diagnostic, not an independent selection result. Outputs
also break down patch, competition, game phase, observed patch age and coverage.
The frozen GAM training history lacks at least one champion/role combination
in 287 later games; the tree's shorter history increases that to 430 games.

## Promotion status and validation

The frozen recency candidate fails **5 of 5,000** coherent sampled +1,000-gold
perturbations, all involving support. The worst reverses the expected direction
by approximately **0.0264 percentage points**. The reference passes this sample;
the selected tree fails eight perturbations. Partial feature constraints do not
guarantee monotonicity when a physical gold change also changes role allocation,
relative share and momentum.

Before promotion, jointly constrain or reparameterize the role-gold effects,
then rerun the full live-input audit. Accuracy also needs fresh evidence: these
dates were already consumed by the existing evaluation registry, which still
requires a fresh promotion block. No model, database contents or registry was
changed in production.

Sixteen targeted tests pass, covering date separation, decay/window weights,
numeric patch ordering, causal observed patch age, category support counted by
game, unseen-category handling, resource direction, calibrated artifact
roundtrips and parameter compatibility. GAM optimizers reported convergence,
tree fits completed, and predictions were finite; final models passed save/load checks.

## Files and reproduction

- `lol_ticker/wpadapt.py`: experimental models, temporal weights, category
  encoding, calibration, serialization and prediction helpers.
- `research/wpx_adapt.py`: search, frozen-finalist diagnostics and monthly replay.
- `tests/test_wpadapt.py`: new correctness tests.
- `requirements-experiments.txt`: optional LightGBM 4.6.0 dependency.
- `data/wpx/adapt_v2/`: plans, trial records, selection, models, predictions,
  `results.json`, `development_audit.json` and `rolling_replay.json`.

Using an environment with the experiment requirements installed:

```bash
python research/wpx_adapt.py
python research/wpx_adapt.py --audit-development
python research/wpx_adapt.py --rolling
python -m unittest tests.test_wpadapt tests.test_wpbench tests.test_wpx_sido
```

macOS requires an available OpenMP runtime. This run used the existing
sklearn-bundled library through `DYLD_LIBRARY_PATH` in a temporary environment.
Search outputs record dataset and implementation hashes; a changed plan requires
a new output directory. The models are experimental artifacts and are not wired
into the production forecast dispatcher.
