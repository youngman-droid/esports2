# Role-specific combat readiness — October 4, 2026

Implemented the first player-state hypothesis: distinguish which roles are
alive, their health, and the gold and levels held by living players. The
historical screen selected the baseline. No new forecast model was promoted.
The existing shadow recorder was reloaded to preserve these richer inputs for
future games.

## Inputs and live capture

`lol_ticker/wpcombat.py` defines twenty blue-minus-red features: five role
alive advantages, five role HP advantages, five living-gold advantages in
thousands, and five living-level advantages. Both sides share role functions.
Alive means strictly positive health. Missing or invalid gold/levels disable
only their own family; all ten valid HP fractions are required for readiness.
Missing, invalid or older-than-90-second observations produce zero correction.
Items are excluded because archived undo events lack sufficient identity for
reliable inventory reconstruction.

`live._combat_block` joins window participants through metadata role and
participant ID, independently of list order. It uses the identical frame's
health, gold and levels; details frames and later observations cannot supply
health. Unknown roles, duplicate identities and wrong-side participants fail
closed. The JSON-safe `state.combat` block includes canonical observations,
feature values, availability, source and timestamps, and is stored by the
existing shadow-state capture. Existing v8/v9 numeric feature vectors are
unchanged; the new readiness model has no production dispatch.

The recorded `age_upper_s=0` live value describes the relationship between the
observation and its own prediction frame. It does not claim zero broadcast or
upstream feed delay; the recorder retains its separate timing measurements.

## Historical experiment

`scripts/wpx_combat.py` uses the immutable corrected archive through September
2, 2026: 15,408 already-consumed games and 500,122 fixed-minute states. It joins
raw telemetry by the archive's accepted `(game_id, hp_observation_ts)`, checks
HP/death agreement, and independently revalidates cumulative-player-gold
lower/upper clock bounds. No nominal feed minute is treated as an observation
clock. The PostgreSQL transaction is explicitly read-only.

Usable readiness covers 20,511 states (4.10%) in 871 games; 4,459 states in 725
games contain an observed death. Living levels are valid on 19,872 states.
None of the accepted observations is within 30 seconds of its prediction.
Historical HP can be carried for up to 90 seconds: it is measured at the
observation, rather than guaranteed current at the prediction instant.

Two fixed, strongly regularized residuals were compared: role alive/HP alone,
and role alive/HP plus living gold/levels. Each adds nonnegative smooth
coefficients to a frozen corrected-core logit; there is no residual intercept
or core refit. RMS transforms and full-population game-balanced weights use
training rows only. Inactive rows contribute constant loss and do not alter
the weighting of supported rows. Both models share the baseline's original
earlier-block Platt transform, so unavailable telemetry returns precisely the
baseline probability.

The original chronological baseline caches were hash-checked, and current
inference reproduced their saved calibration/validation predictions. Residual
settings were fixed at ridge 800 and adjacent-time-knot penalty 70. Selection
used only the most recent pre-May development block: require a negative upper
paired Brier bound and no log-loss regression; otherwise retain baseline. The
earliest development stage has no training telemetry and cannot establish
independent robustness. Earlier stages were retained as diagnostics.

The later frozen diagnostic covers 3,097 games / 99,408 states from May 1
through September 2. Lower game-balanced Brier is better. Intervals resample
whole date/match clusters; these consumed outcomes are exploratory evidence.

| Model | Brier | Change versus corrected core [95% interval] |
|---|---:|---:|
| Corrected core | 0.14417187 | reference |
| Role alive/HP residual | 0.14417198 | +0.00000012 [−0.00000018, +0.00000043] |
| Role alive/HP + living resources | 0.14420459 | +0.00003272 [−0.00000576, +0.00007230] |

On the observed-death subset (339 later games / 2,072 states), the resource
residual worsens game-balanced Brier by **+0.001487**, interval
**[+0.000580, +0.002455]**. Neither residual passed development selection.
These are comparisons to the corrected historical core, not proof about a
replacement for the currently deployed v8 artifact. The findings apply to
this frozen-core, positive-residual formulation, not every possible role-state
parameterization. Sparse, aged observations limit transfer to fresh live data.

## Verification and reproducibility

- 270 tests passed, with one existing PostgreSQL integration class skipped in
  the sandbox. All 29 new feature, live-integration and source-join tests pass.
- 200,000 composed individual-gold, HP recovery/revival and valid-level
  perturbation comparisons across eight stage/model artifacts: zero reversals.
- Every later missing-telemetry prediction equals the baseline exactly.
- Source gold-bracket checks pass for all accepted role-state observations.
- Deployed GAM/comparator artifacts, stack manifest and legacy evaluation
  registries retain their starting hashes.
- Completion metadata hashes models, predictions, reports, evaluated source
  snapshots, baseline metadata and scoring dependencies, with runtime versions.

The completed run is `data/wpx/combat_2026-10-04_verified/`; `report.json`
contains metrics and coverage, `completion.json` contains artifact hashes,
and each stage contains serialized residuals and paired predictions. Earlier
initial/incomplete runs remain separate and are not the completed result.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 scripts/wpx_combat.py --out data/wpx/combat_reproduction
python3 -m unittest tests.test_wpcombat tests.test_live_combat tests.test_wpx_combat
```

Archived health fractions were rounded to three decimals, so extremely low
positive HP can resemble death. Legacy role slots and unnoticed same-champion
remakes retain the canonical archive's explicit identity assumptions. Live
capture preserves unrounded HP and verifies metadata roles. A future serving
adapter must also align the core's existing participant-ID-based gold order
with metadata roles, or require canonical role ordering.

The recorder reload started PID 52224 at October 4, 19:59 PDT and completed
normal maintenance. No live game frame was available at verification, so the
capture path is enabled but no real new combat row is claimed. The previously
registered candidate remains bound to its original source generation; this
task does not alter its source checks or registration and does not repair its
previously observed source mismatch. No combat candidate was registered after
the negative historical screen.
