# Objective opportunities and causal trajectories — October 4, 2026

Two independent research capture/experiment families are implemented. Neither
has production dispatch, replaces a forecast artifact, or changes a frozen
candidate registration. Unavailable extra features yield exactly the frozen
core's probability; no missingness or availability intercept is fitted.

## Objective opportunities

`lol_ticker/wpobjective.py` captures remaining Baron/Elder team-buff seconds,
their interactions with current living-player count/gold, and objective spawn
windows/soul threat when a certified timing profile is provided. Blue and red
share each function. Team buff time is an opportunity proxy: the official
window cannot identify which individual players retained a buff after death.
The module does not invent positions, lane access, summoner cooldowns or
remaining respawn seconds.

The capture tracker reads cumulative objective transitions and retains an
acquisition **interval** between the previous and current game clocks. Gaps
larger than 90 game seconds make the acquisition unknown. Pauses use the
caller's pause-aware game clock. Repeated timestamps are idempotent; older
frames cannot replay transitions. A changed attempt, clock regression or
decreasing objective counter resets history. Missing counters invalidate the
corresponding acquisition/zero-history assertion.

Spawn inference requires a profile with `source`, an exact matching `patch`,
and whichever of these positive timing values are supported by that source:
`dragon_first_s`, `dragon_respawn_s`, `baron_first_s`, `baron_respawn_s`,
`elder_first_after_soul_s`, `elder_respawn_s`. There is no universal default
spawn profile across seasons. A missing or mismatched profile leaves spawn
columns unknown while observed remaining-buff columns can still be captured.
Opportunity weight rises over the final 60 seconds before the **latest**
admissible spawn and is one once the objective is due. Soul threat uses exactly
three non-Elder dragons, separately from Elder acquisitions. After a soul,
elemental-dragon opportunity stops and Elder timing needs its own rule.

Readiness coupling requires age zero and matching current-frame provenance.
An archived HP sample carried for up to 90 seconds cannot certify who is alive
at a later objective opportunity. Historical adaptation accepts only a
caller-certified complete event stream, ignores every event after the
prediction clock, and derives remaining team-buff time from actual kill times.

```python
tracker = wpobjective.new_tracker()
state["objective_research"] = wpobjective.capture(
    state, tracker, attempt_id=attempt_key, timing_rules=certified_rules_or_none)
```

## Recent trajectories

`lol_ticker/wptrend.py` captures changes over 30, 120 and 300 game seconds:
team lead; five canonical role-gold differences; role HP/alive changes; and
kill/tower/Baron/dragon/Elder/inhibitor changes. It also reports lead standard
deviation and lead reversals per window. These side-neutral summaries are
excluded from the initial signed residual rather than being treated as an
arbitrary advantage for blue.

The live role join verifies metadata roles and participant IDs independently
of list order. Role gold remains available during opening frames whose health
is missing. Each channel needs complete valid endpoints. A causal anchor must
precede the target time within declared slack, and all intervening game-clock
gaps must satisfy the declared limit. There is no interpolation from later
frames. Outputs retain actual window duration, anchor clock, sample count,
limits and known masks. Same-frame retries preserve the original result.

Wall-time frames at one paused game-clock instant occupy one history record;
a long pause cannot manufacture a window, reweight volatility or exhaust the
frame cap. Remakes, later game-clock regression and cumulative gold/counter
regression reset history. History is bounded by time and by 1,024 frames.

Standalone defaults are gap 10 seconds and anchor slack one second. The
recorder's sparse-window profile can explicitly use gap 20/slack 15; then a
nominal 30-second window can span 30–45 actual game seconds. Those limits and
actual horizons must be part of a future serving contract. The historical
minute adapter uses gap 90/slack zero: exact 120/300-second windows are
supported, while 30-second windows and carried-HP histories remain unknown.

```python
tracker = wptrend.new_tracker()
observation = wptrend.live_observation(metadata, window_frame, state)
state["trajectory_research"] = wptrend.capture(
    state, tracker, attempt_id=attempt_key, observation=observation,
    max_gap_s=20, anchor_slack_s=15)
```

## Isolated experiments and validation limits

Both modules expose `fit`, `predict`, `save` and `load` wrappers around
`wpresidual`. Their artifacts bind candidate kind, names and extraction
contract. The initial model has no intercept or core refit, uses nonnegative
time-varying coefficients, training-only RMS transforms, and fixed ridge 800
and adjacent-knot penalty 70. The baseline's original earlier-block Platt
transform is shared; scored dates do not refit either stack or calibration.

`scripts/wpx_objective.py` and `scripts/wpx_trend.py` require the hash-matched,
causally audited corrected archive and its saved chronological core caches.
The archive is restricted to already-consumed dates. Resource hashes and
individual/team gold consistency are checked. Objective source-event inputs
must match all corrected objective counts and legacy active-buff indicators;
a contradictory game is withheld. Source event completeness remains an
explicit assumption even after those checks. There are no patch-certified
spawn profiles or synchronous historical readiness in the initial screen.

```sh
python3 scripts/wpx_trend.py --coverage-only --out data/wpx/trend_new_coverage
python3 scripts/wpx_objective.py --coverage-only --read-only-events \
  --out data/wpx/objective_new_coverage
python3 scripts/wpx_trend.py --out data/wpx/trend_new_screen
python3 scripts/wpx_objective.py --events-json /absolute/path/objective_events.json \
  --out data/wpx/objective_new_screen
python3 scripts/wpx_opportunity_audit.py --out data/wpx/trend_new_screen
python3 -m unittest tests.test_wpobjective tests.test_wptrend tests.test_wpresidual
```

The objective database option sets a read-only repeatable-read transaction and
queries only input events for corrected archive game IDs. Event files instead
need `source`, an `events` mapping by game ID, and `complete_game_ids`. New
output paths are mandatory. Models, predictions, source snapshots, plans and
completion hashes remain isolated from production.

The full screen keeps one prespecified candidate, selects baseline
unless the most recent pre-May development block has a negative paired Brier
upper bound and no game-balanced log-loss regression, then reports the frozen
later consumed block. These are exploratory historical diagnostics and cannot
authorize promotion. The extraction/capture APIs and tests establish causal
handling, serialization, reset behavior and fallback; measured accuracy needs
the separate historical screen.

Twenty-five focused objective, trajectory and residual tests passed. Coverage
and accuracy runs are separate: passing these tests alone does not establish a
better forecast. A future composed serving model must additionally verify
physical perturbations jointly with the exact core and preserve the sparse
live sampling/age semantics under its registered protocol.

The completed screens inspected all 500,122 fixed states in 15,408
consumed games. Exact 120-second gold and signed event-difference histories
cover 469,306 states (93.84%); 300-second histories cover 423,082 (84.60%).
Thirty-second and current-HP/alive histories have zero support in these minute
sources. Signed event differences do not require fabricated per-side counts;
their decrease is a legitimate opposing-side event, not a remake signal.

Objective event-prefix checks accept 449,669 states in 13,911 games; 47,547
states have a nonzero remaining-time advantage. They withhold 1,490 games
whose current source events contradict corrected archive objective counts or
active-buff indicators, plus seven without an accepted source stream. Missing
patch profiles and synchronous health leave all spawn/readiness interaction
columns unknown in this initial archive screen. These are coverage findings,
not evidence of accuracy improvement.

## Completed consumed-history screens

The complete runs are saved at `data/wpx/objective_2026-10-04_screen` and
`data/wpx/trend_2026-10-04_screen`. Each binds the actual source files at the
start, checks them again at completion, and saves predictions, per-stage
models, source snapshots and artifact hashes. Both candidates passed the
prespecified development selection rule. The most recent development block
contains 2,075 games; the later consumed block contains 3,097 games. The
comparator is the chronological corrected-history core with its original
calibration, rather than a new scoring-date core fit.

| Extra features | Development game Brier change | Later game Brier change | Later paired 95% interval |
| --- | ---: | ---: | ---: |
| Objective remaining-buff time | −0.00000546 | −0.00000185 | [−0.00000614, +0.00000267] |
| Supported 120/300-second gold and event trajectories | −0.00015485 | −0.00006625 | [−0.00014854, +0.00002000] |

Later baseline game Brier is 0.14417187. Objective and trajectory game log
loss improve by only 0.00000349 and 0.00001939 respectively; state-weighted
log loss increases by 0.00000292 and 0.00027543. Both Brier intervals include
zero, so these screens do not establish an accuracy gain. They also say
nothing about the unsupported spawn, current-health or 30-second columns.
Both completion manifests confirm no production or registration change.

The additional artifact audits each cover 80,000 coherent +100/+1,000
player-gold perturbations over all four chronological stages. They update the
core's physical gold representations together and, for trajectory models,
the supported current-minus-anchor gold changes. Neither family shows a
reversal. Zero-extra fallback is bit-exact, correction side-swap error is zero,
and predictions stay finite within probability bounds. Reports are saved as
`additional_shape_audit.json` inside each run and bind the evaluated plan,
residual models and auditor source. This sampled audit covers only the
historically supported columns; fresh health/spawn interactions still need
joint validation before any serving candidate could use them.

The live recorder's approximate sparse-window contract differs from this
exact-minute historical screen. A future deployment must evaluate and bind
that sampling contract explicitly, collect synchronous readiness and certify
patch-specific spawn rules before claiming support for those columns.
