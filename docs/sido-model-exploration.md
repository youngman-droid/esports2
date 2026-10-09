# SIDO-inspired model exploration — 2026-09-04

Keep the current production model. The new performance priors tested here do
not beat the baseline on validation. The strongest remaining hypothesis is
champion-specific valuation of live resources; that requires a separate
in-game experiment.

## Paper and applicability

[Zhang and Naidu, The SIDO Performance Model for League of Legends, v1](https://arxiv.org/html/2403.04873v1)
models own, ally, and enemy gold/damage contributions using hierarchical player
and champion effects. It uses separate 0–7, 7–15, and 15–25 minute increments;
damage taken helps control combat exposure. Validation emphasizes professional
versus nonprofessional separation and metric quality, rather than prospective
win-probability accuracy. Its high-ranked solo-queue data allow more lineup
variation than professional matches. The authors report uncertainty and weaker
stability in ally/enemy effects. These are useful feature-design ideas, not
evidence that substituting SIDO improves our forecasts. See §§5–7.

## What our model already covers

- `lol_ticker/wpgam.py`: time-varying effects, regularization, player/team Elo,
  recency and series priors, champion outcome effects, an additional champion
  state score, and role-specific gold allocation.
- `lol_ticker/wpx.py:build_rapm`: past-only, recency-weighted gold-margin ratings.
- `scripts/wpx_playerchamp.py`: earlier experiments with player/champion win
  history and GD@15. Merely adding champion familiarity is not a new proposal.

What is missing is a performance rating learned jointly with role/champion
context, and a resource-value effect that depends on the particular champion
holding the gold. The current champion state score has a time curve, while
role-gold coefficients are shared across champions.

## Experiment completed

`scripts/wpx_sido.py` reads PostgreSQL in a consistent read-only transaction.
It fits monthly player ratings using only dates before the month starts, with
150-day exponential decay. Player effects have ridge penalty 60, champion
effects 30; these settings were fixed before validation. This is a Gaussian
ridge approximation, without posterior uncertainty or ally/enemy attribution.

Gold models use phase gold gains per minute, controlling role, own and opposing
lane champions, Elo difference, side, and gold state at phase start. Damage uses
historical full-match damage/minute, controlling damage taken/minute,
gold/minute, duration, role, champions, Elo and side. Only the resulting player
coefficients enter future-game predictions; current-game end statistics never
enter the forecast. Monthly updates continue during the diagnostic period
under the same fixed rule. Same-month games cannot update each other's ratings.

Each team's five player effects are summed; blue minus red is appended to the
existing post-draft pregame model. Rookies receive zero effects. The experiment
retains the baseline's regularization and nonnegative prior coefficients.
It is a pregame screen, not an end-to-end live-GAM comparison.

The available panel contains 154,080 player observations for 0–7 minutes,
154,070 for 7–15, 142,200 for 15–25, and 176,429 usable full-match damage rows.
Games ending before a phase endpoint and incomplete gold panels are excluded
from that phase's rating fit. Missing ratings do not remove forecast games.

There are 9,845 training games, 2,466 validation games beginning 2026-01-16,
and 3,097 diagnostic games from 2026-05-01 through 2026-09-02. Complete dates
stay together at the boundaries. All dates are already consumed according to
the evaluation registry; this is not a fresh promotion test.

| Pregame features | Validation Brier | Change vs baseline |
|---|---:|---:|
| Existing baseline | **0.216997** | — |
| Early gold rating, no champion controls | 0.217686 | +0.000688 |
| Early gold rating, champion controls | 0.217532 | +0.000535 |
| Three phase gold ratings | 0.217443 | +0.000446 |
| Context-adjusted damage rating | 0.217041 | +0.000044 |
| Three phase gold + damage ratings | 0.217461 | +0.000464 |

Lower is better. **Validation selects the baseline.** Damage is only the best
challenger. Its later diagnostic Brier is 0.210954 versus baseline 0.211017:
delta **−0.000063**, with a series-block bootstrap 95% interval of
**[−0.000234, +0.000111]** over 1,179 date/match clusters. Log loss changes from
0.608374 to 0.608194. This does not establish an improvement. Monthly Brier
changes have mixed signs: May +0.000095, June +0.000185, July −0.000474,
August −0.000049; September has only 19 games.

Only 1,879 of the diagnostic games have ten players with at least ten earlier
damage observations. Identity coverage, roster confounding, fixed shrinkage,
and the positivity constraint limit what this screen rules out. It rejects
these particular additions under this protocol, not every possible use of
performance data.

## Next experiments, in priority order

1. **Champion-specific value of gold.** Fit strongly pooled champion deviations
   from role-specific gold effects, varying smoothly with game time. Start with
   one compressed out-of-fold resource score rather than hundreds of unpooled
   interactions. Absolute per-player gold exists in `golgg_timeline` and the
   live window feed, but the present state matrix mostly retains role
   differences. Preserve both sides' individual gold in the experimental
   dataset. Compare to the unchanged v8 GAM on identical minute rows, and
   enforce monotonicity of the total gold effect. This is a new hypothesis,
   not a result of the pregame screen.

2. **Player/champion performance deviations with measured reliability.** Extend
   the joint model with a player×champion effect, pooled toward the player's
   role effect, and patch/league adjustments. Evaluate primarily on new rosters
   and stale-Elo games. Use effective sample size and unseen-champion/rookie
   fallbacks; do not equate a few strong games with mastery. Our earlier
   GD@15/win-rate experiment did not perform this joint adjustment, but this
   screen gives little reason to prioritize another generic player prior.

3. **Team conversion and suppression before individual ally attribution.**
   Model a team's subsequent gold/objective gains conditional on phase-start
   resources, opponent strength, and draft. Use past-only team effects as a
   candidate execution prior. Stable rosters make individual causal credit
   difficult to identify; do not sum several overlapping ally scores and call
   the result independent evidence.

For direct combat-efficiency state features, first establish matched
historical/live cumulative damage and damage-taken coverage. The stored minute
table has gold and CS, while the live scoreboard exposes damage share; neither
alone supplies the required absolute phase damage pair. Full-match statistics
can support historical priors but cannot be copied into earlier live states.

Before any promotion, evaluate the selected in-game candidate with chronological
stacking, identical live-available inputs and minute rows, equal game weights,
earlier-block calibration, and paired series-level intervals. Report early-game,
patch, sparse-champion and roster-change slices, and retain the existing fresh
game registry and live monotonicity gates.

## Reproduction and artifacts

Run `python3 scripts/wpx_sido.py` and
`python3 -m unittest tests.test_wpx_sido`. The script refuses a dataset extending
beyond the registry's consumed date. Three tests passed: same-month/future
target isolation, phase increments/short-game exclusion, and missing-gold
handling. All 160 monthly ridge fits converged.

Local outputs: `data/wpx/sido_screen.json` (metrics, settings, dataset SHA-256),
and `data/wpx/sido_screen.npz` (game-aligned features, reliability counts,
diagnostic predictions). These are ignored by Git under the existing data
policy. Production artifacts, database contents, and the evaluation registry
were not changed.
