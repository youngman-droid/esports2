# esports2 — LoL prediction-market ticker recorder

Collects and stores ticker data for **League of Legends** markets on
**Polymarket** and **Kalshi**: live L2 order books, trades, and price history —
and keeps itself up to date as new games are played. Everything lands in
**PostgreSQL + TimescaleDB** (database `league`), queryable per game.

Python 3.12 in the repo's own venv (the code also runs on 3.9+). No API keys
are needed (all endpoints are public).

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements-experiments.txt -c constraints.txt pytest
```

`requirements.txt` is what the recorder, dashboard and nightly refresh need;
`requirements-experiments.txt` adds LightGBM for `research/`. `constraints.txt`
pins the versions the test suite was last run against. The scripts in
`scripts/` and the launchd agents use `.venv` automatically when it exists.

The dashboard's **Player impact** page (`/players`) ranks recorded worldwide
players with competition-adjusted logistic win RAPM, measured in wins added per 100 games. It includes age cohorts,
a published-age scatter, current form and sampled career peaks from the
2014–2026 Oracle’s Elixir archive. `/players/archive` exposes early results back
to WCG 2010; records without reliable complete player lineups are excluded from individual fits.
See [model, coverage and source notes](docs/player-impact-rankings.md).

## Database setup (one-time)

```bash
brew tap timescale/tap && brew trust timescale/tap && brew install timescaledb
brew services start timescaledb   # or the postgresql@N service it installed
createdb league
```

The collector runs `CREATE EXTENSION timescaledb`, creates the schema, and
converts the time-series tables to hypertables automatically on first connect.
Connection string comes from `$LOL_TICKER_DSN` (default
`postgresql://localhost:5432/league`). If you have data from the earlier SQLite
version, migrate it once with `python3 scripts/migrate_sqlite.py`.

## The one constraint that shapes the design

Neither exchange serves *historical* L2 order books — their book endpoints
return only the current state. So:

- **L2 depth** is captured **live, going forward**, by polling books during games
  (deduplicated: a snapshot is stored only when the book actually changes).
- **Past games** are backfilled with what does exist historically: the full
  **trade tape** (both platforms), Polymarket **price history** (1-min around
  game windows, 10-min over market life), and Kalshi **1-min candlesticks**.

## Quick start

```bash
# 1. initial catalog + full crawl of past LoL events (one-time; the closed-event
#    crawl is what makes old games queryable)
python3 -m lol_ticker discover --all

# 2. pull historical trades/prices for settled markets (long on first run —
#    it's every closed LoL market ever; use --limit to test)
python3 -m lol_ticker backfill

# 3. run the recorder daemon (books + trades + auto-discovery + auto-backfill)
python3 -m lol_ticker record
```

Then, at any time:

```bash
python3 -m lol_ticker status                     # what's stored, what's being watched
python3 -m lol_ticker game LNG 2026-08-09        # find games by team / date terms
python3 -m lol_ticker export LNG 2026-08-09      # dump a game's full record to CSVs
python3 -m lol_ticker dashboard                  # local web UI at 127.0.0.1:8090
```

## Dashboard

`python3 -m lol_ticker dashboard` serves a single-page UI (localhost only).
A **Live game** panel at the top lists in-progress LoL Esports games (official
schedule), polls the live-stats feed every 1–30 s (selectable), scores
**every 1 Hz frame** the feed returns with the odds-free model (Elo prior looked
up automatically) and shows the model's P(blue) next to the current Kalshi /
Polymarket price for the same map (quotes refreshed every 5 s), with a
zoomable running chart of model, state-only model, and both markets on the
game clock, and a live scoreboard (team totals; per player: alive/dead, level,
K/D/A, CS, gold, HP, items) — the same estimate as `python3 -m lol_ticker live`. The feed
publishes one frame per second and lags the broadcast by ~1–3 min, so faster
polling cannot add information.
Player deaths are read from the window frame's `currentHealth` values (the
details payload does not contain health). The panel prints those exact model
inputs, their learned log-odds contribution, a late mass-death warning, and a
same-frame scoreboard consistency alarm. It also compares each side's live
lineup (feed summoner names, team tag stripped) against the roster from that
team's most recent gol.gg game — the lineup its Elo/form prior actually
describes — and shows a **roster change** warning naming who is in and who is
out when they differ. When a lineup differs, the model's player-Elo prior is
**recomputed from the players actually on the rift**: per-player sequential
Elo ratings are persisted at prep time (`oe_player_elo`, the same ratings
whose per-game means train the `pelo` feature, so this only corrects the
live approximation and needs no refit), live summoner names resolve against
them (team tag stripped), and a side is adjusted when at least four of its
five names resolve, unresolved players inheriting the team mean. Unchanged
lineups keep the advanced team value, and teams missing from the rating
tables entirely can still get a player-based prior when their players are
known. The comparison and the adjustment are stored with every prospective
shadow forecast (`state.roster`, `state.pelo_adjustment`), so flagged games
can later be scored separately.
A **Browse** picker below it (season → league → tournament → game, from the
gol.gg tournament catalog; internationals/EMEA Masters under their own
heading) lists each tournament's games; clicking a game loads its markets on
both platforms into the odds chart (pre-ticking that map's winner markets) and
its event timeline — and makes that game the page-wide context: a banner
shows it, the draft simulator loads its real draft (bans/picks in competitive
order, roles, patch, market pre-draft odds), the draft-analysis / event-impact /
odds-free sections re-filter to its league with both teams highlighted, and
the timeline and model-WP views switch to it. Reaching a game via search does
the same. Or search by team/date, click a game, tick markets to chart their odds over time
— Kalshi and Polymarket series can be overlaid on the same chart (solid line =
price history, dashed = recorded L2 midpoint where available). The header pill
shows live Kalshi trading status (red when the exchange is paused, refreshed
every 60 s), and a second pill appears only when the recorder is unhealthy —
the `record` daemon not running, or no book snapshot stored for over six
hours — so a silently dead recorder is visible instead of discovered days
later; exchange outages that overlap a game's window are shaded red on
the chart and listed above it, and the green dashed vertical line marks
scheduled game start. Time-series charts (odds, timeline-vs-odds, model WP)
zoom with the mouse wheel or by dragging a box, pan with shift+drag, and
reset on double-click or the reset button.

The **Calibration** section below the chart evaluates each platform against
reality: every settled market contributes its last traded price before a
chosen cutoff (at game start / 1h / 6h / 24h before) and its resolution, and
the reliability diagram plots predicted probability vs. observed YES frequency
per decile (point size ∝ sample count) with Brier scores per platform. An
optional title filter restricts the sample (e.g. `match`, `map 1`, `LCK`).
Results are computed server-side (`/api/calibration`) and cached 10 minutes.

Two selection artifacts are corrected by construction: Polymarket samples only
conditions where **both** outcome tokens priced before the cutoff (one-sided
samples are winner-biased — losing tokens often stop trading, so their price
series is missing), and Kalshi samples bid/ask midpoints (trade-less candles
close at the bid, which fabricates longshot bias at low prices).

## Draft analysis

The **Draft analysis** section measures how the market re-rates each team after
champion select. Setup: download Oracle's Elixir yearly match CSVs into
`data/oe/oe_<year>.csv` (Google Drive folder linked from
oracleselixir.com/tools/downloads), then run:

```bash
python3 -m lol_ticker draftload
```

This loads per-game teams/picks/winners with actual game-start times, matches
every per-map winner market to its game (team name + map number + start-time
proximity; ~90% match rate), samples each team's win odds 16 min before game
start (pre-draft) and 2 min after (post-draft), and stores the deltas.  The
dashboard then ranks teams by average draft re-rating, champions by the mean
delta of teams that picked them, and same-team champion pairs
(synergies/anti-synergies), with league / platform / min-sample filters.
Games covered by both platforms count once (averaged). Rerun `draftload`
after downloading fresher CSVs to pick up new games (`--skip-deltas` reloads
CSVs and refits the model without re-matching markets; `draftfit` refits only).

### Draft simulator

The **Draft simulator** section walks the competitive draft order (3+3 bans,
6 picks, 2+2 bans, 4 picks). Each action's effect on blue's win probability
comes from a ridge regression (`draft_model` table) of the log-odds draft
re-rating on draft-state indicators, built hierarchically: a base effect per
champion (own/enemy pick, own/enemy ban), plus **role deviations**
(`own_pick:Ashe@sup`) and **patch deviations** (`own_pick:Ashe#16.15`), plus
same-team synergy pairs and cross-team matchups. Interaction features need
≥25 games; the UI has a patch selector and per-pick role selects, and blanks
fall back to the base effect.

It is heavily regularized (λ=1000) because champion identity explains little
of the market's post-draft move: held-out R² is ~0.03 on a random split and
~0.02 on a time-ordered split (train on earlier patches, predict the newest
two) — both shown in the UI. Role and patch features help slightly within an
era and not at all across patches. Treat per-action estimates (typically
±0.2–1 pp) as directional context, not predictions; the market's draft
re-rating is mostly about the specific teams, meta and execution, which
champion-level features do not capture.

## gol.gg detailed game data

`python3 -m lol_ticker golgg` scrapes Games of Legends (gol.gg) for per-game
detail well beyond Oracle's Elixir: the game header (teams, result, duration,
patch, bans/picks, team kills/towers/dragons/barons/gold, first blood/pick,
dragon types), 56 per-player stats (damage breakdown, gold/GPM, vision and
wards, CS/GD/XPD@15, CC, healing/shielding, bounties…), the **full event
timeline** (kills with shutdown bounties, plates, grubs, herald, dragons by
type, atakhan, baron, towers by lane/tier, inhibitors, nexus), **per-minute
gold and CS for every player**, and **item build timelines** (purchase /
sell / undo / destroyed with in-game time) plus final loadout, summoner
spells and runes. Item ids resolve via Riot Data Dragon (`--items`).

Tables: `golgg_tournaments`, `golgg_matches`, `golgg_games`, `golgg_players`
(with `stats` and `loadout` JSONB), `golgg_events`, `golgg_timeline`,
`golgg_builds`, `golgg_items`. Scope flags: `--seasons S15,S16`,
`--regions major` (or a comma list of gol.gg region codes), `--since
YYYY-MM-DD`, `--tournament <substring>`, `--limit N`, `--no-builds`.
The scraper is polite (gzip; 0.5 s spacing for the ~600 KB pages and a
0.15 s lane for the ~2 KB per-player build calls, shared across workers;
descriptive UA; robots.txt-compliant) and resumable — rerun to pick up new
games. Throughput is ~1,000+ games/hour with the default 6 workers; run it
detached, `tail -f data/golgg.log`, or `python3 -m lol_ticker golgg --status`.

### Game timeline vs odds, event impact, pauses

`python3 -m lol_ticker align` links scraped gol.gg games to Oracle's Elixir
games (teams + date + game number) and, for every per-map winner market of a
linked game, aligns the in-game clock with wall clock: starting from the OE
game-start time, each significant event (kills, objectives, towers) is
anchored to the nearest odds jump (trade tape + 1-min series + L2 mids); the
running residual's baseline is the market's reaction lag, a persistent step
above it is a **pause**, and the market's terminal 0.99/0.01 move fixes the
end and sanity-checks total paused time. Results land in `game_alignment`
(start/end wall time, pauses, anchoring quality) and `event_odds` (every
event with wall time, odds before/after, Δ for the market's team and the
swing toward the acting side). Selecting a game in the dashboard links and
aligns it on demand (markets are resolved straight from the catalog by teams +
map number + start window), so freshly scraped games work without waiting for
a bulk `align` run; games whose map had no market (e.g. an unlisted game 3)
show an explicit "no odds series" note instead of a stale chart.

The dashboard's **Game timeline vs odds** section charts a game's odds with
event markers (hover for the event and its swing), pause bands and start/end
lines, plus the event table; **Event impact** aggregates mean/median swing
toward the acting side per event type with platform / phase / league filters.
Caveat: events seconds apart (teamfights) share one odds window, so per-event
swings in clusters are not independent. A **market swing vs outcome-model
WPA** bar chart and table (`/api/impact/compare`) put the two side by side per
event type with the same filters, plus the market/model ratio — the direct
test of whether traders under- or over-react to each kind of event.

### Odds-free evaluation

Two outcome-based models evaluate drafts and events without any market data
(`lol_ticker/wpa.py`, `draft.fit_outcome_model`):

- `python3 -m lol_ticker wpa` — sequential **Elo** over gol.gg games, then an
  in-game **win-probability model** (logistic on gold / kill / tower / dragon /
  baron / inhib / herald / grub / atakhan / plate diffs, game time, side, Elo
  prior; states sampled every minute and at every event; game-level holdout
  log-loss ≈ 0.47 vs 0.68 baseline, ~77% accuracy), then per-event **WPA** =
  WP after − WP before (`event_wpa`). The dashboard shows a game's model WP
  curve on the in-game clock and a WPA-by-event-type table.
  A **calibration** panel refits the model on 80% of games and scores the
  rest (reliability bins, Brier, log-loss, by phase), and — the like-for-like
  test — evaluates the market's odds and the model's WP at the *same event
  instants* in aligned games (`/api/wpa/calibration`). Result so far: the
  model is well calibrated (holdout Brier ≈ 0.15 over all states) but the
  market is clearly sharper at identical moments (Brier ≈ 0.11 vs ≈ 0.13,
  log-loss ≈ 0.34 vs ≈ 0.39 on both platforms).
- **Model exploration and staged fitting**
  (`lol_ticker/wpx.py`, `lol_ticker/wpgam.py`):
  the original pipeline builds a state dataset (per-role gold, CS, momentum, item completion and
  item gold, objective/structure state, players on respawn timers, baron/elder
  buffs, objective timers; priors: sequential team Elo and player ratings over
  all OE games, recent form, margin-RAPM team/player ratings refit monthly on
  earlier games, a leak-free draft-model term) and evaluates a model zoo with
  5-fold game-level CV against the market at the same points, plus a fixed
  minute-mark comparison (`wpx_fair.py`). The production model is now a
  two-stage, shape-constrained, time-varying logistic GAM: a one-row-per-game
  pregame prior (live-available team/player ratings plus strongly shrunk
  champion effects) and a per-game champion-state score (see the v7 notes
  below), followed by six smooth game-time knots over a small
  historical/live feature contract. Gold, momentum, CS, objectives, living
  players, HP and level advantages are constrained to have
  non-negative partial effects at every knot. Inputs are standardized, games
  are balanced in the loss, only causal fixed-minute states are fitted, and an
  earlier-date holdout calibrates the final logits. `wpx gam-eval` evaluates
  the newest 20% of games and reports state- and game-weighted scores with
  game-block bootstrap intervals. `wpx rolling --windows 3` runs strict
  expanding-window stability checks. `wpx all` is disabled during the corrected
  input transition; explicit builds and experiment paths are described below.
  `wpx fit` stages changed artifacts, including same-kind refreshes. Promotion
  requires a frozen, exact-artifact candidate comparison against the incumbent,
  a completed fixed endpoint, sufficient precision and live shape audits.
  Promotion copies the evaluated artifact without refitting and replaces the
  manifest last. Previous artifacts remain as `.previous`.
  `wpx bench` performs a nested chronological comparison of additive ridge
  logistic regression, unconstrained and constrained time-varying GAMs,
  histogram gradient boosting, monotone histogram boosting, and validation-fit
  convex ensembles. Hyperparameters and blend weights use a middle date block;
  its newest block is a chronological diagnostic. For actual promotion, a
  shared `outcome_exposure.json` preserves the union of the legacy registries
  and inspected forward outcomes, preventing already-consumed games from being
  relabeled as fresh. The 100-game collection floor alone does not establish
  adequate precision or an accuracy improvement.

  The v5 live contract models teamfights with the linear death advantage,
  signed squared death advantage, individual side death-count curvature, and
  an interaction between deaths and opened-base pressure. Historical death
  counts are capped to the physical 0–5 range, and every term is constructed
  from fields also available in the live window feed. This fixes the former
  additive-model failure where four dead players could be treated too much
  like one or two deaths, without introducing a hard probability override.
  It also tracks each side's Baron and Elder timers from their cumulative
  counter transitions: respectively 180 and 150 seconds at acquisition,
  pause-aware countdowns, and active exactly while the corresponding timer is
  positive. The matching historical `baron_active` and `elder_buff` terms are
  part of the fitted feature contract. Soul requires four non-Elder dragons;
  Elder kills are tracked separately and cannot accidentally satisfy Soul.

  The time basis carries a sixth knot at 60 minutes: with the surface
  clamped at 45 the model was overconfident in marathon games (calibration
  slope 0.66 on the >45-minute slice, 177 of 98k test states); the extra
  knot improves that slice (slope 0.70, Brier 0.229 → 0.221) at exactly
  zero overall cost (paired delta −0.00001). Live frames record
  `lo_champ_state` so champion-heavy calibration can also be checked
  prospectively.

  The v8 contract adds two **recency prior channels**, both live-servable:
  a second gol.gg team Elo run at K=120 (`elo_gg_fast` — opponent-adjusted
  recent form, where a raw last-N win rate is not) and the current match's
  prior-wins difference (`series_diff`, from the schedule's series score
  live and match siblings historically; measured at +0.10 log-odds per
  prior game controlling Elo over 11,076 consecutive-series pairs, matching
  the ~2.5-point moves markets make on a game win). Gated with nested
  chronological selection: every candidate improved the validation block,
  the pair was selected (0.14477 vs base 0.14515), and the untouched test
  block scores 0.14274 vs 0.14305 (paired −0.00030, 95% interval
  −0.00075..+0.00015 — point-estimate ship, structural rationale: the
  model previously could not see series context at all). Caution from the
  same study: recency does not always move toward the market — faster Elo
  *widened* the model-market gap in the motivating KT–DK case, because the
  market's disagreement there was not about recent results.

  The v7 contract adds the **champion-state channel**: per-champion
  coefficients estimated champscale-style on in-game states — jointly with
  the full exploration feature set, entering as signed presence × min(t, 15
  min), l2=800, fit on all training rows including event-anchored ones
  (restricting to fixed minutes cost 0.0005; the coefficients need
  teamfight-dense samples) — compressed into one signed per-game score that
  becomes a third prior input with its own non-negative smooth time curve.
  Training games receive out-of-fold scores (5 game-level folds) so the
  state model cannot overweight in-sample champion information. This carries
  the in-game champion signal that the pregame outcome fit cannot see:
  walk-forward game Brier 0.14341 → 0.14247, improving every phase and both
  event-aligned slices, and closing the legacy champscale gap to a
  statistically indistinguishable +0.00028.

  The **deployed live forecast is currently the constrained GAM alone**.
  The September 4 **v9 repair is staged**, while the deployed artifact remains
  v8: discrete Elder features survive scaling, five monotone role-gold effects
  replace allocation deviations, and calibration nests the champion learner
  inside earlier dates. Architectural refreshes stage a candidate without
  replacing the incumbent. The recency retest did not improve the later replay;
  see [implementation and validation](docs/model-repairs-2026-09-04.md).
  The [alternative-method reevaluation](docs/alternative-methods-2026-09-04.md)
  uses `research/wpx_methods_v9.py` to compare fourteen families and two
  ensembles with the repaired contract, separate calibration dates, frozen
  development choices and monthly replay. Earlier benchmark files retain their
  original evaluation protocol and should not be mixed with these scores.
  The [matched market comparison](docs/market-comparison-2026-09-04.md)
  reuses those predictions: the rich GAM beats both exchanges at reconstructed
  state times, its advantage is inconclusive at a 45-second market offset,
  and every reevaluated method trails both markets at a 195-second offset.
  The former hard-coded GAM/champscale blend was retired because its legacy
  component had been backtested with historical-only fields that became zero
  live and with event/state-weighted rather than game-balanced loss. Its
  replacement (`champscale_live_v2`) uses only feed-available columns, causal
  fixed-minute rows and equal total weight per game. Nested chronological
  selection now chooses `w_gam=0.75`, but on the 3,102-game diagnostic block
  the GAM scores 0.143094, the comparator 0.144222, and their candidate blend
  0.143143 (candidate-minus-GAM +0.000050, paired interval
  −0.000166..+0.000269), so the manifest rejects it. Even a statistically
  successful candidate cannot deploy from this already-inspected block; the
  forward registry has 72 fresh games and requires 100. The candidate also
  fails the live
  monotonicity sweep (including early towers/teamfight states and late CS,
  Baron and level advantages), independently confirming that it should remain
  a comparator. `predict_live` verifies component and manifest hashes and
  otherwise fails closed to the GAM; it never silently substitutes the legacy
  component when the GAM is unavailable.

  The v6 contract adds three live-derivable inputs. A gol.gg-based team Elo
  joins the pregame stage: the Oracle's Elixir CSV goes stale for weeks at a
  time (Drive quota), and by August 2026 23% of new games had no OE rating —
  the gol.gg Elo covers 99.8% and `live.team_priors` now serves it (falling
  back to the OE Elo when a gol.gg lookup misses). Relative gold share and
  time-since-last-kill join the state features. On the same holdout this
  moves game-weighted Brier from 0.14382 to 0.14341 (paired game-block delta
  −0.00041, 95% interval −0.00117..+0.00031, 55.7% of games improved); the
  gain is concentrated exactly where the failure mode lives — games with a
  missing OE prior improve by −0.00476 (n=412) while fully-covered games are
  flat — so the live benefit under a stale OE feed exceeds the backtest
  average. Contract note: void grubs, herald, Atakhan and plate counts remain
  excluded because the official live window feed does not carry them; the
  respawn-timer features (`baron_up`, `dragon_up`) would need dragon-kill
  clock tracking in the live path before they can enter the contract.

  On the current causal rebuild (15,349 games through 2026-08-29), the
  newest-date diagnostic block is 3,102 games. Historical states use
  official-feed health for exact
  death counts when available instead of the approximate respawn window used
  for the remaining games. The v7 constrained GAM scores game-weighted Brier
  0.143094 and state-weighted Brier 0.153912. Removing the shape constraints or
  swapping model families was tested on the v6 contract: unconstrained GAM
  ±0.00001, monotone boosting +0.00200, ridge +0.00283 — but those bench
  competitors share the GAM's slim feature contract, so they compare model
  *families*, not the best available model. The v7 champion-state channel
  came out of a 2026-08-29 audit of the legacy `champscale_reg(l2=800,cap15)`
  zoo model, which then still beat the v6 GAM (0.14219 vs 0.14341): the edge
  survived fairness controls (GAM's own rows and loss: 0.14250; tuning
  neighbors ~0.14256), its feature set is leak-free (no draft term;
  RAPM/Elo/form time-forward by construction), and attribution showed the
  entire edge was the champion×time terms — champscale *without* its champion
  columns scores 0.14443, worse than the GAM, while grafting cheap
  interaction terms (Baron×deaths, lead×inhib) onto the contract moved
  nothing. After v7 the residual champscale edge is +0.00028 (paired interval
  spans zero). One edge-case mispricing survives in the GAM component: in
  "Baron + ≥3 more enemies dead while 3–7k behind" states (22–35 min) the
  advantaged side historically wins 71% (n=245 states / 86 games), champscale
  says ~79%, the GAM ~49% — too rare (245 of 1.76M states) to affect
  aggregate scores. On event-aligned
  points inside the same newest-date diagnostic block, the v7 GAM scores state
  Brier 0.11915 versus Polymarket's 0.11263, and 0.11929 versus Kalshi's
  0.11196; these event-triggered samples favour the markets and are reported
  separately from fixed-minute accuracy.

  Diagnostics report telemetry coverage and performance by month, patch and
  tournament. On the current diagnostic block, HP is available for 31.0% of
  states / 973 games; 303 games have a missing-or-even OE prior and score
  0.15349 game Brier versus 0.14197 when an OE prior is available. Two strict
  expanding windows score 0.15154 (test 2025-06-02..2026-02-20) and 0.14491
  (2026-02-21..2026-08-29), making regime drift visible instead of hiding it
  in one aggregate. Predictions expose `input_warnings`, clipped feature names
  and a `normal`/`reduced` reliability flag for missing HP/priors, unknown
  champions and out-of-training-range states.

  `python3 -m lol_ticker wpx blend` tests probability-space and log-odds
  averages, market recalibration, positive logistic stacking and time-varying
  blends with nested chronological selection (`lol_ticker/wpblend.py`). At
  event instants (v6 rerun), the selected positive logit stack lowers
  game-weighted Brier from 0.10572 (Polymarket) and 0.10898 (model) to 0.09359
  over 1,866 then-held-out diagnostic games; against Kalshi it lowers
  0.10974 / 0.11428 to
  0.09791 over 420 games. The paired improvements over the markets are 0.01213
  (95% game-block interval 0.00985–0.01442) and 0.01183 (0.00646–0.01727),
  respectively.

  The retrospective latency benchmark is `python3 -m lol_ticker wpx blend-live --lead 45`:
  every causal fixed-minute model state is paired with the last executable quote
  45 seconds later. Polymarket selects a direct-Brier positive logit stack,
  `logit(p) = b₀ + b_market logit(p_market) + b_model logit(p_model)`; it scores 0.14292 versus
  0.15069 for the market and 0.14867 for the model over 1,591 test games.
  Kalshi selects a static positive logit stack and scores 0.13891 versus
  0.14850 / 0.14618 over 344 games. Both improvements over the market have
  paired intervals excluding zero. Repeating the full selection at 30 and 60
  seconds gives the same conclusion. When both exchanges are present at +45 s,
  Kalshi's two-way stack has the best game-weighted Brier (0.14325); the
  three-way stack is only 0.00011 worse and its paired interval crosses zero,
  so that experiment selected the simpler Kalshi/model blend. These results
  are retained for research under `data/wpx/blend_*.json`, but current-match
  exchange quotes are not forecast inputs and this artifact is not deployed.

  Draft-complete probabilities use a separate fit rather than extrapolating
  the in-game coefficients to time zero. The causal `t=0` model is compared
  with the stored post-draft quote at game start +2 minutes (so the market is
  explicitly given 120 seconds of early-game information). Polymarket's logit
  stack scores 0.19662 versus 0.20124 for the market and 0.20520 for the model
  over 1,866 test games; its paired market-minus-blend interval is
  0.00261–0.00667. Kalshi selects market-only Platt recalibration, scoring
  0.20396 versus 0.21021 for raw market and 0.21848 for the model over 420
  games (paired interval 0.00066–0.01195). In other words, the odds-free model
  adds post-draft information to Polymarket on this cohort, but not to Kalshi.
  This remains a retrospective comparison rather than a deployed live input.

  `python3 -m lol_ticker wpx hist-blend` implements the deployable,
  exchange-independent alternative (`lol_ticker/wphist.py`). A constrained
  teacher learns from historical Polymarket/Kalshi probabilities, while live
  inference accepts only the outcome GAM probability and its underlying
  team/draft/game-state features. A middle date block selects the historical
  teacher's log-odds weight; deployment uses its own append-only registry and
  the same fresh ≥100-game paired rule as the live stack, so experiment order
  cannot consume another candidate's ledger. On the earlier v7 diagnostic
  rebuild, the standalone GAM scores
  game-balanced Brier 0.142468 versus 0.142606 for the historical blend
  (blend-minus-model +0.000138, 95% paired interval -0.000125..+0.000406), so
  the historical-odds blend is rejected; the deployed forecast remains the
  standalone constrained GAM described above.

### Prospective shadow scoring

Role-specific combat readiness is now captured in each live frame's
`state.combat`: per-role alive/HP and the gold/levels held by living players,
joined through participant IDs and metadata roles. It is research telemetry;
the deployed forecast is unchanged. The October 4 historical screen selected
the corrected-core baseline rather than either readiness residual. See the
[implementation, limits and results](docs/combat-readiness-2026-10-04.md).
Reproduce into a new directory with
`python3 research/wpx_combat.py --out data/wpx/combat_reproduction`.

Objective opportunities, composition, Fearless context, prior confidence,
recent trajectories and an early SQ signal now have isolated research adapters
and live capture blocks. Experiments retain the incumbent unless supported by
their declared comparison; source coverage and unavailable families are
explicit. See [the six-item expansion](docs/model-expansion-2026-10-04.md) for
capture contracts, source requirements and results.

The corrected-input core candidate registered September 12, 2026 is scored on
the incumbent's exact new frames without requiring market quotes. Its fixed
endpoint is December 1 UTC, with a 2,437-game minimum and an explicit precision
gate. The deployed v8 artifact remains unchanged. See the
[implementation and results report](docs/model-improvements-2026-09-12.md)
for the corrected-data study, frozen artifact and sample-size limitations.
`python3 -m lol_ticker.wpcandidate status` reads candidate progress;
`data/wpx/shadow_candidate_score.json` is refreshed by normal shadow scoring.

`python3 -m lol_ticker shadow record` runs the forward-only evaluator
(`lol_ticker/shadow.py`). The current v12 protocol uses a bounded
background quote cache (12-second lookup budget, maximum quote age 15 seconds).
Quote misses stay missing. Resolution/scoring run off the capture loop;
coverage reports separate upstream feed age from local processing and retain
the 90-second lead limit and 100-game target. Earlier protocol rows remain
queryable. The first live model frame observed in each integer
game minute is written together with contemporaneous blue-oriented market
midpoints used strictly as comparison benchmarks, the independently generated
historical blend when its holdout gate passes, exact feed-to-quote lag, side
assignment, component/stack SHA-256 hashes, blend weight and exact source
revision. A database trigger rejects updates or deletes, while the primary key admits only one row for that game
minute: missing or stale quotes remain missing and historical games are never
backfilled. Outcomes are written later to a separate table; remade attempts
are voided. Outcomes come first from the local gol.gg/feed link or Oracle's
Elixir and, when neither has the game yet, from the **settlement of the very
markets quoted in the row** (the Kalshi market's `result`, or Polymarket's
resolved outcome prices), so scoring does not wait for the nightly scrape.
Outcomes belong to a game rather than a protocol, so rolled-over ledgers keep
resolving and can be scored with `shadow score --protocol <id>`.

The protocol is content-addressed and registered before the first prediction.
Full stack provenance and the GAM-only gate decision start the
`shadow_v8_full_provenance` ledger; all earlier forecasts remain immutable and
queryable. Scores are reported both for the deployment workflow and separately
for each exact `(stack hash, source revision)` cohort, so refreshes cannot mix
model versions silently. Pre-v8 rows are explicitly labeled partial because
their legacy component cannot be reconstructed.
Its primary metric is game-balanced Brier for the deployed independent
forecast versus the raw market on the same `(game, minute)` rows, with a paired
game-block bootstrap interval. Every eligible row uses the forecast selected
at capture; an unavailable optional blend falls back to that row's model
forecast. Scoring and the 100-game cohort use the same quote and timing rules.
Since `shadow_v10_lead_gate` the primary rows
are those whose market quote leads the model's feed frame by at most 90
seconds (`primary_market_lead_s`, registered in the protocol); the feed itself
lags ~45–140 s, and a 2026-09-03 audit found the recorder stalling for
minutes inside the market-quote path, so rows captured after a stall handed
the market minutes of extra game state and dominated the raw aggregate
(at leads under 90 s the model and market were indistinguishable). The
same summary over every quoted row is reported as `all_leads`, and each row
stores per-stage capture timings (`state.capture_timing`; captures over 60 s
log a warning naming the slowest stage). Runs
before 100 resolved games per platform are labeled descriptive, avoiding a
confirmatory claim from repeated early peeking. Model refreshes are allowed,
but every forecast retains the exact artifact versions that generated it.

```bash
python3 -m lol_ticker shadow record --once  # initialize + one live capture pass
python3 -m lol_ticker shadow status
python3 -m lol_ticker shadow resolve        # attach newly available OE/gol.gg outcomes
python3 -m lol_ticker shadow score          # data/wpx/shadow_score.json
```

For a result that cannot yet be matched automatically, use
`shadow resolve --game-id ID --winner blue|red`; manual provenance is stored
with the outcome.
The normal `scripts/update.sh` workflow starts the continuous shadow recorder,
resolves and scores the ledger at the end of every refresh, and the
dashboard's **Prospective shadow** button shows capture progress and scores
without refitting the blend. Exchange markets are matched per event: a Kalshi event must list both teams
(exact names first, then a whole-word tail such as "Meavedron" for
"UP2U Meavedron") and, among several meetings of a team, the one scheduled
nearest the game start within 36 h wins; Kalshi ticker dates can sit a day off
the actual play date, so the window is deliberately wide. Quotes or settlements
from a ticker outside that window are treated as another game's market and
excluded from every scoring cohort (a 2026-09-03 audit found single-team
matching quoting the wrong series and one outcome settled from it). Since
`shadow_v11_market_event_match` captures with a feed frame older than 600 s
are skipped, start-time jitter under 120 s is folded into one attempt instead
of voiding it as a remake, and Polymarket settlement reads closed markets
(Gamma hides them without `closed=true`). The refresh is scheduled nightly at 06:30 local
by the `com.lolticker.update` launchd agent (`sh scripts/install_agents.sh`,
see "Keeping it updated" below).
A failed Oracle's Elixir download (Drive quota) is non-fatal there: the
previous CSV stays in use and the gol.gg scrape, recency Elo and model refit
still run.

  Earlier exploratory findings (Aug 2026, 3.4k–4.8k games):
  event-instant scoring favours the market because events are anchored to its
  own odds jumps; at equal wall time on fixed minute marks the best odds-free
  model (time-interacted logistic + ridge champion-scaling terms) beats both
  markets** — Polymarket 0.1678 vs 0.1703, Kalshi 0.1587 vs 0.1607 — winning
  mid and late game; with the market given 45 s of reaction time it still
  leads by ~0.002–0.003. This is a retrospective state-quality comparison: the
  live feed is delayed, so it is not evidence of a contemporaneous trading
  edge. Gradient boosting underperformed the logistic family
  throughout (game-correlated states). Results render in the dashboard's
  "Odds-free model zoo vs the markets" panel (`data/wpx/results.json`).
- `python3 -m lol_ticker draftfree` — logistic regression of who won on an Elo
  prior plus the same draft features (picks/bans/roles/patch/pairs) over all
  Oracle's Elixir games (sparse gradient descent, λ=300, interaction support
  ≥60). Holdout log-loss improves over Elo-only (≈0.628 vs 0.634), so drafts
  carry real, if modest, outcome signal. Per-(game, team) **draft edge** =
  P(win | Elo + draft) − P(win | Elo) lands in `draft_outcome_games`; the
  dashboard ranks teams and champion effects from it, next to the
  market-based tables for cross-checking. Since 2026-09-17 the fit also takes
  one ridge-penalised `__sq_pair__` input (`draft.SQ_PAIR_ENABLED`).
- `python3 -m lol_ticker sqpairs refresh|build` — solo-queue matchup/synergy
  prior (`lol_ticker/sqpairs.py`). Lolalytics lane-vs-lane matchup and
  teammate-synergy deltas (emerald+, ~25M games/patch), shrunk by n/(n+k) with
  k from the solo-queue data alone, summed over a draft's 25 matchups and
  20 synergies. A pro map on patch P is scored only from patches before P, so
  there is no look-ahead and no pro outcome ever enters. Newest-date gate
  (train < 2026-08-01, 1,972 holdout maps): log loss 0.63310 → 0.63034,
  paired Δ −0.0028 [−0.0050, −0.0005]. `refresh` re-fetches a live patch's
  ~1,240 pages at most weekly (1 req/s, stops on 403/429) and runs as phase S
  of `scripts/update.sh`. Details: `docs/sq-pair-prior-2026-09-17.md`.

`game`/`export` terms match team names, event ids, or market titles; a
`YYYY-MM-DD` term filters by game day (UTC). Export writes one directory per
(platform, event) with `markets.csv`, `book_snapshots.csv`, `trades.csv`,
`candles.csv`, `price_points.csv`, and `outages.csv` when an exchange pause
overlapped the game.

## One-shot refresh

`sh scripts/update.sh` refreshes source data and priors (logs in
`data/update*.log`): discovers new markets and backfills newly settled ones,
re-downloads the current Oracle's Elixir CSV and rebuilds the draft tables,
runs the gol.gg scrape incrementally for the last three weeks, then re-aligns
timelines with odds, refits the odds-free WP/WPA and draft outcome models,
updates source ratings, resolves and scores shadow outcomes, and starts the
`record` and shadow daemons if needed (`--no-record` to skip starting them).
Automatic GAM builds, fitting and historical evaluation are gated while the
corrected-input candidate is evaluated prospectively. The deployed GAM and
frozen candidate artifacts remain fixed. `python3 -m
lol_ticker live` then estimates the in-progress LoL Esports game from the
official live-stats feed.

Corrected historical builds require an explicit new artifact path and an
exclusive cutoff, for example `python3 -m lol_ticker wpx build --out
data/wpx/states_inputs_v2_before_2026-09-03.npz --before 2026-09-03`. Existing
files cannot be overwritten. These builds do not change the default legacy
`states.npz` used by older fit/evaluation commands; use their output explicitly
in the corrected-input experiment. `wpx all` is disabled to prevent combining
the two input contracts accidentally.

## Keeping it updated as new games are played

`record` is the updater. Every loop it:

1. re-discovers markets every 10 min (new match markets appear on both
   platforms days before games);
2. splits open markets into a **fast tier** (game start within −45 min…+8 h,
   polled every ~5 s) and a **slow tier** (everything else, every 5 min);
3. stores changed books and pulls incremental trades for fast-tier markets;
4. checks Kalshi **exchange status** every 60 s (see outages below);
5. auto-backfills markets that just settled.

launchd keeps it running. Install the agents once (and again after moving
the checkout):

```bash
sh scripts/install_agents.sh
```

This generates and loads five agents for the checkout's current location
and its `.venv`: `com.lolticker.record`, `.shadow` and `.dashboard`
(restarted by launchd if they exit, started at login), `.watchdog` (every
5 min) and `.update` (nightly refresh, 06:30). `sh scripts/install_agents.sh
uninstall` removes them; `launchctl print gui/$(id -u)/com.lolticker.record`
shows one's state and last exit code. Logs go to `data/<name>.log`; the nightly
refresh rotates any daemon log over 20 MB into `data/logs/` (gzipped, newest 6
kept, `scripts/rotate_logs.sh`).

The checkout must live outside `~/Documents`, `~/Desktop` and `~/Downloads`
(it lives in `~/Developer/esports2`): macOS privacy protection denies
launchd-spawned processes access to those folders, which is why the
recorder stayed dead after reboots on 2026-09-12 and 2026-10-08 and the
earlier nightly agent exited with `EX_CONFIG` on every run. The installer
refuses to run from them. Artifacts written before the move record absolute
paths under the old location; the production-combination loader maps
`<old root>/data/...` onto this checkout's `data/` (contents stay SHA-256
verified).

**Watchdog.** `python3 -m lol_ticker watchdog` (the `.watchdog` agent) raises
a macOS notification when the record or shadow daemon is not running, when
a LoL Esports game is live but no order book has been stored for 15 min, or
when none has been stored for 6 h. It repeats hourly while a problem lasts
and announces recovery. For alerts on your phone, set a private push URL
(for example `https://ntfy.sh/<hard-to-guess-topic>`) before installing:
`LOL_TICKER_NTFY_URL=... sh scripts/install_agents.sh`.

Without the agents (another machine, or a quick manual restart),
`sh scripts/start_daemons.sh all` starts whichever daemons are missing in
their own session so they outlive the terminal or agent session that
launched them; it leaves daemons that a loaded agent owns to launchd.

## Backing up to the Windows box

`scripts/backup_to_windows.sh` copies everything that is not in git to the
Windows PC's 12 TB drive over Tailscale. The drive is reachable only as an SMB
share (`Code`), so mount it first in Finder (Cmd+K, `smb://100.123.212.8`, tick
"remember in keychain"), then:

```bash
sh scripts/backup_to_windows.sh all /Volumes/Code      # dump (~5 min) + sync; or `dump` / `sync <mount>` separately
```

`dump` writes a `pg_dump -Fc --compress=zstd:9` of `league` (77 GB on disk,
~2.1 GB dumped) plus roles and a manifest (versions, sizes, sha256) to
`~/.cache/esports2-backup`. `sync <mount>` copies it to
`<mount>/esports2-backup/postgres/`, mirrors `data/` (second-level folders with
more than 500 files, e.g. `sq/lolalytics`, ship as one tar under `data/_tar/`
because small files crawl over SMB), adds a git bundle of every commit plus the
working tree under `code/`, verifies size and archive TOC (`VERIFY=full`
re-reads the copy for sha256), and writes restore steps to `README.txt` on the
share: install the same TimescaleDB version first, then
`timescaledb_pre_restore()` / `pg_restore -j 4` / `timescaledb_post_restore()`.
The mirror never deletes on the share and dumps accumulate by timestamp. Not
scheduled; run it after a refresh. Log: `data/backup.log`.

## Kalshi trading pauses

Kalshi halts trading for maintenance (and has scheduled maintenance windows).
The recorder:

- polls `/exchange/status` every 60 s; when trading goes inactive it opens a row
  in the **`outages`** table (`kind='halt'`) and closes it when trading resumes;
- pulls `/exchange/schedule` maintenance windows into `outages`
  (`kind='maintenance'`) at every discovery pass;
- flags any market being recorded near game time during a halt with
  **`markets.outage_affected = 1`**, and backfill applies the same flag to past
  games whose window overlapped a recorded outage.

So a flat stretch in a game's series is distinguishable from "the exchange was
down": check `outage_affected` on the market, or join the game window against
`outages`.

## Data model (TimescaleDB, database `league`)

| table | contents | source |
|---|---|---|
| `markets` | catalog: one row per Polymarket outcome *token* / per Kalshi market ticker; event id, teams/outcome, game start, status, `outage_affected`, raw API json | Gamma API (`tag=league-of-legends`) / Kalshi LoL series (`KXLOLGAME`, `KXLOLMAP`, `KXLOLTOTALMAPS`, …, in `config.py`) |
| `book_snapshots` | L2 depth `[[price, size], …]` in **YES terms**, best-first, stored on change only (`ts_ms` capture time, `hash` dedupe) | CLOB `POST /books` (batched) / Kalshi `GET /markets/{t}/orderbook` |
| `trades` | full trade tape: ts, YES price, size, taker side, raw | data-api `/trades` / Kalshi `/markets/trades` |
| `price_points` | Polymarket price series (fidelity 1 min around game window, 10 min over life) | CLOB `/prices-history` |
| `candles` | Kalshi OHLC + volume + OI (1-min; 60-min for long-lived markets) | `/series/{s}/markets/{t}/candlesticks` |
| `outages` | exchange halts & scheduled maintenance intervals | `/exchange/status`, `/exchange/schedule` |

`book_snapshots`, `trades`, `price_points`, and `candles` are **hypertables**
(7-day chunks); snapshots compress automatically after 30 days. Conventions:
all prices are YES-probabilities in `[0,1]` (Kalshi NO bids are converted to
YES asks at `1 − p`); time columns are `TIMESTAMPTZ`; books are `JSONB`
`[[price, size], …]`; a "game" is a `(platform, event_id)` pair — Polymarket
event slug (`lol-ig1-lng-2026-08-09`) or Kalshi event ticker
(`KXLOLGAME-26AUG090300LNGIG`, whose embedded time is US/Eastern and is parsed
for the true start). The `markets` catalog keeps unix-seconds `BIGINT` fields
(`game_start_ts`, `open_ts`, `close_ts`).

Example — 1-minute best bid/ask for one market with `time_bucket`:

```sql
SELECT time_bucket('1 minute', ts) AS minute,
       last((bids->0->>0)::float, ts) AS best_bid,
       last((asks->0->>0)::float, ts) AS best_ask
FROM book_snapshots
WHERE platform='kalshi' AND market_id='KXLOLGAME-26AUG090300LNGIG-LNG'
GROUP BY minute ORDER BY minute;
```

## Notes & limits

- **Effective fast cadence** is bounded by per-host rate limits in `config.py`
  (Kalshi books are fetched one per request; ~34 live markets ≈ a 4 s sweep).
  Polymarket books are batched 50/request, so its cost stays flat.
- The initial `backfill` of all historical LoL markets is hours of API
  pagination (Polymarket kill/objective props are numerous). New games settle
  incrementally, so steady-state backfill is trivial.
- Backfill anchors price windows to each market's **last trade** — `endDate` on
  long-lived markets is a resolution deadline, not when trading stopped.
- Upgrade path if 5 s snapshots aren't enough: both exchanges expose order-book
  websockets (Polymarket's is unauthenticated; Kalshi's needs an API key). The
  schema already fits deltas-as-snapshots.

### Major-league-only model comparisons

`research/wpx_major.py` builds a season-aware cohort of direct Worlds-slot leagues
and Worlds/MSI/First Stand, then runs all 14 repaired model families, both
ensembles, the 180-day recency comparison, and the matched market diagnostics.
It excludes academy/regional leagues, promotion events, and other cups. Historical
league membership is explicit for 2024–2026; an unknown season fails rather than
silently guessing. Existing historical Elo/form inputs remain available, while
all fitted prior, champion, calibration and state learners use included games.
The current gameplay-state source has no LPL-labeled games because their
timelines are unavailable. The separate postdraft export below includes LPL.

```bash
DYLD_LIBRARY_PATH=/Users/itch/Library/Python/3.9/lib/python/site-packages/sklearn/.dylibs \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/tmp/esports2-methods-v9-env/bin/python research/wpx_major.py \
  --source-cache data/wpx/adapt_cache/408b0e7b37ff020354920067
python3 research/wpx_major_report.py
```

Use `--prepare-only` to inspect `data/wpx/major_v9/cohort.json` before fitting.
Outputs are isolated under `data/wpx/major_v9`; changed cohort inputs require a
new output directory. The report also joins the existing all-league and new
major-only monthly predictions by game/minute, comparing both on identical
major-league rows. These remain retrospective diagnostics; the runner does not
deploy a model or consume a fresh evaluation block.

### Postdraft inputs, including LPL

Completed drafts do not require gameplay timelines. `wpx postdraft-build`
exports one neutral time-zero row per map using validated champions, pregame
team/player ratings and form, and earlier maps in the same series. It reads
no gameplay fields; the current map's result is only a label. Missing priors
and rejected drafts are recorded in a manifest beside the dataset.

```bash
python3 -m lol_ticker wpx postdraft-build \
  --out data/wpx/postdraft_inputs_2026-09-03.npz --before 2026-09-03
```

The output must be new and the cutoff is exclusive. Exporting inputs does not
fit a model; training and calibration still require separate date cutoffs.
The LPL betting replay uses the exact pre-May historical v8 model recovered
from its archived source and inputs, verified against all 99,408 saved held-out
predictions. Run the recovery first, then the replay:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 research/wpx_recover_v8.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 research/wpx_postdraft_bets.py
```

Artifacts are isolated under `data/wpx/postdraft_lpl_2026-09-12`; the replay
adds LPL to the frozen earlier ledger and reports $10/$20/$100 stakes with
historical taker fees. See [the LPL results and assumptions](docs/postdraft-lpl-2026-09-12.md).
