# esports2 — LoL prediction-market ticker recorder

Collects and stores ticker data for **League of Legends** markets on
**Polymarket** and **Kalshi**: live L2 order books, trades, and price history —
and keeps itself up to date as new games are played. Everything lands in
**PostgreSQL + TimescaleDB** (database `league`), queryable per game.

Python 3.9+; install the dependencies in `requirements.txt`. No API keys are
needed (all endpoints are public).

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
same-frame scoreboard consistency alarm.
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
- `python3 -m lol_ticker wpx all` — **model exploration and production fit**
  (`lol_ticker/wpx.py`, `lol_ticker/wpgam.py`):
  builds a state dataset (per-role gold, CS, momentum, item completion and
  item gold, objective/structure state, players on respawn timers, baron/elder
  buffs, objective timers; priors: sequential team Elo and player ratings over
  all OE games, recent form, margin-RAPM team/player ratings refit monthly on
  earlier games, a leak-free draft-model term) and evaluates a model zoo with
  5-fold game-level CV against the market at the same points, plus a fixed
  minute-mark comparison (`wpx_fair.py`). The production model is now a
  two-stage, shape-constrained, time-varying logistic GAM: a one-row-per-game
  pregame prior (live-available team/player ratings plus strongly shrunk
  champion effects), followed by five smooth game-time knots over a small
  historical/live feature contract. Gold, momentum, CS, objectives, living
  players, HP and level advantages are constrained to have
  non-negative partial effects at every knot. Inputs are standardized, games
  are balanced in the loss, only causal fixed-minute states are fitted, and an
  earlier-date holdout calibrates the final logits. `wpx gam-eval` evaluates
  the newest 20% of games and reports state- and game-weighted scores with
  game-block bootstrap intervals; `wpx fit` writes the versioned live artifact.
  `wpx bench` performs a nested chronological comparison of additive ridge
  logistic regression, unconstrained and constrained time-varying GAMs,
  histogram gradient boosting, monotone histogram boosting, and validation-fit
  convex ensembles. Hyperparameters and blend weights use a middle date block;
  the newest test block is untouched until final scoring.

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

  On the current causal rebuild (15,263 games), the strict newest-date holdout
  is 3,064 games. The v6 constrained GAM scores game-weighted Brier 0.14341,
  state-weighted Brier 0.15448, versus 0.21301 / 0.62702 game/log-loss for its
  pregame-only prior (v5: 0.14386 / 0.15483). Removing the shape constraints
  changes game-Brier by only +0.00001 (paired interval includes zero);
  monotone boosting is +0.00200 worse and ridge is +0.00283 worse, with both
  paired intervals excluding zero — but note those bench competitors all share
  the GAM's slim feature contract, so they compare model *families*, not the
  best available model. The strongest legacy spec, `champscale_reg(l2=800,
  cap15)` from the exploration zoo (full feature set plus champion×time
  terms), still beats the deployed GAM on this same holdout: fixed-minute
  state Brier 0.15312 vs 0.15448 and game Brier 0.14219 vs 0.14341, winning
  every phase and event slice (rerun 2026-08-29). The GAM stays deployed as a
  deliberate trade: shape constraints bound live misbehavior, its contract is
  provably identical between the historical fit and the live feed, and the
  shadow protocol declares it — the ~0.0011–0.0014 game-Brier gap is the
  price, and closing it (an in-game champion×time channel distilled into the
  contract) is the known next step. Neither convex ensemble improves on the
  constrained GAM. On event-aligned points inside the same newest-date test
  block, the v6 GAM scores state Brier 0.11814 versus Polymarket's 0.11220,
  and 0.12180 versus Kalshi's 0.11416; these event-triggered samples favour
  the markets and are reported separately from fixed-minute accuracy.

  `python3 -m lol_ticker wpx blend` tests probability-space and log-odds
  averages, market recalibration, positive logistic stacking and time-varying
  blends with nested chronological selection (`lol_ticker/wpblend.py`). At
  event instants (v6 rerun), the selected positive logit stack lowers
  game-weighted Brier from 0.10572 (Polymarket) and 0.10898 (model) to 0.09359
  over 1,866 untouched games; against Kalshi it lowers 0.10974 / 0.11428 to
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
  teacher's log-odds weight and the newest 20% of games is an untouched
  deployment gate. On the current v6 rebuild, the standalone GAM scores
  game-balanced Brier 0.143406 versus 0.143547 for the historical blend
  (blend-minus-model +0.000141, 95% paired interval -0.000052..+0.000345), so
  the historical blend is rejected and the standalone GAM remains deployed.

### Prospective shadow scoring

`python3 -m lol_ticker shadow record` runs the forward-only evaluator
(`lol_ticker/shadow.py`). The first live model frame observed in each integer
game minute is written together with contemporaneous blue-oriented market
midpoints used strictly as comparison benchmarks, the independently generated
historical blend when its holdout gate passes, exact feed-to-quote lag, side
assignment, and SHA-256 hashes of deployed artifacts. A database trigger rejects
updates or deletes, while the primary key admits only one row for that game
minute: missing or stale quotes remain missing and historical games are never
backfilled. Outcomes are written later to a separate table; remade attempts
are voided.

The protocol is content-addressed and registered before the first prediction.
The v6 contract change (gol.gg Elo prior, gold share, kill recency) starts a
fresh `shadow_v6_gg_prior_tempo` ledger; all earlier forecasts remain
immutable and queryable.
Its primary metric is game-balanced Brier for the deployed independent
forecast versus the raw market on the same `(game, minute)` rows, with a paired
game-block bootstrap interval. Runs
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
and the dashboard's **Prospective shadow** button shows capture progress and
scores without refitting the blend.

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
  market-based tables for cross-checking.

`game`/`export` terms match team names, event ids, or market titles; a
`YYYY-MM-DD` term filters by game day (UTC). Export writes one directory per
(platform, event) with `markets.csv`, `book_snapshots.csv`, `trades.csv`,
`candles.csv`, `price_points.csv`, and `outages.csv` when an exchange pause
overlapped the game.

## One-shot refresh

`sh scripts/update.sh` brings everything current in one go (logs in
`data/update*.log`): discovers new markets and backfills newly settled ones,
re-downloads the current Oracle's Elixir CSV and rebuilds the draft tables,
runs the gol.gg scrape incrementally for the last three weeks, then re-aligns
timelines with odds, refits the odds-free WP/WPA and draft outcome models,
rebuilds the exploration dataset and the live model, and starts the `record`
daemon if it isn't running (`--no-record` to skip that). `python3 -m
lol_ticker live` then estimates the in-progress LoL Esports game from the
official live-stats feed.

## Keeping it updated as new games are played

`record` is the updater. Every loop it:

1. re-discovers markets every 10 min (new match markets appear on both
   platforms days before games);
2. splits open markets into a **fast tier** (game start within −45 min…+8 h,
   polled every ~5 s) and a **slow tier** (everything else, every 5 min);
3. stores changed books and pulls incremental trades for fast-tier markets;
4. checks Kalshi **exchange status** every 60 s (see outages below);
5. auto-backfills markets that just settled.

Run it under launchd so it survives reboots:

```bash
cp scripts/com.lolticker.record.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.lolticker.record.plist
```

Logs go to `data/record.log`.

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
