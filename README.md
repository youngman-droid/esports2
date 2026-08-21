# esports2 — LoL prediction-market ticker recorder

Collects and stores ticker data for **League of Legends** markets on
**Polymarket** and **Kalshi**: live L2 order books, trades, and price history —
and keeps itself up to date as new games are played. Everything lands in
**PostgreSQL + TimescaleDB** (database `league`), queryable per game.

Python 3.9+, one dependency (`psycopg`). No API keys (all endpoints are public).

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
Search by team/date, click a game, tick markets to chart their odds over time
— Kalshi and Polymarket series can be overlaid on the same chart (solid line =
price history, dashed = recorded L2 midpoint where available). The header pill
shows live Kalshi trading status (red when the exchange is paused, refreshed
every 60 s); exchange outages that overlap a game's window are shaded red on
the chart and listed above it, and the green dashed vertical line marks
scheduled game start.

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

`game`/`export` terms match team names, event ids, or market titles; a
`YYYY-MM-DD` term filters by game day (UTC). Export writes one directory per
(platform, event) with `markets.csv`, `book_snapshots.csv`, `trades.csv`,
`candles.csv`, `price_points.csv`, and `outages.csv` when an exchange pause
overlapped the game.

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
