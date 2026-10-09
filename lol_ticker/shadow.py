"""Forward-only live shadow scoring for the production win-probability stack.

The recorder writes the first successfully observed model frame for each
integer game minute.  Forecast rows are never updated or backfilled.  Outcomes
are stored separately after a local authoritative result becomes available.
"""
import datetime as dt
import hashlib
import json
import logging
import socket
import urllib.error
import os
import time
import threading

import numpy as np
from psycopg.types.json import Json

from . import config, util, wpbench, wpgam, wphist


log = logging.getLogger("shadow")
RESULT_PATH = os.path.join(wpgam.OUT_DIR, "shadow_score.json")
PROTOCOL_VERSION = "shadow_v13_production_combination"
DEFAULT_INTERVAL_S = 15
CONFIRMATORY_GAMES = 100
# Primary-metric rows: the market quote may lead the model's feed frame by at
# most this much wall-clock time.  The feed itself lags the broadcast by
# roughly 45-140 s; anything beyond that is recorder delay, and rows captured
# after a stall hand the market minutes of extra game state.  (2026-09-03
# audit: at leads under 90 s the model and market were indistinguishable;
# the aggregate gap was driven by leads of 3-20 minutes.)
MAX_MARKET_LEAD_S = 90.0
# Warn when one capture takes longer than this, naming the slowest stage.
SLOW_CAPTURE_S = 60.0
# A feed frame older than this is not a live forecast: the recorder found a
# stalled or finished game and would store minutes-old state against a fresh
# market quote.  (2026-09-03 audit: 62 rows, mostly single late captures.)
MAX_FEED_LAG_S = 600.0
# Successive estimates of the same attempt's start drift by a few seconds as
# the feed back-fills opening frames; starts closer than this are one attempt,
# not a remake (a real remake restarts minutes later).
START_JITTER_S = 120
_SCHEMA_READY = False
_MAINTENANCE_THREAD = None
_RECORDER_SOURCE_REVISION = None

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS shadow_protocols (
        protocol_id TEXT PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        config JSONB NOT NULL
    );""",
    """CREATE TABLE IF NOT EXISTS shadow_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );""",
    """CREATE TABLE IF NOT EXISTS shadow_predictions (
        protocol_id TEXT NOT NULL REFERENCES shadow_protocols(protocol_id),
        game_id TEXT NOT NULL,
        game_start_ts BIGINT NOT NULL,
        minute INT NOT NULL,
        captured_at TIMESTAMPTZ NOT NULL,
        feed_ts BIGINT NOT NULL,
        feed_lag_s DOUBLE PRECISION NOT NULL,
        league TEXT,
        game_num INT,
        blue_team TEXT NOT NULL,
        red_team TEXT NOT NULL,
        model_p DOUBLE PRECISION NOT NULL CHECK (model_p BETWEEN 0 AND 1),
        polymarket_p DOUBLE PRECISION CHECK (polymarket_p BETWEEN 0 AND 1),
        kalshi_p DOUBLE PRECISION CHECK (kalshi_p BETWEEN 0 AND 1),
        polymarket_blend_p DOUBLE PRECISION CHECK (polymarket_blend_p BETWEEN 0 AND 1),
        kalshi_blend_p DOUBLE PRECISION CHECK (kalshi_blend_p BETWEEN 0 AND 1),
        recommended_p DOUBLE PRECISION CHECK (recommended_p BETWEEN 0 AND 1),
        recommended_source TEXT,
        polymarket_lead_s DOUBLE PRECISION,
        kalshi_lead_s DOUBLE PRECISION,
        model_kind TEXT NOT NULL,
        blend_kind TEXT,
        model_sha256 TEXT NOT NULL,
        legacy_component_sha256 TEXT,
        stack_sha256 TEXT NOT NULL,
        live_blend_w_gam DOUBLE PRECISION,
        code_revision TEXT,
        blend_sha256 TEXT,
        state JSONB NOT NULL,
        markets JSONB NOT NULL,
        PRIMARY KEY (protocol_id, game_id, game_start_ts, minute),
        CHECK (minute >= 0)
    );""",
    """CREATE INDEX IF NOT EXISTS idx_shadow_predictions_capture
        ON shadow_predictions (protocol_id, captured_at);""",
    'ALTER TABLE shadow_predictions ADD COLUMN IF NOT EXISTS legacy_component_sha256 TEXT;',
    'ALTER TABLE shadow_predictions ADD COLUMN IF NOT EXISTS stack_sha256 TEXT;',
    'ALTER TABLE shadow_predictions ADD COLUMN IF NOT EXISTS live_blend_w_gam DOUBLE PRECISION;',
    'ALTER TABLE shadow_predictions ADD COLUMN IF NOT EXISTS code_revision TEXT;',
    """CREATE TABLE IF NOT EXISTS shadow_outcomes (
        game_id TEXT NOT NULL,
        game_start_ts BIGINT NOT NULL,
        recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        status TEXT NOT NULL CHECK (status IN ('resolved', 'void')),
        blue_win SMALLINT CHECK (blue_win IN (0, 1)),
        source TEXT NOT NULL,
        evidence JSONB,
        PRIMARY KEY (game_id, game_start_ts),
        CHECK ((status = 'resolved' AND blue_win IS NOT NULL)
            OR (status = 'void' AND blue_win IS NULL))
    );""",
    """CREATE TABLE IF NOT EXISTS shadow_confirmatory_games (
        protocol_id TEXT NOT NULL REFERENCES shadow_protocols(protocol_id),
        platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi')),
        ordinal INT NOT NULL,
        game_id TEXT NOT NULL,
        game_start_ts BIGINT NOT NULL,
        registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        PRIMARY KEY (protocol_id, platform, ordinal),
        UNIQUE (protocol_id, platform, game_id, game_start_ts)
    );""",
    """CREATE OR REPLACE FUNCTION reject_shadow_prediction_mutation()
    RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
        RAISE EXCEPTION 'shadow_predictions is append-only';
    END;
    $$;""",
    """DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_trigger
                       WHERE tgname='shadow_predictions_immutable') THEN
            CREATE TRIGGER shadow_predictions_immutable
            BEFORE UPDATE OR DELETE ON shadow_predictions
            FOR EACH ROW EXECUTE FUNCTION reject_shadow_prediction_mutation();
        END IF;
    END;
    $$;""",
)


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _protocol_config():
    return {
        "version": PROTOCOL_VERSION,
        "started_at": _utc_now(),
        "prospective_only": True,
        "historical_backfill": False,
        "sampling_unit": "first captured live frame per integer game minute",
        "sampling_cadence_s": DEFAULT_INTERVAL_S,
        "forecast_inputs": "team ratings (OE and gol.gg Elo at K=30 and K=120), current-series score, draft, champion-state scores, window-health deaths, stateful Baron/Elder timers, gold share, kill recency, objective remaining time, exact120/300s trajectories, patch composition proxies and pinned prior-patch SQ decaying to zero at20m; no quote from the current match",
        "historical_odds": "offline teacher input only; chronological holdout gate required for deployment",
        "market_price": "blue-oriented midpoint captured only as a comparison benchmark",
        "valid_market": "non-stale and non-settled quote",
        "quote_collection": "bounded background lookups; latest already-observed quote no more than 15 seconds old at capture; cold or expired cache is missing, never backfilled",
        "quote_lookup_budget_s": 12.0,
        "quote_max_age_s": 15.0,
        "primary_market_lead_s": MAX_MARKET_LEAD_S,
        "max_feed_lag_s": MAX_FEED_LAG_S,
        "start_jitter_s": START_JITTER_S,
        "market_matching": "exchange event must list both teams; nearest scheduled start within live.MAX_EVENT_OFFSET_S; quotes and settlements from tickers outside that window are invalid",
        "primary_rows": "rows whose market quote leads the model's feed frame by at most primary_market_lead_s; all-lead rows are reported separately",
        "outcome_sources": "local feed/gol.gg link, Oracle's Elixir, then exchange settlement (Kalshi market result, Polymarket resolution) of the market quoted in the row",
        "primary_metric": "game-balanced Brier on identical independent-forecast/market rows",
        "primary_comparison": "explicitly user-promoted joint combination versus raw platform market; manual promotion is not a claim that the historical statistical gate passed; artifact identity is pinned on every row",
        "uncertainty": "paired game-block bootstrap",
        "confirmatory_games_per_platform": CONFIRMATORY_GAMES,
        "confirmatory_freeze": "first score pass with 100 complete-case resolved games",
        "model_updates": "allowed but artifact SHA-256 is frozen on every row",
        "outcomes": "recorded separately after capture; remade attempts are void",
    }


def _protocol_id(cfg):
    raw = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return "%s-%s" % (PROTOCOL_VERSION, hashlib.sha256(raw).hexdigest()[:12])


def _ensure_tables(conn):
    global _SCHEMA_READY
    if not _SCHEMA_READY:
        # Run the idempotent DDL once per process.  Checking only table/trigger
        # existence used to skip ADD COLUMN migrations on an older schema.
        from . import db
        db.apply_schema(conn, SCHEMA)
        _SCHEMA_READY = True


def ensure_schema(conn):
    _ensure_tables(conn)
    row = conn.execute(
        "SELECT value FROM shadow_meta WHERE key='active_protocol'").fetchone()
    if row:
        pid = row["value"]
        protocol = conn.execute(
            "SELECT protocol_id, created_at, config FROM shadow_protocols WHERE protocol_id=%s",
            (pid,)).fetchone()
        if protocol and (protocol["config"] or {}).get("version") == PROTOCOL_VERSION:
            conn.commit()
            return dict(protocol)
        # A forecast-input change requires a fresh append-only ledger.  Older
        # protocols remain queryable; only the active pointer advances.
    cfg = _protocol_config()
    pid = _protocol_id(cfg)
    conn.execute(
        "INSERT INTO shadow_protocols (protocol_id, config) VALUES (%s,%s) ON CONFLICT DO NOTHING",
        (pid, Json(cfg)))
    conn.execute(
        """INSERT INTO shadow_meta (key, value) VALUES ('active_protocol', %s)
           ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value""", (pid,))
    conn.commit()
    return dict(conn.execute(
        "SELECT protocol_id, created_at, config FROM shadow_protocols WHERE protocol_id=%s",
        (pid,)).fetchone())


def start_protocol(conn, allow_nonempty=False):
    """Register and activate a new protocol without deleting older ledgers."""
    _ensure_tables(conn)
    current = conn.execute(
        "SELECT value FROM shadow_meta WHERE key='active_protocol'").fetchone()
    if current and not allow_nonempty:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM shadow_predictions WHERE protocol_id=%s",
            (current["value"],)).fetchone()["n"]
        if count:
            raise ValueError("active shadow protocol already has forecasts; explicit rollover required")
    cfg = _protocol_config()
    pid = _protocol_id(cfg)
    conn.execute(
        "INSERT INTO shadow_protocols (protocol_id, config) VALUES (%s,%s)",
        (pid, Json(cfg)))
    conn.execute(
        """INSERT INTO shadow_meta (key, value) VALUES ('active_protocol', %s)
           ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value""", (pid,))
    conn.commit()
    return dict(conn.execute(
        "SELECT protocol_id, created_at, config FROM shadow_protocols WHERE protocol_id=%s",
        (pid,)).fetchone())


def active_protocol(conn):
    return ensure_schema(conn)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_revision():
    """Best-effort immutable source identifier for a local deployment."""
    return util.source_revision(config.REPO_ROOT)


def _artifact_versions():
    from . import wpx, wpcombined_prod
    model_path = wpgam.MODEL_PATH
    blend_path = wphist.ARTIFACT_PATH
    model_sha = _sha256(model_path)
    model_kind = wpgam.load_model(model_path)["kind"]
    stack = wpx.load_live_stack()
    deployed_stack = bool(stack.get("deployed"))
    legacy_sha = (_sha256(wpx.LEGACY_LIVE_MODEL_PATH)
                  if deployed_stack else None)
    weight = float(stack.get("w_gam", 1.0)) if deployed_stack else 1.0
    stack_spec = {
        "model_kind": ("%s+%s" % (model_kind, wpx.LEGACY_LIVE_CONTRACT)
                       if deployed_stack else model_kind),
        "model_sha256": model_sha,
        "legacy_component_sha256": legacy_sha,
        "live_blend_w_gam": weight,
    }
    stack_sha = (stack.get("stack_sha256") if deployed_stack else
                 hashlib.sha256(json.dumps(
                     stack_spec, sort_keys=True,
                     separators=(",", ":")).encode()).hexdigest())
    out = {
        "model_kind": stack_spec["model_kind"],
        "model_sha256": model_sha,
        # Freeze the comparator hash only when the holdout-gated manifest
        # actually deploys it; the current rejected stack is GAM-only.
        "live_blend_w_gam": weight,
        "legacy_component_sha256": legacy_sha,
        "stack_sha256": stack_sha,
        # A running process keeps its imported implementation even when files
        # on disk change. Pin that revision until the recorder is restarted.
        "code_revision": _RECORDER_SOURCE_REVISION or _code_revision(),
        "blend_kind": None,
        "blend_sha256": None,
    }
    try:
        combination = wpcombined_prod.load_active(wpx.LIVE_STACK_PATH)
    except (OSError, ValueError, KeyError, TypeError) as error:
        combination = None
        out["combination_error"] = str(error)
    if combination is not None:
        out.update(model_kind=wpcombined_prod.MODEL_KIND,
                   model_sha256=combination["sha256"], stack_sha256=combination["stack_sha256"],
                   live_blend_w_gam=1., legacy_component_sha256=None,
                   base_component_sha256=combination["base_sha256"],
                   joint_component_sha256=combination["joint_sha256"])
    if os.path.exists(blend_path):
        try:
            artifact = wphist.load_artifact(blend_path)
            out["blend_kind"] = artifact.get("kind") if artifact.get("deployed") and combination is None else None
            out["blend_sha256"] = _sha256(blend_path)
        except (ValueError, OSError, KeyError) as exc:
            # A rejected optional comparator must not stop GAM-only capture.
            out["blend_error"] = str(exc)
            log.warning("shadow optional blend unavailable: %s", exc)
    return out


def _orient_game(game, estimate):
    g = dict(game)
    ids = list(g.get("team_ids") or [])
    teams = list(g.get("teams") or [])
    if (estimate.get("blue_team_id") and len(ids) == 2
            and estimate["blue_team_id"] == ids[1]):
        g["teams"] = teams[::-1]
        g["team_ids"] = ids[::-1]
        g["wins"] = list(g.get("wins") or [])[::-1]
    return g


def _valid_market_quotes(markets):
    valid = {}
    for platform in ("polymarket", "kalshi"):
        row = markets.get(platform)
        if not isinstance(row, dict) or row.get("p_blue") is None:
            continue
        if row.get("settled") or row.get("stale"):
            continue
        valid[platform] = float(row["p_blue"])
    return valid


def _already_recorded(conn, protocol_id, game_id, game_start_ts, minute):
    return conn.execute(
        """SELECT 1 FROM shadow_predictions
           WHERE protocol_id=%s AND game_id=%s AND game_start_ts=%s AND minute=%s""",
        (protocol_id, str(game_id), int(game_start_ts), int(minute))).fetchone() is not None


def _canonical_start(conn, protocol_id, game_id, start_ts):
    """Reuse the start already recorded for this attempt when the new estimate
    is within START_JITTER_S of it, so jitter does not split one game into
    several attempts (which the resolver would then void as remakes)."""
    row = conn.execute(
        """SELECT game_start_ts FROM shadow_predictions
           WHERE protocol_id=%s AND game_id=%s
             AND abs(game_start_ts - %s) <= %s
           ORDER BY abs(game_start_ts - %s), captured_at LIMIT 1""",
        (protocol_id, str(game_id), int(start_ts), START_JITTER_S, int(start_ts))).fetchone()
    return int(row["game_start_ts"]) if row else int(start_ts)


class _StageClock:
    """Wall-clock seconds per capture stage, for finding recorder stalls."""

    def __init__(self):
        self.started = time.monotonic()
        self.mark = self.started
        self.stages = {}

    def lap(self, name):
        now = time.monotonic()
        self.stages[name] = round(self.stages.get(name, 0.0) + now - self.mark, 3)
        self.mark = now

    def total(self):
        return round(time.monotonic() - self.started, 3)

    def slowest(self):
        return max(self.stages.items(), key=lambda kv: kv[1]) if self.stages else ("none", 0.0)


def _record_game(conn, protocol, game, versions):
    from . import live, live_quotes
    clock = _StageClock()
    live_quotes.CACHE.prefetch(game)
    if hasattr(live.team_priors, "_cache"):
        del live.team_priors._cache
    priors = live.team_priors(conn, game["teams"])
    priors["series_diff"] = live.series_prior(game)
    clock.lap("priors")
    # Side correction shares this same per-game feed budget; a second pass
    # cannot reset the timer and stall all the games behind it.
    with live.request_deadline(live.FEED_LOOKUP_BUDGET_S):
        estimate = live.estimate_series(
            conn, game["game_id"], priors, since_ts=0, teams=game["teams"],
            research_context=game.get("research_context") or {"series_id": game.get("series_id"),
                                                              "game_num": game.get("number")})
        oriented = _orient_game(game, estimate)
        if oriented["teams"] != game["teams"]:
            priors = live.team_priors(conn, oriented["teams"])
            priors["series_diff"] = live.series_prior(oriented)
            estimate = live.estimate_series(
                conn, oriented["game_id"], priors, since_ts=0,
                teams=oriented["teams"], research_context=oriented.get("research_context")
                or {"series_id": oriented.get("series_id"), "game_num": oriented.get("number")})
    clock.lap("feed_and_model")
    frames = estimate.get("frames") or []
    if not frames or not estimate.get("game_start_ts"):
        _log_slow_capture(clock, game, None)
        return 0
    frame = frames[-1]
    minute = max(0, int(frame.get("clock_s", 0)) // 60)
    start_ts = _canonical_start(conn, protocol["protocol_id"], oriented["game_id"],
                                int(estimate["game_start_ts"]))
    live_quotes.CACHE.request(oriented, start_ts)
    feed_lag = time.time() - float(frame["ts"])
    if feed_lag > MAX_FEED_LAG_S:
        log.info("shadow skip %s minute=%d: feed frame %.0fs old (stale, > %.0fs)",
                 oriented["game_id"], minute, feed_lag, MAX_FEED_LAG_S)
        return 0
    if _already_recorded(conn, protocol["protocol_id"], oriented["game_id"],
                         start_ts, minute):
        _log_slow_capture(clock, game, minute)
        return 0
    markets = live_quotes.CACHE.snapshot(oriented, start_ts)
    clock.lap("markets")
    valid = _valid_market_quotes(markets)
    blend = None
    blend_error = None
    try:
        blend = wphist.predict_live(
            float(frame["p_blue"]), frame,
            estimate.get("blue_champs") or [], estimate.get("red_champs") or [])
    except (OSError, ValueError, KeyError) as exc:
        blend_error = str(exc)
    clock.lap("blend")
    captured = time.time()

    def value(platform, field):
        if platform not in valid:
            return None
        if field == "blend":
            return float(blend["p_blue"]) if blend else None
        return valid[platform]

    def lead(platform):
        row = markets.get(platform) or {}
        ts_ms = row.get("captured_ts_ms")
        return (float(ts_ms) / 1000.0 - float(frame["ts"])) if ts_ms else None

    state = dict(frame)
    state["blue_champs"] = estimate.get("blue_champs") or []
    state["red_champs"] = estimate.get("red_champs") or []
    state["priors"] = priors
    state["priors_effective"] = estimate.get("priors_effective")
    state["roster"] = estimate.get("roster")
    state["pelo_adjustment"] = estimate.get("pelo_adjustment")
    if blend_error:
        state["blend_error"] = blend_error
    # Stage timings up to the insert; the insert itself is logged only.
    state["capture_timing"] = dict(clock.stages, before_insert=clock.total())
    observed_at = estimate.get("feed_observed_at")
    state["capture_timing"]["upstream_feed_age_s"] = estimate.get("upstream_feed_age_s")
    state["capture_timing"]["processing_after_feed_s"] = (
        max(0.0, captured - observed_at) if observed_at is not None else None)
    state["capture_timing"]["quote_cache_ages_s"] = {
        p: r.get("quote_age_s") for p, r in markets.items()}
    inserted = conn.execute(
        """INSERT INTO shadow_predictions (
               protocol_id, game_id, game_start_ts, minute, captured_at,
               feed_ts, feed_lag_s, league, game_num, blue_team, red_team,
               model_p, polymarket_p, kalshi_p, polymarket_blend_p,
               kalshi_blend_p, recommended_p, recommended_source,
               polymarket_lead_s, kalshi_lead_s, model_kind, blend_kind,
               model_sha256, legacy_component_sha256, stack_sha256,
               live_blend_w_gam, code_revision, blend_sha256, state, markets)
           VALUES (%s,%s,%s,%s,to_timestamp(%s),%s,%s,%s,%s,%s,%s,
                   %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                   %s,%s,%s,%s)
           ON CONFLICT DO NOTHING RETURNING minute""",
        (protocol["protocol_id"], str(oriented["game_id"]), start_ts, minute,
         captured, int(frame["ts"]), captured - float(frame["ts"]),
         oriented.get("league"), oriented.get("number"), oriented["teams"][0],
         oriented["teams"][1], float(frame["p_blue"]),
         value("polymarket", "market"), value("kalshi", "market"),
         value("polymarket", "blend"), value("kalshi", "blend"),
         float(blend["p_blue"]) if blend else float(frame["p_blue"]),
         blend.get("source") if blend else "model",
         lead("polymarket"), lead("kalshi"), versions["model_kind"],
         versions["blend_kind"], versions["model_sha256"],
         versions["legacy_component_sha256"], versions["stack_sha256"],
         versions["live_blend_w_gam"], versions["code_revision"],
         versions["blend_sha256"], Json(state), Json(markets))).fetchone()
    conn.commit()
    if inserted is None:
        return 0
    # Only this new frame can enter the bounded, forward-only candidate queue.
    # Its entire immutable state (including effective priors and draft) is the
    # input evidence. Optional inference never runs on the capture thread.
    from . import wpcandidate
    try:
        wpcandidate.submit_capture({"protocol_id": protocol["protocol_id"],
                                    "game_id": str(oriented["game_id"]),
                                    "game_start_ts": start_ts, "minute": minute,
                                    "captured_at": captured, "state": state})
    except Exception:
        log.exception("optional candidate enqueue failed; incumbent frame is recorded")
    clock.lap("insert")
    log.info("shadow captured %s %s vs %s minute=%d feed_lag=%.1fs markets=%s "
             "timing=%s total=%.1fs",
             oriented["game_id"], oriented["teams"][0], oriented["teams"][1],
             minute, captured - float(frame["ts"]), ",".join(sorted(valid)) or "none",
             json.dumps(clock.stages, sort_keys=True), clock.total())
    _log_slow_capture(clock, oriented, minute)
    return 1


def _log_slow_capture(clock, game, minute):
    total = clock.total()
    if total >= SLOW_CAPTURE_S:
        stage, seconds = clock.slowest()
        log.warning("slow shadow capture for %s (%s) minute=%s: %.1fs total, "
                    "slowest stage %s=%.1fs, stages=%s",
                    game.get("game_id"), " vs ".join(game.get("teams") or []),
                    minute, total, stage, seconds,
                    json.dumps(clock.stages, sort_keys=True))


def record_once(conn):
    from . import live
    protocol = active_protocol(conn)
    versions = _artifact_versions()
    games = live.live_games()
    captured = 0
    for game in games:
        try:
            captured += _record_game(conn, protocol, game, versions)
        except Exception:
            conn.rollback()
            log.exception("shadow capture failed for %s", game.get("game_id"))
    # A pass that only read (no new minute, no frames) would otherwise leave
    # the connection idle in transaction holding share locks on the rating
    # tables between polls, which blocks and can deadlock the nightly DDL.
    conn.rollback()
    return {"live_games": len(games), "captured": captured,
            "protocol_id": protocol["protocol_id"]}


def _table_exists(conn, name):
    row = conn.execute("SELECT to_regclass(%s) IS NOT NULL AS present", (name,)).fetchone()
    return bool(row["present"])


def _winner_from_teams(prediction, blue_team, red_team, winner_side):
    from .draft import norm_team
    winner = blue_team if winner_side == "blue" else red_team
    nw = norm_team(winner)
    if nw == norm_team(prediction["blue_team"]):
        return 1
    if nw == norm_team(prediction["red_team"]):
        return 0
    return None


def _automatic_outcome(conn, prediction, exchange=True):
    from .draft import norm_team
    if _table_exists(conn, "feed_games") and _table_exists(conn, "golgg_games"):
        row = conn.execute(
            """SELECT g.game_id, g.blue_team, g.red_team, g.winner_side
               FROM feed_games f JOIN golgg_games g ON g.game_id=f.golgg_game_id
               WHERE f.esports_game_id=%s AND g.winner_side IN ('blue','red')""",
            (prediction["game_id"],)).fetchone()
        if row:
            won = _winner_from_teams(prediction, row["blue_team"], row["red_team"],
                                     row["winner_side"])
            if won is not None:
                return won, "feed_games+golgg", dict(row)
    if _table_exists(conn, "oe_games"):
        rows = conn.execute(
            """SELECT game_id, blue_team, red_team, winner, date_utc, game_num
               FROM oe_games
               WHERE winner IS NOT NULL AND date_utc BETWEEN %s AND %s
                 AND (%s IS NULL OR game_num=%s)
               ORDER BY ABS(date_utc-%s)""",
            (prediction["game_start_ts"] - 6 * 3600,
             prediction["game_start_ts"] + 12 * 3600,
             prediction.get("game_num"), prediction.get("game_num"),
             prediction["game_start_ts"])).fetchall()
        wanted = {norm_team(prediction["blue_team"]), norm_team(prediction["red_team"])}
        for row in rows:
            if {norm_team(row["blue_team"]), norm_team(row["red_team"])} != wanted:
                continue
            winner = norm_team(row["winner"])
            if winner == norm_team(prediction["blue_team"]):
                return 1, "oracle_elixir", dict(row)
            if winner == norm_team(prediction["red_team"]):
                return 0, "oracle_elixir", dict(row)
    if exchange:
        return _exchange_settlement(conn, prediction)
    return None


def _quoted_markets(conn, prediction):
    """Kalshi tickers and Polymarket slugs quoted in this attempt's rows.

    Every row stores the exact markets it compared against, so settlement can
    be read back from the same instruments without any name matching beyond
    the blue/red orientation already recorded with the quote.
    """
    tickers, slugs = {}, {}
    rows = conn.execute(
        """SELECT markets FROM shadow_predictions
           WHERE game_id=%s AND game_start_ts=%s""",
        (str(prediction["game_id"]), int(prediction["game_start_ts"]))).fetchall()
    for row in rows:
        markets = row["markets"] or {}
        kalshi = markets.get("kalshi") or {}
        for sub, detail in (kalshi.get("detail") or {}).items():
            if isinstance(detail, dict) and detail.get("ticker"):
                tickers.setdefault(detail["ticker"], (sub, kalshi.get("source")))
        polymarket = markets.get("polymarket") or {}
        for detail in (polymarket.get("detail") or {}).values():
            if isinstance(detail, dict) and detail.get("slug"):
                slugs.setdefault(detail["slug"], polymarket.get("source"))
    return tickers, slugs


def _settlement_winner(prediction, team_norm, yes_won):
    """Blue-win flag from 'did ``team_norm`` win' in the row's orientation."""
    from .draft import norm_team
    if team_norm == norm_team(prediction["blue_team"]):
        return 1 if yes_won else 0
    if team_norm == norm_team(prediction["red_team"]):
        return 0 if yes_won else 1
    return None


def _slug_date_ok(slug, game_start_ts, max_days=2):
    """Polymarket slugs carry the series date (``lol-png1-vks-2026-08-30``)."""
    import datetime as dt
    import re as _re
    m = _re.search(r"(\d{4})-(\d\d)-(\d\d)", slug or "")
    if not m or game_start_ts is None:
        return True
    try:
        slug_day = dt.date(*map(int, m.groups()))
    except ValueError:
        return True
    game_day = dt.datetime.utcfromtimestamp(int(game_start_ts)).date()
    return abs((slug_day - game_day).days) <= max_days


def _exchange_settlement(conn, prediction):
    """Outcome from the settlement of the exchange markets quoted in the row.

    A per-map market settles on that map; a match-winner market is quoted only
    for the deciding map, where the two coincide.  Network errors are logged
    and leave the game pending for the next pass.
    """
    from . import live
    tickers, slugs = _quoted_markets(conn, prediction)
    for ticker, (sub, source) in tickers.items():
        if not live.ticker_offset_ok(ticker, prediction["game_start_ts"]):
            log.warning("ignoring settlement of %s for game %s: scheduled %.0fh from game start",
                        ticker, prediction["game_id"],
                        (live.ticker_time(ticker) - prediction["game_start_ts"]) / 3600.0)
            continue
        try:
            market = live._get(
                "https://api.elections.kalshi.com/trade-api/v2/markets/%s" % ticker,
                timeout=15)["market"]
        except Exception as exc:
            log.warning("kalshi settlement lookup failed %s: %s", ticker, exc)
            continue
        result = market.get("result")
        if result not in ("yes", "no"):
            continue
        won = _settlement_winner(prediction, sub, result == "yes")
        if won is not None:
            return won, "kalshi_settlement", {
                "ticker": ticker, "result": result, "market_source": source,
                "yes_team": sub, "status": market.get("status")}
    from .draft import norm_team
    for slug, source in slugs.items():
        if not _slug_date_ok(slug, prediction["game_start_ts"]):
            log.warning("ignoring settlement of %s for game %s: slug date far from game start",
                        slug, prediction["game_id"])
            continue
        try:
            # Gamma omits closed markets from /markets?slug= unless asked for
            # them explicitly; without closed=true no settlement was ever read.
            markets = live._get("https://gamma-api.polymarket.com/markets",
                                {"slug": slug, "closed": "true"}, timeout=15) or []
        except Exception as exc:
            log.warning("polymarket settlement lookup failed %s: %s", slug, exc)
            continue
        market = markets[0] if isinstance(markets, list) and markets else None
        if not market or not market.get("closed"):
            continue
        try:
            outcomes = json.loads(market.get("outcomes") or "[]")
            prices = [float(p) for p in json.loads(market.get("outcomePrices") or "[]")]
        except (TypeError, ValueError):
            continue
        if len(outcomes) != len(prices) or not prices or max(prices) < 0.99:
            continue
        winner = norm_team(outcomes[int(np.argmax(prices))])
        won = _settlement_winner(prediction, winner, True)
        if won is not None:
            return won, "polymarket_settlement", {
                "slug": slug, "winner": outcomes[int(np.argmax(prices))],
                "outcome_prices": prices, "market_source": source,
                "uma_status": market.get("umaResolutionStatus")}
    return None


def _insert_outcome(conn, game_id, game_start_ts, status, blue_win, source, evidence):
    conn.execute(
        """INSERT INTO shadow_outcomes
               (game_id, game_start_ts, status, blue_win, source, evidence)
           VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
        (str(game_id), int(game_start_ts), status, blue_win, source,
         Json(evidence or {})))


def resolve_outcomes(conn, game_id=None, winner=None, exchange=True):
    """Attach outcomes to every pending forecast attempt.

    Outcomes belong to a game, not to a protocol, so rolled-over ledgers keep
    resolving and stay scoreable (``score(..., protocol_id=...)``).
    """
    protocol = active_protocol(conn)
    params = []
    where = "o.game_id IS NULL"
    if game_id is not None:
        where += " AND p.game_id=%s"
        params.append(str(game_id))
    pending = conn.execute(
        """SELECT DISTINCT p.game_id, p.game_start_ts, p.game_num,
                  p.blue_team, p.red_team
           FROM shadow_predictions p
           LEFT JOIN shadow_outcomes o USING (game_id, game_start_ts)
           WHERE %s ORDER BY p.game_id, p.game_start_ts""" % where,
        params).fetchall()
    by_game = {}
    for row in pending:
        by_game.setdefault(row["game_id"], []).append(row)
    resolved = voided = 0
    for gid, attempts in by_game.items():
        latest_start = max(r["game_start_ts"] for r in attempts)
        # Starts within START_JITTER_S of the latest are the same attempt seen
        # with slightly different opening frames; only earlier ones are remakes.
        same_attempt = [r for r in attempts
                        if latest_start - r["game_start_ts"] <= START_JITTER_S]
        for row in attempts:
            if row not in same_attempt:
                _insert_outcome(conn, gid, row["game_start_ts"], "void", None,
                                "remake", {"superseded_by_start_ts": latest_start})
                voided += 1
        latest = next(r for r in attempts if r["game_start_ts"] == latest_start)
        if winner is not None:
            blue_win = 1 if winner == "blue" else 0
            found = (blue_win, "manual", {"winner_side": winner})
        else:
            found = _automatic_outcome(conn, latest, exchange=exchange)
        if found:
            for row in same_attempt:
                evidence = dict(found[2] or {})
                if row["game_start_ts"] != latest_start:
                    evidence["canonical_start_ts"] = latest_start
                _insert_outcome(conn, gid, row["game_start_ts"], "resolved",
                                found[0], found[1], evidence)
                resolved += 1
    conn.commit()
    return {"resolved": resolved, "voided": voided, "pending": len(pending),
            "protocol_id": protocol["protocol_id"]}


def _paired_market_improvement(market, forecast, y, gids, bootstrap, seed):
    gids = np.asarray(gids)
    delta = (np.asarray(market) - y) ** 2 - (np.asarray(forecast) - y) ** 2
    ug = np.unique(gids)
    game_delta = np.asarray([delta[gids == g].mean() for g in ug])
    out = {"market_minus_forecast": float(game_delta.mean())}
    if len(game_delta) < 2 or bootstrap <= 0:
        out["ci95"] = None
        return out
    rng = np.random.default_rng(seed)
    draws = rng.choice(game_delta, size=(bootstrap, len(game_delta)), replace=True).mean(axis=1)
    out["ci95"] = [float(np.quantile(draws, 0.025)),
                   float(np.quantile(draws, 0.975))]
    return out


def _per_game_deltas(cohort, market, forecast, y, gids):
    """Each game's mean (market − forecast) squared-error difference, in the
    order the games were first captured, for the dashboard's running chart.
    The mean of ``delta`` over these games is ``market_minus_forecast``."""
    delta = (np.asarray(market) - y) ** 2 - (np.asarray(forecast) - y) ** 2
    out = []
    for g in dict.fromkeys(gids.tolist()):
        mask = gids == g
        first = cohort[int(np.argmax(mask))]
        out.append({"game": g, "league": first.get("league"),
                    "blue_team": first.get("blue_team"), "red_team": first.get("red_team"),
                    "captured_at": str(first.get("captured_at")),
                    "states": int(mask.sum()), "delta": float(delta[mask].mean())})
    return out


def _shadow_summary(p, y, gids, t_min, bootstrap, seed):
    p, y, gids, t_min = (np.asarray(x) for x in (p, y, gids, t_min))
    out = wpbench._basic_metrics(p, y, gids)
    loss = (p - y) ** 2
    ug = np.unique(gids)
    game_loss = np.asarray([loss[gids == g].mean() for g in ug])
    if len(game_loss) >= 2 and bootstrap > 0:
        rng = np.random.default_rng(seed)
        draws = rng.choice(game_loss, size=(bootstrap, len(game_loss)), replace=True).mean(axis=1)
        out["brier_game_ci95"] = [float(np.quantile(draws, 0.025)),
                                    float(np.quantile(draws, 0.975))]
    else:
        out["brier_game_ci95"] = None
    out["states"] = int(len(y))
    out["games"] = int(len(ug))
    out["calibration"] = (wpbench._calibration_diagnostics(p, y, gids)
                          if len(y) >= 20 and len(np.unique(y)) == 2 else None)
    out["by_phase"] = {}
    for name, lo, hi in (("early", 0.0, 15.0), ("mid", 15.0, 25.0),
                         ("late", 25.0, 1e9)):
        mask = (t_min >= lo) & (t_min < hi)
        if mask.any():
            out["by_phase"][name] = wpbench._basic_metrics(
                p[mask], y[mask], gids[mask])
    return out


def _within_lead(row, platform, max_lead_s):
    if max_lead_s is None:
        return True
    lead = row.get(platform + "_lead_s")
    return lead is not None and float(lead) <= float(max_lead_s)


def _market_event_ok(row, platform):
    """False when the row's quote came from a market scheduled far from the
    game (another meeting of one of the teams; pre-v11 single-team matching).
    Such quotes are not this game's market and are excluded from every cohort."""
    from . import live
    markets = row.get("markets") or {}
    detail = (markets.get(platform) or {}).get("detail") or {}
    start = row.get("game_start_ts")
    for d in detail.values():
        if not isinstance(d, dict):
            continue
        if platform == "kalshi" and d.get("ticker") and not live.ticker_offset_ok(d["ticker"], start):
            return False
        if platform == "polymarket" and d.get("slug") and not _slug_date_ok(d["slug"], start):
            return False
    return True


def _eligible_market_rows(rows, platform, max_lead_s):
    return [r for r in rows if r.get(platform + "_p") is not None
            and _market_event_ok(r, platform)
            and _within_lead(r, platform, max_lead_s)]


def _score_platforms(rows, bootstrap=5000, max_lead_s=MAX_MARKET_LEAD_S):
    """Per-platform scores.  The primary cohort keeps only rows whose market
    quote leads the model's frame by at most ``max_lead_s``; the same summary
    over every quoted row is attached as ``all_leads`` for reference."""
    platforms = {}
    for i, platform in enumerate(("polymarket", "kalshi")):
        market_key = platform + "_p"
        blend_key = platform + "_blend_p"
        quoted_rows = _eligible_market_rows(rows, platform, max_lead_s=None)
        cohort = _eligible_market_rows(quoted_rows, platform, max_lead_s)
        excluded = len(quoted_rows) - len(cohort)
        coverage = {
            "resolved_model_rows": len(rows), "valid_quote_rows": len(quoted_rows),
            "eligible_quote_rows": len(cohort),
            "quote_row_fraction": len(quoted_rows) / len(rows) if rows else 0.0,
            "eligible_row_fraction": len(cohort) / len(rows) if rows else 0.0,
            "eligible_games_needed": max(0, CONFIRMATORY_GAMES - len({
                (r["game_id"], r["game_start_ts"]) for r in cohort})),
        }
        timings = [(r.get("state") or {}).get("capture_timing", {}) for r in rows]
        for field in ("upstream_feed_age_s", "processing_after_feed_s"):
            values = [v[field] for v in timings if v.get(field) is not None]
            coverage["median_" + field] = float(np.median(values)) if values else None
        if not cohort:
            platforms[platform] = {
                "games": 0, "states": 0, "confirmatory_ready": False,
                "max_market_lead_s": max_lead_s,
                "coverage": coverage,
                "excluded_lead_rows": excluded}
            if max_lead_s is not None and quoted_rows:
                platforms[platform]["all_leads"] = _score_platforms(
                    quoted_rows, min(bootstrap, 2000), max_lead_s=None)[platform]
            continue
        market = np.asarray([float(r[market_key]) for r in cohort])
        model = np.asarray([float(r["model_p"]) for r in cohort])
        forecasts, sources = [], set()
        for row in cohort:
            prediction = row.get("recommended_p")
            source = row.get("recommended_source")
            if prediction is None:
                prediction = row.get(blend_key)
                source = "blend" if prediction is not None else "model"
            elif source is None:
                source = "blend" if row.get(blend_key) is not None else "model"
            forecasts.append(float(row["model_p"] if prediction is None else prediction))
            sources.add("model" if source == "model" else "blend")
        forecast = np.asarray(forecasts)
        forecast_name = next(iter(sources)) if len(sources) == 1 else "recommended"
        y = np.asarray([float(r["blue_win"]) for r in cohort])
        gids = np.asarray(["%s:%s" % (r["game_id"], r["game_start_ts"])
                           for r in cohort])
        t_min = np.asarray([float(r["minute"]) for r in cohort])
        summaries = {
            "market": _shadow_summary(market, y, gids, t_min,
                                      min(bootstrap, 2000), seed=111 + i),
            "model": _shadow_summary(model, y, gids, t_min,
                                     min(bootstrap, 2000), seed=121 + i),
        }
        if forecast_name != "model":
            summaries[forecast_name] = _shadow_summary(
                forecast, y, gids, t_min, min(bootstrap, 2000), seed=101 + i)
        paired = _paired_market_improvement(
            market, forecast, y, gids, bootstrap, seed=131 + i)
        paired["forecast"] = forecast_name
        paired["per_game"] = _per_game_deltas(cohort, market, forecast, y, gids)
        if forecast_name == "blend":
            paired["market_minus_blend"] = paired["market_minus_forecast"]
        games = len(np.unique(gids))
        platforms[platform] = {
            "games": int(games), "states": len(cohort), "scores": summaries,
            "paired": paired, "confirmatory_target_games": CONFIRMATORY_GAMES,
            "confirmatory_ready": games >= CONFIRMATORY_GAMES,
            "max_market_lead_s": max_lead_s,
            "coverage": coverage,
            "excluded_lead_rows": excluded,
            "median_market_lead_s": float(np.median([
                r[platform + "_lead_s"] for r in cohort
                if r.get(platform + "_lead_s") is not None]))
                if any(r.get(platform + "_lead_s") is not None for r in cohort) else None,
        }
        if max_lead_s is not None and excluded:
            platforms[platform]["all_leads"] = _score_platforms(
                quoted_rows, min(bootstrap, 2000), max_lead_s=None)[platform]
    return platforms


DIVERGENCE_LEAD_BUCKETS_S = ((0, 45), (45, 90), (90, 200), (200, None))
DIVERGENCE_GAPS = (0.05, 0.10)


def divergence_buckets(rows, bootstrap=1000, seed=7):
    """Who was right when the live forecast and a market disagreed.

    For each platform × market-lead bucket × minimum |forecast − market| gap:
    the game-balanced (market − forecast) Brier difference over resolved rows
    (negative = the market was better) with a game-bootstrap 95% interval.
    The live panel looks up the bucket matching the current lead and gap, so a
    divergence is shown with its measured track record rather than as a cue.
    """
    out = []
    for platform in ("polymarket", "kalshi"):
        quoted = [r for r in rows if r.get(platform + "_p") is not None
                  and r.get(platform + "_lead_s") is not None
                  and r.get("blue_win") in (0, 1) and _market_event_ok(r, platform)]
        for lo, hi in DIVERGENCE_LEAD_BUCKETS_S:
            in_lead = [r for r in quoted if r[platform + "_lead_s"] >= lo
                       and (hi is None or r[platform + "_lead_s"] < hi)]
            for gap in DIVERGENCE_GAPS:
                games = {}
                for r in in_lead:
                    f = r["recommended_p"] if r.get("recommended_p") is not None else r["model_p"]
                    m, y = float(r[platform + "_p"]), float(r["blue_win"])
                    if abs(float(f) - m) < gap:
                        continue
                    games.setdefault((r["game_id"], r["game_start_ts"]), []).append(
                        (m - y) ** 2 - (float(f) - y) ** 2)
                d = np.asarray([np.mean(v) for v in games.values()])
                entry = {"platform": platform, "lead_lo_s": lo, "lead_hi_s": hi,
                         "min_gap": gap, "games": int(len(d)),
                         "rows": int(sum(len(v) for v in games.values())),
                         "market_minus_forecast": float(d.mean()) if len(d) else None,
                         "ci95": None}
                if len(d) >= 2 and bootstrap > 0:
                    draws = np.random.default_rng(seed).choice(
                        d, size=(bootstrap, len(d)), replace=True).mean(axis=1)
                    entry["ci95"] = [float(np.quantile(draws, 0.025)),
                                     float(np.quantile(draws, 0.975))]
                out.append(entry)
    return out


def divergence_record(conn, bootstrap=1000):
    """``divergence_buckets`` over every resolved prospective forecast row."""
    rows = conn.execute(
        """SELECT p.game_id, p.game_start_ts, p.model_p, p.recommended_p,
                  p.polymarket_p, p.kalshi_p, p.polymarket_lead_s, p.kalshi_lead_s,
                  p.markets, o.blue_win
           FROM shadow_predictions p JOIN shadow_outcomes o
             USING (game_id, game_start_ts)
           WHERE o.status='resolved' AND p.captured_at < o.recorded_at""").fetchall()
    return {"rows": len(rows), "buckets": divergence_buckets([dict(r) for r in rows], bootstrap)}


def _version_key(row):
    """Stable identity for the exact forecast stack used by one row."""
    stack = row.get("stack_sha256")
    if stack:
        revision = str(row.get("code_revision") or "unknown")
        revision_key = hashlib.sha256(revision.encode()).hexdigest()[:12]
        return "%s@%s" % (str(stack)[:12],
                          revision_key)
    # Backward-compatible identity for v7 rows recorded before stack_sha256 was
    # persisted.  Such a key is explicitly marked partial because the legacy
    # component cannot be recovered from those rows.
    return "partial-%s-%s" % (
        str(row.get("model_sha256") or "none")[:12],
        str(row.get("blend_sha256") or "none")[:12])


def score_rows(rows, bootstrap=5000, max_lead_s=MAX_MARKET_LEAD_S):
    result = {"resolved_rows": len(rows),
              "max_market_lead_s": max_lead_s,
              "platforms": _score_platforms(rows, bootstrap, max_lead_s)}
    versions = {}
    for row in rows:
        key = _version_key(row)
        versions[key] = versions.get(key, 0) + 1
    result["artifact_versions"] = versions
    result["by_artifact_version"] = {}
    for key in sorted(versions):
        cohort = [row for row in rows if _version_key(row) == key]
        sample = cohort[0]
        result["by_artifact_version"][key] = {
            "rows": len(cohort),
            "model_sha256": sample.get("model_sha256"),
            "legacy_component_sha256": sample.get("legacy_component_sha256"),
            "stack_sha256": sample.get("stack_sha256"),
            "live_blend_w_gam": sample.get("live_blend_w_gam"),
            "code_revision": sample.get("code_revision"),
            "platforms": _score_platforms(cohort, min(bootstrap, 2000), max_lead_s),
        }
    return result


def _freeze_confirmatory_games(conn, protocol_id, rows, max_lead_s=MAX_MARKET_LEAD_S):
    """Freeze first eligible games from the same ordered rows used for scoring."""
    frozen = {}
    for platform in ("polymarket", "kalshi"):
        existing = conn.execute(
            """SELECT game_id, game_start_ts FROM shadow_confirmatory_games
               WHERE protocol_id=%s AND platform=%s ORDER BY ordinal""",
            (protocol_id, platform)).fetchall()
        if existing:
            frozen[platform] = [(r["game_id"], r["game_start_ts"]) for r in existing]
            continue
        games = list(dict.fromkeys(
            (r["game_id"], r["game_start_ts"])
            for r in _eligible_market_rows(rows, platform, max_lead_s)))
        if len(games) < CONFIRMATORY_GAMES:
            frozen[platform] = []
            continue
        chosen = games[:CONFIRMATORY_GAMES]
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO shadow_confirmatory_games
                       (protocol_id, platform, ordinal, game_id, game_start_ts)
                   VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                [(protocol_id, platform, i + 1, gid, start)
                 for i, (gid, start) in enumerate(chosen)], returning=False)
        frozen[platform] = chosen
        log.info("froze %s confirmatory cohort at %d games", platform, len(chosen))
    conn.commit()
    return frozen


def _protocol_by_id(conn, protocol_id):
    if protocol_id is None:
        return active_protocol(conn)
    row = conn.execute(
        "SELECT protocol_id, created_at, config FROM shadow_protocols WHERE protocol_id=%s",
        (protocol_id,)).fetchone()
    if not row:
        raise ValueError("unknown shadow protocol %s" % protocol_id)
    return dict(row)


def _expose_resolved_history(conn, protocol, scored_rows):
    """Adopt the resolved ledger before a market-shadow score is published.

    Existing reports predate the shared inventory. Conservatively adopt all
    resolved forecast history across protocols on the next ordinary score pass,
    including frames without valid market quotes. No forecast or legacy
    evaluation registry is rewritten, and exposure failure prevents publication.
    """
    from . import wpexposure
    history = conn.execute("""SELECT DISTINCT p.game_id,p.game_start_ts
        FROM shadow_predictions p JOIN shadow_outcomes o USING(game_id,game_start_ts)
        WHERE o.status='resolved' AND o.blue_win IN (0,1)
          AND p.captured_at < o.recorded_at
        ORDER BY p.game_id,p.game_start_ts""").fetchall()
    resolved = {(str(r["game_id"]), int(r["game_start_ts"])) for r in history}
    resolved.update((str(r["game_id"]), int(r["game_start_ts"])) for r in scored_rows
                    if r.get("blue_win") in (0, 1))
    if not resolved:
        return {"canonical_games_added": 0, "unlinked_games_added": 0,
                "resolved_attempts_adopted": 0, "adoption_scope": "all_resolved_shadow_protocols"}
    resolved = sorted(resolved)
    feed_ids = [game for game, _ in resolved]
    mapping = ({str(r["esports_game_id"]): r["golgg_game_id"] for r in conn.execute(
        "SELECT esports_game_id,golgg_game_id FROM feed_games WHERE esports_game_id=ANY(%s)",
        (sorted(set(feed_ids)),)).fetchall()} if _table_exists(conn, "feed_games") else {})
    protocol_hash = wpexposure.digest({"kind": "market_shadow_exposure_v1",
        "protocol_id": protocol["protocol_id"], "config": protocol.get("config") or {},
        "adoption_scope": "all_resolved_shadow_protocols"})
    exposure = wpexposure.expose(
        [mapping.get(game) for game, _ in resolved],
        [dt.datetime.fromtimestamp(start, dt.timezone.utc).date().isoformat() for _, start in resolved],
        "market-shadow:" + protocol["protocol_id"], protocol_hash, feed_game_ids=feed_ids)
    return dict(exposure, resolved_attempts_adopted=len(resolved),
                adoption_scope="all_resolved_shadow_protocols", protocol_sha256=protocol_hash)


def _refresh_candidate_scores(conn, bootstrap, output_path):
    """Refresh a separate candidate report after preserving the incumbent one."""
    from . import wpcandidate
    import tempfile
    try:
        result = wpcandidate.score(conn, bootstrap=bootstrap)
        if output_path:
            directory = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(directory, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".candidate-score-", dir=directory)
            try:
                with os.fdopen(fd, "w") as handle:
                    json.dump(result, handle, indent=2, sort_keys=True, default=str)
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, output_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return {"status": "ok", "candidates": len(result.get("candidates", [])), "output_path": output_path}
    except wpcandidate.OutcomeExposureError:
        # Candidate outcome inspection must never be relabeled an optional
        # diagnostic failure. Its exposure must succeed before it is published.
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        log.exception("candidate score refresh failed; incumbent report preserved")
        return {"status": "failed", "error": str(exc), "output_path": output_path}


def score(conn, bootstrap=5000, output_path=RESULT_PATH, protocol_id=None):
    """Score one ledger (the active protocol unless ``protocol_id`` names an
    older one).  The primary lead window comes from the protocol's own config
    so an older ledger is scored under the rule it was registered with."""
    protocol = _protocol_by_id(conn, protocol_id)
    max_lead_s = (protocol.get("config") or {}).get("primary_market_lead_s")
    if protocol_id is not None and output_path == RESULT_PATH:
        output_path = None   # never overwrite the active ledger's report
    rows = conn.execute(
        """SELECT p.*, o.blue_win
           FROM shadow_predictions p JOIN shadow_outcomes o
             USING (game_id, game_start_ts)
           WHERE p.protocol_id=%s AND o.status='resolved'
             AND p.captured_at < o.recorded_at
           ORDER BY p.captured_at, p.game_id, p.game_start_ts""",
        (protocol["protocol_id"],)).fetchall()
    row_dicts = [dict(r) for r in rows]
    exposure = _expose_resolved_history(conn, protocol, row_dicts)
    frozen = _freeze_confirmatory_games(conn, protocol["protocol_id"], row_dicts, max_lead_s)
    result = score_rows(row_dicts, bootstrap=bootstrap, max_lead_s=max_lead_s)
    for platform, games in frozen.items():
        if not games:
            continue
        keys = set((str(g), int(t)) for g, t in games)
        fixed_rows = [r for r in row_dicts
                      if (str(r["game_id"]), int(r["game_start_ts"])) in keys]
        fixed = score_rows(fixed_rows, bootstrap=bootstrap,
                           max_lead_s=max_lead_s)["platforms"][platform]
        # Keep existing membership unchanged, but invalid old quotes cannot
        # turn a cohort with fewer eligible games into a confirmatory result.
        if fixed["confirmatory_ready"]:
            result["platforms"][platform]["confirmatory"] = fixed
            result["platforms"][platform]["confirmatory_frozen"] = True
    result.update({
        "kind": "prospective_shadow_score_v1", "scored_at": _utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol": protocol["config"],
        "outcome_exposure": exposure,
    })
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=True, default=str)
    if protocol_id is None:
        candidate_path = (os.path.join(os.path.dirname(os.path.abspath(output_path)), "shadow_candidate_score.json")
                          if output_path else None)
        result["candidate_score_refresh"] = _refresh_candidate_scores(conn, bootstrap, candidate_path)
    return result


def status(conn, protocol_id=None):
    protocol = _protocol_by_id(conn, protocol_id)
    pid = protocol["protocol_id"]
    counts = conn.execute(
        """SELECT COUNT(*) AS rows,
                  COUNT(DISTINCT (game_id, game_start_ts)) AS games,
                  MIN(captured_at) AS first_capture, MAX(captured_at) AS last_capture,
                  COUNT(polymarket_p) AS polymarket_rows,
                  COUNT(kalshi_p) AS kalshi_rows
           FROM shadow_predictions WHERE protocol_id=%s""", (pid,)).fetchone()
    outcomes = conn.execute(
        """SELECT status, COUNT(*) AS n FROM shadow_outcomes o
           WHERE EXISTS (SELECT 1 FROM shadow_predictions p
                         WHERE p.protocol_id=%s AND p.game_id=o.game_id
                           AND p.game_start_ts=o.game_start_ts)
           GROUP BY status""", (pid,)).fetchall()
    out = dict(counts)
    out["outcomes"] = {r["status"]: r["n"] for r in outcomes}
    frozen = conn.execute(
        """SELECT platform, COUNT(*) AS n FROM shadow_confirmatory_games
           WHERE protocol_id=%s GROUP BY platform""", (pid,)).fetchall()
    out["confirmatory_games"] = {r["platform"]: r["n"] for r in frozen}
    out["protocol_id"] = pid
    out["protocol"] = protocol["config"]
    return out


def report_score(result):
    print("prospective shadow protocol", result["protocol_id"])
    lead = result.get("max_market_lead_s")
    if lead is not None:
        print("  primary rows: market lead <= %.0f s over the model's feed frame" % lead)
    for platform in ("polymarket", "kalshi"):
        r = result["platforms"][platform]
        coverage = r.get("coverage") or {}
        if coverage:
            print("  %-11s eligible coverage %d/%d model rows (%.1f%%); %d games remaining" % (
                platform, coverage["eligible_quote_rows"], coverage["resolved_model_rows"],
                100 * coverage["eligible_row_fraction"], coverage["eligible_games_needed"]))
        if not r.get("games"):
            print("  %-11s no resolved complete-case forecasts yet%s" % (
                platform, (" (%d quoted rows beyond the lead window)" % r["excluded_lead_rows"])
                if r.get("excluded_lead_rows") else ""))
            continue
        shown = r.get("confirmatory") or r
        scores = shown["scores"]
        ci = shown["paired"]["ci95"]
        forecast = shown["paired"].get("forecast", "blend")
        print("  %-11s %d games/%d states: %s %.5f market %.5f model %.5f; "
              "market-%s %+.5f%s" %
              (platform, shown["games"], shown["states"], forecast,
               scores[forecast]["brier_game"], scores["market"]["brier_game"],
               scores["model"]["brier_game"], forecast,
               shown["paired"]["market_minus_forecast"],
               (" [%.5f, %.5f]" % tuple(ci)) if ci else ""))
        if r.get("confirmatory_frozen"):
            print("              preregistered confirmatory cohort frozen")
        else:
            print("              descriptive only: %d/%d preregistered games" %
                  (r["games"], r["confirmatory_target_games"]))
        every = r.get("all_leads")
        if every and every.get("games"):
            print("              all leads: %d games/%d states market %.5f model %.5f "
                  "(%d rows beyond the window)" %
                  (every["games"], every["states"],
                   every["scores"]["market"]["brier_game"],
                   every["scores"]["model"]["brier_game"], r.get("excluded_lead_rows", 0)))


def report_status(result):
    print("prospective shadow protocol", result["protocol_id"])
    print("  rows=%d games=%d PM=%d KS=%d outcomes=%s" %
          (result["rows"], result["games"], result["polymarket_rows"],
           result["kalshi_rows"], result["outcomes"]))
    print("  first=%s last=%s" % (result["first_capture"], result["last_capture"]))


def _start_maintenance():
    """Resolution/scoring may use the network; never stall frame collection."""
    global _MAINTENANCE_THREAD
    if _MAINTENANCE_THREAD is not None and _MAINTENANCE_THREAD.is_alive():
        return False

    def maintain():
        from . import db, live
        own = None
        try:
            own = db.connect()
            with live.request_deadline(30):
                resolved = resolve_outcomes(own)
            # Coverage updates even when all currently observed games resolved
            # on an earlier pass and this maintenance pass adds no outcomes.
            score(own)
            log.info("shadow maintenance resolved=%d", resolved["resolved"])
        except Exception:
            if own is not None:
                own.rollback()
            log.exception("shadow maintenance failed; capture continues")
        finally:
            if own is not None:
                own.close()
    _MAINTENANCE_THREAD = threading.Thread(target=maintain, daemon=True)
    _MAINTENANCE_THREAD.start()
    return True


def record(conn, once=False, interval_s=DEFAULT_INTERVAL_S):
    global _RECORDER_SOURCE_REVISION
    _RECORDER_SOURCE_REVISION = _code_revision()
    protocol = active_protocol(conn)
    from . import wpcandidate
    try:
        wpcandidate.start_worker()
    except Exception:
        log.exception("optional candidate startup failed; incumbent capture continues")
    log.info("prospective shadow recorder starting protocol=%s cadence=%ds",
             protocol["protocol_id"], interval_s)
    last_resolve = 0.0
    while True:
        started = time.monotonic()
        try:
            result = record_once(conn)
            if result["captured"] or result["live_games"]:
                log.info("shadow pass live=%d captured=%d",
                         result["live_games"], result["captured"])
            if time.time() - last_resolve >= 300 or once:
                if once:
                    resolve_outcomes(conn)
                    score(conn)
                else:
                    _start_maintenance()
                last_resolve = time.time()
        except KeyboardInterrupt:
            raise
        except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as exc:
            # Transient network trouble (DNS, TLS handshake, timeouts): one
            # line, no traceback; the next pass retries.
            conn.rollback()
            log.warning("shadow recorder pass skipped: network error %s", exc)
        except Exception:
            conn.rollback()
            log.exception("shadow recorder iteration failed; continuing")
        if once:
            return
        elapsed = time.monotonic() - started
        time.sleep(max(1.0, interval_s - elapsed))
