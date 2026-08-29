"""Forward-only live shadow scoring for the production win-probability stack.

The recorder writes the first successfully observed model frame for each
integer game minute.  Forecast rows are never updated or backfilled.  Outcomes
are stored separately after a local authoritative result becomes available.
"""
import datetime as dt
import hashlib
import json
import logging
import os
import time

import numpy as np
from psycopg.types.json import Json

from . import config, wpbench, wpgam, wphist


log = logging.getLogger("shadow")
RESULT_PATH = os.path.join(wpgam.OUT_DIR, "shadow_score.json")
PROTOCOL_VERSION = "shadow_v6_gg_prior_tempo"
DEFAULT_INTERVAL_S = 15
CONFIRMATORY_GAMES = 100
_SCHEMA_READY = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_protocols (
    protocol_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    config JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_predictions (
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
    blend_sha256 TEXT,
    state JSONB NOT NULL,
    markets JSONB NOT NULL,
    PRIMARY KEY (protocol_id, game_id, game_start_ts, minute),
    CHECK (minute >= 0)
);
CREATE INDEX IF NOT EXISTS idx_shadow_predictions_capture
    ON shadow_predictions (protocol_id, captured_at);
CREATE TABLE IF NOT EXISTS shadow_outcomes (
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
);
CREATE TABLE IF NOT EXISTS shadow_confirmatory_games (
    protocol_id TEXT NOT NULL REFERENCES shadow_protocols(protocol_id),
    platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi')),
    ordinal INT NOT NULL,
    game_id TEXT NOT NULL,
    game_start_ts BIGINT NOT NULL,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (protocol_id, platform, ordinal),
    UNIQUE (protocol_id, platform, game_id, game_start_ts)
);
CREATE OR REPLACE FUNCTION reject_shadow_prediction_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'shadow_predictions is append-only';
END;
$$;
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger
                   WHERE tgname='shadow_predictions_immutable') THEN
        CREATE TRIGGER shadow_predictions_immutable
        BEFORE UPDATE OR DELETE ON shadow_predictions
        FOR EACH ROW EXECUTE FUNCTION reject_shadow_prediction_mutation();
    END IF;
END;
$$;
"""


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
        "forecast_inputs": "team ratings (OE and gol.gg Elo), draft, window-health deaths, stateful Baron/Elder timers, gold share, kill recency, and game state only; no quote from the current match",
        "historical_odds": "offline teacher input only; chronological holdout gate required for deployment",
        "market_price": "blue-oriented midpoint captured only as a comparison benchmark",
        "valid_market": "non-stale and non-settled quote",
        "primary_metric": "game-balanced Brier on identical independent-forecast/market rows",
        "primary_comparison": "deployed historical-odds distillation (or standalone model) versus raw platform market",
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
        ready = conn.execute(
            """SELECT to_regclass('public.shadow_protocols') IS NOT NULL
                          AND to_regclass('public.shadow_predictions') IS NOT NULL
                          AND to_regclass('public.shadow_outcomes') IS NOT NULL
                          AND to_regclass('public.shadow_confirmatory_games') IS NOT NULL
                          AND EXISTS (SELECT 1 FROM pg_trigger
                                      WHERE tgname='shadow_predictions_immutable')
                       AS ready""").fetchone()["ready"]
        if ready:
            _SCHEMA_READY = True
            return
        conn.execute(SCHEMA)
        conn.commit()
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


def _artifact_versions():
    model_path = wpgam.MODEL_PATH
    blend_path = wphist.ARTIFACT_PATH
    out = {
        "model_kind": wpgam.MODEL_KIND,
        "model_sha256": _sha256(model_path),
        "blend_kind": None,
        "blend_sha256": None,
    }
    if os.path.exists(blend_path):
        artifact = wphist.load_artifact(blend_path)
        out["blend_kind"] = artifact.get("kind") if artifact.get("deployed") else None
        out["blend_sha256"] = _sha256(blend_path)
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


def _record_game(conn, protocol, game, versions):
    from . import live
    if hasattr(live.team_priors, "_cache"):
        del live.team_priors._cache
    priors = live.team_priors(conn, game["teams"])
    estimate = live.estimate_series(
        conn, game["game_id"],
        {k: v for k, v in priors.items() if k in live.PRIOR_KEYS},
        since_ts=0)
    oriented = _orient_game(game, estimate)
    if oriented["teams"] != game["teams"]:
        priors = live.team_priors(conn, oriented["teams"])
        estimate = live.estimate_series(
            conn, oriented["game_id"],
            {k: v for k, v in priors.items() if k in live.PRIOR_KEYS},
            since_ts=0)
    frames = estimate.get("frames") or []
    if not frames or not estimate.get("game_start_ts"):
        return 0
    frame = frames[-1]
    minute = max(0, int(frame.get("clock_s", 0)) // 60)
    start_ts = int(estimate["game_start_ts"])
    if _already_recorded(conn, protocol["protocol_id"], oriented["game_id"],
                         start_ts, minute):
        return 0
    markets = live.market_prices(
        conn, oriented["teams"], oriented.get("number") or 1,
        deciding=oriented.get("deciding", False))
    valid = _valid_market_quotes(markets)
    blend = None
    blend_error = None
    try:
        blend = wphist.predict_live(
            float(frame["p_blue"]), frame,
            estimate.get("blue_champs") or [], estimate.get("red_champs") or [])
    except (OSError, ValueError, KeyError) as exc:
        blend_error = str(exc)
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
    if blend_error:
        state["blend_error"] = blend_error
    conn.execute(
        """INSERT INTO shadow_predictions (
               protocol_id, game_id, game_start_ts, minute, captured_at,
               feed_ts, feed_lag_s, league, game_num, blue_team, red_team,
               model_p, polymarket_p, kalshi_p, polymarket_blend_p,
               kalshi_blend_p, recommended_p, recommended_source,
               polymarket_lead_s, kalshi_lead_s, model_kind, blend_kind,
               model_sha256, blend_sha256, state, markets)
           VALUES (%s,%s,%s,%s,to_timestamp(%s),%s,%s,%s,%s,%s,%s,
                   %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT DO NOTHING""",
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
         versions["blend_sha256"], Json(state), Json(markets)))
    conn.commit()
    log.info("shadow captured %s %s vs %s minute=%d feed_lag=%.1fs markets=%s",
             oriented["game_id"], oriented["teams"][0], oriented["teams"][1],
             minute, captured - float(frame["ts"]), ",".join(sorted(valid)) or "none")
    return 1


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


def _automatic_outcome(conn, prediction):
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
    return None


def _insert_outcome(conn, game_id, game_start_ts, status, blue_win, source, evidence):
    conn.execute(
        """INSERT INTO shadow_outcomes
               (game_id, game_start_ts, status, blue_win, source, evidence)
           VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
        (str(game_id), int(game_start_ts), status, blue_win, source,
         Json(evidence or {})))


def resolve_outcomes(conn, game_id=None, winner=None):
    protocol = active_protocol(conn)
    params = [protocol["protocol_id"]]
    where = "p.protocol_id=%s AND o.game_id IS NULL"
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
        for row in attempts:
            if row["game_start_ts"] != latest_start:
                _insert_outcome(conn, gid, row["game_start_ts"], "void", None,
                                "remake", {"superseded_by_start_ts": latest_start})
                voided += 1
                continue
            if winner is not None:
                blue_win = 1 if winner == "blue" else 0
                found = (blue_win, "manual", {"winner_side": winner})
            else:
                found = _automatic_outcome(conn, row)
            if found:
                _insert_outcome(conn, gid, row["game_start_ts"], "resolved",
                                found[0], found[1], found[2])
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


def score_rows(rows, bootstrap=5000):
    result = {"resolved_rows": len(rows), "platforms": {}}
    for i, platform in enumerate(("polymarket", "kalshi")):
        market_key = platform + "_p"
        blend_key = platform + "_blend_p"
        market_rows = [r for r in rows if r.get(market_key) is not None]
        blend_rows = [r for r in market_rows if r.get(blend_key) is not None]
        cohort = blend_rows or market_rows
        if not cohort:
            result["platforms"][platform] = {
                "games": 0, "states": 0, "confirmatory_ready": False}
            continue
        market = np.asarray([float(r[market_key]) for r in cohort])
        model = np.asarray([float(r["model_p"]) for r in cohort])
        blend = (np.asarray([float(r[blend_key]) for r in cohort])
                 if blend_rows else None)
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
        if blend is not None:
            summaries["blend"] = _shadow_summary(
                blend, y, gids, t_min, min(bootstrap, 2000), seed=101 + i)
        forecast_name = "blend" if blend is not None else "model"
        forecast = blend if blend is not None else model
        paired = _paired_market_improvement(
            market, forecast, y, gids, bootstrap, seed=131 + i)
        paired["forecast"] = forecast_name
        if forecast_name == "blend":
            paired["market_minus_blend"] = paired["market_minus_forecast"]
        games = len(np.unique(gids))
        result["platforms"][platform] = {
            "games": int(games), "states": len(cohort), "scores": summaries,
            "paired": paired, "confirmatory_target_games": CONFIRMATORY_GAMES,
            "confirmatory_ready": games >= CONFIRMATORY_GAMES,
            "median_market_lead_s": float(np.median([
                r[platform + "_lead_s"] for r in cohort
                if r.get(platform + "_lead_s") is not None]))
                if any(r.get(platform + "_lead_s") is not None for r in cohort) else None,
        }
    versions = {}
    for row in rows:
        key = "%s+%s" % (row["model_sha256"][:12],
                         (row.get("blend_sha256") or "none")[:12])
        versions[key] = versions.get(key, 0) + 1
    result["artifact_versions"] = versions
    return result


def _freeze_confirmatory_games(conn, protocol_id):
    frozen = {}
    for platform in ("polymarket", "kalshi"):
        existing = conn.execute(
            """SELECT game_id, game_start_ts FROM shadow_confirmatory_games
               WHERE protocol_id=%s AND platform=%s ORDER BY ordinal""",
            (protocol_id, platform)).fetchall()
        if existing:
            frozen[platform] = [(r["game_id"], r["game_start_ts"]) for r in existing]
            continue
        market_col = platform + "_p"
        games = conn.execute(
            """SELECT p.game_id, p.game_start_ts, MIN(p.captured_at) AS first_capture
               FROM shadow_predictions p JOIN shadow_outcomes o
                 USING (game_id, game_start_ts)
               WHERE p.protocol_id=%%s AND o.status='resolved'
                 AND p.%s IS NOT NULL
                 AND p.captured_at < o.recorded_at
               GROUP BY p.game_id, p.game_start_ts
               ORDER BY first_capture, p.game_id, p.game_start_ts""" %
            market_col, (protocol_id,)).fetchall()
        if len(games) < CONFIRMATORY_GAMES:
            frozen[platform] = []
            continue
        chosen = games[:CONFIRMATORY_GAMES]
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO shadow_confirmatory_games
                       (protocol_id, platform, ordinal, game_id, game_start_ts)
                   VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                [(protocol_id, platform, i + 1, r["game_id"], r["game_start_ts"])
                 for i, r in enumerate(chosen)], returning=False)
        frozen[platform] = [(r["game_id"], r["game_start_ts"]) for r in chosen]
        log.info("froze %s confirmatory cohort at %d games", platform, len(chosen))
    conn.commit()
    return frozen


def score(conn, bootstrap=5000, output_path=RESULT_PATH):
    protocol = active_protocol(conn)
    frozen = _freeze_confirmatory_games(conn, protocol["protocol_id"])
    rows = conn.execute(
        """SELECT p.*, o.blue_win
           FROM shadow_predictions p JOIN shadow_outcomes o
             USING (game_id, game_start_ts)
           WHERE p.protocol_id=%s AND o.status='resolved'
             AND p.captured_at < o.recorded_at
           ORDER BY p.captured_at""", (protocol["protocol_id"],)).fetchall()
    result = score_rows([dict(r) for r in rows], bootstrap=bootstrap)
    row_dicts = [dict(r) for r in rows]
    for platform, games in frozen.items():
        if not games:
            continue
        keys = set((str(g), int(t)) for g, t in games)
        fixed_rows = [r for r in row_dicts
                      if (str(r["game_id"]), int(r["game_start_ts"])) in keys]
        fixed = score_rows(fixed_rows, bootstrap=bootstrap)["platforms"][platform]
        result["platforms"][platform]["confirmatory"] = fixed
        result["platforms"][platform]["confirmatory_frozen"] = True
    result.update({
        "kind": "prospective_shadow_score_v1", "scored_at": _utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol": protocol["config"],
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, default=str)
    return result


def status(conn):
    protocol = active_protocol(conn)
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
    for platform in ("polymarket", "kalshi"):
        r = result["platforms"][platform]
        if not r.get("games"):
            print("  %-11s no resolved complete-case forecasts yet" % platform)
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


def report_status(result):
    print("prospective shadow protocol", result["protocol_id"])
    print("  rows=%d games=%d PM=%d KS=%d outcomes=%s" %
          (result["rows"], result["games"], result["polymarket_rows"],
           result["kalshi_rows"], result["outcomes"]))
    print("  first=%s last=%s" % (result["first_capture"], result["last_capture"]))


def record(conn, once=False, interval_s=DEFAULT_INTERVAL_S):
    protocol = active_protocol(conn)
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
                resolved = resolve_outcomes(conn)
                if resolved["resolved"] or once:
                    score(conn)
                last_resolve = time.time()
        except KeyboardInterrupt:
            raise
        except Exception:
            conn.rollback()
            log.exception("shadow recorder iteration failed; continuing")
        if once:
            return
        elapsed = time.monotonic() - started
        time.sleep(max(1.0, interval_s - elapsed))
