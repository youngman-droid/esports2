"""Tiny local dashboard: search games, chart historical odds, Kalshi status.

Serves on 127.0.0.1 only.  Endpoints:
  /                    the page
  /api/status          Kalshi trading status + recent outages
  /api/search?q=...    games matching team/date terms
  /api/game?platform=&event_id=      market list + outages for one event
  /api/series?platform=&market_id=   odds series for one market
  /api/calibration?horizon=&filter=   reliability bins + Brier per platform
"""
import concurrent.futures
import json
import logging
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config, db, kalshi, query

_conn = None
_conn_lock = threading.Lock()

PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


def _db():
    global _conn
    if _conn is None:
        _conn = db.connect()
    return _conn


def api_status():
    """Live Kalshi trading status, falling back to the outages table.

    Also reports recorder health: whether a ``lol_ticker record`` process is
    alive and how old the newest stored book snapshot is, so a silently dead
    daemon shows up in the header instead of as a mysteriously flat chart.
    """
    out = {"kalshi_trading_active": None, "halted_since": None, "recent_outages": []}
    try:
        import subprocess
        out["record_daemon_running"] = subprocess.run(
            ["pgrep", "-f", "lol_ticker record"], capture_output=True,
            timeout=5).returncode == 0
    except Exception:
        out["record_daemon_running"] = None
    with _conn_lock:
        conn = _db()
        row = conn.execute(
            "SELECT extract(epoch from now() - max(ts)) AS age FROM book_snapshots").fetchone()
        out["record_snapshot_age_s"] = round(row["age"]) if row and row["age"] is not None else None
        out["halted_since"] = db.ongoing_outage(conn, "kalshi")
        rows = conn.execute(
            """SELECT kind, start_ts, end_ts FROM outages WHERE platform='kalshi'
               ORDER BY start_ts DESC LIMIT 10""").fetchall()
        out["recent_outages"] = [dict(r) for r in rows]
        conn.rollback()
    try:
        active, _ = kalshi.fetch_exchange_status()
        out["kalshi_trading_active"] = active
    except Exception:
        # API unreachable: infer from the recorder's outage tracking
        out["kalshi_trading_active"] = out["halted_since"] is None
        out["status_source"] = "db-fallback"
    return out


def api_search(params):
    q = (params.get("q") or [""])[0].strip()
    if not q:
        return []
    with _conn_lock:
        conn = _db()
        games = query.find_games(conn, q.split())
        conn.rollback()
    return games


def api_game(params):
    platform = (params.get("platform") or [""])[0]
    event_id = (params.get("event_id") or [""])[0]
    with _conn_lock:
        conn = _db()
        markets = [dict(r) for r in conn.execute(
            """SELECT m.market_id, m.title, m.outcome, m.status, m.result,
                      m.game_start_ts, m.outage_affected,
                      (SELECT COUNT(*) FROM trades t WHERE t.platform = m.platform
                       AND t.market_id = m.market_id) AS n_trades
               FROM markets m WHERE m.platform = %s AND m.event_id = %s
               ORDER BY n_trades DESC, market_id""", (platform, event_id))]
        g = min((m["game_start_ts"] for m in markets if m["game_start_ts"]),
                default=None)
        outages = []
        if g:
            outages = [dict(r) for r in db.outages_overlapping(
                conn, platform, g - config.FAST_BEFORE, g + config.FAST_AFTER)]
        conn.rollback()
    return {"markets": markets, "game_start_ts": g, "outages": outages}


def api_series(params):
    platform = (params.get("platform") or [""])[0]
    market_id = (params.get("market_id") or [""])[0]
    with _conn_lock:
        conn = _db()
        if platform == "polymarket":
            price = conn.execute(
                """SELECT DISTINCT ON (ts) EXTRACT(EPOCH FROM ts)::bigint AS t,
                          price AS p
                   FROM price_points WHERE platform = %s AND market_id = %s
                   ORDER BY ts, fidelity""", (platform, market_id)).fetchall()
        else:
            price = conn.execute(
                """SELECT DISTINCT ON (ts) EXTRACT(EPOCH FROM ts)::bigint AS t,
                          close AS p
                   FROM candles WHERE platform = %s AND market_id = %s
                     AND close IS NOT NULL
                   ORDER BY ts, period_min""", (platform, market_id)).fetchall()
        mid = conn.execute(
            """SELECT (EXTRACT(EPOCH FROM ts))::bigint AS t,
                      (bids->0->0)::text::float AS bb,
                      (asks->0->0)::text::float AS ba
               FROM book_snapshots WHERE platform = %s AND market_id = %s
               ORDER BY ts""", (platform, market_id)).fetchall()
        trades = conn.execute(
            """SELECT EXTRACT(EPOCH FROM ts)::bigint AS t, price AS p
               FROM trades WHERE platform = %s AND market_id = %s
               ORDER BY ts""", (platform, market_id)).fetchall()
        conn.rollback()
    mids = [[r["t"], round((r["bb"] + r["ba"]) / 2, 4)]
            for r in mid if r["bb"] is not None and r["ba"] is not None]
    return {
        "price": [[r["t"], r["p"]] for r in price if r["p"] is not None],
        "mid": mids,
        "trades": [[r["t"], r["p"]] for r in trades if r["p"] is not None],
    }


_cal_cache = {}  # (horizon, filter) -> (computed_at, payload)
CAL_CACHE_TTL = 600


def api_calibration(params):
    """Reliability bins + Brier score per platform.

    Prediction = last observed YES price at least `horizon` seconds before the
    game's scheduled start.  Outcome = the market's settled result.  Note both
    sides of a market enter the sample (Kalshi's two team markets, Polymarket's
    Yes/No tokens), which is symmetric around 0.5 by construction.
    """
    # negative horizon = sample after scheduled start (post-draft / early game)
    horizon = max(-2 * 3600, min(30 * 86400, int((params.get("horizon") or ["0"])[0])))
    n_bins = max(2, min(100, int((params.get("bins") or ["100"])[0])))
    filt = "%" + (params.get("filter") or [""])[0].strip() + "%"
    key = (horizon, filt.lower(), n_bins)
    hit = _cal_cache.get(key)
    if hit and time.time() - hit[0] < CAL_CACHE_TTL:
        return hit[1]
    with _conn_lock:
        conn = _db()
        out = {"kalshi": _bins(_kalshi_pairs(conn, horizon, filt), n_bins),
               "polymarket": _bins(_pm_pairs(conn, horizon, filt), n_bins),
               "horizon": horizon, "bins": n_bins}
        conn.rollback()
    _cal_cache[key] = (time.time(), out)
    return out


def _kalshi_pairs(conn, horizon, filt):
    # one scan over candles, keeping each market's last candle before cutoff.
    # Sample the bid/ask midpoint when the book is quoted: candle closes fall
    # back to yes_bid for trade-less periods, which understates fair value and
    # fakes a longshot bias at the low end.
    rows = conn.execute(
        """
        WITH last_price AS (
            SELECT DISTINCT ON (c.market_id) c.market_id,
                   COALESCE(((c.raw->'yes_bid'->>'close_dollars')::float
                             + (c.raw->'yes_ask'->>'close_dollars')::float) / 2,
                            c.close) AS p
            FROM candles c
            JOIN markets m ON m.platform = c.platform AND m.market_id = c.market_id
            WHERE c.platform = 'kalshi' AND c.close IS NOT NULL
              AND m.status IN ('settled', 'finalized')
              AND m.result IN ('yes', 'no') AND m.game_start_ts IS NOT NULL
              AND c.ts <= to_timestamp(m.game_start_ts - %(h)s)
              AND (m.title ILIKE %(f)s OR m.event_id ILIKE %(f)s)
            ORDER BY c.market_id, c.ts DESC
        )
        SELECT m.result, lp.p
        FROM last_price lp
        JOIN markets m ON m.platform = 'kalshi' AND m.market_id = lp.market_id
        """, {"h": horizon, "f": filt}).fetchall()
    return [(r["p"], 1.0 if r["result"] == "yes" else 0.0) for r in rows]


def _pm_pairs(conn, horizon, filt):
    # only conditions where BOTH outcome tokens have a pre-cutoff price enter
    # the sample: one-sided sampling is winner-biased (losing tokens often
    # stop trading, so their price series is missing) and wrecks calibration
    rows = conn.execute(
        """
        WITH last_price AS (
            SELECT DISTINCT ON (pp.market_id) pp.market_id, pp.price AS p
            FROM price_points pp
            JOIN markets m ON m.platform = pp.platform AND m.market_id = pp.market_id
            WHERE pp.platform = 'polymarket' AND m.status = 'closed'
              AND m.game_start_ts IS NOT NULL
              AND m.raw->>'outcomePrices' IS NOT NULL
              AND pp.ts <= to_timestamp(m.game_start_ts - %(h)s)
              AND (m.title ILIKE %(f)s OR m.event_id ILIKE %(f)s)
            ORDER BY pp.market_id, pp.ts DESC
        ),
        paired AS (
            SELECT m.condition_id
            FROM last_price lp
            JOIN markets m ON m.platform = 'polymarket' AND m.market_id = lp.market_id
            GROUP BY m.condition_id HAVING COUNT(*) >= 2
        )
        SELECT m.outcome, m.raw->>'outcomes' AS outs,
               m.raw->>'outcomePrices' AS prices, lp.p
        FROM last_price lp
        JOIN markets m ON m.platform = 'polymarket' AND m.market_id = lp.market_id
        WHERE m.condition_id IN (SELECT condition_id FROM paired)
        """, {"h": horizon, "f": filt}).fetchall()
    pairs = []
    for r in rows:
        try:
            outs = json.loads(r["outs"])
            prices = [float(x) for x in json.loads(r["prices"])]
            idx = outs.index(r["outcome"])
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        y = prices[idx]
        if y > 0.99:
            pairs.append((r["p"], 1.0))
        elif y < 0.01:
            pairs.append((r["p"], 0.0))
        # anything between is a voided/fair-price resolution: excluded
    return pairs


def _bins(pairs, n_bins=10):
    bins = [{"n": 0, "sum_p": 0.0, "sum_y": 0.0} for _ in range(n_bins)]
    brier = 0.0
    for p, y in pairs:
        if p is None:
            continue
        b = bins[min(int(p * n_bins), n_bins - 1)]
        b["n"] += 1
        b["sum_p"] += p
        b["sum_y"] += y
        brier += (p - y) ** 2
    n = sum(b["n"] for b in bins)
    return {
        "n": n,
        "brier": round(brier / n, 5) if n else None,
        "bins": [{
            "lo": round(i / n_bins, 4), "hi": round((i + 1) / n_bins, 4), "n": b["n"],
            "mean_pred": round(b["sum_p"] / b["n"], 4) if b["n"] else None,
            "obs_freq": round(b["sum_y"] / b["n"], 4) if b["n"] else None,
        } for i, b in enumerate(bins)],
    }


# one market observation per (game, team) even when both platforms cover it
_PER_GAME = """
    per_game AS (
        SELECT oe_game_id, team, MIN(league) AS league,
               AVG(delta) AS delta, AVG(pre_p) AS pre_p,
               AVG(post_p) AS post_p, MAX(won) AS won
        FROM draft_deltas
        WHERE (%(plat)s = '' OR platform = %(plat)s)
          AND league ILIKE ANY(%(lgs)s)
        GROUP BY oe_game_id, team
    )
"""


def _draft_params(params, default_min):
    # league accepts a comma-separated list: "LCK, LPL, LEC" -> any of them
    leagues = [t.strip() for t in (params.get("league") or [""])[0].split(",")
               if t.strip()]
    return {
        "plat": (params.get("platform") or [""])[0],
        "lgs": ["%" + lg + "%" for lg in leagues] or ["%"],
        "minn": max(1, int((params.get("min_n") or [str(default_min)])[0])),
    }


def api_draft_teams(params):
    p = _draft_params(params, 8)
    with _conn_lock:
        conn = _db()
        rows = conn.execute("WITH " + _PER_GAME + """
            SELECT team, COUNT(*) AS n,
                   ROUND(AVG(delta)::numeric, 4) AS avg_delta,
                   ROUND(AVG(pre_p)::numeric, 3) AS avg_pre,
                   ROUND(AVG(post_p)::numeric, 3) AS avg_post,
                   ROUND(AVG(won::float)::numeric, 3) AS win_rate,
                   ROUND(AVG(won - post_p)::numeric, 4) AS post_edge
            FROM per_game GROUP BY team HAVING COUNT(*) >= %(minn)s
            ORDER BY avg_delta DESC""", p).fetchall()
        conn.rollback()
    return [dict(r) for r in rows]


def api_draft_champions(params):
    p = _draft_params(params, 10)
    with _conn_lock:
        conn = _db()
        rows = conn.execute("WITH " + _PER_GAME + """
            SELECT pk.champion, COUNT(*) AS n,
                   ROUND(AVG(g.delta)::numeric, 4) AS avg_delta,
                   ROUND(AVG(g.won::float)::numeric, 3) AS win_rate
            FROM per_game g
            JOIN oe_picks pk ON pk.game_id = g.oe_game_id AND pk.team = g.team
            GROUP BY pk.champion HAVING COUNT(*) >= %(minn)s
            ORDER BY avg_delta DESC""", p).fetchall()
        conn.rollback()
    return [dict(r) for r in rows]


def api_draft_pairs(params):
    p = _draft_params(params, 6)
    with _conn_lock:
        conn = _db()
        rows = conn.execute("WITH " + _PER_GAME + """
            SELECT p1.champion AS c1, p2.champion AS c2, COUNT(*) AS n,
                   ROUND(AVG(g.delta)::numeric, 4) AS avg_delta,
                   ROUND(AVG(g.won::float)::numeric, 3) AS win_rate
            FROM per_game g
            JOIN oe_picks p1 ON p1.game_id = g.oe_game_id AND p1.team = g.team
            JOIN oe_picks p2 ON p2.game_id = g.oe_game_id AND p2.team = g.team
                            AND p1.champion < p2.champion
            GROUP BY p1.champion, p2.champion HAVING COUNT(*) >= %(minn)s
            ORDER BY avg_delta DESC""", p).fetchall()
        conn.rollback()
    return [dict(r) for r in rows]


_model_cache = {"loaded_at": 0, "model": None}


def _draft_model():
    from . import draft
    if time.time() - _model_cache["loaded_at"] > 600 or _model_cache["model"] is None:
        with _conn_lock:
            conn = _db()
            _model_cache["model"] = draft.load_model(conn)
            conn.rollback()
        _model_cache["loaded_at"] = time.time()
    return _model_cache["model"]


def api_draft_simulate(params):
    """GET ?pre=0.5&seq=<json [[side, action, champion], ...]>"""
    from . import draft
    model = _draft_model()
    pre = float((params.get("pre") or ["0.5"])[0])
    pre = min(0.99, max(0.01, pre))
    patch = (params.get("patch") or [""])[0].strip()
    try:
        seq = json.loads((params.get("seq") or ["[]"])[0])
        actions = [(str(a[0]).upper()[:1], str(a[1]).lower(), (a[2] or "").strip(),
                    (a[3] if len(a) > 3 else "") or "")
                   for a in seq]
    except (ValueError, TypeError, IndexError):
        return {"error": "bad seq"}
    out = draft.simulate(model, pre, actions, patch)
    meta = model.get("__meta__")
    meta_t = model.get("__meta_time__")
    out["model"] = {"r2": round(meta[0], 3) if meta else None,
                    "r2_time": (round(meta_t[0], 3) if meta_t and meta_t[0] is not None
                                else None),
                    "rows": meta[1] if meta else 0,
                    "features": len([f for f in model if not f.startswith("__")])}
    return out


def api_draft_patches(params):
    from . import draft
    return draft.model_patches(_draft_model())


def api_draft_champlist(params):
    model = _draft_model()
    champs = sorted({f.split(":", 1)[1] for f in model
                     if f.startswith("own_pick:") or f.startswith("own_ban:")})
    return champs


def api_align_games(params):
    """Aligned games available for the timeline view (newest first)."""
    with _conn_lock:
        conn = _db()
        rows = conn.execute("""
            SELECT a.game_id, a.platform, a.market_id, a.team, a.team_side, a.quality,
                   a.n_matched, a.n_events, a.start_wall, a.end_wall, a.duration_s,
                   jsonb_array_length(a.pauses) AS n_pauses,
                   g.blue_team, g.red_team, g.date, g.game_num, g.trname, g.winner_side
            FROM game_alignment a JOIN golgg_games g ON g.game_id = a.game_id
            ORDER BY g.date DESC, a.game_id DESC, a.platform LIMIT 300""").fetchall()
        conn.rollback()
    return [dict(r) for r in rows]


def api_align_game(params):
    """Full alignment for one (game, platform, market): odds series + events."""
    from . import align
    game_id = int((params.get("game_id") or ["0"])[0])
    platform = (params.get("platform") or [""])[0]
    market_id = (params.get("market_id") or [""])[0]
    with _conn_lock:
        conn = _db()
        a = align.align_game(conn, game_id, platform, market_id)
        conn.rollback()
    if not a:
        return {"error": "no alignment possible (missing odds or link)"}
    a["series"] = [[t, p] for t, p in a["series"]]
    return a


def api_align_impact(params):
    from . import align
    with _conn_lock:
        conn = _db()
        rows = align.event_impact(
            conn, platform=(params.get("platform") or [""])[0],
            league=(params.get("league") or [""])[0].strip(),
            min_n=int((params.get("min_n") or ["5"])[0]),
            phase=(params.get("phase") or [""])[0] or None)
        conn.rollback()
    return rows


REGION_LABELS = {
    "KR": "LCK (Korea)", "CN": "LPL (China)", "EUW": "LEC (EMEA)", "NA": "LCS / LTA North",
    "LTA": "LTA (Americas)", "BR": "CBLOL / LTA South", "VN": "VCS (Vietnam)",
    "PCS": "PCS / LCP", "JP": "LJL (Japan)", "TR": "TCL (Türkiye)", "LAT": "LLA (LatAm)",
    "OCE": "LCO (Oceania)", "WR": "International / EMEA Masters",
    "AL": "Arabian League", "FR": "LFL (France)", "DE": "Prime League (DACH)",
    "ES": "Superliga (Spain)", "IT": "PG Nationals (Italy)", "PL": "Ultraliga (Poland)",
    "PT": "LPLOL (Portugal)", "CZ": "Hitpoint (CZ/SK)", "GR": "GLL (Greece)",
    "RS": "EBL (Balkans)", "BE": "Elite Series (Benelux)", "NL": "Elite Series (Benelux)",
}


def api_picker_tournaments(params):
    with _conn_lock:
        conn = _db()
        rows = conn.execute("""
            SELECT t.trname, t.season, t.region, t.nbgames, t.first_game, t.last_game,
                   (SELECT COUNT(*) FROM golgg_games g WHERE g.trname = t.trname) AS scraped,
                   (SELECT COUNT(DISTINCT a.game_id) FROM game_alignment a
                      JOIN golgg_games g2 ON g2.game_id = a.game_id
                     WHERE g2.trname = t.trname) AS aligned
            FROM golgg_tournaments t ORDER BY t.season DESC, t.region, t.last_game DESC""").fetchall()
        conn.rollback()
    out = []
    for r in rows:
        d = dict(r)
        d["region_label"] = REGION_LABELS.get(d["region"], d["region"])
        d["first_game"] = str(d["first_game"]) if d["first_game"] else None
        d["last_game"] = str(d["last_game"]) if d["last_game"] else None
        out.append(d)
    return out


def api_picker_games(params):
    trname = (params.get("trname") or [""])[0]
    with _conn_lock:
        conn = _db()
        games = [dict(r) for r in conn.execute("""
            SELECT g.game_id, g.match_id, g.game_num, g.date, g.duration_s, g.blue_team,
                   g.red_team, g.winner_side, g.patch, g.oe_game_id,
                   (SELECT json_agg(json_build_array(a.platform, a.market_id, a.team, a.quality))
                      FROM game_alignment a WHERE a.game_id = g.game_id) AS alignments,
                   (SELECT json_agg(DISTINCT jsonb_build_array(m.platform, m.event_id))
                      FROM draft_deltas d JOIN markets m
                        ON m.platform = d.platform AND m.market_id = d.market_id
                     WHERE d.oe_game_id = g.oe_game_id) AS events
            FROM golgg_games g WHERE g.trname = %s
            ORDER BY g.date DESC, g.match_id DESC, g.game_num""", (trname,))]
        conn.rollback()
    for g in games:
        g["date"] = str(g["date"]) if g["date"] else None
    return games


def api_wpa_game(params):
    from . import wpa
    gid = int((params.get("game_id") or ["0"])[0])
    with _conn_lock:
        conn = _db()
        r = wpa.game_wp(conn, gid)
        conn.rollback()
    return r or {"error": "no model or game data"}


def api_wpa_impact(params):
    from . import wpa
    with _conn_lock:
        conn = _db()
        rows = wpa.event_impact(conn, league=(params.get("league") or [""])[0].strip(),
                                min_n=int((params.get("min_n") or ["10"])[0]),
                                phase=(params.get("phase") or [""])[0] or None)
        _, meta = wpa.load_model(conn)
        conn.rollback()
    return {"rows": rows, "meta": meta}


def api_draftfree_teams(params):
    p = _draft_params(params, 8)
    with _conn_lock:
        conn = _db()
        rows = conn.execute("""
            SELECT team, COUNT(*) AS n,
                   ROUND(AVG(edge)::numeric, 4) AS avg_edge,
                   ROUND(AVG(p_elo)::numeric, 3) AS avg_p_elo,
                   ROUND(AVG(p_full)::numeric, 3) AS avg_p_full,
                   ROUND(AVG(won::float)::numeric, 3) AS win_rate
            FROM draft_outcome_games WHERE league ILIKE ANY(%(lgs)s)
            GROUP BY team HAVING COUNT(*) >= %(minn)s ORDER BY avg_edge DESC""", p).fetchall()
        meta = conn.execute("SELECT value FROM draft_outcome_meta WHERE key='fit'").fetchone()
        conn.rollback()
    return {"rows": [dict(r) for r in rows], "meta": json.loads(meta["value"]) if meta else None}


def api_draftfree_champions(params):
    minn = max(1, int((params.get("min_n") or ["20"])[0]))
    kind = (params.get("kind") or ["own_pick"])[0]
    with _conn_lock:
        conn = _db()
        rows = conn.execute("""
            SELECT feature, coef, n FROM draft_outcome_model
            WHERE feature LIKE %s AND feature NOT LIKE '%%@%%' AND feature NOT LIKE '%%#%%'
              AND n >= %s ORDER BY coef DESC""", (kind + ":%", minn)).fetchall()
        conn.rollback()
    out = []
    for r in rows:
        name = r["feature"].split(":", 1)[1]
        out.append({"name": name, "coef": round(r["coef"], 4), "n": r["n"],
                    # log-odds -> approx win-prob effect at 50%
                    "dp": round((1 / (1 + 2.718281828 ** (-r["coef"])) - 0.5), 4)})
    return out


def api_impact_compare(params):
    """Market swing vs outcome-model WPA per event type, same filters."""
    from . import align, wpa
    phase = (params.get("phase") or [""])[0] or None
    league = (params.get("league") or [""])[0].strip()
    platform = (params.get("platform") or [""])[0]
    min_n = int((params.get("min_n") or ["10"])[0])
    with _conn_lock:
        conn = _db()
        mk = {r["action"]: r for r in align.event_impact(
            conn, platform=platform, league=league, min_n=min_n, phase=phase)}
        md = {r["action"]: r for r in wpa.event_impact(conn, league=league, min_n=min_n, phase=phase)}
        conn.rollback()
    out = []
    for a in sorted(set(mk) | set(md)):
        if a in ("plate", "nexus"):
            continue
        m, d = mk.get(a), md.get(a)
        out.append({
            "action": a,
            "market_n": m["n"] if m else 0,
            "market_mean": float(m["mean_swing"]) if m else None,
            "market_median": float(m["median_swing"]) if m else None,
            "model_n": d["n"] if d else 0,
            "model_mean": float(d["mean_wpa"]) if d else None,
            "model_median": float(d["median_wpa"]) if d else None,
            "ratio": (round(float(m["mean_swing"]) / float(d["mean_wpa"]), 2)
                      if m and d and d["mean_wpa"] and abs(float(d["mean_wpa"])) > 1e-4 else None),
        })
    out.sort(key=lambda r: -(r["model_mean"] if r["model_mean"] is not None else -9))
    return out


def api_picker_game(params):
    """Everything the page needs to make one gol.gg game the global context."""
    gid = int((params.get("game_id") or ["0"])[0])
    from . import align
    with _conn_lock:
        conn = _db()
        # link to OE and align against its markets on demand, so freshly
        # scraped games work without waiting for a bulk `align` run
        try:
            align.ensure_aligned(conn, gid)
        except Exception:
            conn.rollback()
        g = conn.execute("""SELECT g.*, og.league AS oe_league, og.date_utc AS oe_start
                            FROM golgg_games g LEFT JOIN oe_games og ON og.game_id = g.oe_game_id
                            WHERE g.game_id = %s""", (gid,)).fetchone()
        if not g:
            conn.rollback()
            return {"error": "unknown game"}
        players = [dict(r) for r in conn.execute(
            """SELECT side, role, champion, player, slot FROM golgg_players
               WHERE game_id = %s ORDER BY slot""", (gid,))]
        pre = conn.execute("""SELECT team, AVG(pre_p) AS pre_p FROM draft_deltas
                              WHERE oe_game_id = %s GROUP BY team""", (g["oe_game_id"],)).fetchall()
        ev_set = set()
        for m in align.linked_markets(conn, gid):
            eid = m.get("event_id")
            if not eid:
                row = conn.execute("SELECT event_id FROM markets WHERE platform=%s AND market_id=%s",
                                   (m["platform"], m["market_id"])).fetchone()
                eid = row["event_id"] if row else None
            if eid:
                ev_set.add((m["platform"], eid))
        events = [{"platform": p_, "event_id": e_} for p_, e_ in sorted(ev_set)]
        aligns = conn.execute("""SELECT platform, market_id, team, quality FROM game_alignment
                                 WHERE game_id = %s ORDER BY quality DESC NULLS LAST""", (gid,)).fetchall()
        conn.rollback()
    from .draft import norm_team
    pre_blue = None
    for r in pre:
        if norm_team(r["team"]) == norm_team(g["blue_team"]):
            pre_blue = float(r["pre_p"])
    elo_blue = None
    if g["elo_blue_pre"] is not None and g["elo_red_pre"] is not None:
        elo_blue = 1 / (1 + 10 ** ((g["elo_red_pre"] - g["elo_blue_pre"]) / 400))
    d = {k: g[k] for k in ("game_id", "match_id", "game_num", "trname", "patch", "duration_s",
                           "blue_team", "red_team", "winner_side", "blue_bans", "red_bans",
                           "blue_picks", "red_picks", "oe_game_id", "oe_league")}
    d["date"] = str(g["date"]) if g["date"] else None
    d["players"] = players
    d["pre_blue_market"] = pre_blue
    d["pre_blue_elo"] = round(elo_blue, 4) if elo_blue is not None else None
    d["events"] = [[r["platform"], r["event_id"]] for r in events]
    d["alignments"] = [[r["platform"], r["market_id"], r["team"], r["quality"]] for r in aligns]
    return d


def api_picker_resolve(params):
    """gol.gg games behind a market event (for search -> context propagation)."""
    platform = (params.get("platform") or [""])[0]
    event_id = (params.get("event_id") or [""])[0]
    with _conn_lock:
        conn = _db()
        rows = conn.execute("""
            SELECT DISTINCT g.game_id, g.game_num, g.blue_team, g.red_team, g.date
            FROM markets m JOIN draft_deltas d ON d.platform = m.platform AND d.market_id = m.market_id
            JOIN golgg_games g ON g.oe_game_id = d.oe_game_id
            WHERE m.platform = %s AND m.event_id = %s ORDER BY g.game_num""", (platform, event_id)).fetchall()
        conn.rollback()
    out = [dict(r) for r in rows]
    for r in out:
        r["date"] = str(r["date"]) if r["date"] else None
    return out


_wpcal_cache = {"at": 0, "val": None}


def api_wpa_calibration(params):
    """Model holdout calibration + market-vs-model at identical instants (cached 10 min)."""
    from . import wpa
    if _wpcal_cache["val"] and time.time() - _wpcal_cache["at"] < 600:
        return _wpcal_cache["val"]
    with _conn_lock:
        conn = _db()
        out = {"model": wpa.calibration(conn), "vs_market": wpa.compare_market(conn)}
        conn.rollback()
    _wpcal_cache.update(at=time.time(), val=out)
    return out


def api_wpx_results(params):
    path = os.path.join(config.REPO_ROOT, "data", "wpx", "results.json")
    if not os.path.exists(path):
        return {"error": "no results yet: run python3 -m lol_ticker wpx all"}
    with open(path) as f:
        out = json.load(f)
    benchmark = os.path.join(config.REPO_ROOT, "data", "wpx", "method_benchmark.json")
    if "method_benchmark" not in out and os.path.exists(benchmark):
        with open(benchmark) as f:
            out["method_benchmark"] = json.load(f)
    for key, filename in (("blend_benchmark", "blend_benchmark.json"),
                          ("blend_latency45", "blend_latency45.json"),
                          ("historical_blend", "historical_blend.json"),
                          ("live_stack_benchmark", "live_stack_benchmark.json"),
                          ("rolling_origin", "rolling_origin.json"),
                          ("walk_forward_diagnostics", "walk_forward_diagnostics.json"),
                          ("live_stack", "live_stack.json")):
        blend_path = os.path.join(config.REPO_ROOT, "data", "wpx", filename)
        if key not in out and os.path.exists(blend_path):
            with open(blend_path) as f:
                out[key] = json.load(f)
    return out


def api_shadow_results(params):
    from . import shadow
    with _conn_lock:
        conn = _db()
        out = {"status": shadow.status(conn)}
        conn.rollback()
    if os.path.exists(shadow.RESULT_PATH):
        with open(shadow.RESULT_PATH) as f:
            score = json.load(f)
        if score.get("protocol_id") == out["status"]["protocol_id"]:
            out["score"] = score
    return out


def api_live_games(params):
    from . import live
    try:
        return live.live_games()
    except Exception as e:
        return {"error": str(e)}


def api_live_estimate(params):
    from . import live
    gid = (params.get("game_id") or [""])[0] or None
    global _live_games_cache
    ts_g, games = _live_games_cache
    if time.time() - ts_g > 20 or (gid and not any(x["game_id"] == gid for x in games)):
        try:
            games = live.live_games(); _live_games_cache = (time.time(), games)
        except Exception as e:
            if not games:
                return {"error": "schedule lookup failed: %s" % e}
    g = next((x for x in games if x["game_id"] == gid), None) if gid else (games[0] if games else None)
    if not g:
        return {"error": "no in-progress game", "games": games}
    with _conn_lock:
        conn = _db()
        if hasattr(live.team_priors, "_cache"):
            del live.team_priors._cache
        priors = live.team_priors(conn, g["teams"])
        priors["series_diff"] = live.series_prior(g)
        try:
            r = live.estimate(conn, g["game_id"], priors, teams=g["teams"],
                              team_ids=g.get("team_ids"))
        except Exception as e:
            conn.rollback()
            return {"error": "feed/model failed: %s" % e}
        mk = live.market_prices(conn, g["teams"], g.get("number") or 1, deciding=g.get("deciding", False))
        conn.rollback()
    r["game"] = g
    r["priors"] = priors
    r["markets"] = mk
    if r.get("state"):
        try:
            from . import wphist
            r["blend_status"] = _historical_blend_status(wphist.status())
            blend = wphist.predict_live(
                r["p_blue"], r["state"], r["state"].get("blue_champs", []),
                r["state"].get("red_champs", []))
            if blend:
                r["blend"] = blend
        except (OSError, ValueError, KeyError) as e:
            r["blend_error"] = str(e)
    r["ts"] = int(time.time())
    return r


_live_mk_cache = {}  # game_id -> (ts, markets)
_live_games_cache = (0, [])


def _historical_blend_status(status):
    """Small live-API view of the offline deployment decision."""
    meta = status.get("meta") or {}
    paired = ((meta.get("test") or {}).get("paired") or {})
    return {
        "available": bool(status.get("available")),
        "deployed": bool(status.get("deployed")),
        "kind": status.get("kind"),
        "alpha": status.get("alpha"),
        "uses_live_odds": False,
        "holdout_blend_minus_model": paired.get("blend_minus_model"),
        "holdout_ci95": paired.get("ci95"),
    }


def _live_quote_loop():
    """Background: keep exchange quotes fresh (~every 3 s) for every in-progress
    game so requests never wait on the exchanges and never serve stale numbers
    without saying so."""
    from . import live
    global _live_games_cache
    while True:
        try:
            ts, games = _live_games_cache
            if time.time() - ts > 20:
                games = live.live_games(); _live_games_cache = (time.time(), games)
            def _one(g):
                try:
                    mk = live.market_prices(None, g["teams"], g.get("number") or 1, deciding=g.get("deciding", False))
                    _live_mk_cache[g["game_id"]] = (time.time(), mk)
                except Exception as e:
                    logging.getLogger("dashboard").warning("quote refresh failed: %s", e)
            if games:   # several games at once (e.g. two regions live) -> quote them concurrently
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                    list(ex.map(_one, games))
        except Exception as e:
            logging.getLogger("dashboard").warning("live games refresh failed: %s", e)
        time.sleep(3)


threading.Thread(target=_live_quote_loop, daemon=True).start()


def api_live_series(params):
    """New 1 Hz frames since `since` (unix ts) with model P per frame; market
    quotes refreshed at most every 5 s."""
    from . import live
    gid = (params.get("game_id") or [""])[0] or None
    since = int((params.get("since") or ["0"])[0])
    global _live_games_cache
    ts_g, games = _live_games_cache
    if time.time() - ts_g > 20 or (gid and not any(x["game_id"] == gid for x in games)):
        try:
            games = live.live_games(); _live_games_cache = (time.time(), games)
        except Exception as e:
            if not games:
                return {"error": "schedule lookup failed: %s" % e}
    g = next((x for x in games if x["game_id"] == gid), None) if gid else (games[0] if games else None)
    if not g:
        return {"error": "no in-progress game", "games": games}
    with _conn_lock:
        conn = _db()
        if hasattr(live.team_priors, "_cache"):
            del live.team_priors._cache
        priors = live.team_priors(conn, g["teams"])
        priors["series_diff"] = live.series_prior(g)
        try:
            r = live.estimate_series(conn, g["game_id"], priors, since, teams=g["teams"])
            # the feed is the authority on sides: if it disagrees with the schedule's side info, swap
            ids = g.get("team_ids") or []
            if r.get("blue_team_id") and len(ids) == 2 and r["blue_team_id"] == ids[1]:
                g["teams"] = g["teams"][::-1]; g["team_ids"] = ids[::-1]; g["wins"] = (g.get("wins") or [])[::-1]
                _live_mk_cache.pop(g["game_id"], None)
                priors = live.team_priors(conn, g["teams"])
                priors["series_diff"] = live.series_prior(g)
                r = live.estimate_series(conn, g["game_id"], priors, since, teams=g["teams"])
        except Exception as e:
            conn.rollback()
            return {"error": "feed/model failed: %s" % e}
        conn.rollback()
    # quotes come from the background refresher; report their age so the page can flag staleness
    mk_ts, mk = _live_mk_cache.get(g["game_id"], (0, {}))
    r["markets_age_s"] = round(time.time() - mk_ts, 1) if mk_ts else None
    if r.get("frames"):
        try:
            from . import wphist
            latest = r["frames"][-1]
            r["blend_status"] = _historical_blend_status(wphist.status())
            blend = wphist.predict_live(
                latest["p_blue"], latest, r.get("blue_champs", []),
                r.get("red_champs", []))
            if blend:
                r["blend"] = blend
        except (OSError, ValueError, KeyError) as e:
            r["blend_error"] = str(e)
    merged_priors = dict(priors)
    merged_priors.update(r.get("priors_effective") or {})
    r["game"] = g; r["priors"] = merged_priors; r["markets"] = mk; r["ts"] = int(time.time())
    return r


ROUTES = {
    "/api/status": lambda p: api_status(),
    "/api/live/series": api_live_series,
    "/api/live/games": api_live_games,
    "/api/live/estimate": api_live_estimate,
    "/api/wpx/results": api_wpx_results,
    "/api/shadow/results": api_shadow_results,
    "/api/wpa/calibration": api_wpa_calibration,
    "/api/picker/game": api_picker_game,
    "/api/picker/resolve": api_picker_resolve,
    "/api/impact/compare": api_impact_compare,
    "/api/wpa/game": api_wpa_game,
    "/api/wpa/impact": api_wpa_impact,
    "/api/draftfree/teams": api_draftfree_teams,
    "/api/draftfree/champions": api_draftfree_champions,
    "/api/picker/tournaments": api_picker_tournaments,
    "/api/picker/games": api_picker_games,
    "/api/align/games": api_align_games,
    "/api/align/game": api_align_game,
    "/api/align/impact": api_align_impact,
    "/api/draft/simulate": api_draft_simulate,
    "/api/draft/champlist": api_draft_champlist,
    "/api/draft/patches": api_draft_patches,
    "/api/search": api_search,
    "/api/game": api_game,
    "/api/series": api_series,
    "/api/calibration": api_calibration,
    "/api/draft/teams": api_draft_teams,
    "/api/draft/champions": api_draft_champions,
    "/api/draft/pairs": api_draft_pairs,
}


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        # CORS/private-network preflight (Chrome sends this for localhost
        # requests from other origins, e.g. the page opened as a file)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            with open(PAGE_PATH, "rb") as f:
                body = f.read()
            self._send(200, body, "text/html; charset=utf-8")
            return
        route = ROUTES.get(parsed.path)
        if not route:
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        try:
            data = route(urllib.parse.parse_qs(parsed.query))
            self._send(200, json.dumps(data, default=str).encode(),
                       "application/json")
        except Exception as e:
            try:
                with _conn_lock:
                    _db().rollback()
            except Exception:
                pass
            self._send(500, json.dumps({"error": str(e)}).encode(),
                       "application/json")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # localhost-only server; CORS open so the page also works when opened
        # as a file/preview rather than served from this host
        self.send_header("Access-Control-Allow-Origin", "*")
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away (page reload / poll cancelled) - not an error

    def log_message(self, fmt, *args):  # quiet
        pass


def serve(port=8090):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("dashboard: http://127.0.0.1:%d" % port)
    server.serve_forever()
