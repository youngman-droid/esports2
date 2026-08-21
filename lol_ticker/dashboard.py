"""Tiny local dashboard: search games, chart historical odds, Kalshi status.

Serves on 127.0.0.1 only.  Endpoints:
  /                    the page
  /api/status          Kalshi trading status + recent outages
  /api/search?q=...    games matching team/date terms
  /api/game?platform=&event_id=      market list + outages for one event
  /api/series?platform=&market_id=   odds series for one market
  /api/calibration?horizon=&filter=   reliability bins + Brier per platform
"""
import json
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
    """Live Kalshi trading status, falling back to the outages table."""
    out = {"kalshi_trading_active": None, "halted_since": None, "recent_outages": []}
    with _conn_lock:
        conn = _db()
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


ROUTES = {
    "/api/status": lambda p: api_status(),
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
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quiet
        pass


def serve(port=8090):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("dashboard: http://127.0.0.1:%d" % port)
    server.serve_forever()
