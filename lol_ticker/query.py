"""Game-centric querying: find a game, pull its full ticker record, export CSVs.

A "game" is a (platform, event_id) pair: a Polymarket event slug like
lol-t1-drx-2026-05-20 or a Kalshi event ticker like KXLOLGAME-26AUG090300LNGIG.
"""
import calendar
import csv
import os
import re
import time

from . import config, db

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def find_games(conn, terms):
    """Search markets by team/date terms; return per-event summaries.

    Non-date terms must each match event_id, title, or outcome (case-insensitive).
    A YYYY-MM-DD term restricts to games starting that UTC day.
    """
    where, params = [], []
    for term in terms:
        if _DATE_RE.match(term):
            day0 = calendar.timegm(time.strptime(term, "%Y-%m-%d"))
            where.append("(game_start_ts BETWEEN %s AND %s)")
            params += [day0, day0 + 86400]
        else:
            where.append("(event_id ILIKE %s OR title ILIKE %s OR outcome ILIKE %s)")
            params += ["%" + term + "%"] * 3
    sql = """
        SELECT platform, event_id, COUNT(*) AS n_markets,
               MIN(game_start_ts) AS game_start_ts, MIN(title) AS sample_title,
               MAX(outage_affected) AS outage_affected
        FROM markets WHERE event_id IS NOT NULL
    """
    if where:
        sql += " AND " + " AND ".join(where)
    sql += " GROUP BY platform, event_id ORDER BY MIN(game_start_ts) DESC NULLS LAST LIMIT 100"
    events = [dict(r) for r in conn.execute(sql, params)]
    for ev in events:
        ev.update(coverage(conn, ev["platform"], ev["event_id"]))
    return events


def coverage(conn, platform, event_id):
    """Row counts of stored ticker data for one event."""
    out = {}
    for name, table in (("snapshots", "book_snapshots"), ("trades", "trades"),
                        ("candles", "candles"), ("price_points", "price_points")):
        out[name] = conn.execute(
            """SELECT COUNT(*) AS c FROM %s t WHERE t.platform = %%s
               AND t.market_id IN (SELECT market_id FROM markets
                   WHERE platform = %%s AND event_id = %%s)""" % table,
            (platform, platform, event_id)).fetchone()["c"]
    return out


EXPORT_QUERIES = {
    "markets.csv": """
        SELECT platform, market_id, event_id, series, condition_id, title,
               outcome, game_start_ts, open_ts, close_ts, status, result,
               outage_affected
        FROM markets WHERE platform = %(p)s AND event_id = %(e)s
        ORDER BY market_id""",
    "book_snapshots.csv": """
        SELECT s.platform, s.market_id,
               (EXTRACT(EPOCH FROM s.ts) * 1000)::bigint AS ts_ms,
               (EXTRACT(EPOCH FROM s.source_ts) * 1000)::bigint AS source_ts_ms,
               s.bids::text AS bids, s.asks::text AS asks
        FROM book_snapshots s
        WHERE s.platform = %(p)s AND s.market_id IN
              (SELECT market_id FROM markets WHERE platform = %(p)s AND event_id = %(e)s)
        ORDER BY s.market_id, s.ts""",
    "trades.csv": """
        SELECT t.platform, t.market_id, t.trade_id,
               EXTRACT(EPOCH FROM t.ts)::bigint AS ts, t.price, t.size, t.side
        FROM trades t
        WHERE t.platform = %(p)s AND t.market_id IN
              (SELECT market_id FROM markets WHERE platform = %(p)s AND event_id = %(e)s)
        ORDER BY t.market_id, t.ts""",
    "candles.csv": """
        SELECT c.platform, c.market_id, c.period_min,
               EXTRACT(EPOCH FROM c.ts)::bigint AS ts, c.open, c.high, c.low,
               c.close, c.volume, c.open_interest
        FROM candles c
        WHERE c.platform = %(p)s AND c.market_id IN
              (SELECT market_id FROM markets WHERE platform = %(p)s AND event_id = %(e)s)
        ORDER BY c.market_id, c.period_min, c.ts""",
    "price_points.csv": """
        SELECT pp.platform, pp.market_id, pp.fidelity,
               EXTRACT(EPOCH FROM pp.ts)::bigint AS ts, pp.price
        FROM price_points pp
        WHERE pp.platform = %(p)s AND pp.market_id IN
              (SELECT market_id FROM markets WHERE platform = %(p)s AND event_id = %(e)s)
        ORDER BY pp.market_id, pp.fidelity, pp.ts""",
}


def export_game(conn, platform, event_id, out_dir):
    """Dump every stored series for one event as CSVs. Returns written paths."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", "%s_%s" % (platform, event_id))
    dest = os.path.join(out_dir, safe)
    os.makedirs(dest, exist_ok=True)
    written = []
    params = {"p": platform, "e": event_id}
    for fname, sql in EXPORT_QUERIES.items():
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            continue
        path = os.path.join(dest, fname)
        with open(path, "w", newline="") as f:
            cols = list(rows[0].keys())
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r[c] for c in cols])
        written.append(path)

    # outages overlapping the game window, so gaps are explainable
    g = conn.execute(
        "SELECT MIN(game_start_ts) AS g FROM markets WHERE platform = %s AND event_id = %s",
        (platform, event_id)).fetchone()["g"]
    if g:
        rows = db.outages_overlapping(conn, platform,
                                      g - config.FAST_BEFORE, g + config.FAST_AFTER)
        if rows:
            path = os.path.join(dest, "outages.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["platform", "kind", "start_ts", "end_ts", "note"])
                for r in rows:
                    w.writerow([r["platform"], r["kind"], r["start_ts"],
                                r["end_ts"], r["note"]])
            written.append(path)
    return written
