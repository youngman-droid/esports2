"""Orchestration: discovery, live L2 recording, and historical backfill."""
import json
import logging
import time

from . import config, db, kalshi, polymarket, util

log = logging.getLogger("collector")

TERMINAL_KALSHI = {"settled", "finalized", "closed"}


def now_s():
    return int(time.time())


# ---------------------------------------------------------------- discovery

def discover(conn, include_closed_pm=False, pm_closed_pages=2):
    """Refresh the market catalog from both platforms.

    Rows are buffered and written in short batches so no write transaction is
    ever held open across network fetches (other processes share this db).
    """
    t = now_s()
    n_pm = n_k = 0
    buf = []

    def flush():
        for row in buf:
            db.upsert_market(conn, row, t)
        conn.commit()
        del buf[:]

    def take(rows):
        nonlocal n_pm, n_k
        for row in rows:
            if row["platform"] == "polymarket":
                n_pm += 1
            else:
                n_k += 1
            buf.append(row)
            if len(buf) >= 500:
                flush()

    # each source is independent: one platform's network failure must not
    # abort the others (rows flush every 500, so partial progress persists)
    sources = [
        ("pm-open", lambda: polymarket.discover(closed=False)),
        # closed events: first pages (newest first) keep statuses current; a
        # full keyset-paginated crawl happens on `discover --all`
        ("pm-closed", lambda: polymarket.discover(
            closed=True, deep=include_closed_pm, max_pages=pm_closed_pages)),
        ("kalshi", lambda: kalshi.discover(
            statuses=("open", "unopened", "closed", "settled"))),
    ]
    failed = []
    for name, source in sources:
        try:
            take(source())
            flush()
        except Exception:
            log.exception("discover source %s failed; continuing", name)
            failed.append(name)
            flush()
    # scheduled maintenance windows -> outages table
    try:
        for start, end, raw in kalshi.fetch_maintenance_windows():
            db.open_outage(conn, "kalshi", "maintenance", start, raw)
            if end:
                conn.execute(
                    """UPDATE outages SET end_ts = %s WHERE platform='kalshi'
                       AND kind='maintenance' AND start_ts = %s""", (end, start))
    except Exception:
        log.exception("failed to fetch kalshi maintenance schedule")
    conn.commit()
    log.info("discover: upserted %d polymarket, %d kalshi market rows%s",
             n_pm, n_k, (" (FAILED: %s)" % ", ".join(failed)) if failed else "")
    return n_pm, n_k


# ---------------------------------------------------------------- backfill

def backfill(conn, limit=None, min_volume=0.0):
    """Capture trade + price history for terminal markets not yet backfilled."""
    rows = conn.execute(
        """SELECT * FROM markets WHERE backfilled = 0 AND (
               (platform = 'polymarket' AND status = 'closed') OR
               (platform = 'kalshi' AND status IN ('settled', 'finalized'))
           ) ORDER BY close_ts DESC""").fetchall()
    if limit:
        rows = rows[:limit]
    done = 0
    for m in rows:
        try:
            _backfill_one(conn, m)
            conn.commit()
            done += 1
        except Exception:
            log.exception("backfill failed for %s/%s", m["platform"], m["market_id"])
            conn.rollback()  # clear any aborted transaction state
    log.info("backfill: completed %d of %d pending markets", done, len(rows))
    return done


LONG_LIVED = 21 * 86400  # season/tournament markets: coarser price backfill


def _backfill_one(conn, m):
    platform, mid = m["platform"], m["market_id"]
    start = m["open_ts"] or (m["game_start_ts"] or now_s()) - 7 * 86400
    # close_ts can be a future resolution deadline; history only exists up to now
    end = min(m["close_ts"] or now_s(), now_s())
    start = min(start, end)
    life = max(0, end - start)
    if platform == "polymarket":
        # trades are the full-life tick record and are per condition id; pull
        # once per condition (both outcome tokens arrive together)
        if m["condition_id"]:
            already = conn.execute(
                """SELECT COUNT(*) AS c FROM markets WHERE platform='polymarket'
                   AND condition_id = %s AND backfilled = 1""",
                (m["condition_id"],)).fetchone()["c"]
            if not already:
                db.insert_trades(conn, polymarket.fetch_trades(m["condition_id"]))
            # trading may have stopped long before close_ts (early resolution);
            # anchor the price window to the last real trade
            last = conn.execute(
                """SELECT EXTRACT(EPOCH FROM MAX(ts))::bigint AS t FROM trades
                   WHERE platform='polymarket'
                   AND market_id IN (SELECT market_id FROM markets
                       WHERE platform='polymarket' AND condition_id = %s)""",
                (m["condition_id"],)).fetchone()["t"]
            if last:
                end = min(end, last + 86400)
        # price series: whole life at 10-min for match markets; long-lived
        # season markets only get the final weeks (trades cover the rest)
        coarse_start = start if life <= LONG_LIVED else max(start, end - LONG_LIVED)
        pts = polymarket.fetch_price_history(mid, coarse_start, end, fidelity=10)
        db.insert_price_points(conn, platform, mid, 10, pts)
        # 1-min fidelity around the game window when known
        if m["game_start_ts"]:
            g = m["game_start_ts"]
            fine = polymarket.fetch_price_history(
                mid, max(start, g - 3600), min(end, g + config.FAST_AFTER), fidelity=1)
            db.insert_price_points(conn, platform, mid, 1, fine)
    else:
        db.insert_trades(conn, kalshi.fetch_trades(mid))
        last = conn.execute(
            """SELECT EXTRACT(EPOCH FROM MAX(ts))::bigint AS t FROM trades
               WHERE platform='kalshi' AND market_id = %s""",
            (mid,)).fetchone()["t"]
        if last:
            end = min(end, last + 86400)
        if life <= LONG_LIVED:
            candles = kalshi.fetch_candles(m["series"], mid, start, end, period_min=1)
            db.insert_candles(conn, "kalshi", mid, 1, candles)
        else:
            db.insert_candles(conn, "kalshi", mid, 60,
                              kalshi.fetch_candles(m["series"], mid, start, end, period_min=60))
            db.insert_candles(conn, "kalshi", mid, 1,
                              kalshi.fetch_candles(m["series"], mid, max(start, end - 3 * 86400), end, period_min=1))
    # flag markets whose game window overlapped an exchange outage
    if m["game_start_ts"]:
        overlaps = db.outages_overlapping(
            conn, platform, m["game_start_ts"] - config.FAST_BEFORE,
            m["game_start_ts"] + config.FAST_AFTER)
        if overlaps:
            db.set_market_fields(conn, platform, mid, outage_affected=1)
    db.set_market_fields(conn, platform, mid, backfilled=1)
    log.info("backfilled %s/%s", platform, mid)


# ---------------------------------------------------------------- recording

def _watchlist(conn):
    """Split open markets into fast (near game time) and slow tiers."""
    t = now_s()
    rows = conn.execute(
        """SELECT platform, market_id, condition_id, game_start_ts, series,
                  last_trade_ts
           FROM markets
           WHERE (platform = 'polymarket' AND status = 'open')
              OR (platform = 'kalshi' AND status IN ('open', 'active'))"""
    ).fetchall()
    fast, slow = [], []
    for r in rows:
        g = r["game_start_ts"]
        if g and (g - config.FAST_BEFORE) <= t <= (g + config.FAST_AFTER):
            fast.append(r)
        else:
            slow.append(r)
    return fast, slow


def _poll_books(conn, markets, last_hashes):
    """Fetch current books for the given markets; store only changed books."""
    t_ms = int(time.time() * 1000)
    stored = 0
    pm_tokens = [m["market_id"] for m in markets if m["platform"] == "polymarket"]
    books = [("polymarket", b) for b in (polymarket.fetch_books(pm_tokens) if pm_tokens else [])]
    for m in markets:
        if m["platform"] == "kalshi":
            book = kalshi.fetch_book(m["market_id"])
            if book:
                books.append(("kalshi", book))
    for platform, b in books:
        key = (platform, b["market_id"])
        h = util.book_hash(b["bids"], b["asks"])
        if last_hashes.get(key) == h:
            continue
        db.insert_snapshot(conn, platform, b["market_id"], t_ms,
                           b.get("source_ts_ms"), b["bids"], b["asks"], h)
        last_hashes[key] = h
        stored += 1
    conn.commit()
    return stored


def _poll_trades(conn, markets):
    """Incremental trades for fast-tier markets."""
    new = 0
    seen_conditions = set()
    for m in markets:
        cutoff = max(0, (m["last_trade_ts"] or 0) - 60)
        if m["platform"] == "polymarket":
            c = m["condition_id"]
            if not c or c in seen_conditions:
                continue
            seen_conditions.add(c)
            rows = polymarket.fetch_trades(c, after_ts=cutoff, max_pages=4)
        else:
            rows = kalshi.fetch_trades(m["market_id"], min_ts=cutoff, max_pages=4)
        if rows:
            new += max(0, db.insert_trades(conn, rows))
            newest = max(r[3] for r in rows)
            db.set_market_fields(conn, m["platform"], m["market_id"],
                                 last_trade_ts=newest)
    conn.commit()
    return new


def _check_kalshi_status(conn, fast_markets):
    """Track Kalshi trading halts; flag fast-tier markets seen during a halt.

    Returns True while trading is halted.
    """
    t = now_s()
    try:
        active, raw = kalshi.fetch_exchange_status()
    except Exception as e:
        log.warning("kalshi exchange status check failed: %s", e)
        return db.ongoing_outage(conn, "kalshi") is not None
    ongoing = db.ongoing_outage(conn, "kalshi")
    if not active and ongoing is None:
        log.warning("kalshi trading HALTED — opening outage record")
        db.open_outage(conn, "kalshi", "halt", t,
                       json.dumps(raw, separators=(",", ":")))
    elif active and ongoing is not None:
        log.info("kalshi trading resumed after %ds halt", t - ongoing)
        db.close_outage(conn, "kalshi", t)
    if not active:
        # any market being recorded near game time during the halt is affected
        for m in fast_markets:
            if m["platform"] == "kalshi":
                db.set_market_fields(conn, "kalshi", m["market_id"], outage_affected=1)
    conn.commit()
    return not active


def record(conn, once=False):
    """Main loop: discover -> poll books (tiered) -> incremental trades -> backfill."""
    last_hashes = db.last_book_hashes(conn)
    last_discover = last_slow = last_trades = last_backfill = last_status = 0.0
    log.info("record loop starting (fast=%ds slow=%ds)", config.FAST_POLL, config.SLOW_POLL)
    while True:
        loop_start = time.monotonic()
        t = time.time()
        try:
            if t - last_discover >= config.DISCOVER_EVERY:
                discover(conn)
                last_discover = t
            fast, slow = _watchlist(conn)
            if t - last_status >= config.STATUS_EVERY:
                _check_kalshi_status(conn, fast)
                last_status = t
            n = _poll_books(conn, fast, last_hashes)
            if n:
                log.info("stored %d changed books (%d fast-tier markets)", n, len(fast))
            if t - last_slow >= config.SLOW_POLL:
                n = _poll_books(conn, slow, last_hashes)
                log.info("slow tier: %d/%d books changed", n, len(slow))
                last_slow = t
            if fast and t - last_trades >= config.TRADES_EVERY:
                n = _poll_trades(conn, fast)
                if n:
                    log.info("stored %d new trades", n)
                last_trades = t
            if t - last_backfill >= config.DISCOVER_EVERY:
                backfill(conn, limit=20)
                last_backfill = t
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception("record loop iteration failed; continuing")
            try:
                conn.rollback()  # clear any aborted transaction state
            except Exception:
                pass
        if once:
            return
        elapsed = time.monotonic() - loop_start
        time.sleep(max(0.5, config.FAST_POLL - elapsed))


# ---------------------------------------------------------------- status

def status(conn):
    out = []
    for platform in ("polymarket", "kalshi"):
        r = conn.execute(
            """SELECT COUNT(*) AS n,
                      SUM(CASE WHEN status IN ('open','active') THEN 1 ELSE 0 END) AS open_n,
                      SUM(backfilled) AS bf
               FROM markets WHERE platform = %s""", (platform,)).fetchone()
        s = conn.execute(
            """SELECT COUNT(*) AS n,
                      (EXTRACT(EPOCH FROM MAX(ts)) * 1000)::bigint AS latest
               FROM book_snapshots WHERE platform = %s""", (platform,)).fetchone()
        tr = conn.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE platform = %s",
            (platform,)).fetchone()
        out.append((platform, r["n"], r["open_n"], r["bf"], s["n"], s["latest"], tr["n"]))
    fast, slow = _watchlist(conn)
    return out, fast, slow
