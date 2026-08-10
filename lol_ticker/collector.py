"""Orchestration: discovery, live L2 recording, and historical backfill."""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, db, http, kalshi, polymarket, util

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

LONG_LIVED = 21 * 86400  # season/tournament markets: coarser price backfill


def backfill(conn, limit=None, workers=8):
    """Capture trade + price history for terminal markets not yet backfilled.

    Workers do the network fetches concurrently (per-host rate limiting still
    applies globally); only this thread touches the database.  Polymarket
    markets are grouped by condition so the shared trade tape is fetched once,
    and zero-volume markets skip their history requests entirely.
    """
    rows = conn.execute(
        """SELECT platform, market_id, event_id, series, condition_id,
                  game_start_ts, open_ts, close_ts,
                  raw->>'volumeNum' AS vol_a, raw->>'volume' AS vol_b,
                  raw->>'volume_fp' AS vol_c
           FROM markets WHERE backfilled = 0 AND (
               (platform = 'polymarket' AND status = 'closed') OR
               (platform = 'kalshi' AND status IN ('settled', 'finalized'))
           ) ORDER BY close_ts DESC""").fetchall()
    if limit:
        rows = rows[:limit]
    if not rows:
        return 0
    k_tasks = []  # each: [market dict]
    pm_groups = {}
    for m in rows:
        m = dict(m)
        if m["platform"] == "kalshi":
            k_tasks.append([m])
        else:
            pm_groups.setdefault(m["condition_id"] or m["market_id"], []).append(m)
    p_tasks = list(pm_groups.values())
    log.info("backfill: %d markets pending (%d kalshi + %d polymarket tasks, "
             "%d workers)", len(rows), len(k_tasks), len(p_tasks), workers)

    done = failed = 0
    t0 = time.time()
    last_log = t0
    # separate pools so Kalshi's slower rate budget can't starve Polymarket:
    # each platform saturates its own request budget concurrently
    k_pool = ThreadPoolExecutor(max_workers=3)
    p_pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {k_pool.submit(_fetch_history, "kalshi", g): g for g in k_tasks}
        futures.update(
            {p_pool.submit(_fetch_history, "polymarket", g): g for g in p_tasks})
        for fut in as_completed(futures):
            group = futures[fut]
            try:
                _store_history(conn, fut.result())
                conn.commit()
                done += len(group)
            except Exception:
                conn.rollback()
                failed += len(group)
                log.exception("backfill failed for %s/%s",
                              group[0]["platform"], group[0]["market_id"])
            if time.time() - last_log >= 30:
                last_log = time.time()
                n = done + failed
                rate = n / max(1e-9, last_log - t0)
                log.info("backfill: %d/%d markets (%.1f/s, ~%dmin left)",
                         n, len(rows), rate, (len(rows) - n) / max(rate, 1e-9) / 60)
        k_pool.shutdown()
        p_pool.shutdown()
    except (KeyboardInterrupt, SystemExit):
        # ^C: drop queued tasks and unblock in-flight waits so exit is prompt;
        # everything not yet stored stays pending and resumes next run
        log.info("backfill interrupted at %d/%d — cancelling queued tasks",
                 done + failed, len(rows))
        http.SHUTDOWN.set()
        k_pool.shutdown(wait=False, cancel_futures=True)
        p_pool.shutdown(wait=False, cancel_futures=True)
        conn.rollback()
        raise
    log.info("backfill: completed %d of %d pending markets (%d failed)",
             done, len(rows), failed)
    return done


def _market_volume(m):
    """Reported lifetime volume, extracted from the raw exchange json in SQL."""
    for key in ("vol_a", "vol_b", "vol_c"):
        v = m.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None  # unknown -> don't skip


def _fetch_history(kind, group):
    """Network-only: fetch a task group's full history.  Runs in a worker."""
    payload = {"trades": [], "points": [], "candles": [], "markets": group}
    if kind == "polymarket":
        m0 = group[0]
        vol = sum(_market_volume(m) or 1 for m in group)
        if m0["condition_id"] and vol > 0:
            payload["trades"] = polymarket.fetch_trades(m0["condition_id"])
        by_token = {}
        for r in payload["trades"]:
            by_token.setdefault(r[2], []).append(r[3])
        for m in group:
            token_ts = by_token.get(m["market_id"])
            if not token_ts:
                continue  # price history mirrors trades: no trades, no series
            start, end = _history_window(m, last_trade=max(token_ts))
            life = max(0, end - start)
            coarse_start = start if life <= LONG_LIVED else max(start, end - LONG_LIVED)
            payload["points"].append(
                (m["market_id"], 10,
                 polymarket.fetch_price_history(m["market_id"], coarse_start, end, 10)))
            if m["game_start_ts"]:
                g = m["game_start_ts"]
                fine = polymarket.fetch_price_history(
                    m["market_id"], max(start, g - 3600),
                    min(end, g + config.FAST_AFTER), 1)
                payload["points"].append((m["market_id"], 1, fine))
    else:
        m = group[0]
        vol = _market_volume(m)
        if vol == 0.0:
            return payload  # nothing ever traded; candles would be empty too
        payload["trades"] = kalshi.fetch_trades(m["market_id"])
        last = max((r[3] for r in payload["trades"]), default=None)
        start, end = _history_window(m, last_trade=last)
        life = max(0, end - start)
        if life <= LONG_LIVED:
            payload["candles"].append(
                (m["market_id"], 1,
                 kalshi.fetch_candles(m["series"], m["market_id"], start, end, 1)))
        else:
            payload["candles"].append(
                (m["market_id"], 60,
                 kalshi.fetch_candles(m["series"], m["market_id"], start, end, 60)))
            payload["candles"].append(
                (m["market_id"], 1,
                 kalshi.fetch_candles(m["series"], m["market_id"],
                                      max(start, end - 3 * 86400), end, 1)))
    return payload


def _history_window(m, last_trade=None):
    """[start, end] bounds for history fetches: clamp future resolution
    deadlines to now, and anchor the end to the last real trade if known."""
    start = m["open_ts"] or (m["game_start_ts"] or now_s()) - 7 * 86400
    end = min(m["close_ts"] or now_s(), now_s())
    if last_trade:
        end = min(end, last_trade + 86400)
    return min(start, end), end


def _store_history(conn, payload):
    """DB-only: persist a fetched payload.  Runs on the main thread."""
    db.insert_trades(conn, payload["trades"])
    platform = payload["markets"][0]["platform"]
    for market_id, fidelity, pts in payload["points"]:
        db.insert_price_points(conn, platform, market_id, fidelity, pts)
    for market_id, period, rows in payload["candles"]:
        db.insert_candles(conn, platform, market_id, period, rows)
    for m in payload["markets"]:
        if m["game_start_ts"]:
            overlaps = db.outages_overlapping(
                conn, platform, m["game_start_ts"] - config.FAST_BEFORE,
                m["game_start_ts"] + config.FAST_AFTER)
            if overlaps:
                db.set_market_fields(conn, platform, m["market_id"], outage_affected=1)
        db.set_market_fields(conn, platform, m["market_id"], backfilled=1)


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
        try:
            if m["platform"] == "polymarket":
                c = m["condition_id"]
                if not c or c in seen_conditions:
                    continue
                seen_conditions.add(c)
                rows = polymarket.fetch_trades(c, after_ts=cutoff, max_pages=4)
            else:
                rows = kalshi.fetch_trades(m["market_id"], min_ts=cutoff, max_pages=4)
        except Exception as e:
            log.warning("incremental trades failed for %s/%s: %s",
                        m["platform"], m["market_id"], e)
            continue
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
                backfill(conn, limit=20, workers=4)
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
