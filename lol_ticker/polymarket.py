"""Polymarket: discovery via Gamma API, books via CLOB, trades via data-api.

market_id here is the CLOB token id (one per outcome).  Trades are fetched per
conditionId and attributed to whichever token they executed on.
"""
import json
import logging

from . import config, util
from .http import get_json, post_json, HttpError

log = logging.getLogger("pm")

PLATFORM = "polymarket"


def _parse_json_field(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return []
    return v or []


MAX_OFFSET = 2000  # gamma rejects deeper offsets

# NOTE: gamma's /events/keyset endpoint is unusable for tag-filtered crawls:
# passing the cursor together with filters wedges it on the same page, and
# passing the cursor alone silently drops the tag filter.  The deep crawl
# instead slices /events into endDate month-windows, each far below the
# 2000-row offset cap.
EARLIEST_WINDOW = "2023-01-01"


def discover(closed=False, max_pages=100, deep=False):
    """Yield normalized market rows for LoL events (one row per token).

    Shallow refresh: offset pagination, newest endDate first (bounded by the
    API's ~2000 offset cap).  deep=True: full historical crawl via endDate
    month-windows.
    """
    if deep:
        for lo, hi in _month_windows():
            for row in _paged_events(closed, extra={
                    "end_date_min": lo, "end_date_max": hi}):
                yield row
    else:
        for row in _paged_events(closed, max_pages=max_pages):
            yield row


def _paged_events(closed, extra=None, max_pages=None):
    offset = 0
    pages = 0
    while max_pages is None or pages < max_pages:
        if offset >= MAX_OFFSET:
            log.warning("offset cap reached (params=%s); remaining events skipped",
                        extra)
            return
        params = {
            "tag_slug": config.PM_TAG_SLUG,
            "closed": "true" if closed else "false",
            "limit": 100,
            "offset": offset,
            "order": "endDate",
            "ascending": "false",
        }
        params.update(extra or {})
        events = get_json(config.PM_GAMMA + "/events", params)
        if not events:
            return
        for row in _normalize_events(events):
            yield row
        if len(events) < 100:
            return
        offset += 100
        pages += 1


def _month_windows(start=EARLIEST_WINDOW):
    """Yield (lo, hi) ISO endDate windows, month-sized, up to now + 2 years
    (closed markets can carry far-future resolution deadlines)."""
    from datetime import datetime, timedelta, timezone
    cur = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    horizon = datetime.now(timezone.utc) + timedelta(days=730)
    while cur < horizon:
        nxt = (cur + timedelta(days=32)).replace(day=1)
        yield (cur.strftime("%Y-%m-%dT00:00:00Z"),
               nxt.strftime("%Y-%m-%dT00:00:00Z"))
        cur = nxt


def _normalize_events(events):
    for ev in events:
        for m in ev.get("markets", []):
            tokens = _parse_json_field(m.get("clobTokenIds"))
            outcomes = _parse_json_field(m.get("outcomes"))
            if not tokens or not m.get("enableOrderBook", True):
                continue
            game_start = (util.parse_ts(m.get("gameStartTime"))
                          or util.parse_ts(ev.get("startTime"))
                          or util.parse_ts(ev.get("eventDate")))
            for i, token in enumerate(tokens):
                yield {
                    "platform": PLATFORM,
                    "market_id": str(token),
                    "event_id": ev.get("slug"),
                    "series": config.PM_TAG_SLUG,
                    "condition_id": m.get("conditionId"),
                    "title": m.get("question"),
                    "outcome": outcomes[i] if i < len(outcomes) else str(i),
                    "game_start_ts": game_start,
                    "open_ts": util.parse_ts(m.get("startDate")),
                    "close_ts": util.parse_ts(m.get("endDate")),
                    "status": "closed" if m.get("closed") else "open",
                    "result": m.get("umaResolutionStatus"),
                    # keep raw compact: market json without nested event blob
                    "raw": {k: v for k, v in m.items() if k != "events"},
                }


def fetch_books(token_ids):
    """Batch-fetch current L2 books.  Returns list of dicts:
    {market_id, source_ts_ms, bids, asks} with bids desc / asks asc, YES side."""
    out = []
    for i in range(0, len(token_ids), 50):
        chunk = token_ids[i:i + 50]
        try:
            books = post_json(config.PM_CLOB + "/books",
                              [{"token_id": t} for t in chunk])
        except HttpError as e:
            log.warning("books batch failed: %s", e)
            continue
        for b in books or []:
            bids = sorted(
                ([float(x["price"]), float(x["size"])] for x in b.get("bids", [])),
                key=lambda r: -r[0])
            asks = sorted(
                ([float(x["price"]), float(x["size"])] for x in b.get("asks", [])),
                key=lambda r: r[0])
            out.append({
                "market_id": str(b.get("asset_id")),
                "source_ts_ms": int(b["timestamp"]) if b.get("timestamp") else None,
                "bids": bids,
                "asks": asks,
            })
    return out


PRICE_CHUNK = {1: 12 * 3600, 10: 5 * 86400}  # max request span by fidelity (API caps span)


def fetch_price_history(token_id, start_ts, end_ts, fidelity=10):
    """Return [{'t': ts, 'p': price}, ...] over [start_ts, end_ts], chunked.

    The API rejects long startTs/endTs spans, and the interval=max shorthand
    returns nothing for closed markets, so explicit chunked ranges it is.
    """
    step = PRICE_CHUNK.get(fidelity, 86400)
    out = []
    t0 = start_ts
    while t0 < end_ts:
        t1 = min(t0 + step, end_ts)
        try:
            data = get_json(config.PM_CLOB + "/prices-history", {
                "market": token_id, "startTs": t0, "endTs": t1,
                "fidelity": fidelity,
            })
        except HttpError as e:
            log.warning("prices-history %s failed: %s", token_id, e)
            break
        out.extend(data.get("history", []))
        t0 = t1
    return out


def fetch_trades(condition_id, after_ts=0, max_pages=40):
    """Fetch trades for a condition, newest first, until older than after_ts.

    Returns rows: (platform, trade_id, market_id, ts, price, size, side, raw).
    """
    rows = []
    offset = 0
    for _ in range(max_pages):
        try:
            page = get_json(config.PM_DATA + "/trades", {
                "market": condition_id, "limit": 500, "offset": offset,
            })
        except HttpError as e:
            log.warning("trades %s failed: %s", condition_id, e)
            break
        if not page:
            break
        done = False
        for t in page:
            ts = int(t.get("timestamp") or 0)
            if ts and ts < after_ts:
                done = True
                continue
            trade_id = util.trade_hash(
                t.get("transactionHash") or "",
                t.get("proxyWallet"), t.get("asset"), ts,
                t.get("price"), t.get("size"), t.get("side"))
            rows.append((
                PLATFORM, trade_id, str(t.get("asset")), ts,
                float(t.get("price") or 0), float(t.get("size") or 0),
                (t.get("side") or "").lower(),
                json.dumps(t, separators=(",", ":")),
            ))
        if done or len(page) < 500:
            break
        offset += 500
    return rows
