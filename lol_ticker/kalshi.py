"""Kalshi: discovery, orderbooks, candlesticks, trades (public endpoints).

market_id is the Kalshi market ticker.  Books are normalized to YES terms:
Kalshi returns resting YES bids and NO bids; a NO bid at p is a YES ask at 1-p.
"""
import json
import logging
import re
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    ET = None

from . import config, util
from .http import get_json, HttpError

log = logging.getLogger("kalshi")

PLATFORM = "kalshi"

CANDLE_MAX_PERIODS = 4900  # API caps ~5000 periods per request

# event tickers embed the scheduled game start in US/Eastern:
# KXLOLGAME-26AUG090300LNGIG -> 2026 Aug 09 03:00 ET
_TICKER_DT = re.compile(r"-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(\d{2})(\d{2})")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def ticker_start_ts(ticker):
    """Scheduled game start (unix seconds) parsed from an event/market ticker."""
    m = _TICKER_DT.search(ticker or "")
    if not m or ET is None:
        return None
    yy, mon, dd, hh, mi = (m.group(1), _MONTHS[m.group(2)], m.group(3),
                           m.group(4), m.group(5))
    try:
        dt = datetime(2000 + int(yy), mon, int(dd), int(hh), int(mi), tzinfo=ET)
    except ValueError:
        return None
    return int(dt.timestamp())


def _num(v):
    """Kalshi values arrive as dollar strings ('0.2400'), cents ints, or fp strings."""
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    if isinstance(v, int):
        return v / 100.0  # cents -> dollars
    return float(v)


def discover(statuses=("open", "unopened"), series_list=None):
    """Yield normalized market rows for all configured LoL series."""
    for series in (series_list or config.KALSHI_SERIES):
        for status in statuses:
            cursor = None
            while True:
                params = {"series_ticker": series, "status": status, "limit": 200}
                if cursor:
                    params["cursor"] = cursor
                try:
                    data = get_json(config.KALSHI + "/markets", params)
                except HttpError as e:
                    log.warning("markets %s/%s failed: %s", series, status, e)
                    break
                for m in data.get("markets", []):
                    yield normalize_market(m, series)
                cursor = data.get("cursor")
                if not cursor or not data.get("markets"):
                    break


def normalize_market(m, series):
    # occurrence_datetime is the expected match END; the ticker holds the start
    game_start = (ticker_start_ts(m.get("event_ticker") or m["ticker"])
                  or util.parse_ts(m.get("occurrence_datetime"))
                  or util.parse_ts(m.get("expected_expiration_time")))
    return {
        "platform": PLATFORM,
        "market_id": m["ticker"],
        "event_id": m.get("event_ticker"),
        "series": series,
        "condition_id": None,
        "title": m.get("title"),
        "outcome": m.get("yes_sub_title"),
        "game_start_ts": game_start,
        "open_ts": util.parse_ts(m.get("open_time")),
        "close_ts": util.parse_ts(m.get("close_time")),
        "status": m.get("status"),
        "result": m.get("result"),
        "raw": m,
    }


def fetch_market(ticker):
    data = get_json(config.KALSHI + "/markets/" + ticker)
    return data.get("market")


def fetch_book(ticker, depth=100):
    """Return {market_id, source_ts_ms, bids, asks} in YES terms, or None."""
    try:
        data = get_json(config.KALSHI + "/markets/%s/orderbook" % ticker,
                        {"depth": depth})
    except HttpError as e:
        log.warning("orderbook %s failed: %s", ticker, e)
        return None
    ob = data.get("orderbook_fp") or data.get("orderbook") or {}
    yes_levels = ob.get("yes_dollars") or ob.get("yes") or []
    no_levels = ob.get("no_dollars") or ob.get("no") or []
    is_fp = bool(ob.get("yes_dollars") is not None or ob.get("no_dollars") is not None)

    def lvl(price, size):
        p = float(price) if is_fp else float(price) / 100.0
        s = float(size)
        return [p, s]

    bids = sorted((lvl(p, s) for p, s in yes_levels), key=lambda r: -r[0])
    asks = sorted(([round(1.0 - lvl(p, s)[0], 4), float(s)] for p, s in no_levels),
                  key=lambda r: r[0])
    return {"market_id": ticker, "source_ts_ms": None, "bids": bids, "asks": asks}


def fetch_candles(series, ticker, start_ts, end_ts, period_min=1):
    """Return normalized 1-min candles over [start_ts, end_ts], chunked."""
    out = []
    step = CANDLE_MAX_PERIODS * period_min * 60
    t0 = start_ts
    while t0 < end_ts:
        t1 = min(t0 + step, end_ts)
        try:
            data = get_json(
                config.KALSHI + "/series/%s/markets/%s/candlesticks" % (series, ticker),
                {"start_ts": t0, "end_ts": t1, "period_interval": period_min})
        except HttpError as e:
            log.warning("candles %s failed: %s", ticker, e)
            break
        for c in data.get("candlesticks", []):
            price = c.get("price") or {}
            # fall back to yes_bid OHLC when no trades printed in the period
            fb = c.get("yes_bid") or {}

            def field(name):
                for src in (price, fb):
                    v = src.get(name + "_dollars", src.get(name))
                    if v is not None:
                        return _num(v)
                return None

            out.append({
                "ts": int(c.get("end_period_ts") or 0),
                "open": field("open"),
                "high": field("high"),
                "low": field("low"),
                "close": field("close"),
                "volume": _num(c.get("volume_fp")) if c.get("volume_fp") is not None else float(c.get("volume") or 0),
                "open_interest": _num(c.get("open_interest_fp")) if c.get("open_interest_fp") is not None else float(c.get("open_interest") or 0),
                "raw": json.dumps(c, separators=(",", ":")),
            })
        t0 = t1
    return out


def fetch_exchange_status():
    """Return (trading_active, raw) for exchange index 0 (where LoL markets live)."""
    data = get_json(config.KALSHI + "/exchange/status")
    active = bool(data.get("exchange_active")) and bool(data.get("trading_active"))
    for idx in data.get("exchange_index_statuses", []):
        if idx.get("exchange_index") == 0:
            active = active and bool(idx.get("trading_active")) and bool(idx.get("exchange_active"))
    return active, data


def fetch_maintenance_windows():
    """Return scheduled maintenance windows: [(start_ts, end_ts, raw), ...]."""
    data = get_json(config.KALSHI + "/exchange/schedule")
    out = []
    for w in (data.get("schedule") or {}).get("maintenance_windows", []):
        start = util.parse_ts(w.get("start_datetime") or w.get("start_time"))
        end = util.parse_ts(w.get("end_datetime") or w.get("end_time"))
        if start:
            out.append((start, end, json.dumps(w, separators=(",", ":"))))
    return out


def fetch_trades(ticker, min_ts=0, max_pages=100):
    """Return trade rows: (platform, trade_id, market_id, ts, price, size, side, raw)."""
    rows = []
    cursor = None
    for _ in range(max_pages):
        params = {"ticker": ticker, "limit": 1000}
        if min_ts:
            params["min_ts"] = min_ts
        if cursor:
            params["cursor"] = cursor
        try:
            data = get_json(config.KALSHI + "/markets/trades", params)
        except HttpError as e:
            log.warning("trades %s failed: %s", ticker, e)
            break
        trades = data.get("trades", [])
        for t in trades:
            ts = util.parse_ts(t.get("created_time")) or 0
            price = _num(t.get("yes_price_dollars"))
            if price is None:
                price = _num(t.get("yes_price"))
            size = _num(t.get("count_fp"))
            if size is None:
                size = float(t.get("count") or 0)
            rows.append((
                PLATFORM, t.get("trade_id"), ticker, ts, price, size,
                t.get("taker_side") or "",
                json.dumps(t, separators=(",", ":")),
            ))
        cursor = data.get("cursor")
        if not cursor or not trades:
            break
    return rows
