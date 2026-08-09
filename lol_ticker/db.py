"""PostgreSQL + TimescaleDB storage layer.

Conventions:
  - platform is 'polymarket' or 'kalshi'
  - market_id: Polymarket = CLOB token id (one row per outcome token);
    Kalshi = market ticker (e.g. KXLOLGAME-26AUG090300LNGIG-LNG)
  - all prices are YES-side probabilities in [0, 1]
  - the Python API speaks unix epoch seconds (snapshots: milliseconds);
    columns are TIMESTAMPTZ and converted at this boundary
  - book_snapshots / trades / price_points / candles are hypertables
"""
import json

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    platform        TEXT NOT NULL,
    market_id       TEXT NOT NULL,
    event_id        TEXT,           -- pm: event slug; kalshi: event_ticker
    series          TEXT,           -- kalshi series ticker; pm: tag slug
    condition_id    TEXT,           -- pm conditionId (shared by both outcome tokens)
    title           TEXT,
    outcome         TEXT,           -- pm outcome label; kalshi yes_sub_title
    game_start_ts   BIGINT,
    open_ts         BIGINT,
    close_ts        BIGINT,
    status          TEXT,           -- open | closed | settled/finalized ...
    result          TEXT,
    last_trade_ts   BIGINT NOT NULL DEFAULT 0,  -- newest trade already stored
    backfilled      SMALLINT NOT NULL DEFAULT 0,
    outage_affected SMALLINT NOT NULL DEFAULT 0,
    first_seen_ts   BIGINT,
    last_seen_ts    BIGINT,
    raw             JSONB,
    PRIMARY KEY (platform, market_id)
);
CREATE INDEX IF NOT EXISTS idx_markets_event ON markets (platform, event_id);
CREATE INDEX IF NOT EXISTS idx_markets_backfill
    ON markets (backfilled, close_ts DESC);

CREATE TABLE IF NOT EXISTS book_snapshots (
    platform    TEXT NOT NULL,
    market_id   TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,   -- capture time
    source_ts   TIMESTAMPTZ,            -- exchange-reported book timestamp
    bids        JSONB NOT NULL,         -- [[price, size], ...] best first, YES side
    asks        JSONB NOT NULL,
    hash        TEXT,
    PRIMARY KEY (platform, market_id, ts)
);

CREATE TABLE IF NOT EXISTS trades (
    platform    TEXT NOT NULL,
    trade_id    TEXT NOT NULL,
    market_id   TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    price       DOUBLE PRECISION,       -- YES price
    size        DOUBLE PRECISION,
    side        TEXT,                   -- taker side
    raw         JSONB,
    PRIMARY KEY (platform, trade_id, ts)  -- hypertable keys must include ts
);
CREATE INDEX IF NOT EXISTS idx_trades_market ON trades (platform, market_id, ts);

-- Polymarket /prices-history points (last-trade price series)
CREATE TABLE IF NOT EXISTS price_points (
    platform    TEXT NOT NULL,
    market_id   TEXT NOT NULL,
    fidelity    INT NOT NULL,           -- minutes between points as requested
    ts          TIMESTAMPTZ NOT NULL,
    price       DOUBLE PRECISION,
    PRIMARY KEY (platform, market_id, fidelity, ts)
);

-- Kalshi candlesticks
CREATE TABLE IF NOT EXISTS candles (
    platform    TEXT NOT NULL,
    market_id   TEXT NOT NULL,
    period_min  INT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,   -- end of period
    open DOUBLE PRECISION, high DOUBLE PRECISION,
    low DOUBLE PRECISION, close DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    open_interest DOUBLE PRECISION,
    raw         JSONB,
    PRIMARY KEY (platform, market_id, period_min, ts)
);

-- Exchange trading halts / maintenance windows (Kalshi pauses nightly, etc.)
CREATE TABLE IF NOT EXISTS outages (
    platform    TEXT NOT NULL,
    kind        TEXT NOT NULL,      -- 'halt' (observed) | 'maintenance' (scheduled)
    start_ts    BIGINT NOT NULL,
    end_ts      BIGINT,             -- NULL while ongoing
    note        TEXT,
    PRIMARY KEY (platform, kind, start_ts)
);
"""

HYPERTABLES = ["book_snapshots", "trades", "price_points", "candles"]


def connect(dsn=None):
    # DDL runs in autocommit so a failed optional statement can't poison a
    # shared transaction; the connection is handed to the app transactional.
    conn = psycopg.connect(dsn or config.PG_DSN, row_factory=dict_row, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    conn.execute(SCHEMA)
    for table in HYPERTABLES:
        conn.execute(
            "SELECT create_hypertable(%s, 'ts', if_not_exists => TRUE, "
            "migrate_data => TRUE)", (table,))
    # compress book snapshots older than 30 days (they dominate volume)
    try:
        conn.execute("""
            ALTER TABLE book_snapshots SET (timescaledb.compress,
                timescaledb.compress_segmentby = 'platform, market_id')""")
        conn.execute("""
            SELECT add_compression_policy('book_snapshots', INTERVAL '30 days',
                                          if_not_exists => TRUE)""")
    except psycopg.Error:
        pass  # compression unavailable on this build; fine
    conn.autocommit = False
    return conn


def upsert_market(conn, m, now):
    """m: dict with keys matching markets columns (subset ok)."""
    conn.execute(
        """
        INSERT INTO markets (platform, market_id, event_id, series, condition_id,
                             title, outcome, game_start_ts, open_ts, close_ts,
                             status, result, first_seen_ts, last_seen_ts, raw)
        VALUES (%(platform)s, %(market_id)s, %(event_id)s, %(series)s,
                %(condition_id)s, %(title)s, %(outcome)s, %(game_start_ts)s,
                %(open_ts)s, %(close_ts)s, %(status)s, %(result)s,
                %(now)s, %(now)s, %(raw)s)
        ON CONFLICT (platform, market_id) DO UPDATE SET
            event_id = COALESCE(EXCLUDED.event_id, markets.event_id),
            series = COALESCE(EXCLUDED.series, markets.series),
            condition_id = COALESCE(EXCLUDED.condition_id, markets.condition_id),
            title = COALESCE(EXCLUDED.title, markets.title),
            outcome = COALESCE(EXCLUDED.outcome, markets.outcome),
            game_start_ts = COALESCE(EXCLUDED.game_start_ts, markets.game_start_ts),
            open_ts = COALESCE(EXCLUDED.open_ts, markets.open_ts),
            close_ts = COALESCE(EXCLUDED.close_ts, markets.close_ts),
            status = COALESCE(EXCLUDED.status, markets.status),
            result = COALESCE(EXCLUDED.result, markets.result),
            last_seen_ts = EXCLUDED.last_seen_ts,
            raw = COALESCE(EXCLUDED.raw, markets.raw)
        """,
        {
            "platform": m["platform"], "market_id": m["market_id"],
            "event_id": m.get("event_id"), "series": m.get("series"),
            "condition_id": m.get("condition_id"), "title": m.get("title"),
            "outcome": m.get("outcome"), "game_start_ts": m.get("game_start_ts"),
            "open_ts": m.get("open_ts"), "close_ts": m.get("close_ts"),
            "status": m.get("status"), "result": m.get("result"),
            "now": now,
            "raw": Json(m["raw"]) if m.get("raw") else None,
        },
    )


def last_book_hashes(conn):
    """{(platform, market_id): hash} of each market's most recent snapshot."""
    rows = conn.execute(
        """
        SELECT DISTINCT ON (platform, market_id) platform, market_id, hash
        FROM book_snapshots ORDER BY platform, market_id, ts DESC
        """).fetchall()
    return {(r["platform"], r["market_id"]): r["hash"] for r in rows}


def insert_snapshot(conn, platform, market_id, ts_ms, source_ts_ms, bids, asks, book_hash):
    conn.execute(
        """INSERT INTO book_snapshots
           (platform, market_id, ts, source_ts, bids, asks, hash)
           VALUES (%s, %s, to_timestamp(%s), to_timestamp(%s), %s, %s, %s)
           ON CONFLICT DO NOTHING""",
        (platform, market_id, ts_ms / 1000.0,
         source_ts_ms / 1000.0 if source_ts_ms else None,
         Json(bids), Json(asks), book_hash),
    )


def insert_trades(conn, rows):
    """rows: iterable of (platform, trade_id, market_id, ts, price, size, side, raw).
    Returns number of new rows."""
    rows = list(rows)
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO trades
               (platform, trade_id, market_id, ts, price, size, side, raw)
               VALUES (%s, %s, %s, to_timestamp(%s), %s, %s, %s, %s)
               ON CONFLICT DO NOTHING""",
            [(p, tid, mid, ts, price, size, side, Json(json.loads(raw)))
             for (p, tid, mid, ts, price, size, side, raw) in rows],
            returning=False,
        )
        return cur.rowcount


def insert_price_points(conn, platform, market_id, fidelity, points):
    if not points:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO price_points (platform, market_id, fidelity, ts, price)
               VALUES (%s, %s, %s, to_timestamp(%s), %s)
               ON CONFLICT DO NOTHING""",
            [(platform, market_id, fidelity, p["t"], p["p"]) for p in points],
            returning=False,
        )


def insert_candles(conn, platform, market_id, period_min, rows):
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO candles
               (platform, market_id, period_min, ts, open, high, low, close,
                volume, open_interest, raw)
               VALUES (%s, %s, %s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (platform, market_id, period_min, ts) DO UPDATE SET
                   open = EXCLUDED.open, high = EXCLUDED.high,
                   low = EXCLUDED.low, close = EXCLUDED.close,
                   volume = EXCLUDED.volume,
                   open_interest = EXCLUDED.open_interest, raw = EXCLUDED.raw""",
            [(platform, market_id, period_min, c["ts"], c["open"], c["high"],
              c["low"], c["close"], c["volume"], c["open_interest"],
              Json(json.loads(c["raw"])))
             for c in rows],
            returning=False,
        )


def open_outage(conn, platform, kind, start_ts, note=None):
    conn.execute(
        """INSERT INTO outages (platform, kind, start_ts, end_ts, note)
           VALUES (%s, %s, %s, NULL, %s) ON CONFLICT DO NOTHING""",
        (platform, kind, start_ts, note))


def close_outage(conn, platform, kind, end_ts):
    conn.execute(
        """UPDATE outages SET end_ts = %s WHERE platform = %s AND kind = %s
           AND end_ts IS NULL""", (end_ts, platform, kind))


def ongoing_outage(conn, platform, kind="halt"):
    row = conn.execute(
        """SELECT start_ts FROM outages WHERE platform = %s AND kind = %s
           AND end_ts IS NULL ORDER BY start_ts DESC LIMIT 1""",
        (platform, kind)).fetchone()
    return row["start_ts"] if row else None


def outages_overlapping(conn, platform, start_ts, end_ts):
    return conn.execute(
        """SELECT * FROM outages WHERE platform = %s
           AND start_ts <= %s
           AND COALESCE(end_ts, EXTRACT(EPOCH FROM now())::bigint) >= %s
           ORDER BY start_ts""", (platform, end_ts, start_ts)).fetchall()


def set_market_fields(conn, platform, market_id, **fields):
    sets = ", ".join("%s = %%s" % k for k in fields)
    conn.execute(
        "UPDATE markets SET %s WHERE platform = %%s AND market_id = %%s" % sets,
        list(fields.values()) + [platform, market_id],
    )
