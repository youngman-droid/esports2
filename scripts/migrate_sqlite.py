"""One-time migration of the legacy sqlite db (data/league.db) into TimescaleDB.

Usage: python3 scripts/migrate_sqlite.py [sqlite_path]
Idempotent: everything inserts with ON CONFLICT handling.
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lol_ticker import config, db  # noqa: E402


def main():
    sqlite_path = sys.argv[1] if len(sys.argv) > 1 else config.SQLITE_PATH
    if not os.path.exists(sqlite_path):
        print("no sqlite db at", sqlite_path)
        return 1
    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    dst = db.connect()

    n = 0
    for r in src.execute("SELECT * FROM markets"):
        m = dict(r)
        m["raw"] = json.loads(m["raw"]) if m.get("raw") else None
        db.upsert_market(dst, m, m.get("last_seen_ts") or 0)
        db.set_market_fields(dst, m["platform"], m["market_id"],
                             backfilled=m.get("backfilled") or 0,
                             last_trade_ts=m.get("last_trade_ts") or 0,
                             outage_affected=m.get("outage_affected") or 0)
        n += 1
        if n % 2000 == 0:
            dst.commit()
            print("markets:", n)
    dst.commit()
    print("markets done:", n)

    for r in src.execute("SELECT * FROM book_snapshots"):
        db.insert_snapshot(dst, r["platform"], r["market_id"], r["ts_ms"],
                           r["source_ts_ms"], json.loads(r["bids"]),
                           json.loads(r["asks"]), r["hash"])
    dst.commit()
    print("snapshots done")

    rows = [(r["platform"], r["trade_id"], r["market_id"], r["ts"], r["price"],
             r["size"], r["side"], r["raw"] or "{}")
            for r in src.execute("SELECT * FROM trades")]
    db.insert_trades(dst, rows)
    dst.commit()
    print("trades done:", len(rows))

    for r in src.execute("SELECT platform, market_id, fidelity FROM price_points GROUP BY 1,2,3"):
        pts = [{"t": x["ts"], "p": x["price"]} for x in src.execute(
            "SELECT ts, price FROM price_points WHERE platform=? AND market_id=? AND fidelity=?",
            (r["platform"], r["market_id"], r["fidelity"]))]
        db.insert_price_points(dst, r["platform"], r["market_id"], r["fidelity"], pts)
    dst.commit()
    print("price_points done")

    for r in src.execute("SELECT platform, market_id, period_min FROM candles GROUP BY 1,2,3"):
        cs = [dict(x) for x in src.execute(
            "SELECT * FROM candles WHERE platform=? AND market_id=? AND period_min=?",
            (r["platform"], r["market_id"], r["period_min"]))]
        for c in cs:
            c["raw"] = c["raw"] or "{}"
        db.insert_candles(dst, r["platform"], r["market_id"], r["period_min"], cs)
    dst.commit()
    print("candles done")

    for r in src.execute("SELECT * FROM outages"):
        db.open_outage(dst, r["platform"], r["kind"], r["start_ts"], r["note"])
        if r["end_ts"]:
            dst.execute(
                """UPDATE outages SET end_ts = %s WHERE platform = %s
                   AND kind = %s AND start_ts = %s""",
                (r["end_ts"], r["platform"], r["kind"], r["start_ts"]))
    dst.commit()
    print("outages done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
