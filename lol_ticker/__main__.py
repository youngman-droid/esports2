import argparse
import logging
import sys
from datetime import datetime, timezone

from . import collector, config, db, query


def main():
    p = argparse.ArgumentParser(
        prog="lol_ticker",
        description="Historical L2/ticker data collector for Polymarket & Kalshi "
                    "League of Legends markets")
    p.add_argument("--dsn", default=config.PG_DSN,
                   help="postgres dsn (default: %(default)s or $LOL_TICKER_DSN)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="refresh the market catalog")
    d.add_argument("--all", action="store_true",
                   help="crawl ALL closed Polymarket LoL events (initial setup)")

    b = sub.add_parser("backfill", help="pull trades + price history for settled markets")
    b.add_argument("--limit", type=int, default=None, help="max markets this run")
    b.add_argument("--workers", type=int, default=8, help="concurrent fetch workers")

    r = sub.add_parser("record", help="poll live L2 books (daemon)")
    r.add_argument("--once", action="store_true", help="single pass, then exit")

    sub.add_parser("status", help="summarize stored data")

    g = sub.add_parser("game", help="find games by team / date (YYYY-MM-DD) terms")
    g.add_argument("terms", nargs="+")

    e = sub.add_parser("export", help="export a game's full ticker record as CSVs")
    e.add_argument("terms", nargs="+", help="team / date terms or an exact event id")
    e.add_argument("--out", default="exports", help="output directory")

    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    conn = db.connect(args.dsn)

    if args.cmd == "discover":
        collector.discover(conn, include_closed_pm=args.all)
    elif args.cmd == "backfill":
        try:
            collector.backfill(conn, limit=args.limit, workers=args.workers)
        except KeyboardInterrupt:
            print("\nstopped (progress saved; rerun to resume)")
    elif args.cmd == "record":
        try:
            collector.record(conn, once=args.once)
        except KeyboardInterrupt:
            print("\nstopped")
    elif args.cmd == "status":
        rows, fast, slow = collector.status(conn)
        print("%-11s %8s %6s %10s %10s %9s  %s" % (
            "platform", "markets", "open", "backfilled", "snapshots", "trades", "latest snapshot (UTC)"))
        for platform, n, open_n, bf, snaps, latest, ntr in rows:
            latest_s = (datetime.fromtimestamp(latest / 1000, tz=timezone.utc)
                        .strftime("%Y-%m-%d %H:%M:%S") if latest else "-")
            print("%-11s %8d %6d %10d %10d %9d  %s" % (
                platform, n, open_n or 0, bf or 0, snaps, ntr, latest_s))
        print("\nwatchlist: %d fast-tier (near game time), %d slow-tier" % (len(fast), len(slow)))
        for m in fast[:20]:
            print("  FAST %s %s" % (m["platform"], m["market_id"]))
    elif args.cmd == "game":
        _print_games(query.find_games(conn, args.terms))
    elif args.cmd == "export":
        games = query.find_games(conn, args.terms)
        if not games:
            print("no matching games")
            return 1
        for ev in games:
            paths = query.export_game(conn, ev["platform"], ev["event_id"], args.out)
            print("%s/%s -> %d files" % (ev["platform"], ev["event_id"], len(paths)))
            for pth in paths:
                print("   ", pth)
    return 0


def _print_games(games):
    if not games:
        print("no matching games")
        return
    for ev in games:
        start = (datetime.fromtimestamp(ev["game_start_ts"], tz=timezone.utc)
                 .strftime("%Y-%m-%d %H:%M") if ev["game_start_ts"] else "?")
        flag = "  [OUTAGE-AFFECTED]" if ev["outage_affected"] else ""
        print("%-10s %-40s start=%s  markets=%d%s" % (
            ev["platform"], ev["event_id"], start, ev["n_markets"], flag))
        print("           snapshots=%d trades=%d candles=%d price_points=%d  (%s)" % (
            ev["snapshots"], ev["trades"], ev["candles"], ev["price_points"],
            (ev["sample_title"] or "")[:60]))


if __name__ == "__main__":
    sys.exit(main())
