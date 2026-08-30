import argparse
import json
import logging
import os
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

    w = sub.add_parser("dashboard", help="serve the local web dashboard")
    w.add_argument("--port", type=int,
                   default=int(os.environ.get("PORT", "8090")),
                   help="default: $PORT or 8090")

    dr = sub.add_parser("draftload",
                        help="load Oracle's Elixir CSVs and build draft deltas")
    dr.add_argument("csvs", nargs="*", default=[],
                    help="OE csv paths (default: data/oe/oe_*.csv)")
    dr.add_argument("--skip-deltas", action="store_true",
                    help="reload CSVs and refit the model without re-matching markets")
    sub.add_parser("draftfit", help="(re)fit the draft simulator model only")

    gg = sub.add_parser("golgg", help="scrape detailed game data from gol.gg")
    gg.add_argument("--seasons", default="S16", help="comma list, e.g. S15,S16")
    gg.add_argument("--regions", default=None,
                    help="comma list of gol.gg region codes (default: all); "
                         "'major' = KR,CN,EUW,NA,LTA,PCS,VN,BR,WR,INT")
    gg.add_argument("--since", default=None, help="only matches on/after YYYY-MM-DD")
    gg.add_argument("--tournament", default=None, help="substring filter on tournament name")
    gg.add_argument("--workers", type=int, default=8)
    gg.add_argument("--limit", type=int, default=None, help="max games this run")
    gg.add_argument("--no-builds", action="store_true", help="skip item build timelines")
    gg.add_argument("--items", action="store_true", help="also refresh item id->name table")
    gg.add_argument("--status", action="store_true", help="print scrape progress and exit")

    wp = sub.add_parser("wpa", help="odds-free: Elo + in-game win-probability model + event WPA")
    wp.add_argument("--rebuild", action="store_true")
    sub.add_parser("draftfree", help="odds-free draft model: outcomes on Elo + draft features")
    lv = sub.add_parser("live", help="live win probability from the LoL Esports stats feed")
    lv.add_argument("game_id", nargs="?", default=None, help="lolesports game id (default: the in-progress game)")
    lv.add_argument("--blue-elo", type=float, default=None, help="blue minus red Elo diff (optional prior)")

    sh = sub.add_parser("shadow", help="prospective live forecast ledger and Brier scoring")
    sh.add_argument("action", choices=["record", "resolve", "score", "status"])
    sh.add_argument("--once", action="store_true", help="one capture/resolve pass, then exit")
    sh.add_argument("--interval", type=int, default=15, help="recorder polling interval in seconds")
    sh.add_argument("--game-id", default=None, help="manual resolution: LoL Esports game id")
    sh.add_argument("--winner", choices=["blue", "red"], default=None,
                    help="manual resolution winner side (requires --game-id)")
    sh.add_argument("--bootstrap", type=int, default=5000,
                    help="paired game-block bootstrap draws for score")

    wx = sub.add_parser("wpx", help="odds-free modeling and historical evaluation")
    wx.add_argument("step", choices=["prep", "build", "fit", "gam-eval", "bench",
                                     "rolling", "stack-eval", "hist-blend", "blend",
                                     "blend-live", "eval", "all"])
    wx.add_argument("--lead", type=int, default=45,
                    help="market lead in seconds for blend-live (default: 45)")
    wx.add_argument("--windows", type=int, default=3,
                    help="expanding chronological windows for rolling evaluation")

    al = sub.add_parser("align", help="align gol.gg timelines with odds; build event swings")
    al.add_argument("--rebuild", action="store_true", help="recompute existing alignments too")

    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # Cached-model operations are deliberately usable without PostgreSQL.
    # ``prep``, ``build`` and ``all`` still need the source database.
    cached_wpx = args.cmd == "wpx" and args.step in (
        "fit", "gam-eval", "rolling", "stack-eval", "bench",
        "hist-blend", "blend", "eval")
    conn = None if cached_wpx else db.connect(args.dsn)

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
    elif args.cmd == "dashboard":
        from . import dashboard
        try:
            dashboard.serve(port=args.port)
        except KeyboardInterrupt:
            print("\nstopped")
    elif args.cmd == "draftload":
        import glob
        from . import draft
        paths = args.csvs or sorted(glob.glob(
            os.path.join(config.REPO_ROOT, "data", "oe", "oe_*.csv")))
        if not paths:
            print("no OE csvs found; pass paths or put them in data/oe/")
            return 1
        draft.load_oe(conn, paths)
        if not args.skip_deltas:
            draft.build_deltas(conn)
        draft.fit_model(conn)
    elif args.cmd == "draftfit":
        from . import draft
        draft.fit_model(conn)
    elif args.cmd == "wpa":
        from . import wpa
        wpa.elo(conn)
        wpa.fit(conn)
        wpa.build_wpa(conn, rebuild=args.rebuild)
    elif args.cmd == "draftfree":
        from . import draft
        draft.fit_outcome_model(conn)
    elif args.cmd == "live":
        import json as _json
        from . import live
        priors = {"elo_oe": args.blue_elo / 400.0} if args.blue_elo is not None else None
        r = live.estimate(conn, args.game_id, priors)
        if "error" in r:
            print(r["error"]); return 1
        st = r["state"]
        print("%s | clock %d:%02d | patch %s" % (r["meta"].get("teams") or r["meta"].get("game_id"), st["clock_s"] // 60, st["clock_s"] % 60, st["patch"]))
        print("blue %s  vs  red %s" % (st["blue_champs"], st["red_champs"]))
        print("gold %d vs %d (%+.1fk)  kills %d-%d  towers %d-%d  dragons %d-%d %s/%s  barons %d-%d  inhibs %d-%d  items done diff %+d" % (
            st["gold_blue"], st["gold_red"], st["gold_diff_k"], st["kills_blue"], st["kills_red"], st["towers_blue"], st["towers_red"],
            st["drag_blue"], st["drag_red"], st["dragon_types_blue"], st["dragon_types_red"], st["barons_blue"], st["barons_red"],
            st["inhib_blue"], st["inhib_red"], st["items_done_diff"]))
        print("P(blue wins) = %.1f%%   (state-only, no champion terms: %.1f%%)" % (100 * r["p_blue"], 100 * r["p_blue_no_champ"]))
        if r["unknown_champions"]:
            print("unknown champions (no scaling term):", r["unknown_champions"])
    elif args.cmd == "shadow":
        from . import shadow
        if args.winner and not args.game_id:
            p.error("shadow resolve --winner requires --game-id")
        if args.action == "record":
            shadow.record(conn, once=args.once, interval_s=max(5, args.interval))
        elif args.action == "resolve":
            r = shadow.resolve_outcomes(conn, game_id=args.game_id, winner=args.winner)
            print("resolved={resolved} voided={voided} pending={pending} protocol={protocol_id}".format(**r))
        elif args.action == "score":
            shadow.resolve_outcomes(conn)
            r = shadow.score(conn, bootstrap=max(0, args.bootstrap))
            shadow.report_score(r)
        else:
            shadow.report_status(shadow.status(conn))
    elif args.cmd == "wpx":
        from . import wpx
        if args.step in ("prep", "all"):
            wpx.prep(conn)
        if args.step in ("build", "all"):
            wpx.build(conn)
        if args.step in ("fit", "all"):
            path = wpx.fit_full()
            print("constrained live model:", path)
        if args.step in ("gam-eval", "all"):
            from . import wpgam
            wpgam.report_walk_forward(wpgam.evaluate_walk_forward())
        if args.step == "rolling":
            from . import wpgam
            out = wpgam.evaluate_rolling(windows=max(1, args.windows))
            print(json.dumps(out, indent=2))
        if args.step == "stack-eval":
            from . import wpdeploy
            wpdeploy.report(wpdeploy.run())
        if args.step in ("bench", "all"):
            from . import wpbench
            wpbench.report(wpbench.run())
        if args.step in ("hist-blend", "all"):
            from . import wphist
            wphist.report(wphist.run())
        if args.step in ("blend", "all"):
            from . import wpblend
            wpblend.report(wpblend.run())
        if args.step in ("blend-live", "all"):
            from . import wpblend
            wpblend.report(wpblend.run_latency(conn, lag_s=args.lead))
        if args.step in ("eval", "all"):
            wpx.report(wpx.evaluate())
    elif args.cmd == "align":
        from . import align
        align.build_all(conn, rebuild=args.rebuild)
    elif args.cmd == "golgg":
        from . import golgg
        if args.status:
            st, per_tr = golgg.status(conn)
            print("tournaments %d (%d games in scope) · matches indexed %d (%d games listed)" % (
                st["tournaments"], st["games_in_scope"], st["matches"], st["games_listed"]))
            print("games scraped %d · players %d · events %s · timeline rows %s · build events %s" % (
                st["games"], st["players"], f"{st['events']:,}", f"{st['timeline_rows']:,}",
                f"{st['build_events']:,}"))
            if st["games"] and st["t0"] and st["t1"] and st["t1"] > st["t0"]:
                print("rate %.0f games/h · game dates %s → %s" % (
                    st["games"] / (st["t1"] - st["t0"]) * 3600, st["d0"], st["d1"]))
            for x in per_tr[:15]:
                print("  %4d/%-4d %-4s %s" % (x["done"], x["nbgames"], x["region"], x["trname"]))
            if len(per_tr) > 15:
                print("  … %d more tournaments" % (len(per_tr) - 15))
            return 0
        regions = None
        if args.regions:
            regions = (golgg.MAJOR_REGIONS if args.regions == "major"
                       else [r.strip() for r in args.regions.split(",") if r.strip()])
        if args.items:
            golgg.load_items(conn)
        try:
            golgg.sync(conn, seasons=[s.strip() for s in args.seasons.split(",")],
                       regions=regions, since=args.since, tournament=args.tournament,
                       workers=args.workers, limit=args.limit,
                       with_builds=not args.no_builds)
        except KeyboardInterrupt:
            print("\nstopped (progress saved; rerun to resume)")
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
