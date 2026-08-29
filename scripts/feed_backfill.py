"""Backfill 1-Hz live-stats feed frames (HP / level / gold per player) for
historical esports games, matched to gol.gg games. The feed serves months of
history; per-player currentHealth/maxHealth/level come from the `window`
endpoint alone (~1 request per game-minute).

Tables:
  feed_games   (esports_game_id PK, golgg_game_id, league, game_num, start_iso, status)
  feed_minutes (esports_game_id, minute, ts, data jsonb)  -- hp fractions, levels, gold per player

Usage: python3 scripts/feed_backfill.py [--discover-only] [--limit N]
Resumable: skips games already scraped/unavailable.
"""
import argparse, datetime as dt, json, logging, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from lol_ticker import db
from lol_ticker.live import _get, _iso, _ts, FEED
from lol_ticker.draft import norm_team
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("feedbf")
conn = db.connect()
conn.execute("""CREATE TABLE IF NOT EXISTS feed_games (
    esports_game_id TEXT PRIMARY KEY, golgg_game_id INT, league TEXT, game_num INT,
    start_iso TEXT, teams TEXT, status TEXT DEFAULT 'pending')""")
conn.execute("""CREATE TABLE IF NOT EXISTS feed_minutes (
    esports_game_id TEXT NOT NULL, minute INT NOT NULL, ts BIGINT, data JSONB,
    PRIMARY KEY (esports_game_id, minute))""")
conn.commit()
ap = argparse.ArgumentParser(); ap.add_argument("--discover-only", action="store_true")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=1); a = ap.parse_args()

# ---------------- discovery ----------------
def discover():
    known = {r["esports_game_id"] for r in conn.execute("SELECT esports_game_id FROM feed_games")}
    gg = conn.execute("""SELECT game_id, game_num, date, blue_team, red_team FROM golgg_games
                         WHERE date >= '2025-06-01'""").fetchall()
    by_key = defaultdict(list)
    for g in gg:
        by_key[(frozenset((norm_team(g["blue_team"]), norm_team(g["red_team"]))), g["game_num"])].append(g)
    leagues = _get("https://esports-api.lolesports.com/persisted/gw/getLeagues", {"hl": "en-US"}, key=True)["data"]["leagues"]
    n_new = 0
    for lg in leagues:
        try:
            trs = _get("https://esports-api.lolesports.com/persisted/gw/getTournamentsForLeague",
                       {"hl": "en-US", "leagueId": lg["id"]}, key=True)["data"]["leagues"]
            trs = trs[0]["tournaments"] if trs else []
        except Exception:
            continue
        for tr in trs:
            if (tr.get("endDate") or "2000") < "2025-06-01":
                continue
            try:
                evs = _get("https://esports-api.lolesports.com/persisted/gw/getCompletedEvents",
                           {"hl": "en-US", "tournamentId": tr["id"]}, key=True)["data"]["schedule"]["events"]
            except Exception as e:
                log.warning("completed events failed %s: %s", tr.get("slug"), e); continue
            for e in evs:
                m = e.get("match") or {}
                teams = [t.get("name") for t in m.get("teams", [])]
                if len(teams) != 2: continue
                key = frozenset((norm_team(teams[0]), norm_team(teams[1])))
                edate = dt.datetime.fromisoformat(e["startTime"].replace("Z", "+00:00")).date()
                # completed-events game entries carry only ids, in game order; the
                # number of games actually played = sum of gameWins
                wins = [((t_.get("result") or {}).get("gameWins") or 0) for t_ in m.get("teams", [])]
                played = sum(wins) if any(wins) else len(e.get("games") or [])
                for i_g, g in enumerate((e.get("games") or [])[:played]):
                    if g.get("id") in known: continue
                    number = g.get("number") or (i_g + 1)
                    cands = [x for x in by_key.get((key, number), [])
                             if x["date"] and abs((x["date"] - edate).days) <= 1]
                    gid = cands[0]["game_id"] if len(cands) == 1 else None
                    conn.execute("""INSERT INTO feed_games (esports_game_id, golgg_game_id, league, game_num, start_iso, teams)
                                    VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                                 (g["id"], gid, lg.get("slug"), number, e["startTime"], " vs ".join(teams)))
                    known.add(g["id"]); n_new += 1
            conn.commit()
        time.sleep(0.2)
    log.info("discovery: %d new feed games (linked: %s)", n_new,
             conn.execute("SELECT count(*) n FROM feed_games WHERE golgg_game_id IS NOT NULL").fetchone()["n"])

# ---------------- scrape ----------------
def hp_row(f):
    out = {}
    for side, kk in (("blueTeam", "b"), ("redTeam", "r")):
        ps = f[side]["participants"]
        out["hp" + kk] = [round(p["currentHealth"] / p["maxHealth"], 3) if p.get("maxHealth") else None for p in ps]
        out["lv" + kk] = [p.get("level") for p in ps]
        out["gd" + kk] = [p.get("totalGold") for p in ps]
    out["state"] = f.get("gameState")
    return out

def scrape_game(egid, http_pause=0.12):
    try:
        w0 = _get(FEED + "/window/%s" % egid, allow_empty=True, timeout=20)
    except Exception:
        w0 = None
    if not w0 or not w0.get("frames"):
        return "unavailable"
    t0 = _ts(w0["frames"][0]["rfc460Timestamp"])
    rows = []
    empty = 0
    for minute in range(0, 75):
        st = int((t0 + minute * 60) // 10 * 10)
        try:
            w = _get(FEED + "/window/%s" % egid, {"startingTime": _iso(st)}, allow_empty=True, timeout=20)
        except Exception:
            w = None
        time.sleep(http_pause)
        if not w or not w.get("frames"):
            empty += 1
            if empty >= 3 and minute > 15: break
            continue
        empty = 0
        f = w["frames"][-1]
        rows.append((egid, minute, int(_ts(f["rfc460Timestamp"])), json.dumps(hp_row(f))))
        if f.get("gameState") == "finished": break
    if not rows:
        return "unavailable", []
    return "scraped", rows

discover()
if a.discover_only:
    sys.exit(0)
todo = [r["esports_game_id"] for r in conn.execute("""SELECT esports_game_id FROM feed_games
                       WHERE status = 'pending' AND golgg_game_id IS NOT NULL
                       ORDER BY start_iso DESC""")]
if a.limit: todo = todo[:a.limit]
log.info("scraping %d games with %d workers", len(todo), a.workers)
t_start = time.time()
stats = {"ok": 0, "un": 0, "i": 0}
slock = threading.Lock()
qlock = threading.Lock()
queue = list(todo)
def worker():
    wconn = db.connect()
    while True:
        with qlock:
            if not queue: return
            egid = queue.pop(0)
        try:
            st, rows = scrape_game(egid, http_pause=0.05)
        except Exception as e:
            log.warning("scrape %s failed: %s", egid, e); st, rows = "pending", []
        if rows:
            with wconn.cursor() as cur:
                cur.executemany("""INSERT INTO feed_minutes (esports_game_id, minute, ts, data)
                                   VALUES (%s,%s,%s,%s) ON CONFLICT (esports_game_id, minute) DO NOTHING""", rows, returning=False)
        if st != "pending":
            wconn.execute("UPDATE feed_games SET status=%s WHERE esports_game_id=%s", (st, egid))
        wconn.commit()
        with slock:
            stats["i"] += 1; stats["ok"] += st == "scraped"; stats["un"] += st == "unavailable"
            if stats["i"] % 50 == 0:
                rate = stats["i"] / max(1e-9, time.time() - t_start) * 3600
                log.info("%d/%d scraped=%d unavailable=%d (%.0f games/h)", stats["i"], len(todo), stats["ok"], stats["un"], rate)
threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, a.workers))]
for th in threads: th.start()
for th in threads: th.join()
log.info("done: %d scraped, %d unavailable", stats["ok"], stats["un"])
