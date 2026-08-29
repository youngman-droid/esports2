"""gol.gg (Games of Legends) scraper: detailed per-game data.

Per game we store: header (teams, result, duration, patch, bans/picks, team
objective counts), 57 per-player stats (damage, gold, vision, CS@15, ...),
the full event timeline (kills w/ bounties, plates, grubs, herald, dragons by
type, atakhan, baron, towers by lane/tier, inhibs, nexus), per-minute gold and
CS per player, and item build timelines (purchases/sells/undo) with the final
loadout, summoner spells and runes.

Polite by design: shared request spacing (GOLGG_INTERVAL), a descriptive UA,
resumable (games already stored are skipped).  robots.txt permits these pages.
"""
import gzip
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from . import config, http

log = logging.getLogger("golgg")

BASE = "https://gol.gg"
GOLGG_INTERVAL = 0.22       # seconds between page requests (shared across workers)
GOLGG_AJAX_INTERVAL = 0.06  # the per-player build calls are ~2 KB JSON: lighter lane
config.MIN_INTERVAL["gol.gg"] = GOLGG_INTERVAL
config.MIN_INTERVAL["gol.gg:ajax"] = GOLGG_AJAX_INTERVAL
UA = "Mozilla/5.0 (compatible; lol-ticker-research/1.0; private esports research)"

MAJOR_REGIONS = ["KR", "CN", "EUW", "NA", "LTA", "PCS", "VN", "BR", "WR", "INT"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS golgg_tournaments (
    trname      TEXT PRIMARY KEY,
    season      TEXT,
    region      TEXT,
    nbgames     INT,
    first_game  DATE,
    last_game   DATE,
    synced_at   BIGINT
);
CREATE TABLE IF NOT EXISTS golgg_matches (
    match_id    INT PRIMARY KEY,      -- gol.gg id of the match (= its first game)
    trname      TEXT,
    team1       TEXT, team2 TEXT,
    score       TEXT,
    patch       TEXT,
    date        DATE,
    game_ids    JSONB,                -- ordered list of game ids
    fetched_at  BIGINT
);
CREATE TABLE IF NOT EXISTS golgg_games (
    game_id     INT PRIMARY KEY,
    match_id    INT,
    game_num    INT,
    trname      TEXT,
    date        DATE,
    patch       TEXT,
    duration_s  INT,
    blue_team   TEXT, red_team TEXT,
    winner_side TEXT,                 -- 'blue' | 'red'
    blue_kills INT, blue_towers INT, blue_dragons INT, blue_barons INT, blue_gold INT,
    red_kills INT,  red_towers INT,  red_dragons INT,  red_barons INT,  red_gold INT,
    blue_first_blood BOOLEAN, blue_first_pick BOOLEAN,
    blue_bans JSONB, red_bans JSONB, blue_picks JSONB, red_picks JSONB,
    blue_dragon_types JSONB, red_dragon_types JSONB,
    fetched_at  BIGINT
);
CREATE INDEX IF NOT EXISTS idx_golgg_games_date ON golgg_games (date);
CREATE TABLE IF NOT EXISTS golgg_players (
    game_id     INT NOT NULL,
    player_id   INT NOT NULL,
    side        TEXT, role TEXT, slot INT,      -- slot 0-9: blue top..sup, red top..sup
    player      TEXT, team TEXT, champion TEXT,
    kills INT, deaths INT, assists INT, cs INT, gold INT, level INT,
    stats       JSONB,                -- every row of the full-stats table
    loadout     JSONB,                -- final items, spells, runes (ajax part 0)
    PRIMARY KEY (game_id, player_id)
);
CREATE TABLE IF NOT EXISTS golgg_events (
    game_id INT NOT NULL, seq INT NOT NULL,
    time_s INT, side TEXT, player TEXT, champion TEXT,
    action TEXT,                      -- kill|plate|grubs|herald|dragon:<type>|atakhan|baron|tower|inhib|nexus
    target TEXT, target_champion TEXT, bounty INT,
    PRIMARY KEY (game_id, seq)
);
CREATE TABLE IF NOT EXISTS golgg_timeline (
    game_id INT NOT NULL, slot INT NOT NULL, minute INT NOT NULL,
    gold INT, cs INT,
    PRIMARY KEY (game_id, slot, minute)
);
CREATE TABLE IF NOT EXISTS golgg_builds (
    game_id INT NOT NULL, player_id INT NOT NULL, seq INT NOT NULL,
    build_time INT, event TEXT, item_id INT,
    PRIMARY KEY (game_id, player_id, seq)
);
CREATE TABLE IF NOT EXISTS golgg_items (
    item_id INT PRIMARY KEY, name TEXT, version TEXT
);
"""


def ensure_schema(conn):
    conn.execute(SCHEMA)
    conn.commit()


# ------------------------------------------------------------------ fetching

def _fetch(url, data=None, retries=3, timeout=60, referer=None, lane=None):
    """GET/POST with the shared throttle (per lane); gzip; returns decoded text."""
    body = urllib.parse.urlencode(data).encode() if data else None
    last = None
    for attempt in range(retries + 1):
        http._throttle(url, lane)
        req = urllib.request.Request(url, data=body, headers={
            "User-Agent": UA, "Accept": "text/html,application/json",
            "Accept-Encoding": "gzip",
            "Referer": referer or BASE + "/",
            **({"Content-Type": "application/x-www-form-urlencoded"} if body else {}),
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                http._rate_limited(lane or "gol.gg")
            elif 400 <= e.code < 500:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        if attempt < retries and http.SHUTDOWN.wait(2 ** attempt):
            raise http.ShuttingDown()
    raise last


def list_tournaments(season):
    """[{trname, region, nbgames, firstgame, lastgame}] for a season like 'S16'."""
    txt = _fetch(BASE + "/tournament/ajax.trlist.php", {"season": season})
    return json.loads(txt)


def list_matches(trname):
    """Played matches of a tournament: [{match_id, team1, team2, score, patch, date}]."""
    url = BASE + "/tournament/tournament-matchlist/%s/" % urllib.parse.quote(trname)
    soup = BeautifulSoup(_fetch(url), "lxml")
    out = []
    table = soup.find("table")
    if not table:
        return out
    for tr in table.find_all("tr")[1:]:
        a = tr.find("a", href=re.compile(r"/game/stats/(\d+)/page-summary"))
        if not a:
            continue  # preview rows = not yet played
        mid = int(re.search(r"/game/stats/(\d+)/", a["href"]).group(1))
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        out.append({"match_id": mid, "team1": tds[1] if len(tds) > 1 else None,
                    "team2": tds[3] if len(tds) > 3 else None,
                    "score": tds[2] if len(tds) > 2 else None,
                    "patch": tds[5] if len(tds) > 5 else None,
                    "date": tds[6] if len(tds) > 6 else None})
    return out


def match_game_ids(match_id):
    """Ordered game ids from a match's summary page."""
    soup = BeautifulSoup(_fetch(BASE + "/game/stats/%d/page-summary/" % match_id), "lxml")
    ids = []
    for a in soup.find_all("a", href=re.compile(r"/game/stats/(\d+)/page-game")):
        if re.match(r"Game \d+", a.get_text(" ", strip=True)):
            gid = int(re.search(r"/game/stats/(\d+)/", a["href"]).group(1))
            if gid not in ids:
                ids.append(gid)
    return ids


# ------------------------------------------------------------------- parsing

def _num(s):
    s = (s or "").replace(",", "").strip()
    m = re.match(r"^-?\d+(\.\d+)?", s)
    if not m:
        return None
    v = float(m.group(0))
    if s.endswith("k"):
        v *= 1000
    return int(v) if v == int(v) else v


def _mmss(s):
    m = re.match(r"(\d+):(\d+)", s or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def parse_game_page(html):
    soup = BeautifulSoup(html, "lxml")
    out = {"players": []}
    txt = soup.get_text(" ", strip=True)
    m = re.search(r"Game Time (\d+:\d+)", txt)
    out["duration_s"] = _mmss(m.group(1)) if m else None
    m = re.search(r"\bv(\d+\.\d+)", txt)
    out["patch"] = m.group(1) if m else None
    for side in ("blue", "red"):
        hdr = soup.find("div", class_="%s-line-header" % side)
        if not hdr:
            continue
        block = hdr.find_parent("div", class_="col-sm-6") or hdr.parent
        name = re.sub(r"\s*-\s*(WIN|LOSS)\s*$", "", hdr.get_text(" ", strip=True))
        out[side + "_team"] = name
        if "WIN" in hdr.get_text():
            out["winner_side"] = side
        vals = {}
        for sb in block.find_all("span", class_="score-box"):
            img = sb.find("img")
            if img:
                vals[(img.get("alt") or "").lower()] = _num(sb.get_text(" ", strip=True))
        out[side + "_kills"] = vals.get("kills")
        out[side + "_towers"] = vals.get("towers")
        out[side + "_dragons"] = vals.get("dragons")
        out[side + "_barons"] = vals.get("nashor")
        out[side + "_gold"] = vals.get("team gold")
        out[side + "_first_blood"] = bool(block.find("img", alt="First Blood"))
        out[side + "_first_pick"] = bool(block.find("img", alt="First Pick"))
        champs = [img for img in block.find_all("img", src=re.compile("champion"))]
        bans = [i.get("alt") for i in champs if i.parent and "black_link" in (i.parent.get("class") or [])]
        picks = [i.get("alt") for i in champs if not (i.parent and "black_link" in (i.parent.get("class") or []))]
        out[side + "_bans"] = bans[:5]
        out[side + "_picks"] = picks[:5]
        out[side + "_dragon_types"] = [i.get("alt") for i in block.find_all("img", src=re.compile(r"-dragon\.png"))]
    # players: two tables (blue, red); rows with a player link
    tabs = soup.find_all("table", class_="playersInfosLine")
    slot = 0
    for side, t in zip(("blue", "red"), tabs):
        for tr in t.find_all("tr"):
            a = tr.find("a", href=re.compile(r"/players/player-stats/(\d+)/"))
            champ = tr.find("img", src=re.compile("champion"))
            if not (a and champ):
                continue
            tds = tr.find_all("td", recursive=False)
            kda = next((td.get_text(strip=True) for td in tds if re.match(r"^\d+/\d+/\d+$", td.get_text(strip=True))), None)
            k = d = as_ = None
            if kda:
                k, d, as_ = (int(x) for x in kda.split("/"))
            out["players"].append({
                "player_id": int(re.search(r"/players/player-stats/(\d+)/", a["href"]).group(1)),
                "player": a.get_text(strip=True), "champion": champ.get("alt"),
                "side": side, "slot": slot, "kills": k, "deaths": d, "assists": as_,
                "team": out.get(side + "_team"),
            })
            slot += 1
    return out


def parse_fullstats(html):
    """{player_name: {stat_label: value_text}} in table column order (10 players)."""
    soup = BeautifulSoup(html, "lxml")
    t = soup.find("table", class_="completestats")
    if not t:
        return []
    rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            for tr in t.find_all("tr")]
    rows = [r for r in rows if len(r) >= 11]
    labels = [r[0] for r in rows]
    cols = []
    for j in range(1, 11):
        cols.append({labels[i]: rows[i][j] for i in range(len(rows)) if labels[i]})
    return cols


_ACTION_BY_IMG = [
    (r"kill-icon", "kill"), (r"Voidgrubs|grubs", "grubs"), (r"herald", "herald"),
    (r"nashor|baron", "baron"), (r"atakhan", "atakhan"), (r"tower", "tower"),
    (r"inhib", "inhib"), (r"nexus", "nexus"),
]


def parse_timeline(html):
    soup = BeautifulSoup(html, "lxml")
    events = []
    table = None
    for t in soup.find_all("table"):
        first = t.find("tr")
        if first and "Events" in first.get_text():
            table = t
            break
    if table is None:
        tabs = soup.find_all("table")
        table = tabs[1] if len(tabs) > 1 else None
    if table is not None:
        seq = 0
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 7:
                continue
            tm = _mmss(cells[0].get_text(strip=True))
            if tm is None:
                continue
            side_img = cells[1].find("img")
            side = ("blue" if side_img and "blue" in side_img.get("src", "") else
                    "red" if side_img else None)
            player = cells[2].get_text(strip=True) or None
            champ = (cells[3].find("img") or {}).get("alt") if cells[3].find("img") else None
            act_txt = cells[4].get_text(strip=True)
            act_img = cells[4].find("img")
            action, bounty = None, None
            if act_img:
                src = (act_img.get("src") or "") + " " + (act_img.get("alt") or "")
                dm = re.search(r"([a-z]+)-dragon", src)
                if dm:
                    action = "dragon:" + dm.group(1)
                else:
                    for pat, name in _ACTION_BY_IMG:
                        if re.search(pat, src, re.I):
                            action = name
                            break
                    action = action or src.split("/")[-1].split(".")[0]
            elif act_txt.upper() == "PLATE":
                action = "plate"
            elif act_txt.isdigit():
                action, bounty = "kill", int(act_txt)
            elif act_txt:
                action = act_txt.lower()
            tgt_img = cells[5].find("img")
            events.append({
                "seq": seq, "time_s": tm, "side": side, "player": player, "champion": champ,
                "action": action, "target": cells[6].get_text(" ", strip=True) or None,
                "target_champion": tgt_img.get("alt") if tgt_img else None, "bounty": bounty,
            })
            seq += 1
    # per-minute gold / cs series from the chart scripts
    series = {}
    for s in soup.find_all("script"):
        t = s.string or ""
        for var in ("golddatas", "csdatas"):
            m = re.search(r"var %s\s*=\s*\{(.*?)\};" % var, t, re.S)
            if not m:
                continue
            body = m.group(1)
            datasets = re.findall(r"data:\s*\[([^\]]*)\]", body)
            series[var] = [[_num(x) for x in ds.split(",") if x.strip()] for ds in datasets]
    return events, series


def fetch_build(game_id, player_id):
    txt = _fetch(BASE + "/game/ajax.build.php",
                 {"game_id": game_id, "player_id": player_id},
                 referer=BASE + "/game/stats/%d/page-builds/" % game_id,
                 lane="gol.gg:ajax")
    data = json.loads(txt)
    loadout = data[0][0] if data and data[0] else None
    events = data[1] if len(data) > 1 else []
    return loadout, events


# --------------------------------------------------------------- game fetch

def fetch_game(game_id, with_builds=True, inner=4):
    """Network-only: fetch and parse everything for one game.

    The three pages are fetched concurrently, then all ten build calls; the
    shared throttle lanes still pace requests globally, so this only overlaps
    server latency instead of paying it 13 times in sequence.
    """
    with ThreadPoolExecutor(max_workers=inner) as ex:
        f_game = ex.submit(_fetch, BASE + "/game/stats/%d/page-game/" % game_id)
        f_fs = ex.submit(_fetch, BASE + "/game/stats/%d/page-fullstats/" % game_id)
        f_tl = ex.submit(_fetch, BASE + "/game/stats/%d/page-timeline/" % game_id)
        g = parse_game_page(f_game.result())
        if not g.get("players"):
            raise ValueError("no players parsed for game %d" % game_id)
        builds = {}
        if with_builds:
            for p in g["players"]:
                builds[p["player_id"]] = ex.submit(fetch_build, game_id, p["player_id"])
        stats = parse_fullstats(f_fs.result())
        events, series = parse_timeline(f_tl.result())
        for i, p in enumerate(g["players"]):
            p["stats"] = stats[i] if i < len(stats) else {}
            p["role"] = p["stats"].get("Role")
            p["cs"] = _num(p["stats"].get("CS"))
            p["gold"] = _num(p["stats"].get("Golds"))
            p["level"] = _num(p["stats"].get("Level"))
            p["loadout"], p["build"] = None, []
            if with_builds:
                try:
                    p["loadout"], p["build"] = builds[p["player_id"]].result()
                except Exception as e:  # builds are optional; keep the game
                    log.warning("build fetch failed for game %d player %d: %s",
                                game_id, p["player_id"], e)
    g["events"], g["series"], g["game_id"] = events, series, game_id
    return g


def store_game(conn, g, match_id=None, game_num=None, trname=None, date=None):
    now = int(time.time())
    conn.execute("""
        INSERT INTO golgg_games (game_id, match_id, game_num, trname, date, patch,
            duration_s, blue_team, red_team, winner_side,
            blue_kills, blue_towers, blue_dragons, blue_barons, blue_gold,
            red_kills, red_towers, red_dragons, red_barons, red_gold,
            blue_first_blood, blue_first_pick, blue_bans, red_bans, blue_picks,
            red_picks, blue_dragon_types, red_dragon_types, fetched_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (game_id) DO UPDATE SET fetched_at = EXCLUDED.fetched_at,
            winner_side = EXCLUDED.winner_side, duration_s = EXCLUDED.duration_s""",
        (g["game_id"], match_id, game_num, trname, date, g.get("patch"),
         g.get("duration_s"), g.get("blue_team"), g.get("red_team"), g.get("winner_side"),
         g.get("blue_kills"), g.get("blue_towers"), g.get("blue_dragons"), g.get("blue_barons"), g.get("blue_gold"),
         g.get("red_kills"), g.get("red_towers"), g.get("red_dragons"), g.get("red_barons"), g.get("red_gold"),
         g.get("blue_first_blood"), g.get("blue_first_pick"),
         json.dumps(g.get("blue_bans")), json.dumps(g.get("red_bans")),
         json.dumps(g.get("blue_picks")), json.dumps(g.get("red_picks")),
         json.dumps(g.get("blue_dragon_types")), json.dumps(g.get("red_dragon_types")), now))
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO golgg_players (game_id, player_id, side, role, slot, player, team,
                champion, kills, deaths, assists, cs, gold, level, stats, loadout)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (game_id, player_id) DO UPDATE SET stats = EXCLUDED.stats,
                loadout = EXCLUDED.loadout""",
            [(g["game_id"], p["player_id"], p["side"], p.get("role"), p["slot"], p["player"],
              p.get("team"), p["champion"], p.get("kills"), p.get("deaths"), p.get("assists"),
              p.get("cs"), p.get("gold"), p.get("level"), json.dumps(p.get("stats")),
              json.dumps(p.get("loadout")))
             for p in g["players"]], returning=False)
        cur.execute("DELETE FROM golgg_events WHERE game_id = %s", (g["game_id"],))
        cur.executemany("""
            INSERT INTO golgg_events (game_id, seq, time_s, side, player, champion, action,
                target, target_champion, bounty)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(g["game_id"], e["seq"], e["time_s"], e["side"], e["player"], e["champion"],
              e["action"], e["target"], e["target_champion"], e["bounty"])
             for e in g["events"]], returning=False)
        gold = g["series"].get("golddatas") or []
        cs = g["series"].get("csdatas") or []
        rows = []
        for slot in range(min(10, len(gold))):
            for minute, val in enumerate(gold[slot]):
                c = cs[slot][minute] if slot < len(cs) and minute < len(cs[slot]) else None
                rows.append((g["game_id"], slot, minute, val, c))
        if rows:
            cur.execute("DELETE FROM golgg_timeline WHERE game_id = %s", (g["game_id"],))
            cur.executemany("""INSERT INTO golgg_timeline (game_id, slot, minute, gold, cs)
                               VALUES (%s,%s,%s,%s,%s)""", rows, returning=False)
        brows = []
        for p in g["players"]:
            for i, ev in enumerate(p.get("build") or []):
                brows.append((g["game_id"], p["player_id"], i, _num(ev.get("build_time")),
                              ev.get("typeEvent"), _num(ev.get("itemId"))))
        if brows:
            cur.execute("DELETE FROM golgg_builds WHERE game_id = %s", (g["game_id"],))
            cur.executemany("""INSERT INTO golgg_builds (game_id, player_id, seq, build_time,
                               event, item_id) VALUES (%s,%s,%s,%s,%s,%s)""",
                            brows, returning=False)
    conn.commit()


# ------------------------------------------------------------------- sync

def sync(conn, seasons=("S16",), regions=None, since=None, tournament=None,
         workers=3, limit=None, with_builds=True):
    """Discover tournaments -> matches -> games and scrape what's missing.

    Streaming: each tournament's games are submitted to the worker pool as soon
    as its matches are indexed, so results accumulate from the start instead of
    after a long indexing phase.  Fully resumable.
    """
    from concurrent.futures import wait, FIRST_COMPLETED
    ensure_schema(conn)
    now = int(time.time())
    regions = set(regions) if regions else None
    trs = []
    for season in seasons:
        for t in list_tournaments(season):
            if regions and t["region"] not in regions:
                continue
            if since and t["lastgame"] < since:
                continue
            if tournament and tournament.lower() not in t["trname"].lower():
                continue
            t["season"] = season
            trs.append(t)
            conn.execute("""INSERT INTO golgg_tournaments (trname, season, region, nbgames,
                               first_game, last_game, synced_at)
                            VALUES (%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (trname) DO UPDATE SET nbgames = EXCLUDED.nbgames,
                               last_game = EXCLUDED.last_game, synced_at = EXCLUDED.synced_at""",
                         (t["trname"], season, t["region"], int(t["nbgames"]),
                          t["firstgame"], t["lastgame"], now))
    conn.commit()
    # most recent tournaments first: they overlap our market coverage most
    trs.sort(key=lambda t: t["lastgame"], reverse=True)
    log.info("golgg: %d tournaments selected (%d games listed)",
             len(trs), sum(int(t["nbgames"]) for t in trs))
    have_games = {r["game_id"] for r in conn.execute("SELECT game_id FROM golgg_games")}
    known_matches = {r["match_id"]: r for r in conn.execute(
        "SELECT match_id, game_ids, score FROM golgg_matches")}

    import queue
    import threading
    from . import db as _db
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {}          # future -> (gid, mid, game_num, trname, date)
    stats = {"done": 0, "failed": 0, "submitted": 0, "matches": 0}
    t0 = time.time()
    last_log = [t0]
    work = queue.Queue(maxsize=workers * 6)
    SENTINEL = None
    stop = threading.Event()

    def indexer():
        """Walk tournaments -> matches -> game ids on its own DB connection."""
        iconn = _db.connect()
        try:
            for t in trs:
                if stop.is_set():
                    break
                try:
                    matches = list_matches(t["trname"])
                except Exception as e:
                    log.warning("match list failed for %s: %s", t["trname"], e)
                    continue
                matches = [m for m in matches if not (since and m["date"] and m["date"] < since)]
                # resolve game ids: known matches from the db, the rest via
                # summary pages fetched concurrently (the indexer was the bottleneck)
                need = [m for m in matches
                        if not (known_matches.get(m["match_id"])
                                and known_matches[m["match_id"]]["game_ids"]
                                and known_matches[m["match_id"]]["score"] == m["score"])]
                fetched = {}
                with ThreadPoolExecutor(max_workers=4) as ipool:
                    futs = {ipool.submit(match_game_ids, m["match_id"]): m["match_id"] for m in need}
                    for f in futs:
                        try:
                            fetched[futs[f]] = f.result()
                        except Exception as e:
                            log.warning("summary failed for match %d: %s", futs[f], e)
                for m in matches:
                    if stop.is_set():
                        break
                    km = known_matches.get(m["match_id"])
                    if km and km["game_ids"] and km["score"] == m["score"]:
                        gids = km["game_ids"]
                    else:
                        gids = fetched.get(m["match_id"])
                        if gids is None:
                            continue
                        iconn.execute("""INSERT INTO golgg_matches (match_id, trname, team1, team2,
                                           score, patch, date, game_ids, fetched_at)
                                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                        ON CONFLICT (match_id) DO UPDATE SET score = EXCLUDED.score,
                                           game_ids = EXCLUDED.game_ids, fetched_at = EXCLUDED.fetched_at""",
                                      (m["match_id"], t["trname"], m["team1"], m["team2"], m["score"],
                                       m["patch"], m["date"] or None, json.dumps(gids), now))
                        iconn.commit()
                    stats["matches"] += 1
                    for i, gid in enumerate(gids):
                        if gid in have_games:
                            continue
                        have_games.add(gid)
                        work.put((gid, m["match_id"], i + 1, t["trname"], m["date"] or None))
        except Exception:
            log.exception("golgg: indexer failed")
        finally:
            work.put(SENTINEL)
            try:
                iconn.close()
            except Exception:
                pass

    def drain(block=False):
        """Store finished games; optionally block until at least one finishes."""
        if not pending:
            return
        done_set, _ = wait(list(pending), timeout=None if block else 0,
                           return_when=FIRST_COMPLETED)
        for fut in done_set:
            gid, mid, gn, tr, dt = pending.pop(fut)
            try:
                store_game(conn, fut.result(), mid, gn, tr, dt)
                stats["done"] += 1
            except Exception:
                conn.rollback()
                stats["failed"] += 1
                log.exception("golgg: game %d failed", gid)

    def progress(force=False):
        if force or time.time() - last_log[0] >= 30:
            last_log[0] = time.time()
            n = stats["done"] + stats["failed"]
            el = max(1e-9, time.time() - t0)
            log.info("golgg: %d games scraped (%d failed), %d queued, %d matches indexed, "
                     "%.0f games/h", stats["done"], stats["failed"], len(pending),
                     stats["matches"], n / el * 3600)

    threading.Thread(target=indexer, name="golgg-indexer", daemon=True).start()
    try:
        while True:
            try:
                item = work.get(timeout=1.0)
            except queue.Empty:
                drain()
                progress()
                continue
            if item is SENTINEL:
                break
            if limit and stats["submitted"] >= limit:
                stop.set()
                continue
            pending[pool.submit(fetch_game, item[0], with_builds)] = item
            stats["submitted"] += 1
            while len(pending) > workers * 3:
                drain(block=True)
                progress()
            drain()
            progress()
        while pending:
            drain(block=True)
            progress()
        pool.shutdown()
    except (KeyboardInterrupt, SystemExit):
        log.info("golgg: interrupted — %d scraped, %d queued (progress saved)",
                 stats["done"], len(pending))
        stop.set()
        http.SHUTDOWN.set()
        pool.shutdown(wait=False, cancel_futures=True)
        conn.rollback()
        raise
    progress(force=True)
    log.info("golgg: done %d games (%d failed)", stats["done"], stats["failed"])
    return stats["done"]


def status(conn):
    """Progress summary from the database."""
    ensure_schema(conn)
    r = conn.execute("""SELECT
        (SELECT COUNT(*) FROM golgg_tournaments) AS tournaments,
        (SELECT COALESCE(SUM(nbgames),0) FROM golgg_tournaments) AS games_in_scope,
        (SELECT COUNT(*) FROM golgg_matches) AS matches,
        (SELECT COALESCE(SUM(jsonb_array_length(game_ids)),0) FROM golgg_matches) AS games_listed,
        (SELECT COUNT(*) FROM golgg_games) AS games,
        (SELECT COUNT(*) FROM golgg_players) AS players,
        (SELECT COUNT(*) FROM golgg_events) AS events,
        (SELECT COUNT(*) FROM golgg_timeline) AS timeline_rows,
        (SELECT COUNT(*) FROM golgg_builds) AS build_events,
        (SELECT MIN(fetched_at) FROM golgg_games) AS t0,
        (SELECT MAX(fetched_at) FROM golgg_games) AS t1,
        (SELECT MIN(date) FROM golgg_games) AS d0,
        (SELECT MAX(date) FROM golgg_games) AS d1""").fetchone()
    per_tr = conn.execute("""SELECT t.trname, t.region, t.nbgames, COUNT(g.game_id) AS done
        FROM golgg_tournaments t LEFT JOIN golgg_games g ON g.trname = t.trname
        GROUP BY t.trname, t.region, t.nbgames ORDER BY done DESC, t.nbgames DESC""").fetchall()
    conn.rollback()
    return dict(r), [dict(x) for x in per_tr]


def load_items(conn):
    """Item id -> name from Riot's Data Dragon (one request)."""
    ensure_schema(conn)
    versions = json.loads(_fetch("https://ddragon.leagueoflegends.com/api/versions.json"))
    ver = versions[0]
    data = json.loads(_fetch(
        "https://ddragon.leagueoflegends.com/cdn/%s/data/en_US/item.json" % ver))
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO golgg_items (item_id, name, version) VALUES (%s,%s,%s)
                           ON CONFLICT (item_id) DO UPDATE SET name = EXCLUDED.name,
                           version = EXCLUDED.version""",
                        [(int(k), v.get("name"), ver) for k, v in data["data"].items()],
                        returning=False)
    conn.commit()
    log.info("golgg: %d items loaded (ddragon %s)", len(data["data"]), ver)
    return len(data["data"])
