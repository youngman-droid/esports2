"""Align gol.gg in-game timelines with market odds; measure event impact.

Pipeline:
  link_games()   golgg_games <-> oe_games (teams + date + game number), which
                 connects each scraped game to its per-map winner markets via
                 draft_deltas.
  align_game()   for one (game, market): estimate wall-clock game start/end,
                 detect pauses, and attribute an odds swing to every event.
  build_all()    persist alignments + per-event swings for every linked game.

Alignment idea: the odds series (trade tape, plus 1-min points) jumps when
something happens.  Starting from the OE start time (or terminal-move time
minus game duration), we walk the significant events in order, look for the
largest odds jump near each event's expected wall time, and keep a running
clock offset.  A persistent step in that offset is a pause (the in-game clock
stopped while wall clock ran).  The end is the nexus / terminal move.
"""
import json
import logging
import statistics
import time

from . import draft

log = logging.getLogger("align")

SCHEMA = (
    'ALTER TABLE golgg_games ADD COLUMN IF NOT EXISTS oe_game_id TEXT;',
    'CREATE INDEX IF NOT EXISTS idx_golgg_games_oe ON golgg_games (oe_game_id);',
    """CREATE TABLE IF NOT EXISTS game_alignment (
        game_id     INT NOT NULL,          -- golgg game id
        platform    TEXT NOT NULL,
        market_id   TEXT NOT NULL,
        team        TEXT,                  -- team the market refers to (YES side)
        team_side   TEXT,                  -- 'blue' | 'red'
        start_wall  BIGINT,                -- wall clock at in-game 0:00
        end_wall    BIGINT,                -- wall clock at nexus / terminal move
        duration_s  INT,
        pauses      JSONB,                 -- [{game_time_s, wall_ts, length_s}]
        n_events    INT, n_matched INT,
        quality     REAL,                  -- matched / significant events
        built_at    BIGINT,
        PRIMARY KEY (game_id, platform, market_id)
    );""",
    """CREATE TABLE IF NOT EXISTS event_odds (
        game_id     INT NOT NULL, platform TEXT NOT NULL, market_id TEXT NOT NULL,
        seq         INT NOT NULL,
        time_s      INT, wall_ts BIGINT,
        action      TEXT, side TEXT, player TEXT, target TEXT,
        p_before    REAL, p_after REAL,
        dp          REAL,                  -- market team's odds change
        dp_actor    REAL,                  -- swing toward the acting side (+ = helped actor)
        matched     BOOLEAN,               -- event was anchored to an observed jump
        PRIMARY KEY (game_id, platform, market_id, seq)
    );""",
    'CREATE INDEX IF NOT EXISTS idx_event_odds_action ON event_odds (action);',
)

SIGNIFICANT = {"kill", "baron", "herald", "atakhan", "tower", "inhib", "nexus",
               "grubs"}  # dragons matched via prefix
BEFORE_S, AFTER_S = 20, 60      # odds window around an event
SEARCH_BACK, SEARCH_FWD = 75, 150  # seconds around expected wall time
JUMP_MIN = 0.015                # minimum |dp| to count as an observed jump
PAUSE_MIN = 90                  # residual step above the lag baseline that counts as a pause


def ensure_schema(conn):
    from . import db
    db.apply_schema(conn, SCHEMA)


# ------------------------------------------------------------------ linking

def link_games(conn):
    """Fill golgg_games.oe_game_id by team names + date (+/-1d) + game number."""
    ensure_schema(conn)
    oe = conn.execute("""SELECT game_id, date_utc, game_num, blue_team, red_team
                         FROM oe_games WHERE date_utc IS NOT NULL""").fetchall()
    idx = {}
    for g in oe:
        day = g["date_utc"] // 86400
        key = (frozenset((draft.norm_team(g["blue_team"]), draft.norm_team(g["red_team"]))),
               g["game_num"])
        for d in (day - 1, day, day + 1):
            idx.setdefault((key, d), []).append(g)
    rows = conn.execute("""SELECT game_id, date, game_num, blue_team, red_team
                           FROM golgg_games WHERE oe_game_id IS NULL AND date IS NOT NULL""").fetchall()
    n = 0
    for r in rows:
        day = int(time.mktime(r["date"].timetuple())) // 86400
        key = (frozenset((draft.norm_team(r["blue_team"]), draft.norm_team(r["red_team"]))),
               r["game_num"])
        cands = idx.get((key, day), [])
        if len(cands) >= 1:
            # nearest by day if several (same teams twice in window is rare)
            best = min(cands, key=lambda g: abs(g["date_utc"] // 86400 - day))
            conn.execute("UPDATE golgg_games SET oe_game_id = %s WHERE game_id = %s",
                         (best["game_id"], r["game_id"]))
            n += 1
    conn.commit()
    log.info("align: linked %d gol.gg games to OE games (%d unlinked remain)",
             n, len(rows) - n)
    return n


def link_one(conn, game_id):
    """Link a single gol.gg game to its OE game (same rule as link_games)."""
    r = conn.execute("""SELECT game_id, date, game_num, blue_team, red_team, oe_game_id
                        FROM golgg_games WHERE game_id = %s""", (game_id,)).fetchone()
    if not r or r["oe_game_id"] or not r["date"]:
        return r["oe_game_id"] if r else None
    day = int(time.mktime(r["date"].timetuple())) // 86400
    teams = (draft.norm_team(r["blue_team"]), draft.norm_team(r["red_team"]))
    cands = conn.execute("""SELECT game_id, date_utc, blue_team, red_team FROM oe_games
                            WHERE game_num = %s AND date_utc BETWEEN %s AND %s""",
                         (r["game_num"], (day - 1) * 86400, (day + 2) * 86400)).fetchall()
    best = None
    for g in cands:
        if frozenset((draft.norm_team(g["blue_team"]), draft.norm_team(g["red_team"]))) == frozenset(teams):
            if best is None or abs(g["date_utc"] // 86400 - day) < abs(best["date_utc"] // 86400 - day):
                best = g
    if best:
        conn.execute("UPDATE golgg_games SET oe_game_id = %s WHERE game_id = %s",
                     (best["game_id"], game_id))
        conn.commit()
        return best["game_id"]
    return None


def ensure_aligned(conn, game_id):
    """Compute + store alignments for a game on demand (link first if needed)."""
    oe = link_one(conn, game_id)
    if not oe:
        return []
    have = {(r["platform"], r["market_id"]) for r in conn.execute(
        "SELECT platform, market_id FROM game_alignment WHERE game_id = %s", (game_id,))}
    made = []
    for m in linked_markets(conn, game_id):
        if (m["platform"], m["market_id"]) in have:
            continue
        try:
            a = align_game(conn, game_id, m["platform"], m["market_id"])
        except Exception:
            log.exception("align on demand failed for %s", game_id)
            conn.rollback()
            a = None
        if a:
            store_alignment(conn, a)
            conn.commit()
            made.append((m["platform"], m["market_id"]))
    return made


def markets_for_game(conn, game_id):
    """Per-map winner markets for a gol.gg game, resolved directly from the
    markets catalog via its OE game (teams + map number + start-time window),
    independent of whether draft_deltas captured a pre/post pair for them."""
    g = conn.execute("""SELECT g.game_id, g.game_num, g.blue_team, g.red_team, g.duration_s,
                               og.date_utc, og.game_id AS oe_game_id
                        FROM golgg_games g JOIN oe_games og ON og.game_id = g.oe_game_id
                        WHERE g.game_id = %s""", (game_id,)).fetchone()
    if not g or not g["date_utc"]:
        return []
    teams = {draft.norm_team(g["blue_team"]), draft.norm_team(g["red_team"])}
    n = g["game_num"]
    t0, t1 = g["date_utc"] - 14 * 3600, g["date_utc"] + 4 * 3600
    rows = conn.execute("""
        SELECT platform, market_id, outcome, event_id, game_start_ts, title FROM markets
        WHERE game_start_ts BETWEEN %s AND %s AND outcome IS NOT NULL
          AND ((platform = 'kalshi' AND series = 'KXLOLMAP' AND title ILIKE %s)
               OR (platform = 'polymarket' AND title ~* %s))""",
        (t0, t1, "%% map %d %%" % n, "Game %d Winner" % n)).fetchall()
    out = []
    for r in rows:
        if draft.norm_team(r["outcome"]) not in teams:
            continue
        out.append({"platform": r["platform"], "market_id": r["market_id"],
                    "team": r["outcome"], "event_id": r["event_id"],
                    "game_start": r["game_start_ts"], "blue_team": g["blue_team"],
                    "red_team": g["red_team"], "duration_s": g["duration_s"],
                    "date_utc": g["date_utc"]})
    return out


def linked_markets(conn, game_id):
    """Markets for a gol.gg game: draft_deltas links plus direct resolution."""
    rows = [dict(r) for r in conn.execute("""
        SELECT d.platform, d.market_id, d.team, d.game_start, g.blue_team, g.red_team,
               g.duration_s, og.date_utc
        FROM golgg_games g
        JOIN oe_games og ON og.game_id = g.oe_game_id
        JOIN draft_deltas d ON d.oe_game_id = g.oe_game_id
        WHERE g.game_id = %s""", (game_id,))]
    seen = {(r["platform"], r["market_id"]) for r in rows}
    for m in markets_for_game(conn, game_id):
        if (m["platform"], m["market_id"]) not in seen:
            rows.append(m)
            seen.add((m["platform"], m["market_id"]))
    return rows


# --------------------------------------------------------------- odds series

def odds_series(conn, platform, market_id, t0, t1):
    """Sorted [(ts, p)] from trades (exact seconds) merged with 1-min series."""
    pts = []
    for r in conn.execute("""SELECT EXTRACT(EPOCH FROM ts)::bigint AS t, price AS p FROM trades
                             WHERE platform=%s AND market_id=%s
                               AND ts BETWEEN to_timestamp(%s) AND to_timestamp(%s)""",
                          (platform, market_id, t0, t1)):
        if r["p"] is not None:
            pts.append((r["t"], float(r["p"])))
    if platform == "polymarket":
        q = """SELECT EXTRACT(EPOCH FROM ts)::bigint AS t, price AS p FROM price_points
               WHERE platform=%s AND market_id=%s AND ts BETWEEN to_timestamp(%s) AND to_timestamp(%s)"""
    else:
        q = """SELECT EXTRACT(EPOCH FROM ts)::bigint AS t,
                      COALESCE(((raw->'yes_bid'->>'close_dollars')::float
                                + (raw->'yes_ask'->>'close_dollars')::float)/2, close) AS p
               FROM candles WHERE platform=%s AND market_id=%s AND period_min = 1
                 AND ts BETWEEN to_timestamp(%s) AND to_timestamp(%s)"""
    for r in conn.execute(q, (platform, market_id, t0, t1)):
        if r["p"] is not None:
            pts.append((r["t"], float(r["p"])))
    # L2 mids at 5s when recorded live
    for r in conn.execute("""SELECT EXTRACT(EPOCH FROM ts)::bigint AS t,
                                    ((bids->0->>0)::float + (asks->0->>0)::float)/2 AS p
                             FROM book_snapshots WHERE platform=%s AND market_id=%s
                               AND ts BETWEEN to_timestamp(%s) AND to_timestamp(%s)
                               AND jsonb_array_length(bids) > 0 AND jsonb_array_length(asks) > 0""",
                          (platform, market_id, t0, t1)):
        if r["p"] is not None:
            pts.append((r["t"], float(r["p"])))
    pts.sort()
    return pts


def _price_at(series, ts, side="before"):
    """Last price at/before ts (side='before') or first at/after (side='after')."""
    if not series:
        return None
    lo, hi = 0, len(series)
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] < ts:
            lo = mid + 1
        else:
            hi = mid
    if side == "before":
        return series[lo - 1][1] if lo > 0 else series[0][1]
    return series[lo][1] if lo < len(series) else series[-1][1]


def _jump(series, ts):
    a = _price_at(series, ts - BEFORE_S, "before")
    b = _price_at(series, ts + AFTER_S, "before")
    return (b - a) if a is not None and b is not None else 0.0


def _terminal_time(series, t_from, strict=False):
    """First time after t_from the price goes and stays beyond the terminal
    band (0.95/0.05, or 0.985/0.015 when strict)."""
    hi, lo, keep = (0.985, 0.015, 8) if strict else (0.95, 0.05, 6)
    for i, (t, p) in enumerate(series):
        if t < t_from:
            continue
        if p >= hi or p <= lo:
            rest = [q for _, q in series[i:i + keep]]
            if len(rest) >= 3 and (all(q >= hi - 0.03 for q in rest) or all(q <= lo + 0.03 for q in rest)):
                return t
    return None


# ---------------------------------------------------------------- alignment

def align_game(conn, game_id, platform, market_id):
    g = conn.execute("SELECT * FROM golgg_games WHERE game_id = %s", (game_id,)).fetchone()
    link = next((m for m in linked_markets(conn, game_id)
                 if m["platform"] == platform and m["market_id"] == market_id), None)
    if not g or not link:
        return None
    team_side = ("blue" if draft.norm_team(link["team"]) == draft.norm_team(g["blue_team"])
                 else "red")
    duration = g["duration_s"] or 0
    events = [dict(r) for r in conn.execute(
        "SELECT * FROM golgg_events WHERE game_id = %s ORDER BY seq", (game_id,))]
    start0 = link["date_utc"]  # OE actual start (best prior)
    series = odds_series(conn, platform, market_id, start0 - 1800, start0 + duration + 3 * 3600)
    if len(series) < 10:
        return None
    start_est = start0
    offset = 0.0                  # wall = start_est + time_s + offset (pauses)
    lag = None                    # market reaction lag baseline (s)
    pauses = []
    matched = 0
    recent = []
    n_sig = 0
    out_events = []
    anchors = set()
    for e in events:
        t = e["time_s"]
        sig = (e["action"] in SIGNIFICANT or (e["action"] or "").startswith("dragon"))
        expected = start_est + t + offset
        wall = expected + (lag or 0)
        is_matched = False
        if sig and e["action"] != "nexus":
            n_sig += 1
            cands = []
            for cand in range(int(expected - SEARCH_BACK), int(expected + SEARCH_FWD) + 1, 5):
                cands.append((abs(_jump(series, cand)), cand))
            best_j = max(j for j, _ in cands) if cands else 0.0
            if best_j >= JUMP_MIN:
                # the window [ts-BEFORE, ts+AFTER] brackets a move over a range of
                # ts; take that range's midpoint and shift by the window asymmetry
                top = sorted(c for j, c in cands if j >= 0.9 * best_j)
                best_ts = top[len(top) // 2] + (AFTER_S - BEFORE_S) // 2
                is_matched = True
                matched += 1
                anchors.add(best_ts // 30)
                resid = best_ts - expected
                recent.append(resid)
                recent = recent[-5:]
                med = statistics.median(recent)
                if matched <= 3 and len(recent) >= 2 and med <= -45:
                    # early events consistently before expectation: the prior
                    # start time is late — shift the start, not the drift
                    start_est += med
                    recent = []
                    wall = start_est + t + offset
                elif len(recent) >= 3:
                    # market reaction lag is a roughly constant positive residual;
                    # learn it as a baseline, and call a pause only on a step
                    # well above it that persists
                    if lag is None:
                        lag = med
                    adj = med - lag
                    if adj >= PAUSE_MIN:
                        pauses.append({"game_time_s": t, "wall_ts": int(expected + lag),
                                       "length_s": int(adj)})
                        offset += adj
                        recent = []
                    else:
                        lag = lag + 0.25 * adj     # slow baseline tracking
                    wall = start_est + t + offset + lag
                else:
                    wall = best_ts
        out_events.append((e, wall, is_matched))
    # end of game.  A strict terminal move (0.99/0.01, persistent) is the most
    # reliable end marker when present; its distance from start+duration is
    # the total paused time, which sanity-checks the residual-based pauses.
    lag0 = lag or 0.0
    end_clock = start_est + duration + offset + lag0
    term = _terminal_time(series, start_est + duration - 240, strict=True)
    if term is not None and -120 <= term - (start_est + duration + lag0) <= 2400:
        total_pause = term - (start_est + duration + lag0)
        if total_pause < PAUSE_MIN:
            pauses = []           # no room for a real pause: residual wobble only
            offset = 0.0
        elif pauses:
            # scale detected pauses so they sum to the observed total
            tot = sum(p["length_s"] for p in pauses) or 1
            for p in pauses:
                p["length_s"] = int(round(p["length_s"] * total_pause / tot))
            offset = float(sum(p["length_s"] for p in pauses))
        end_est = term
    else:
        end_est = end_clock
    final_events = []
    for e, wall, is_matched in out_events:
        if e["action"] == "nexus":
            wall = end_est
        p_before = _price_at(series, wall - BEFORE_S, "before")
        p_after = _price_at(series, wall + AFTER_S, "before")
        dp = (p_after - p_before) if p_before is not None and p_after is not None else None
        actor_sign = 1 if e["side"] == team_side else (-1 if e["side"] else 0)
        final_events.append({
            "seq": e["seq"], "time_s": e["time_s"], "wall_ts": int(wall), "action": e["action"],
            "side": e["side"], "player": e["player"], "target": e["target"],
            "p_before": p_before, "p_after": p_after, "dp": dp,
            "dp_actor": (dp * actor_sign) if dp is not None and actor_sign else None,
            "matched": is_matched,
        })
    out_events = final_events
    # quality: share of significant events anchored to a distinct jump
    quality = round(min(1.0, len(anchors) / max(1, n_sig) * 2), 3) if n_sig else None
    return {
        "game_id": game_id, "platform": platform, "market_id": market_id,
        "team": link["team"], "team_side": team_side,
        "start_wall": int(start_est), "end_wall": int(end_est), "duration_s": duration,
        "pauses": pauses, "n_events": len(events), "n_matched": matched,
        "quality": quality,
        "events": out_events,
        "series": series,
    }


def store_alignment(conn, a):
    conn.execute("""
        INSERT INTO game_alignment (game_id, platform, market_id, team, team_side,
            start_wall, end_wall, duration_s, pauses, n_events, n_matched, quality, built_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (game_id, platform, market_id) DO UPDATE SET
            start_wall = EXCLUDED.start_wall, end_wall = EXCLUDED.end_wall,
            pauses = EXCLUDED.pauses, n_matched = EXCLUDED.n_matched,
            quality = EXCLUDED.quality, built_at = EXCLUDED.built_at""",
        (a["game_id"], a["platform"], a["market_id"], a["team"], a["team_side"],
         a["start_wall"], a["end_wall"], a["duration_s"], json.dumps(a["pauses"]),
         a["n_events"], a["n_matched"], a["quality"], int(time.time())))
    with conn.cursor() as cur:
        cur.execute("DELETE FROM event_odds WHERE game_id=%s AND platform=%s AND market_id=%s",
                    (a["game_id"], a["platform"], a["market_id"]))
        cur.executemany("""
            INSERT INTO event_odds (game_id, platform, market_id, seq, time_s, wall_ts,
                action, side, player, target, p_before, p_after, dp, dp_actor, matched)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(a["game_id"], a["platform"], a["market_id"], e["seq"], e["time_s"], e["wall_ts"],
              e["action"], e["side"], e["player"], e["target"], e["p_before"], e["p_after"],
              e["dp"], e["dp_actor"], e["matched"]) for e in a["events"]],
            returning=False)
    conn.commit()


def build_all(conn, rebuild=False):
    ensure_schema(conn)
    link_games(conn)
    pairs = conn.execute("""
        SELECT g.game_id, d.platform, d.market_id
        FROM golgg_games g JOIN draft_deltas d ON d.oe_game_id = g.oe_game_id
        %s""" % ("" if rebuild else """
        WHERE NOT EXISTS (SELECT 1 FROM game_alignment a WHERE a.game_id = g.game_id
                          AND a.platform = d.platform AND a.market_id = d.market_id)""")).fetchall()
    n = 0
    for r in pairs:
        try:
            a = align_game(conn, r["game_id"], r["platform"], r["market_id"])
            if a:
                store_alignment(conn, a)
                n += 1
        except Exception:
            conn.rollback()
            log.exception("align failed for %s/%s", r["game_id"], r["market_id"])
    log.info("align: built %d alignments (%d candidate pairs)", n, len(pairs))
    return n


def event_impact(conn, platform="", league="", min_n=5, phase=None):
    """Average odds swing toward the acting side per event type."""
    where = ["e.dp_actor IS NOT NULL"]
    params = {"minn": min_n}
    if platform:
        where.append("e.platform = %(plat)s")
        params["plat"] = platform
    if league:
        where.append("g.trname ILIKE %(lg)s")
        params["lg"] = "%" + league + "%"
    if phase == "early":
        where.append("e.time_s < 900")
    elif phase == "mid":
        where.append("e.time_s BETWEEN 900 AND 1500")
    elif phase == "late":
        where.append("e.time_s > 1500")
    rows = conn.execute("""
        SELECT e.action, COUNT(*) AS n,
               ROUND(AVG(e.dp_actor)::numeric, 4) AS mean_swing,
               ROUND(AVG(ABS(e.dp))::numeric, 4) AS mean_abs,
               ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY e.dp_actor)::numeric, 4) AS median_swing,
               ROUND(AVG(CASE WHEN e.matched THEN 1.0 ELSE 0.0 END)::numeric, 3) AS matched_share
        FROM event_odds e JOIN golgg_games g ON g.game_id = e.game_id
        WHERE %s
        GROUP BY e.action HAVING COUNT(*) >= %%(minn)s
        ORDER BY mean_swing DESC""" % " AND ".join(where), params).fetchall()
    return [dict(r) for r in rows]
