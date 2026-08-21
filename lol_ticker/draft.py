"""Draft analysis: join per-game pick data (Oracle's Elixir) to market odds.

For every per-map winner market we sample the team's win probability just
before champion select (game start − 16 min) and just after the game begins
(game start + 2 min, using the game's actual start time from OE).  The change
is the market's re-rating of the team after seeing the draft.

Tables:
  oe_games   one row per game: teams, winner, actual start time, league
  oe_picks   ten rows per game: team, champion, role, pick order
  draft_deltas  one row per (market, matched game): pre/post odds + result
"""
import csv
import logging
import re
import time
from datetime import datetime, timezone

log = logging.getLogger("draft")

PRE_OFFSET = 16 * 60    # sample this long before actual game start (pre-draft)
POST_OFFSET = 2 * 60    # and this long after (post-draft, minimal game info)
MATCH_WINDOW = 12 * 3600  # market series start vs game start tolerance

SCHEMA = """
CREATE TABLE IF NOT EXISTS oe_games (
    game_id     TEXT PRIMARY KEY,
    league      TEXT,
    date_utc    BIGINT,          -- actual game start (draft ends here)
    game_num    INT,
    patch       TEXT,
    blue_team   TEXT,
    red_team    TEXT,
    winner      TEXT
);
CREATE INDEX IF NOT EXISTS idx_oe_games_date ON oe_games (date_utc);

CREATE TABLE IF NOT EXISTS oe_picks (
    game_id     TEXT NOT NULL,
    team        TEXT NOT NULL,
    champion    TEXT NOT NULL,
    position    TEXT,
    pick_order  INT,
    PRIMARY KEY (game_id, team, champion)
);

CREATE TABLE IF NOT EXISTS oe_bans (
    game_id     TEXT NOT NULL,
    team        TEXT NOT NULL,      -- the team that banned
    champion    TEXT NOT NULL,
    ban_order   INT,
    PRIMARY KEY (game_id, team, champion)
);

-- ridge-regression coefficients: log-odds effect on a team's win prob
CREATE TABLE IF NOT EXISTS draft_model (
    feature     TEXT PRIMARY KEY,   -- e.g. own_pick:Ashe, syn:Ashe|Rumble
    coef        DOUBLE PRECISION,
    n           INT
);

CREATE TABLE IF NOT EXISTS draft_deltas (
    platform    TEXT NOT NULL,
    market_id   TEXT NOT NULL,
    oe_game_id  TEXT NOT NULL,
    team        TEXT,            -- OE team name (canonical)
    opponent    TEXT,
    league      TEXT,
    game_start  BIGINT,
    game_num    INT,
    pre_p       REAL,
    post_p      REAL,
    delta       REAL,
    won         INT,
    PRIMARY KEY (platform, market_id)
);
CREATE INDEX IF NOT EXISTS idx_dd_game ON draft_deltas (oe_game_id);
"""

_SUFFIXES = {"esports", "esport", "e-sports", "gaming", "team", "club", "gg"}


def norm_team(name):
    words = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower()).split()
    core = [w for w in words if w not in _SUFFIXES]
    return " ".join(core or words)


def ensure_schema(conn):
    conn.execute(SCHEMA)
    conn.commit()


# ------------------------------------------------------------------ OE load

def load_oe(conn, paths):
    """Load Oracle's Elixir yearly CSVs (team + player rows) into oe_*."""
    ensure_schema(conn)
    n_games = 0
    for path in paths:
        games = {}   # game_id -> row dict
        picks = []
        bans = []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                gid = r["gameid"]
                if not gid:
                    continue
                ts = _parse_dt(r["date"])
                g = games.setdefault(gid, {
                    "game_id": gid, "league": r["league"], "date_utc": ts,
                    "game_num": int(float(r["game"] or 0)),
                    "patch": r["patch"], "blue": None, "red": None, "winner": None,
                })
                if r["position"] == "team":
                    side = r["side"].lower()
                    g["blue" if side == "blue" else "red"] = r["teamname"]
                    if r["result"] == "1":
                        g["winner"] = r["teamname"]
                    for i in range(1, 6):
                        champ = (r.get("pick%d" % i) or "").strip()
                        if champ:
                            picks.append((gid, r["teamname"], champ, None, i))
                        ban = (r.get("ban%d" % i) or "").strip()
                        if ban:
                            bans.append((gid, r["teamname"], ban, i))
                elif r.get("champion"):
                    picks.append((gid, r["teamname"], r["champion"],
                                  r["position"], None))
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO oe_games (game_id, league, date_utc, game_num,
                       patch, blue_team, red_team, winner)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (game_id) DO UPDATE SET
                       winner = EXCLUDED.winner, date_utc = EXCLUDED.date_utc""",
                [(g["game_id"], g["league"], g["date_utc"], g["game_num"],
                  g["patch"], g["blue"], g["red"], g["winner"])
                 for g in games.values()], returning=False)
            # player rows carry the role; team rows carry pick order — merge
            cur.executemany(
                """INSERT INTO oe_picks (game_id, team, champion, position, pick_order)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (game_id, team, champion) DO UPDATE SET
                       position = COALESCE(EXCLUDED.position, oe_picks.position),
                       pick_order = COALESCE(EXCLUDED.pick_order, oe_picks.pick_order)""",
                picks, returning=False)
            cur.executemany(
                """INSERT INTO oe_bans (game_id, team, champion, ban_order)
                   VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                bans, returning=False)
        conn.commit()
        n_games += len(games)
        log.info("loaded %s: %d games", path, len(games))
    return n_games


def _parse_dt(s):
    try:
        return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                   .replace(tzinfo=timezone.utc).timestamp())
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------- market <-> OE join

_PM_GAME_RE = re.compile(r"Game (\d+) Winner", re.I)
_K_MAP_RE = re.compile(r"map (\d+)", re.I)


def _market_candidates(conn):
    """Per-map winner markets: (platform, market_id, team, series_start, map_n)."""
    out = []
    for r in conn.execute(
            """SELECT market_id, title, outcome, game_start_ts FROM markets
               WHERE platform = 'kalshi' AND series = 'KXLOLMAP'
                 AND status IN ('settled', 'finalized') AND result IN ('yes','no')
                 AND game_start_ts IS NOT NULL AND outcome IS NOT NULL"""):
        m = _K_MAP_RE.search(r["title"] or "")
        if m:
            out.append(("kalshi", r["market_id"], r["outcome"],
                        r["game_start_ts"], int(m.group(1))))
    for r in conn.execute(
            """SELECT market_id, title, outcome, game_start_ts FROM markets
               WHERE platform = 'polymarket' AND status = 'closed'
                 AND title ~* 'Game [0-9]+ Winner'
                 AND game_start_ts IS NOT NULL AND outcome NOT IN ('Yes','No')"""):
        m = _PM_GAME_RE.search(r["title"] or "")
        if m:
            out.append(("polymarket", r["market_id"], r["outcome"],
                        r["game_start_ts"], int(m.group(1))))
    return out


def _match_games(conn, candidates):
    """Match each market to an OE game by team, map number, and start time."""
    games = conn.execute(
        "SELECT * FROM oe_games WHERE winner IS NOT NULL AND date_utc IS NOT NULL"
    ).fetchall()
    by_key = {}
    for g in games:
        for team in (g["blue_team"], g["red_team"]):
            by_key.setdefault((norm_team(team), g["game_num"]), []).append(g)
    matched = []
    for platform, market_id, team, series_start, map_n in candidates:
        key = (norm_team(team), map_n)
        best, best_d = None, None
        # map N starts roughly (N-1) games after the series start
        expected = series_start + (map_n - 1) * 70 * 60
        for g in by_key.get(key, []):
            d = abs(g["date_utc"] - expected)
            if d <= MATCH_WINDOW and (best is None or d < best_d):
                best, best_d = g, d
        if best:
            canonical = (best["blue_team"]
                         if norm_team(best["blue_team"]) == key[0]
                         else best["red_team"])
            opponent = (best["red_team"] if canonical == best["blue_team"]
                        else best["blue_team"])
            matched.append((platform, market_id, best, canonical, opponent))
    return matched


def _price_at(conn, platform, market_id, ts, earliest=None):
    """Last known YES price at ts (None if none, or if older than earliest)."""
    if platform == "polymarket":
        row = conn.execute(
            """SELECT EXTRACT(EPOCH FROM ts)::bigint AS t, price AS p
               FROM price_points WHERE platform='polymarket' AND market_id = %s
                 AND ts <= to_timestamp(%s) ORDER BY ts DESC LIMIT 1""",
            (market_id, ts)).fetchone()
    else:
        row = conn.execute(
            """SELECT EXTRACT(EPOCH FROM ts)::bigint AS t,
                      COALESCE(((raw->'yes_bid'->>'close_dollars')::float
                                + (raw->'yes_ask'->>'close_dollars')::float) / 2,
                               close) AS p
               FROM candles WHERE platform='kalshi' AND market_id = %s
                 AND close IS NOT NULL AND ts <= to_timestamp(%s)
               ORDER BY ts DESC LIMIT 1""", (market_id, ts)).fetchone()
    if not row or row["p"] is None:
        return None
    if earliest is not None and row["t"] < earliest:
        return None
    return row["p"]


def build_deltas(conn):
    """(Re)compute draft_deltas for all matchable markets."""
    ensure_schema(conn)
    candidates = _market_candidates(conn)
    matched = _match_games(conn, candidates)
    log.info("draft: %d candidate markets, %d matched to OE games",
             len(candidates), len(matched))
    rows, skipped = [], 0
    for platform, market_id, g, team, opponent in matched:
        start = g["date_utc"]
        pre = _price_at(conn, platform, market_id, start - PRE_OFFSET)
        # post price must be fresh (from the draft window), not a stale carry
        post = _price_at(conn, platform, market_id, start + POST_OFFSET,
                         earliest=start - PRE_OFFSET)
        if pre is None or post is None:
            skipped += 1
            continue
        rows.append((platform, market_id, g["game_id"], team, opponent,
                     g["league"], start, g["game_num"], pre, post,
                     round(post - pre, 4), 1 if g["winner"] == team else 0))
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO draft_deltas (platform, market_id, oe_game_id, team,
                   opponent, league, game_start, game_num, pre_p, post_p,
                   delta, won)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (platform, market_id) DO UPDATE SET
                   pre_p = EXCLUDED.pre_p, post_p = EXCLUDED.post_p,
                   delta = EXCLUDED.delta, won = EXCLUDED.won,
                   oe_game_id = EXCLUDED.oe_game_id""",
            rows, returning=False)
    conn.commit()
    log.info("draft: stored %d deltas (%d skipped: no draft-window prices)",
             len(rows), skipped)
    return len(rows)


# ------------------------------------------------------------ draft model

import json
import math

MIN_PAIR_N = 25      # synergy / matchup pairs need this many games to get a feature
RIDGE_LAMBDA = 1000.0  # heavy shrinkage: best on both random and time-ordered holdouts
P_CLAMP = 0.02       # keep logit finite on near-certain markets


def _logit(p):
    p = min(1 - P_CLAMP, max(P_CLAMP, p))
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _training_rows(conn):
    """One row per (game, team): log-odds draft delta + own/enemy picks (with
    roles), bans, and the game's patch."""
    rows = conn.execute("""
        WITH per_game AS (
            SELECT oe_game_id, team, opponent, AVG(pre_p) AS pre_p, AVG(post_p) AS post_p
            FROM draft_deltas GROUP BY oe_game_id, team, opponent)
        SELECT g.oe_game_id, g.team, g.opponent, g.pre_p, g.post_p, og.patch,
               (SELECT json_agg(json_build_array(champion, position)) FROM oe_picks p
                 WHERE p.game_id = g.oe_game_id AND p.team = g.team) AS own_picks,
               (SELECT json_agg(json_build_array(champion, position)) FROM oe_picks p
                 WHERE p.game_id = g.oe_game_id AND p.team = g.opponent) AS enemy_picks,
               (SELECT array_agg(champion) FROM oe_bans b
                 WHERE b.game_id = g.oe_game_id AND b.team = g.team) AS own_bans,
               (SELECT array_agg(champion) FROM oe_bans b
                 WHERE b.game_id = g.oe_game_id AND b.team = g.opponent) AS enemy_bans
        FROM per_game g JOIN oe_games og ON og.game_id = g.oe_game_id""").fetchall()
    out = []
    for r in rows:
        if not r["own_picks"] or not r["enemy_picks"]:
            continue
        out.append({
            "y": _logit(r["post_p"]) - _logit(r["pre_p"]),
            "patch": r["patch"] or "",
            "own_picks": sorted({c for c, _ in r["own_picks"]}),
            "enemy_picks": sorted({c for c, _ in r["enemy_picks"]}),
            "own_roles": {c: (pos or "") for c, pos in r["own_picks"]},
            "enemy_roles": {c: (pos or "") for c, pos in r["enemy_picks"]},
            "own_bans": sorted(set(r["own_bans"] or [])),
            "enemy_bans": sorted(set(r["enemy_bans"] or [])),
        })
    return out


def _features(row):
    """Feature names active for one (team perspective) draft state.

    Hierarchical: a base indicator per champion, plus deviations for the
    champion in a role (own_pick:Ashe@sup) and in a patch (own_pick:Ashe#16.10).
    Ridge shrinks sparse deviations toward the base effect.
    """
    patch = row.get("patch") or ""
    own_roles = row.get("own_roles") or {}
    enemy_roles = row.get("enemy_roles") or {}
    f = []
    for c in row["own_picks"]:
        f.append("own_pick:" + c)
        if own_roles.get(c):
            f.append("own_pick:%s@%s" % (c, own_roles[c]))
        if patch:
            f.append("own_pick:%s#%s" % (c, patch))
    for c in row["enemy_picks"]:
        f.append("enemy_pick:" + c)
        if enemy_roles.get(c):
            f.append("enemy_pick:%s@%s" % (c, enemy_roles[c]))
        if patch:
            f.append("enemy_pick:%s#%s" % (c, patch))
    for c in row["own_bans"]:
        f.append("own_ban:" + c)
        if patch:
            f.append("own_ban:%s#%s" % (c, patch))
    for c in row["enemy_bans"]:
        f.append("enemy_ban:" + c)
        if patch:
            f.append("enemy_ban:%s#%s" % (c, patch))
    op = row["own_picks"]
    for i in range(len(op)):
        for j in range(i + 1, len(op)):
            a, b = sorted((op[i], op[j]))
            f.append("syn:%s|%s" % (a, b))
    for a in op:
        for b in row["enemy_picks"]:
            f.append("vs:%s|%s" % (a, b))
    return f


def _is_interaction(f):
    return (f.startswith("syn:") or f.startswith("vs:")
            or "@" in f or "#" in f)


def fit_model(conn, lam=RIDGE_LAMBDA):
    """Ridge regression of log-odds draft delta on draft-state indicators."""
    import numpy as np
    rows = _training_rows(conn)
    if len(rows) < 100:
        log.warning("draft model: only %d rows, skipping fit", len(rows))
        return 0
    counts = {}
    feats_per_row = []
    for r in rows:
        fs = _features(r)
        feats_per_row.append(fs)
        for f in fs:
            counts[f] = counts.get(f, 0) + 1
    # interaction features (pairs, role/patch deviations) need support;
    # base single-champion features are always kept
    keep = [f for f, n in counts.items()
            if n >= MIN_PAIR_N or not _is_interaction(f)]
    idx = {f: i for i, f in enumerate(sorted(keep))}
    X = np.zeros((len(rows), len(idx)), dtype=np.float64)
    y = np.array([r["y"] for r in rows])
    for i, fs in enumerate(feats_per_row):
        for f in fs:
            j = idx.get(f)
            if j is not None:
                X[i, j] = 1.0
    def ridge(Xm, ym):
        return np.linalg.solve(Xm.T @ Xm + lam * np.eye(Xm.shape[1]), Xm.T @ ym)

    def r2(Xm, ym, b):
        ss_res = float(np.sum((ym - Xm @ b) ** 2))
        ss_tot = float(np.sum((ym - ym.mean()) ** 2))
        return 1 - ss_res / ss_tot if ss_tot else 0.0

    # honest quality estimates: a fixed random 80/20 holdout, and a
    # time-ordered one (train on earlier patches, test on the newest two);
    # then refit on everything
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(rows))
    cut = int(0.8 * len(rows))
    tr, te = perm[:cut], perm[cut:]
    r2_out = r2(X[te], y[te], ridge(X[tr], y[tr]))
    patches = sorted({r["patch"] for r in rows if r["patch"]},
                     key=lambda x: [int(t) if t.isdigit() else 0 for t in x.split(".")])
    newest = set(patches[-2:])
    te_t = np.array([i for i, r in enumerate(rows) if r["patch"] in newest])
    tr_t = np.array([i for i, r in enumerate(rows) if r["patch"] not in newest])
    r2_time = (r2(X[te_t], y[te_t], ridge(X[tr_t], y[tr_t]))
               if len(te_t) > 50 and len(tr_t) > 50 else None)
    beta = ridge(X, y)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM draft_model")
        cur.executemany(
            "INSERT INTO draft_model (feature, coef, n) VALUES (%s, %s, %s)",
            [(f, float(beta[j]), counts[f]) for f, j in idx.items()] +
            [("__meta__", r2_out, len(rows)),
             ("__meta_time__", r2_time, len(te_t))], returning=False)
    conn.commit()
    log.info("draft model: %d rows, %d features, lambda=%g, holdout R^2 random=%.3f "
             "time-split=%s", len(rows), len(idx), lam, r2_out,
             "%.3f" % r2_time if r2_time is not None else "n/a")
    return len(idx)


def load_model(conn):
    rows = conn.execute("SELECT feature, coef, n FROM draft_model").fetchall()
    return {r["feature"]: (r["coef"], r["n"]) for r in rows}


# competitive draft order: (side, action) for the 20 steps
DRAFT_ORDER = (
    [("B", "ban"), ("R", "ban")] * 3 +
    [("B", "pick"), ("R", "pick"), ("R", "pick"), ("B", "pick"), ("B", "pick"), ("R", "pick")] +
    [("R", "ban"), ("B", "ban")] * 2 +
    [("R", "pick"), ("B", "pick"), ("B", "pick"), ("R", "pick")]
)


def predict_logodds(model, state, patch=""):
    """Blue's predicted log-odds draft delta for a partial state, plus the
    list of (feature, coef, n) terms that contributed."""
    row = {"own_picks": sorted(state["B"]["pick"]), "enemy_picks": sorted(state["R"]["pick"]),
           "own_roles": state["B"]["role"], "enemy_roles": state["R"]["role"],
           "own_bans": sorted(state["B"]["ban"]), "enemy_bans": sorted(state["R"]["ban"]),
           "patch": patch}
    total, terms = 0.0, []
    for f in _features(row):
        if f in model:
            total += model[f][0]
            terms.append((f, model[f][0], model[f][1]))
    return total, terms


def simulate(model, pre_blue, actions, patch=""):
    """Walk a sequence of (side, action, champion, role); return per-step
    estimates.  Role may be '' (base champion effect only)."""
    state = {"B": {"ban": [], "pick": [], "role": {}}, "R": {"ban": [], "pick": [], "role": {}}}
    base = _logit(pre_blue)
    prev_p = pre_blue
    steps = []
    for k, act in enumerate(actions):
        side, action, champ = act[0], act[1], act[2]
        role = act[3] if len(act) > 3 else ""
        if not champ:
            continue
        state[side][action].append(champ)
        if action == "pick" and role:
            state[side]["role"][champ] = role
        lo, terms = predict_logodds(model, state, patch)
        p = _sigmoid(base + lo)
        new_terms = [(f, round(c, 4), n) for f, c, n in terms if champ in f]
        steps.append({
            "step": k + 1, "side": side, "action": action, "champion": champ,
            "role": role, "dp_blue": round(p - prev_p, 4), "p_blue": round(p, 4),
            "known": any(champ in f for f, _, _ in terms),
            "terms": sorted(new_terms, key=lambda t: -abs(t[1]))[:6],
        })
        prev_p = p
    return {"pre_blue": pre_blue, "p_blue": round(prev_p, 4), "patch": patch,
            "steps": steps, "order": DRAFT_ORDER}


def model_patches(model):
    """Patches present in the model (from #patch features), newest first."""
    ps = {f.split("#", 1)[1] for f in model if "#" in f}
    return sorted(ps, key=lambda x: [int(t) if t.isdigit() else t for t in x.split(".")],
                  reverse=True)
