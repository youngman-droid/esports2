"""Odds-free model exploration: richer states, stronger priors, several model
classes, evaluated out-of-sample at the exact instants where market odds exist.

CLI:  python3 -m lol_ticker wpx prep   # item gold, OE players, sequential Elo/player ratings
      python3 -m lol_ticker wpx build  # state dataset -> data/wpx/states.npz
      python3 -m lol_ticker wpx eval   # model zoo, game-level CV, vs market
"""
import csv
import glob
import json
import logging
import math
import os
import time
import urllib.request

import numpy as np

from . import config

np.seterr(all="ignore")
log = logging.getLogger("wpx")
OUT_DIR = os.path.join(config.REPO_ROOT, "data", "wpx")

SCHEMA = """
ALTER TABLE golgg_items ADD COLUMN IF NOT EXISTS gold INT;
CREATE TABLE IF NOT EXISTS oe_players (
    game_id TEXT NOT NULL, team TEXT NOT NULL, position TEXT, player TEXT, player_id TEXT,
    PRIMARY KEY (game_id, team, position)
);
CREATE TABLE IF NOT EXISTS oe_ratings (
    game_id TEXT PRIMARY KEY,       -- OE game
    elo_blue REAL, elo_red REAL,    -- sequential team Elo before the game
    pelo_blue REAL, pelo_red REAL,  -- mean sequential player rating before the game
    n_known_blue INT, n_known_red INT
);
ALTER TABLE oe_ratings ADD COLUMN IF NOT EXISTS form_blue REAL;   -- last-10 win rate
ALTER TABLE oe_ratings ADD COLUMN IF NOT EXISTS form_red REAL;
ALTER TABLE oe_ratings ADD COLUMN IF NOT EXISTS ngames_blue INT;  -- games played so far
ALTER TABLE oe_ratings ADD COLUMN IF NOT EXISTS ngames_red INT;
ALTER TABLE oe_games ADD COLUMN IF NOT EXISTS blue_gold INT;
ALTER TABLE oe_games ADD COLUMN IF NOT EXISTS red_gold INT;
ALTER TABLE oe_games ADD COLUMN IF NOT EXISTS gamelength INT;
ALTER TABLE oe_ratings ADD COLUMN IF NOT EXISTS rapm_team REAL;    -- blue - red, gold/min margin units
ALTER TABLE oe_ratings ADD COLUMN IF NOT EXISTS rapm_player REAL;  -- blue - red, player ridge ratings summed
"""


def ensure_schema(conn):
    conn.execute(SCHEMA)
    conn.commit()


# ------------------------------------------------------------------ prep

def load_item_gold(conn):
    ver = json.load(urllib.request.urlopen(
        "https://ddragon.leagueoflegends.com/api/versions.json", timeout=30))[0]
    items = json.load(urllib.request.urlopen(
        "https://ddragon.leagueoflegends.com/cdn/%s/data/en_US/item.json" % ver, timeout=60))["data"]
    rows = [(int(k), v.get("name"), ver, int((v.get("gold") or {}).get("total") or 0))
            for k, v in items.items()]
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO golgg_items (item_id, name, version, gold) VALUES (%s,%s,%s,%s)
                           ON CONFLICT (item_id) DO UPDATE SET gold = EXCLUDED.gold,
                               name = COALESCE(golgg_items.name, EXCLUDED.name)""", rows, returning=False)
    conn.commit()
    log.info("wpx: item gold loaded for %d items (ddragon %s)", len(rows), ver)


def load_oe_players(conn, paths=None):
    paths = paths or sorted(glob.glob(os.path.join(config.REPO_ROOT, "data", "oe", "oe_*.csv")))
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["position"] in ("top", "jng", "mid", "bot", "sup") and r["gameid"]:
                    rows.append((r["gameid"], r["teamname"], r["position"],
                                 r.get("playername") or None, r.get("playerid") or None))
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO oe_players (game_id, team, position, player, player_id)
                           VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""", rows, returning=False)
    conn.commit()
    log.info("wpx: loaded %d OE player rows", len(rows))


def load_oe_margins(conn, paths=None):
    """Team total gold + game length per OE game (margin for RAPM)."""
    paths = paths or sorted(glob.glob(os.path.join(config.REPO_ROOT, "data", "oe", "oe_*.csv")))
    rows = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["position"] != "team" or not r["gameid"]:
                    continue
                g = rows.setdefault(r["gameid"], {"b": None, "r": None, "len": None})
                try:
                    gold = int(float(r.get("totalgold") or 0))
                except ValueError:
                    gold = None
                try:
                    g["len"] = int(float(r.get("gamelength") or 0)) or g["len"]
                except ValueError:
                    pass
                if r["side"].lower() == "blue":
                    g["b"] = gold
                else:
                    g["r"] = gold
    with conn.cursor() as cur:
        cur.executemany("UPDATE oe_games SET blue_gold=%s, red_gold=%s, gamelength=%s WHERE game_id=%s",
                        [(g["b"], g["r"], g["len"], gid) for gid, g in rows.items()], returning=False)
    conn.commit()
    log.info("wpx: margins loaded for %d OE games", len(rows))


def build_rapm(conn, half_life_days=150.0, lam_team=2.0, lam_player=8.0, refit_days=30):
    """Time-aware margin RAPM: for each calendar window, ridge-regress the
    per-minute gold margin of every EARLIER game (exp-decayed) on team (+1 blue,
    -1 red) and on player indicators; store the rating difference for the
    games in that window.  Leak-free by construction."""
    games = conn.execute("""SELECT game_id, blue_team, red_team, date_utc, blue_gold, red_gold, gamelength
                            FROM oe_games WHERE date_utc IS NOT NULL AND blue_gold IS NOT NULL
                              AND red_gold IS NOT NULL AND gamelength IS NOT NULL AND gamelength > 600
                            ORDER BY date_utc, game_id""").fetchall()
    players = {}
    for r in conn.execute("SELECT game_id, team, player_id, player FROM oe_players"):
        players.setdefault(r["game_id"], {}).setdefault(r["team"], []).append(r["player_id"] or r["player"])
    margins = np.array([(g["blue_gold"] - g["red_gold"]) / (g["gamelength"] / 60.0) for g in games])
    dates = np.array([g["date_utc"] for g in games], dtype=np.float64)
    teams = sorted({g["blue_team"] for g in games} | {g["red_team"] for g in games})
    tid = {t: i for i, t in enumerate(teams)}
    pl_ids = sorted({p for g in games for tm in players.get(g["game_id"], {}).values() for p in tm if p})
    pid = {p: i for i, p in enumerate(pl_ids)}
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr
    # design rows
    tr_rows, tr_cols, tr_vals = [], [], []
    pr_rows, pr_cols, pr_vals = [], [], []
    for i, g in enumerate(games):
        tr_rows += [i, i]; tr_cols += [tid[g["blue_team"]], tid[g["red_team"]]]; tr_vals += [1.0, -1.0]
        for p in players.get(g["game_id"], {}).get(g["blue_team"], []):
            if p: pr_rows.append(i); pr_cols.append(pid[p]); pr_vals.append(1.0)
        for p in players.get(g["game_id"], {}).get(g["red_team"], []):
            if p: pr_rows.append(i); pr_cols.append(pid[p]); pr_vals.append(-1.0)
    T = csr_matrix((tr_vals, (tr_rows, tr_cols)), shape=(len(games), len(teams)))
    P = csr_matrix((pr_vals, (pr_rows, pr_cols)), shape=(len(games), len(pl_ids)))
    out = {}
    t_start = dates[0]
    step = refit_days * 86400
    cur_t = t_start + step
    n_windows = 0
    while cur_t <= dates[-1] + step:
        window = (dates >= cur_t - step) & (dates < cur_t)
        prior = dates < cur_t - step
        if window.any() and prior.sum() >= 200:
            w = np.exp(-(cur_t - step - dates[prior]) / 86400.0 / half_life_days * math.log(2))
            sw = np.sqrt(w)
            # weighted ridge via augmented least squares (lsqr with damp)
            Tw = T[prior].multiply(sw[:, None]).tocsr(); yw = margins[prior] * sw
            beta_t = lsqr(Tw, yw, damp=math.sqrt(lam_team))[0]
            Pw = P[prior].multiply(sw[:, None]).tocsr()
            beta_p = lsqr(Pw, yw, damp=math.sqrt(lam_player))[0]
            for i in np.where(window)[0]:
                g = games[i]
                rt = beta_t[tid[g["blue_team"]]] - beta_t[tid[g["red_team"]]]
                bp = [beta_p[pid[p]] for p in players.get(g["game_id"], {}).get(g["blue_team"], []) if p]
                rp = [beta_p[pid[p]] for p in players.get(g["game_id"], {}).get(g["red_team"], []) if p]
                out[g["game_id"]] = (float(rt), float(sum(bp) - sum(rp)))
            n_windows += 1
        cur_t += step
    with conn.cursor() as cur:
        cur.executemany("UPDATE oe_ratings SET rapm_team=%s, rapm_player=%s WHERE game_id=%s",
                        [(v[0], v[1], k) for k, v in out.items()], returning=False)
    conn.commit()
    log.info("wpx: RAPM ratings for %d games over %d windows (%d teams, %d players)",
             len(out), n_windows, len(teams), len(pl_ids))


def build_ratings(conn, k_team=30.0, k_player=24.0, base=1500.0):
    """Sequential (leak-free) team Elo and player ratings over all OE games,
    recorded as the pre-game values for each game."""
    games = conn.execute("""SELECT game_id, blue_team, red_team, winner, date_utc FROM oe_games
                            WHERE winner IS NOT NULL AND date_utc IS NOT NULL
                            ORDER BY date_utc, game_id""").fetchall()
    players = {}
    for r in conn.execute("SELECT game_id, team, player_id, player FROM oe_players"):
        players.setdefault(r["game_id"], {}).setdefault(r["team"], []).append(r["player_id"] or r["player"])
    team_r, pl_r = {}, {}
    hist = {}   # team -> recent results (1/0), most recent last
    out = []
    for g in games:
        b, rd = g["blue_team"], g["red_team"]
        eb, er = team_r.get(b, base), team_r.get(rd, base)
        hb, hr = hist.get(b, []), hist.get(rd, [])
        fb = sum(hb[-10:]) / len(hb[-10:]) if hb else 0.5
        fr = sum(hr[-10:]) / len(hr[-10:]) if hr else 0.5
        pb = [pl_r.get(p, base) for p in players.get(g["game_id"], {}).get(b, []) if p]
        pr = [pl_r.get(p, base) for p in players.get(g["game_id"], {}).get(rd, []) if p]
        mpb = sum(pb) / len(pb) if pb else base
        mpr = sum(pr) / len(pr) if pr else base
        out.append((g["game_id"], eb, er, mpb, mpr, len(pb), len(pr), fb, fr, len(hb), len(hr)))
        sb = 1.0 if g["winner"] == b else 0.0
        hist.setdefault(b, []).append(sb); hist.setdefault(rd, []).append(1 - sb)
        exp_t = 1 / (1 + 10 ** ((er - eb) / 400))
        team_r[b] = eb + k_team * (sb - exp_t)
        team_r[rd] = er + k_team * ((1 - sb) - (1 - exp_t))
        exp_p = 1 / (1 + 10 ** ((mpr - mpb) / 400))
        for p in players.get(g["game_id"], {}).get(b, []):
            if p:
                pl_r[p] = pl_r.get(p, base) + k_player * (sb - exp_p)
        for p in players.get(g["game_id"], {}).get(rd, []):
            if p:
                pl_r[p] = pl_r.get(p, base) + k_player * ((1 - sb) - (1 - exp_p))
    with conn.cursor() as cur:
        cur.execute("DELETE FROM oe_ratings")
        cur.executemany("""INSERT INTO oe_ratings (game_id, elo_blue, elo_red, pelo_blue, pelo_red,
                           n_known_blue, n_known_red, form_blue, form_red, ngames_blue, ngames_red)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", out, returning=False)
    conn.commit()
    log.info("wpx: ratings for %d OE games (%d teams, %d players)", len(out), len(team_r), len(pl_r))


def prep(conn, skip_static=False):
    ensure_schema(conn)
    if not skip_static:
        try:
            load_item_gold(conn)
        except Exception as e:
            log.warning("item gold load failed: %s", e)
        load_oe_players(conn)
        build_ratings(conn)
    load_oe_margins(conn)
    build_rapm(conn)
    # draft-model term for gol.gg games, fitted WITHOUT those games (leak-free)
    from . import draft
    excl = {r["oe_game_id"] for r in conn.execute(
        "SELECT oe_game_id FROM golgg_games WHERE oe_game_id IS NOT NULL")}
    draft.fit_outcome_model(conn, exclude=excl, table="draft_outcome_oos")


# ----------------------------------------------------------------- states

ROLE_SLOTS = ["top", "jng", "mid", "bot", "sup"]
DRAGON_ELEMS = ["fire", "mountain", "ocean", "cloud", "hextech", "chemtech"]
COUNT_KEYS = ["kill", "tower", "dragon", "baron", "inhib", "herald", "grubs", "atakhan", "plate", "elder"]

FEATURE_NAMES = (
    ["bias", "t", "t2", "elo_oe", "pelo_oe", "draft", "elo_gg"] +
    ["gold_k", "gold_k_x_t", "gold_mom", "cs_k"] +
    ["gold_%s" % r for r in ROLE_SLOTS] +
    ["d_" + k for k in COUNT_KEYS] +
    ["d_kill_x_t", "soul", "baron_active", "kills_2m", "items_done", "item_gold_k",
     "drag_blue", "drag_red", "towers_blue", "towers_red", "inhib_blue", "inhib_red",
     "dead_blue", "dead_red", "nexus_tw_blue", "nexus_tw_red", "gold_rel", "gold_k_x_t2",
     "lead_x_inhib", "elder_buff", "t_since_kill", "form_diff", "exp_diff",
     "baron_up", "dragon_up", "baron_up_x_dead", "dead_diff_x_t", "rapm_team", "rapm_player",
     "hp_pool", "hp_low_b", "hp_low_r", "lvl_k", "has_hp"]
)


def _load_game(conn, gid):
    g = conn.execute("""SELECT g.*, r.elo_blue AS oe_elo_b, r.elo_red AS oe_elo_r,
                               r.pelo_blue AS oe_pelo_b, r.pelo_red AS oe_pelo_r,
                               r.form_blue, r.form_red, r.ngames_blue, r.ngames_red,
                               r.rapm_team, r.rapm_player,
                               dg.p_full AS draft_p_full, dg.p_elo AS draft_p_elo
                        FROM golgg_games g
                        LEFT JOIN oe_ratings r ON r.game_id = g.oe_game_id
                        LEFT JOIN draft_outcome_oos dg ON dg.oe_game_id = g.oe_game_id
                             AND dg.team = (SELECT blue_team FROM oe_games WHERE game_id = g.oe_game_id)
                        WHERE g.game_id = %s""", (gid,)).fetchone()
    if not g:
        return None
    gold = {}   # minute -> [blue roles 5], [red roles 5]
    cs = {}
    for r in conn.execute("SELECT slot, minute, gold, cs FROM golgg_timeline WHERE game_id=%s", (gid,)):
        gold.setdefault(r["minute"], [[0] * 5, [0] * 5])[0 if r["slot"] < 5 else 1][r["slot"] % 5] += (r["gold"] or 0)
        cs.setdefault(r["minute"], [0, 0])[0 if r["slot"] < 5 else 1] += (r["cs"] or 0)
    ev = [dict(r) for r in conn.execute(
        "SELECT seq, time_s, action, side, player, target FROM golgg_events WHERE game_id=%s ORDER BY seq", (gid,))]
    g["champs"] = [r["champion"] for r in conn.execute(
        "SELECT champion FROM golgg_players WHERE game_id=%s ORDER BY slot", (gid,))]
    # player -> side, to know which side a kill's victim belongs to
    pside = {r["player"]: r["side"] for r in conn.execute(
        "SELECT player, side FROM golgg_players WHERE game_id=%s", (gid,))}
    for e in ev:
        e["victim_side"] = pside.get(e.get("target"))
    # item purchases with gold value, by side and time
    items = [dict(r) for r in conn.execute("""
        SELECT b.build_time AS t, b.event, COALESCE(i.gold, 0) AS gold, p.side
        FROM golgg_builds b JOIN golgg_players p ON p.game_id = b.game_id AND p.player_id = b.player_id
        LEFT JOIN golgg_items i ON i.item_id = b.item_id
        WHERE b.game_id = %s AND b.event IN ('ITEM_PURCHASED', 'ITEM_SOLD') ORDER BY b.build_time""", (gid,))]
    return dict(g), gold, cs, ev, items


def _interp(series_by_min, t_s, f):
    """Latest observed minute value at or before ``t_s``.

    Linear interpolation used to read the following minute for event-time
    states.  That made a nominally pre-event prediction depend on future gold
    and CS.  A live model can only carry the latest observation forward.
    """
    if not series_by_min:
        return 0.0
    minute = int(math.floor(max(0.0, t_s) / 60.0))
    observed = [m for m in series_by_min if m <= minute]
    if not observed:
        return 0.0
    return f(series_by_min[max(observed)])


def _hp_feats(hpmap, t_s):
    """Latest HP feed observation at or before ``t_s`` (max age 90 s).

    New builds attach an exact game-clock timestamp to each row.  The fallback
    for old callers is previous-minute-only; neither path can select m+1.
    """
    if hpmap:
        timed = [r for r in hpmap.values() if isinstance(r, dict) and "_clock_s" in r
                 and r["_clock_s"] <= t_s and t_s - r["_clock_s"] <= 90]
        if timed:
            wrapped = max(timed, key=lambda r: r["_clock_s"])
            row = wrapped.get("data") or {}
        else:
            m = int(math.floor(t_s / 60.0)) - 1
            row = hpmap.get(m)
        if row:
            hpb = [x for x in (row.get("hpb") or []) if x is not None]
            hpr = [x for x in (row.get("hpr") or []) if x is not None]
            lvb, lvr = row.get("lvb") or [], row.get("lvr") or []
            if len(hpb) == 5 and len(hpr) == 5:
                lvl = (sum(lvb) - sum(lvr)) / 5.0 if len(lvb) == 5 and len(lvr) == 5 else 0.0
                return [sum(hpb) - sum(hpr), float(sum(1 for x in hpb if x < 0.3)),
                        float(sum(1 for x in hpr if x < 0.3)), lvl, 1.0]
    return [0.0, 0.0, 0.0, 0.0, 0.0]


def _state(g, gold, cs, ev, items, idx, t_s, hpmap=None):
    t = t_s / 60.0
    f_total = lambda v: (sum(v[0]) - sum(v[1])) / 1000.0
    gk = _interp(gold, t_s, f_total)
    gk_prev = _interp(gold, max(0, t_s - 120), f_total)
    role_g = [_interp(gold, t_s, (lambda i: (lambda v: (v[0][i] - v[1][i]) / 1000.0))(i)) for i in range(5)]
    csk = _interp(cs, t_s, lambda v: (v[0] - v[1]) / 100.0)
    c = {k: 0 for k in COUNT_KEYS}
    drag = [0, 0]; tow = [0, 0]; inh = [0, 0]
    last_baron = {"blue": -1e9, "red": -1e9}
    last_elder = {"blue": -1e9, "red": -1e9}
    last_baron_any = -1e9; last_dragon_any = -1e9
    kills_recent = 0
    dead = [0, 0]          # players currently on a respawn timer, per side
    nexus_tw = [0, 0]      # nexus turrets lost, per side (taken by the other side)
    last_kill_t = -1e9
    respawn = min(60.0, 8.0 + 1.6 * t)   # rough respawn-timer scale by game minute
    for e in ev[:idx]:
        a = e["action"] or ""; side = e["side"]
        if side not in ("blue", "red"):
            continue
        sgn = 1 if side == "blue" else -1
        et = e["time_s"] or 0
        key = "dragon" if a.startswith("dragon") else a
        if a == "dragon:elder":
            key = "elder"
            last_elder[side] = et
        if key in c:
            c[key] += sgn
        if key == "dragon":
            drag[0 if side == "blue" else 1] += 1
            last_dragon_any = et
        if key == "tower":
            tow[0 if side == "blue" else 1] += 1
            if "NEX" in (e.get("target") or "").upper():
                nexus_tw[1 if side == "blue" else 0] += 1
        if key == "inhib":
            inh[0 if side == "blue" else 1] += 1
        if key == "baron":
            last_baron[side] = et
            last_baron_any = et
        if key == "kill":
            last_kill_t = max(last_kill_t, et)
            if et >= t_s - 120:
                kills_recent += sgn
            if et >= t_s - respawn:
                vs = e.get("victim_side")
                if vs in ("blue", "red"):
                    dead[0 if vs == "blue" else 1] += 1
    soul = (1 if drag[0] >= 4 else 0) - (1 if drag[1] >= 4 else 0)
    baron_active = (1 if t_s - last_baron["blue"] <= 180 else 0) - (1 if t_s - last_baron["red"] <= 180 else 0)
    items_done = 0; item_gold = 0.0
    for it in items:
        if it["t"] > t_s:
            break
        sgn = 1 if it["side"] == "blue" else -1
        if it["event"] == "ITEM_PURCHASED":
            item_gold += sgn * it["gold"]
            if it["gold"] >= 2200:
                items_done += sgn
        elif it["event"] == "ITEM_SOLD":
            item_gold -= sgn * it["gold"]
    elo_oe = ((g["oe_elo_b"] or 1500) - (g["oe_elo_r"] or 1500)) / 400.0
    pelo_oe = ((g["oe_pelo_b"] or 1500) - (g["oe_pelo_r"] or 1500)) / 400.0
    draft = 0.0
    if g.get("draft_p_full") is not None and g.get("draft_p_elo") is not None:
        pf = min(0.99, max(0.01, g["draft_p_full"])); pe = min(0.99, max(0.01, g["draft_p_elo"]))
        draft = math.log(pf / (1 - pf)) - math.log(pe / (1 - pe))
    elo_gg = ((g.get("elo_blue_pre") or 1500) - (g.get("elo_red_pre") or 1500)) / 400.0
    total_gold = _interp(gold, t_s, lambda v: (sum(v[0]) + sum(v[1])) / 1000.0)
    gold_rel = gk / total_gold if total_gold > 1 else 0.0
    elder_buff = (1 if t_s - last_elder["blue"] <= 150 else 0) - (1 if t_s - last_elder["red"] <= 150 else 0)
    lead_x_inhib = gk * (inh[0] - inh[1])
    t_since_kill = min(10.0, (t_s - last_kill_t) / 60.0) if last_kill_t > -1e8 else 10.0
    baron_up = 1.0 if (t_s >= 20 * 60 and t_s - last_baron_any >= 360) else 0.0
    dragon_up = 1.0 if (t_s >= 5 * 60 and t_s - last_dragon_any >= 300) else 0.0
    dead_diff = dead[1] - dead[0]          # enemies dead minus own dead (blue view)
    x = [1.0, t / 30.0, (t / 30.0) ** 2, elo_oe, pelo_oe, draft, elo_gg,
         gk, gk * t / 30.0, gk - gk_prev, csk] + role_g + [c[k] for k in COUNT_KEYS] + \
        [c["kill"] * t / 30.0, soul, baron_active, kills_recent, items_done, item_gold / 1000.0,
         drag[0], drag[1], tow[0], tow[1], inh[0], inh[1],
         dead[0], dead[1], nexus_tw[0], nexus_tw[1], gold_rel, gk * (t / 30.0) ** 2,
         lead_x_inhib, elder_buff, t_since_kill,
         ((g.get("form_blue") if g.get("form_blue") is not None else 0.5) -
          (g.get("form_red") if g.get("form_red") is not None else 0.5)),
         (math.log1p(g.get("ngames_blue") or 0) - math.log1p(g.get("ngames_red") or 0)),
         baron_up, dragon_up, baron_up * dead_diff, dead_diff * t / 30.0,
         (g.get("rapm_team") or 0.0) / 100.0, (g.get("rapm_player") or 0.0) / 100.0] + _hp_feats(hpmap, t_s)
    return x


def build(conn, sample_every_s=60):
    """Assemble the state dataset + market probs at event instants; cache to npz."""
    os.makedirs(OUT_DIR, exist_ok=True)
    gids = [r["game_id"] for r in conn.execute(
        """SELECT game_id FROM golgg_games WHERE winner_side IS NOT NULL AND duration_s IS NOT NULL
           AND EXISTS (SELECT 1 FROM golgg_timeline t WHERE t.game_id = golgg_games.game_id)
           ORDER BY game_id""")]
    # market probs at event instants, blue-oriented, averaged per platform
    mk = {}
    for r in conn.execute("""
        SELECT o.game_id, o.seq, o.platform, o.p_before, a.team_side
        FROM event_odds o JOIN game_alignment a ON a.game_id = o.game_id AND a.platform = o.platform
                                                AND a.market_id = o.market_id
        WHERE o.p_before IS NOT NULL"""):
        p = r["p_before"] if r["team_side"] == "blue" else 1 - r["p_before"]
        mk.setdefault((r["game_id"], r["seq"]), {}).setdefault(r["platform"], []).append(p)
    g2e = {r["golgg_game_id"]: r["esports_game_id"] for r in conn.execute(
        "SELECT golgg_game_id, esports_game_id FROM feed_games WHERE status = 'scraped' AND golgg_game_id IS NOT NULL")} \
        if conn.execute("SELECT to_regclass('feed_games') r").fetchone()["r"] else {}
    hp_all = {}
    if g2e:
        hp_rows = list(conn.execute("SELECT esports_game_id, minute, ts, data FROM feed_minutes"))
        first_ts = {}
        for r in hp_rows:
            if r["ts"] is not None:
                first_ts[r["esports_game_id"]] = min(first_ts.get(r["esports_game_id"], r["ts"]), r["ts"])
        for r in hp_rows:
            t0_hp = first_ts.get(r["esports_game_id"])
            wrapped = {"data": r["data"], "_clock_s": (r["ts"] - t0_hp) if t0_hp is not None and r["ts"] is not None else r["minute"] * 60}
            hp_all.setdefault(r["esports_game_id"], {})[r["minute"]] = wrapped
        log.info("wpx: HP feed minutes for %d games", len(hp_all))
    X, y, gid_arr, t_arr, seq_arr, pm_arr, ks_arr, date_arr = [], [], [], [], [], [], [], []
    C = []   # champion ids per state: 10 slots (blue 0-4, red 5-9)
    champ_ids = {}
    t0 = time.time()
    for n, gid in enumerate(gids):
        d = _load_game(conn, gid)
        if not d:
            continue
        g, gold, cs, ev, items = d
        hpmap = hp_all.get(g2e.get(gid))
        won = 1.0 if g["winner_side"] == "blue" else 0.0
        times = sorted(set([(t_s, -1) for t_s in range(0, g["duration_s"] + 1, sample_every_s)] +
                           [(e["time_s"], e["seq"]) for e in ev if e["time_s"] is not None]))
        j = 0
        for t_s, seq in times:
            while j < len(ev) and (ev[j]["time_s"] or 0) < t_s:
                j += 1
            # events at exactly t_s (seq >= 0) are "before" the event state
            jj = j
            while seq >= 0 and jj < len(ev) and ev[jj]["seq"] < seq:
                jj += 1
            idx = jj if seq >= 0 else j
            X.append(_state(g, gold, cs, ev, items, idx, max(0, t_s - (1 if seq >= 0 else 0)), hpmap))
            C.append([champ_ids.setdefault(c, len(champ_ids)) for c in (g.get("champs") or [])][:10] + [-1] * (10 - len(g.get("champs") or [])))
            y.append(won); gid_arr.append(gid); t_arr.append(t_s); seq_arr.append(seq)
            date_arr.append(str(g.get("date") or ""))
            m = mk.get((gid, seq)) if seq >= 0 else None
            pm_arr.append(np.mean(m["polymarket"]) if m and "polymarket" in m else np.nan)
            ks_arr.append(np.mean(m["kalshi"]) if m and "kalshi" in m else np.nan)
        if n % 300 == 0:
            log.info("wpx: built %d/%d games, %d states (%.0fs)", n, len(gids), len(y), time.time() - t0)
    path = os.path.join(OUT_DIR, "states.npz")
    np.savez_compressed(path, X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.float32),
                        gid=np.array(gid_arr), t=np.array(t_arr), seq=np.array(seq_arr),
                        date=np.array(date_arr),
                        pm=np.array(pm_arr, dtype=np.float32), ks=np.array(ks_arr, dtype=np.float32),
                        names=np.array(FEATURE_NAMES), C=np.array(C, dtype=np.int16),
                        champ_names=np.array([c for c, _ in sorted(champ_ids.items(), key=lambda kv: kv[1])]))
    log.info("wpx: saved %s (%d states, %d games, %d PM points, %d Kalshi points)",
             path, len(y), len(set(gid_arr)), int(np.sum(~np.isnan(pm_arr))), int(np.sum(~np.isnan(ks_arr))))
    return path


# ------------------------------------------------------------------- eval

def _brier(p, y):
    return float(np.mean((p - y) ** 2))


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _folds(gids, k=5, seed=11):
    ug = np.unique(gids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    fold_of = {g: i % k for i, g in enumerate(ug)}
    return np.array([fold_of[g] for g in gids])


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _ridge_logit(X, y, l2=1.0, iters=30):
    beta = np.zeros(X.shape[1]); reg = np.full(X.shape[1], l2); reg[0] = 0
    for _ in range(iters):
        p = _sigmoid(X @ beta); W = p * (1 - p) + 1e-9
        step = np.linalg.solve((X * W[:, None]).T @ X + np.diag(reg), X.T @ (p - y) + reg * beta)
        beta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta


ALREADY_INTERACTED = {"t", "t2", "gold_k_x_t", "d_kill_x_t", "dead_diff_x_t", "lead_x_inhib",
                      "baron_up_x_dead", "rel_lead"}


def model_zoo(names, C=None, n_champs=0):
    """Return {model_name: fit_predict}; fit_predict(Xtr, ytr, Xte, itr, ite)."""
    idx = {n: i for i, n in enumerate(names)}
    base_cols = [idx[c] for c in ["bias", "t", "elo_gg", "gold_k", "gold_k_x_t", "d_kill", "d_tower", "d_dragon",
                                  "d_baron", "d_inhib", "d_herald", "d_grubs", "d_atakhan", "d_plate", "d_kill_x_t", "soul"]]
    rich_cols = [i for i, n in enumerate(names) if n not in ("draft",)]
    all_cols = list(range(len(names)))

    def logit(cols, l2=1.0):
        def fp(Xtr, ytr, Xte, itr=None, ite=None):
            b = _ridge_logit(Xtr[:, cols].astype(np.float64), ytr, l2)
            return _sigmoid(Xte[:, cols].astype(np.float64) @ b)
        return fp

    def hgb(cols, **kw):
        from sklearn.ensemble import HistGradientBoostingClassifier
        def fp(Xtr, ytr, Xte, itr=None, ite=None):
            m = HistGradientBoostingClassifier(max_iter=kw.get("iters", 400), learning_rate=kw.get("lr", 0.05),
                                               max_leaf_nodes=kw.get("leaves", 31), l2_regularization=kw.get("l2", 1.0),
                                               min_samples_leaf=kw.get("msl", 100), early_stopping=False, random_state=0)
            m.fit(Xtr[:, cols], ytr)
            return m.predict_proba(Xte[:, cols])[:, 1]
        return fp

    def champ_scale_matrix(rows_idx, Xm, cols_t, t_cap=None, base=False):
        """Signed champion presence x game time (capped): +1 own (blue), -1 enemy.
        base=True gives presence alone (no time factor)."""
        S = np.zeros((len(rows_idx), n_champs), dtype=np.float32)
        Cm = C[rows_idx]
        tt = Xm[:, cols_t].astype(np.float32)
        if t_cap is not None:
            tt = np.minimum(tt, t_cap)
        if base:
            tt = np.ones_like(tt)
        for k in range(10):
            sgn = 1.0 if k < 5 else -1.0
            ids = Cm[:, k]
            ok = ids >= 0
            S[np.where(ok)[0], ids[ok]] += sgn * tt[ok]
        return S

    def logit_xt_champ(cols, l2=1.0, l2_champ=30.0, t_cap=None, with_base=False, l2_base=100.0):
        """logit_xt plus champion-scaling columns (champion presence x time), ridge-penalized harder."""
        xcols_ = [c for c in cols if names[c] not in ALREADY_INTERACTED and names[c] != "bias"]
        def fp(Xtr, ytr, Xte, itr=None, ite=None):
            def expand(Xm, ii):
                B = Xm[:, cols].astype(np.float64)
                tt = Xm[:, t_col:t_col + 1].astype(np.float64)
                S = champ_scale_matrix(ii, Xm, t_col, t_cap=t_cap).astype(np.float64)
                parts = [B, Xm[:, xcols_].astype(np.float64) * tt, S]
                if with_base:
                    parts.append(champ_scale_matrix(ii, Xm, t_col, base=True).astype(np.float64))
                return np.hstack(parts)
            A = expand(Xtr, itr)
            beta = np.zeros(A.shape[1]); reg = np.full(A.shape[1], l2); reg[0] = 0
            nc = n_champs * (2 if with_base else 1)
            reg[-nc:] = l2_champ
            if with_base:
                reg[-n_champs:] = l2_base
            for _ in range(30):
                p = _sigmoid(A @ beta); W = p * (1 - p) + 1e-9
                step = np.linalg.solve((A * W[:, None]).T @ A + np.diag(reg), A.T @ (p - ytr) + reg * beta)
                beta -= step
                if np.max(np.abs(step)) < 1e-6:
                    break
            return _sigmoid(expand(Xte, ite) @ beta)
        return fp

    prior_cols = [idx[c] for c in ["bias", "elo_oe", "pelo_oe", "draft", "elo_gg"]]

    def hgb_reg(cols, **kw):
        from sklearn.ensemble import HistGradientBoostingClassifier
        def fp(Xtr, ytr, Xte, itr=None, ite=None):
            m = HistGradientBoostingClassifier(max_iter=kw.get("iters", 250), learning_rate=kw.get("lr", 0.05),
                                               max_leaf_nodes=kw.get("leaves", 15), l2_regularization=10.0,
                                               min_samples_leaf=kw.get("msl", 3000), max_bins=64,
                                               early_stopping=False, random_state=0)
            m.fit(Xtr[:, cols], ytr)
            return m.predict_proba(Xte[:, cols])[:, 1]
        return fp

    t_col = idx["t"]

    def logit_xt(cols, l2=1.0):
        """Logistic with (non-interaction) features also interacted with game time."""
        xcols_ = [c for c in cols if names[c] not in ALREADY_INTERACTED and names[c] != "bias"]
        def expand(Xm):
            B = Xm[:, cols].astype(np.float64)
            tt = Xm[:, t_col:t_col + 1].astype(np.float64)
            return np.hstack([B, Xm[:, xcols_].astype(np.float64) * tt])
        def fp(Xtr, ytr, Xte, itr=None, ite=None):
            b = _ridge_logit(expand(Xtr), ytr, l2)
            return _sigmoid(expand(Xte) @ b)
        return fp

    def logit_piecewise(cols, l2=1.0, cuts=(900, 1500)):
        """Separate logistic fits for early / mid / late game."""
        def fp(Xtr, ytr, Xte, itr=None, ite=None):
            ttr = Xtr[:, t_col] * 30 * 60; tte = Xte[:, t_col] * 30 * 60
            out = np.zeros(len(Xte))
            bounds = [(-1, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 1e12)]
            for lo, hi in bounds:
                mtr = (ttr > lo) & (ttr <= hi); mte = (tte > lo) & (tte <= hi)
                if mtr.sum() < 100 or not mte.any():
                    continue
                b = _ridge_logit(Xtr[mtr][:, cols].astype(np.float64), ytr[mtr], l2)
                out[mte] = _sigmoid(Xte[mte][:, cols].astype(np.float64) @ b)
            return out
        return fp

    state_cols = [i for i, n in enumerate(names) if n not in ("draft", "elo_oe", "pelo_oe", "elo_gg")]

    def blend(fa, fb, w=0.5):
        def fp(Xtr, ytr, Xte, itr=None, ite=None):
            return w * fa(Xtr, ytr, Xte, itr, ite) + (1 - w) * fb(Xtr, ytr, Xte, itr, ite)
        return fp

    def calibrated(inner_fp, method="isotonic"):
        """Recalibrate a model with an inner 3-fold OOF fit (nested, no leakage)."""
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        def fp(Xtr, ytr, Xte, itr=None, ite=None):
            n = len(ytr)
            rng = np.random.default_rng(3)
            f3 = rng.integers(0, 3, n)
            inner = np.zeros(n)
            for k3 in range(3):
                a, b = f3 != k3, f3 == k3
                inner[b] = inner_fp(Xtr[a], ytr[a], Xtr[b], itr[a] if itr is not None else None, itr[b] if itr is not None else None)
            raw = inner_fp(Xtr, ytr, Xte, itr, ite)
            if method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(inner, ytr)
                return iso.predict(raw)
            z = np.log(np.clip(inner, 1e-4, 1 - 1e-4) / (1 - np.clip(inner, 1e-4, 1 - 1e-4)))
            lr = LogisticRegression(C=1e6).fit(z[:, None], ytr)
            zr = np.log(np.clip(raw, 1e-4, 1 - 1e-4) / (1 - np.clip(raw, 1e-4, 1 - 1e-4)))
            return lr.predict_proba(zr[:, None])[:, 1]
        return fp

    prior2_cols = [idx[c] for c in ["bias", "elo_oe", "pelo_oe", "elo_gg", "form_diff", "exp_diff", "rapm_team", "rapm_player"] if c in idx]

    def blend3(fa, fb, fc):
        def fp(Xtr, ytr, Xte, itr=None, ite=None):
            return (fa(Xtr, ytr, Xte, itr, ite) + fb(Xtr, ytr, Xte, itr, ite) + fc(Xtr, ytr, Xte, itr, ite)) / 3.0
        return fp

    return {
        "prior2(+rapm)": logit(prior2_cols),
        "ens3(xt,champ,piecewise)": blend3(logit_xt(rich_cols), logit_xt_champ(rich_cols, l2_champ=200.0, t_cap=25 / 30.0), logit_piecewise(rich_cols)),
        "ens(xt,champ)+isotonic": calibrated(blend(logit_xt(rich_cols), logit_xt_champ(rich_cols, l2_champ=200.0, t_cap=25 / 30.0)), "isotonic"),
        "champscale_reg(l2=200,cap25)": logit_xt_champ(rich_cols, l2_champ=200.0, t_cap=25 / 30.0),
        # time-forward validated 2026-08-23: heavier shrink + earlier cap generalize better to future metas
        "champscale_reg(l2=800,cap15)": logit_xt_champ(rich_cols, l2_champ=800.0, t_cap=15 / 30.0),
        "champscale_reg(l2=100,cap20)": logit_xt_champ(rich_cols, l2_champ=100.0, t_cap=20 / 30.0),
        "champscale+base(cap25)": logit_xt_champ(rich_cols, l2_champ=200.0, t_cap=25 / 30.0, with_base=True),
        "ens(xt, champscale_reg)": blend(logit_xt(rich_cols), logit_xt_champ(rich_cols, l2_champ=200.0, t_cap=25 / 30.0)),
        "logit_xt+champscale": logit_xt_champ(rich_cols),
        "hgb_noES(31,msl300)": hgb(rich_cols, iters=300, lr=0.05, leaves=31, msl=300, l2=5.0),
        "logit_xt+isotonic": calibrated(logit_xt(rich_cols), "isotonic"),
        "logit_xt+platt": calibrated(logit_xt(rich_cols), "platt"),
        "hgb_state_only(no priors)": hgb_reg(state_cols, leaves=15, msl=2000, iters=300),
        "blend(logit_xt, hgb_state)": blend(logit_xt(rich_cols), hgb_reg(state_cols, leaves=15, msl=2000, iters=300)),
        "logit_rich_v3": logit(rich_cols),
        "logit_rich_v3_xt": logit_xt(rich_cols),
        "logit_piecewise_v3": logit_piecewise(rich_cols),
        "hgb_reg_v3": hgb_reg(rich_cols),
        "hgb_v3(leaves31,msl500)": hgb_reg(rich_cols, leaves=31, msl=500, iters=300),
        "prior_only(elo+pelo+draft)": logit(prior_cols),
        "logit_base(old feats, gg-elo)": logit(base_cols),
        "logit_rich(no draft)": logit(rich_cols),
        "logit_rich+draft": logit(all_cols),
        "logit_rich+draft(l2=10)": logit(all_cols, l2=10.0),
        "hgb_reg(no draft)": hgb_reg(rich_cols),
        "hgb_reg+draft": hgb_reg(all_cols),
        "hgb_reg+draft(msl=8000)": hgb_reg(all_cols, msl=8000, leaves=11),
    }


def evaluate(path=None, k=5, only=None):
    path = path or os.path.join(OUT_DIR, "states.npz")
    d = np.load(path, allow_pickle=True)
    X, y, gid, t, pm, ks = d["X"], d["y"], d["gid"], d["t"], d["pm"], d["ks"]
    names = list(d["names"])
    C = d["C"] if "C" in d.files else None
    n_champs = int(C.max()) + 1 if C is not None else 0
    folds = _folds(gid, k)
    has_pm, has_ks = ~np.isnan(pm), ~np.isnan(ks)
    results = []
    log.info("wpx: %d states / %d games; PM points %d, Kalshi points %d",
             len(y), len(np.unique(gid)), has_pm.sum(), has_ks.sum())
    results.append({"model": "MARKET polymarket", "all": None, "pm": _brier(pm[has_pm], y[has_pm]),
                    "ks": None, "pm_ll": _logloss(pm[has_pm], y[has_pm]), "ks_ll": None})
    results.append({"model": "MARKET kalshi", "all": None, "pm": None, "ks": _brier(ks[has_ks], y[has_ks]),
                    "pm_ll": None, "ks_ll": _logloss(ks[has_ks], y[has_ks])})
    zoo = model_zoo(names, C, n_champs)
    for name, fp in zoo.items():
        if only and name not in only:
            continue
        oof = np.zeros(len(y))
        t0 = time.time()
        for f in range(k):
            tr, te = folds != f, folds == f
            oof[te] = fp(X[tr], y[tr], X[te], np.where(tr)[0], np.where(te)[0])
        r = {"model": name, "all": _brier(oof, y), "pm": _brier(oof[has_pm], y[has_pm]),
             "ks": _brier(oof[has_ks], y[has_ks]), "pm_ll": _logloss(oof[has_pm], y[has_pm]),
             "ks_ll": _logloss(oof[has_ks], y[has_ks]), "secs": round(time.time() - t0, 1)}
        for ph, lo, hi in (("early", 0, 900), ("mid", 900, 1500), ("late", 1500, 10 ** 9)):
            m = has_pm & (t >= lo) & (t < hi)
            r["pm_" + ph] = _brier(oof[m], y[m]) if m.any() else None
            r["mkt_pm_" + ph] = _brier(pm[m], y[m]) if m.any() else None
        results.append(r)
        log.info("wpx: %-30s all=%.4f  PM=%.4f  KS=%.4f  (%.0fs)", name, r["all"], r["pm"], r["ks"], r["secs"])
        np.save(os.path.join(OUT_DIR, "oof_%s.npy" % name.split("(")[0].replace("+", "_")), oof)
    return results


def report(results):
    f = lambda v: ("%.4f" % v) if v is not None else "   -  "
    print("%-32s %9s %11s %11s %9s %9s | %7s %7s %7s" % ("model", "Brier all", "Brier @PM", "Brier @KS", "LL @PM", "LL @KS", "PMearly", "PMmid", "PMlate"))
    for r in results:
        print("%-32s %9s %11s %11s %9s %9s | %7s %7s %7s" % (r["model"], f(r["all"]), f(r["pm"]), f(r["ks"]), f(r["pm_ll"]), f(r["ks_ll"]),
              f(r.get("pm_early")), f(r.get("pm_mid")), f(r.get("pm_late"))))
    mk = [r for r in results if r.get("mkt_pm_early") is not None]
    if mk:
        r = mk[0]
        print("%-32s %9s %11s %11s %9s %9s | %7s %7s %7s" % ("MARKET polymarket by phase", "", "", "", "", "", f(r["mkt_pm_early"]), f(r["mkt_pm_mid"]), f(r["mkt_pm_late"])))


# -------------------------------------------------------------- live use

# The optimized live blend and prospective shadow protocol declare wpgam_v2 as
# their base-model contract.  Keep the legacy artifact for explicit comparison
# and as a missing-artifact fallback, but production live scores must use the
# same constrained GAM that the blend was trained against.
LEGACY_LIVE_MODEL_PATH = os.path.join(OUT_DIR, "model_live.npz")
GAM_LIVE_MODEL_PATH = os.path.join(OUT_DIR, "model_live_gam.npz")
LIVE_MODEL_PATH = GAM_LIVE_MODEL_PATH


def fit_full_legacy(l2=1.0, l2_champ=800.0, t_cap=15 / 30.0,
                    path=LEGACY_LIVE_MODEL_PATH):
    """Fit the best exploration model (xt + champion scaling) on ALL states and
    save coefficients + feature/champion vocab for live prediction."""
    d = np.load(os.path.join(OUT_DIR, "states.npz"), allow_pickle=True)
    X, y, names, C = d["X"], d["y"], list(d["names"]), d["C"]
    champ_names = list(d["champ_names"])
    n_champs = len(champ_names)
    idx = {n: i for i, n in enumerate(names)}
    rich_cols = [i for i, n in enumerate(names) if n != "draft"]
    xcols = [i for i in rich_cols if names[i] not in ALREADY_INTERACTED and names[i] != "bias"]
    t_col = idx["t"]

    def expand(Xm, Cm):
        B = Xm[:, rich_cols].astype(np.float64)
        t_raw = Xm[:, t_col:t_col + 1].astype(np.float64)
        tt = np.minimum(t_raw, t_cap)
        S = np.zeros((len(Xm), n_champs))
        for k in range(10):
            sgn = 1.0 if k < 5 else -1.0
            ids = Cm[:, k]; ok = ids >= 0
            S[np.where(ok)[0], ids[ok]] += sgn * tt[ok, 0]
        return np.hstack([B, Xm[:, xcols].astype(np.float64) * t_raw, S])

    A = expand(X, C)
    beta = np.zeros(A.shape[1]); reg = np.full(A.shape[1], l2); reg[0] = 0; reg[-n_champs:] = l2_champ
    for _ in range(30):
        p = _sigmoid(A @ beta); W = p * (1 - p) + 1e-9
        step = np.linalg.solve((A * W[:, None]).T @ A + np.diag(reg), A.T @ (p - y) + reg * beta)
        beta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    np.savez(path, beta=beta, names=np.array(names), rich_cols=np.array(rich_cols),
             xcols=np.array(xcols), champ_names=np.array(champ_names), t_cap=t_cap)
    # sanity: net effect of +1k gold at several game times must stay positive
    gi = rich_cols.index(idx["gold_k"]); gxi = len(rich_cols) + xcols.index(idx["gold_k"])
    for tm in (10, 20, 30, 40):
        tt = tm / 30.0
        net = beta[gi] + beta[gxi] * tt + (beta[rich_cols.index(idx["gold_k_x_t"])] * tt if "gold_k_x_t" in names else 0)
        log.info("wpx: net log-odds per +1k gold at %d min = %+.3f", tm, net)
    log.info("wpx: live model saved (%d states, %d features)", len(y), A.shape[1])
    return path


def fit_full(*_args, **_kwargs):
    """Fit the production causal constrained model.

    The old exploration fit remains available as :func:`fit_full_legacy`.
    Positional tuning arguments from that model are intentionally ignored so
    scheduled refresh scripts can migrate without changing their call site.
    """
    from . import wpgam
    dataset_path = _kwargs.pop("dataset_path", None)
    _kwargs.pop("model_path", None)
    if _kwargs:
        raise TypeError("unknown constrained-model options: %s" % sorted(_kwargs))
    fit_full_legacy(*_args)
    return wpgam.fit_full(dataset_path=dataset_path,
                          model_path=GAM_LIVE_MODEL_PATH)


def live_vector(state, names):
    """Feature vector for a live state, mirroring _state() exactly (by name).

    state keys (missing -> 0): t_min, gold_diff_k, gold_diff_prev_k, gold_blue, gold_red,
    cs_diff_k, gold_role[5], kills/towers/dragons/barons/inhibs/heralds/grubs/atakhans/
    plates/elders (blue-red diffs), drag_blue/red, towers_blue/red, inhib_blue/red,
    nexus_tw_blue/red, dead_blue/red, baron_active, elder_buff, kills_2m, items_done_diff,
    item_gold_diff_k, t_since_kill_min, baron_up, dragon_up, elo_oe, pelo_oe, elo_gg,
    form_diff, exp_diff, rapm_team, rapm_player.
    """
    # A key explicitly set to None (e.g. team_priors' elo_gg when the gol.gg
    # lookup misses) must behave like a missing key, not poison the vector.
    s = {k: v for k, v in dict(state).items() if v is not None}
    t = s.get("t_min", 0.0) / 30.0
    gk = s.get("gold_diff_k", 0.0); gk_prev = s.get("gold_diff_prev_k", gk)
    role = s.get("gold_role") or [gk / 5.0] * 5
    total_gold_k = (s.get("gold_blue", 0) + s.get("gold_red", 0)) / 1000.0
    dead_b, dead_r = s.get("dead_blue", 0), s.get("dead_red", 0)
    dead_diff = dead_r - dead_b
    inh_b, inh_r = s.get("inhib_blue", 0), s.get("inhib_red", 0)
    drag_b, drag_r = s.get("drag_blue", 0), s.get("drag_red", 0)
    v = {
        "bias": 1.0, "t": t, "t2": t * t,
        "elo_oe": s.get("elo_oe", 0.0), "pelo_oe": s.get("pelo_oe", 0.0), "draft": 0.0,
        "elo_gg": s.get("elo_gg", s.get("elo_oe", 0.0)),
        "gold_k": gk, "gold_k_x_t": gk * t, "gold_mom": gk - gk_prev, "cs_k": s.get("cs_diff_k", 0.0),
        "gold_top": role[0], "gold_jng": role[1], "gold_mid": role[2], "gold_bot": role[3], "gold_sup": role[4],
        "d_kill": s.get("kills", 0), "d_tower": s.get("towers", 0), "d_dragon": s.get("dragons", 0),
        "d_baron": s.get("barons", 0), "d_inhib": s.get("inhibs", inh_b - inh_r), "d_herald": s.get("heralds", 0),
        "d_grubs": s.get("grubs", 0), "d_atakhan": s.get("atakhans", 0), "d_plate": s.get("plates", 0),
        "d_elder": s.get("elders", 0), "d_kill_x_t": s.get("kills", 0) * t,
        "soul": (1 if drag_b >= 4 else 0) - (1 if drag_r >= 4 else 0),
        "baron_active": s.get("baron_active", 0), "kills_2m": s.get("kills_2m", 0),
        "items_done": s.get("items_done_diff", 0), "item_gold_k": s.get("item_gold_diff_k", 0.0),
        "drag_blue": drag_b, "drag_red": drag_r,
        "towers_blue": s.get("towers_blue", 0), "towers_red": s.get("towers_red", 0),
        "inhib_blue": inh_b, "inhib_red": inh_r, "dead_blue": dead_b, "dead_red": dead_r,
        "nexus_tw_blue": s.get("nexus_tw_blue", 0), "nexus_tw_red": s.get("nexus_tw_red", 0),
        "gold_rel": (gk / total_gold_k) if total_gold_k > 1 else 0.0,
        "gold_k_x_t2": gk * t * t, "lead_x_inhib": gk * (inh_b - inh_r),
        "elder_buff": s.get("elder_buff", s.get("elder_active", 0)),
        "t_since_kill": s.get("t_since_kill_min", 10.0),
        "form_diff": s.get("form_diff", 0.0), "exp_diff": s.get("exp_diff", 0.0),
        "baron_up": s.get("baron_up", 0), "dragon_up": s.get("dragon_up", 0),
        "hp_pool": s.get("hp_pool", 0.0), "hp_low_b": s.get("hp_low_b", 0.0),
        "hp_low_r": s.get("hp_low_r", 0.0), "lvl_k": s.get("lvl_k", 0.0), "has_hp": s.get("has_hp", 0.0),
        "baron_up_x_dead": s.get("baron_up", 0) * dead_diff, "dead_diff_x_t": dead_diff * t,
        "rapm_team": s.get("rapm_team", 0.0) / 100.0, "rapm_player": s.get("rapm_player", 0.0) / 100.0,
    }
    missing = [n for n in names if n not in v]
    if missing:
        raise KeyError("live_vector lacks features: %s" % missing)
    return np.array([v[n] for n in names], dtype=np.float64)


PRIOR_FEATURES = {"elo_oe", "pelo_oe", "form_diff", "elo_gg", "draft", "rapm_team", "rapm_player", "exp_diff"}
# Live priors are clipped to the training 1st-99th percentile so an unusually
# lopsided matchup (e.g. a new team with a 390-Elo gap and 0.1 vs 0.9 form)
# doesn't linearly extrapolate the prior term beyond anything the fit has seen.
PRIOR_CLIP = {"elo_oe": 0.9, "pelo_oe": 0.9, "form_diff": 0.6}


def _predict_live_legacy(state, blue_champs=(), red_champs=(), path=LEGACY_LIVE_MODEL_PATH):
    """Win probability for BLUE from an observable live state (see live_vector)."""
    m = np.load(path, allow_pickle=True)
    names = list(m["names"]); beta = m["beta"]; rich_cols = list(m["rich_cols"])
    xcols = list(m["xcols"]) if "xcols" in m.files else [c for c in rich_cols if names[c] not in ALREADY_INTERACTED and names[c] != "bias"]
    champ_names = list(m["champ_names"]); t_cap = float(m["t_cap"])
    state = dict(state)
    for k, lim in PRIOR_CLIP.items():
        if state.get(k) is not None:
            state[k] = max(-lim, min(lim, float(state[k])))
    x = live_vector(state, names)
    t = x[names.index("t")]
    B = x[rich_cols]
    Xt = x[xcols] * t
    tt = min(t, t_cap)
    S = np.zeros(len(champ_names))
    norm = lambda c: (c or "").lower().replace("'", "").replace(" ", "").replace(".", "")
    lut = {norm(nm): i for i, nm in enumerate(champ_names)}
    unknown = []
    for c in blue_champs:
        i = lut.get(norm(c), -1)
        if i >= 0: S[i] += tt
        elif c: unknown.append(c)
    for c in red_champs:
        i = lut.get(norm(c), -1)
        if i >= 0: S[i] -= tt
        elif c: unknown.append(c)
    a = np.concatenate([B, Xt, S])
    p = float(_sigmoid(a @ beta))
    # log-odds breakdown: prior (team strength) / game state / champions / time
    nB, nX = len(rich_cols), len(xcols)
    lo_prior = lo_time = lo_state = 0.0
    for i, c in enumerate(rich_cols):
        v = B[i] * beta[i]
        if names[c] in PRIOR_FEATURES: lo_prior += v
        elif names[c] in ("bias", "t", "t2"): lo_time += v
        else: lo_state += v
    for i, c in enumerate(xcols):
        v = Xt[i] * beta[nB + i]
        if names[c] in PRIOR_FEATURES: lo_prior += v
        else: lo_state += v
    lo_champ = float(S @ beta[nB + nX:])
    return {"p_blue": round(p, 4), "unknown_champions": unknown,
            "lo_prior": round(lo_prior, 3), "lo_state": round(lo_state, 3), "lo_champ": round(lo_champ, 3), "lo_time": round(lo_time, 3)}


# Deployed forecast = logit blend of the constrained GAM (v7 champion-state
# contract) and the legacy champscale model.  The weight was selected on the
# chronological validation block (nested, test untouched) on 2026-08-29:
# untouched-test game Brier 0.14179 vs 0.14247 for the GAM alone, paired
# 95% interval -0.00125..-0.00011.  The selection curve is flat for
# w_gam in 0.25..0.5, so a fixed weight is robust to redeployment drift.
LIVE_BLEND_W_GAM = 0.45


def predict_live(state, blue_champs=(), red_champs=(), path=LIVE_MODEL_PATH,
                 blend=True):
    """Deployed live score: GAM + legacy champscale logit blend.

    Falls back to the GAM alone when the legacy artifact is missing, and to
    the legacy model alone when the GAM artifact is missing.  The log-odds
    breakdown fields always describe the GAM component.
    """
    gam_out = None
    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as model:
                kind = str(model["kind"].item()) if "kind" in model.files else ""
            if kind:
                from . import wpgam
                gam_out = wpgam.predict_live(state, blue_champs, red_champs, path)
        except (KeyError, ValueError, OSError):
            if path != LEGACY_LIVE_MODEL_PATH:
                raise
    if gam_out is None:
        legacy = path if path != LIVE_MODEL_PATH else LEGACY_LIVE_MODEL_PATH
        return _predict_live_legacy(state, blue_champs, red_champs, legacy)
    if not blend or path != LIVE_MODEL_PATH or not os.path.exists(LEGACY_LIVE_MODEL_PATH):
        return gam_out
    legacy_out = _predict_live_legacy(state, blue_champs, red_champs)
    clip = lambda p: min(1.0 - 1e-6, max(1e-6, float(p)))
    logit = lambda p: math.log(clip(p) / (1.0 - clip(p)))
    w = LIVE_BLEND_W_GAM
    p = _sigmoid(w * logit(gam_out["p_blue"]) + (1.0 - w) * logit(legacy_out["p_blue"]))
    out = dict(gam_out)
    out.update({
        "p_blue": round(float(p), 4),
        "p_gam": gam_out["p_blue"], "p_legacy": legacy_out["p_blue"],
        "blend_w_gam": w,
        "unknown_champions": sorted(set(gam_out.get("unknown_champions") or [])
                                    | set(legacy_out.get("unknown_champions") or [])),
        "model_kind": "%s+champscale_w%.2f" % (gam_out.get("model_kind", ""), w),
    })
    return out
