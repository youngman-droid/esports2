"""Odds-free evaluation: in-game win-probability model and event WPA; Elo.

Uses only outcomes and in-game state (gol.gg timelines), no market data:
  elo()          sequential Elo over gol.gg games -> pre-game rating diff
  fit()          logistic win-probability model on sampled game states
  build_wpa()    per-event Win Probability Added = WP(after) - WP(before)
  game_wp()      a game's model WP curve (per minute + at events)
"""
import json
import logging
import math
import time

import numpy as np

np.seterr(all="ignore")  # saturated logits overflow harmlessly in the sigmoid

log = logging.getLogger("wpa")

SCHEMA = (
    'ALTER TABLE golgg_games ADD COLUMN IF NOT EXISTS elo_blue_pre REAL;',
    'ALTER TABLE golgg_games ADD COLUMN IF NOT EXISTS elo_red_pre REAL;',
    'ALTER TABLE golgg_games ADD COLUMN IF NOT EXISTS elo_blue_pre_fast REAL; -- K=120 recency variant',
    'ALTER TABLE golgg_games ADD COLUMN IF NOT EXISTS elo_red_pre_fast REAL;',
    """CREATE TABLE IF NOT EXISTS wp_model (
        feature TEXT PRIMARY KEY, coef DOUBLE PRECISION
    );""",
    'CREATE TABLE IF NOT EXISTS wp_meta (key TEXT PRIMARY KEY, value TEXT);',
    """CREATE TABLE IF NOT EXISTS event_wpa (
        game_id INT NOT NULL, seq INT NOT NULL,
        time_s INT, action TEXT, side TEXT, player TEXT, target TEXT,
        wp_before REAL, wp_after REAL,
        wpa REAL,              -- change in blue WP
        wpa_actor REAL,        -- swing toward the acting side
        PRIMARY KEY (game_id, seq)
    );""",
    'CREATE INDEX IF NOT EXISTS idx_event_wpa_action ON event_wpa (action);',
)

COUNT_ACTIONS = ["kill", "tower", "dragon", "baron", "inhib", "herald", "grubs",
                 "atakhan", "plate"]
FEATURES = (["bias", "blue_side", "elo_diff", "t", "gold_k", "gold_k_x_t"] +
            ["d_" + a for a in COUNT_ACTIONS] + ["d_kill_x_t", "d_dragon_soul"])


def ensure_schema(conn):
    from . import db
    db.apply_schema(conn, SCHEMA)


# ---------------------------------------------------------------------- Elo

def elo(conn, k=30.0, k_fast=120.0, base=1500.0):
    """Sequential team Elo at two adaptation speeds.

    K=30 carries long-run strength; the K=120 variant tracks recent form with
    opponent adjustment (unlike a raw last-N win rate) and feeds the model's
    elo_gg_fast prior channel.
    """
    ensure_schema(conn)
    games = conn.execute("""SELECT game_id, blue_team, red_team, winner_side, date, match_id,
                                   game_num FROM golgg_games WHERE winner_side IS NOT NULL
                            ORDER BY date, match_id, game_num""").fetchall()
    r, rf = {}, {}
    upd = []
    for g in games:
        b, rd = r.get(g["blue_team"], base), r.get(g["red_team"], base)
        fb, fr = rf.get(g["blue_team"], base), rf.get(g["red_team"], base)
        upd.append((b, rd, fb, fr, g["game_id"]))
        sb = 1.0 if g["winner_side"] == "blue" else 0.0
        eb = 1 / (1 + 10 ** ((rd - b) / 400))
        r[g["blue_team"]] = b + k * (sb - eb)
        r[g["red_team"]] = rd + k * ((1 - sb) - (1 - eb))
        ef = 1 / (1 + 10 ** ((fr - fb) / 400))
        rf[g["blue_team"]] = fb + k_fast * (sb - ef)
        rf[g["red_team"]] = fr + k_fast * ((1 - sb) - (1 - ef))
    with conn.cursor() as cur:
        cur.executemany("""UPDATE golgg_games SET elo_blue_pre=%s, elo_red_pre=%s,
                           elo_blue_pre_fast=%s, elo_red_pre_fast=%s WHERE game_id=%s""",
                        upd, returning=False)
    conn.commit()
    log.info("wpa: elo over %d games, %d teams (K=%.0f and K=%.0f)",
             len(games), len(r), k, k_fast)
    return r


# ------------------------------------------------------------- game states

def _game_data(conn, game_id):
    g = conn.execute("SELECT * FROM golgg_games WHERE game_id=%s", (game_id,)).fetchone()
    if not g:
        return None
    tl = conn.execute("SELECT slot, minute, gold FROM golgg_timeline WHERE game_id=%s",
                      (game_id,)).fetchall()
    gold = {}
    for r in tl:
        if r["gold"] is None:
            continue
        side = 0 if r["slot"] < 5 else 1
        gold.setdefault(r["minute"], [0, 0])[side] += r["gold"]
    ev = [dict(r) for r in conn.execute(
        "SELECT seq, time_s, action, side, player, target FROM golgg_events WHERE game_id=%s ORDER BY seq",
        (game_id,))]
    return dict(g), gold, ev


def _gold_diff_at(gold, t_s):
    """Blue-red gold diff (k) at t seconds, linearly interpolated by minute."""
    if not gold:
        return 0.0
    m = t_s / 60.0
    lo = int(math.floor(m))
    hi = lo + 1
    def d(k):
        v = gold.get(k)
        return (v[0] - v[1]) / 1000.0 if v else None
    dlo, dhi = d(lo), d(hi)
    if dlo is None and dhi is None:
        mx = max(gold)
        return d(min(lo, mx)) or 0.0
    if dhi is None:
        return dlo
    if dlo is None:
        return dhi
    return dlo + (dhi - dlo) * (m - lo)


def _counts_before(events, idx):
    """Cumulative blue-red diffs of each action type over events[:idx]."""
    c = {a: 0 for a in COUNT_ACTIONS}
    dragons = [0, 0]
    for e in events[:idx]:
        a = e["action"] or ""
        key = "dragon" if a.startswith("dragon") else a
        if key in c and e["side"] in ("blue", "red"):
            c[key] += 1 if e["side"] == "blue" else -1
            if key == "dragon":
                dragons[0 if e["side"] == "blue" else 1] += 1
    soul = (1 if dragons[0] >= 4 else 0) - (1 if dragons[1] >= 4 else 0)
    return c, soul


def _features(g, gold, events, idx, t_s):
    c, soul = _counts_before(events, idx)
    t = t_s / 60.0
    gk = _gold_diff_at(gold, t_s)
    elo_diff = ((g.get("elo_blue_pre") or 1500) - (g.get("elo_red_pre") or 1500)) / 400.0
    x = [1.0, 1.0, elo_diff, t / 30.0, gk, gk * t / 30.0]
    x += [c[a] for a in COUNT_ACTIONS]
    x += [c["kill"] * t / 30.0, soul]
    return x


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------- model

def fit(conn, l2=1.0, sample_every_s=60, holdout=0.2, max_games=None):
    """Fit logistic WP model on states sampled every minute + at each event."""
    ensure_schema(conn)
    gids = [r["game_id"] for r in conn.execute(
        """SELECT game_id FROM golgg_games WHERE winner_side IS NOT NULL AND duration_s IS NOT NULL
           AND EXISTS (SELECT 1 FROM golgg_timeline t WHERE t.game_id = golgg_games.game_id)
           ORDER BY game_id""")]
    if max_games:
        gids = gids[:max_games]
    X, y, grp = [], [], []
    for gid in gids:
        d = _game_data(conn, gid)
        if not d:
            continue
        g, gold, ev = d
        won = 1.0 if g["winner_side"] == "blue" else 0.0
        times = sorted(set(list(range(0, g["duration_s"] + 1, sample_every_s)) +
                           [e["time_s"] for e in ev if e["time_s"] is not None]))
        j = 0
        for t_s in times:
            while j < len(ev) and (ev[j]["time_s"] or 0) <= t_s:
                j += 1
            X.append(_features(g, gold, ev, j, t_s))
            y.append(won)
            grp.append(gid)
    X = np.array(X)
    y = np.array(y)
    grp = np.array(grp)
    # game-level holdout
    rng = np.random.default_rng(11)
    ug = np.unique(grp)
    test_g = set(rng.choice(ug, size=int(len(ug) * holdout), replace=False).tolist())
    te = np.array([g in test_g for g in grp])
    tr = ~te

    def train(Xm, ym):
        beta = np.zeros(Xm.shape[1])
        reg = l2 * np.ones(Xm.shape[1]); reg[0] = 0.0  # no penalty on bias
        for _ in range(25):  # IRLS / Newton
            p = _sigmoid(Xm @ beta)
            W = p * (1 - p) + 1e-9
            grad = Xm.T @ (p - ym) + reg * beta
            H = (Xm * W[:, None]).T @ Xm + np.diag(reg)
            step = np.linalg.solve(H, grad)
            beta -= step
            if np.max(np.abs(step)) < 1e-6:
                break
        return beta

    def metrics(Xm, ym, beta):
        p = np.clip(_sigmoid(Xm @ beta), 1e-6, 1 - 1e-6)
        ll = -np.mean(ym * np.log(p) + (1 - ym) * np.log(1 - p))
        acc = np.mean((p > 0.5) == (ym > 0.5))
        base = -np.mean(ym * np.log(ym.mean()) + (1 - ym) * np.log(1 - ym.mean()))
        return ll, acc, base

    beta_tr = train(X[tr], y[tr])
    ll, acc, base = metrics(X[te], y[te], beta_tr)
    beta = train(X, y)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM wp_model")
        cur.executemany("INSERT INTO wp_model (feature, coef) VALUES (%s, %s)",
                        [(f, float(b)) for f, b in zip(FEATURES, beta)], returning=False)
        cur.execute("""INSERT INTO wp_meta (key, value) VALUES ('fit', %s)
                       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
                    (json.dumps({"games": len(gids), "rows": int(len(y)),
                                 "holdout_logloss": round(float(ll), 4),
                                 "holdout_acc": round(float(acc), 4),
                                 "baseline_logloss": round(float(base), 4),
                                 "fitted_at": int(time.time())}),))
    conn.commit()
    log.info("wpa: fit on %d games / %d states; holdout logloss %.4f (baseline %.4f), acc %.3f",
             len(gids), len(y), ll, base, acc)
    return dict(zip(FEATURES, beta))


def load_model(conn):
    beta = {r["feature"]: r["coef"] for r in conn.execute("SELECT feature, coef FROM wp_model")}
    m = conn.execute("SELECT value FROM wp_meta WHERE key='fit'").fetchone()
    meta = json.loads(m["value"]) if m else None
    return beta, meta


def _wp(beta, x):
    return float(_sigmoid(sum(beta.get(f, 0.0) * v for f, v in zip(FEATURES, x))))


def game_wp(conn, game_id, beta=None):
    """Model WP curve: per-minute points + per-event before/after."""
    if beta is None:
        beta, _ = load_model(conn)
    d = _game_data(conn, game_id)
    if not d or not beta:
        return None
    g, gold, ev = d
    curve = []
    j = 0
    for t_s in range(0, (g["duration_s"] or 0) + 1, 60):
        while j < len(ev) and (ev[j]["time_s"] or 0) <= t_s:
            j += 1
        curve.append([t_s, round(_wp(beta, _features(g, gold, ev, j, t_s)), 4)])
    evs = []
    for i, e in enumerate(ev):
        t_s = e["time_s"] or 0
        before = _wp(beta, _features(g, gold, ev, i, max(0, t_s - 1)))
        after = _wp(beta, _features(g, gold, ev, i + 1, t_s + 60))
        sgn = 1 if e["side"] == "blue" else (-1 if e["side"] == "red" else 0)
        evs.append({"seq": e["seq"], "time_s": t_s, "action": e["action"], "side": e["side"],
                    "player": e["player"], "target": e["target"],
                    "wp_before": round(before, 4), "wp_after": round(after, 4),
                    "wpa": round(after - before, 4),
                    "wpa_actor": round((after - before) * sgn, 4) if sgn else None})
    return {"game_id": game_id, "blue_team": g["blue_team"], "red_team": g["red_team"],
            "winner_side": g["winner_side"], "duration_s": g["duration_s"],
            "curve": curve, "events": evs}


def build_wpa(conn, rebuild=False):
    ensure_schema(conn)
    beta, _ = load_model(conn)
    if not beta:
        log.warning("wpa: no model fitted yet")
        return 0
    gids = [r["game_id"] for r in conn.execute(
        "SELECT game_id FROM golgg_games WHERE winner_side IS NOT NULL" +
        ("" if rebuild else " AND NOT EXISTS (SELECT 1 FROM event_wpa w WHERE w.game_id = golgg_games.game_id)"))]
    n = 0
    for gid in gids:
        r = game_wp(conn, gid, beta)
        if not r:
            continue
        with conn.cursor() as cur:
            cur.execute("DELETE FROM event_wpa WHERE game_id=%s", (gid,))
            cur.executemany("""INSERT INTO event_wpa (game_id, seq, time_s, action, side, player,
                               target, wp_before, wp_after, wpa, wpa_actor)
                               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                            [(gid, e["seq"], e["time_s"], e["action"], e["side"], e["player"],
                              e["target"], e["wp_before"], e["wp_after"], e["wpa"], e["wpa_actor"])
                             for e in r["events"]], returning=False)
        conn.commit()
        n += 1
    log.info("wpa: built event WPA for %d games", n)
    return n


def event_impact(conn, league="", min_n=10, phase=None):
    where = ["w.wpa_actor IS NOT NULL"]
    params = {"minn": min_n}
    if league:
        where.append("g.trname ILIKE %(lg)s")
        params["lg"] = "%" + league + "%"
    if phase == "early":
        where.append("w.time_s < 900")
    elif phase == "mid":
        where.append("w.time_s BETWEEN 900 AND 1500")
    elif phase == "late":
        where.append("w.time_s > 1500")
    rows = conn.execute("""
        SELECT w.action, COUNT(*) AS n,
               ROUND(AVG(w.wpa_actor)::numeric, 4) AS mean_wpa,
               ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY w.wpa_actor)::numeric, 4) AS median_wpa,
               ROUND(AVG(ABS(w.wpa))::numeric, 4) AS mean_abs
        FROM event_wpa w JOIN golgg_games g ON g.game_id = w.game_id
        WHERE %s GROUP BY w.action HAVING COUNT(*) >= %%(minn)s
        ORDER BY mean_wpa DESC""" % " AND ".join(where), params).fetchall()
    return [dict(r) for r in rows]


# ----------------------------------------------------------- calibration

def _bins(pairs, n_bins=10):
    bins = [{"n": 0, "sp": 0.0, "sy": 0.0} for _ in range(n_bins)]
    brier = ll = 0.0
    for p, y in pairs:
        b = bins[min(int(p * n_bins), n_bins - 1)]
        b["n"] += 1; b["sp"] += p; b["sy"] += y
        brier += (p - y) ** 2
        pc = min(1 - 1e-6, max(1e-6, p))
        ll -= y * math.log(pc) + (1 - y) * math.log(1 - pc)
    n = len(pairs)
    return {"n": n, "brier": round(brier / n, 5) if n else None,
            "logloss": round(ll / n, 5) if n else None,
            "bins": [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": b["n"],
                      "mean_pred": round(b["sp"] / b["n"], 4) if b["n"] else None,
                      "obs_freq": round(b["sy"] / b["n"], 4) if b["n"] else None}
                     for i, b in enumerate(bins)]}


def calibration(conn, n_bins=10, holdout=0.2, sample_every_s=60):
    """Refit on a game-level train split and score the held-out games' states."""
    gids = [r["game_id"] for r in conn.execute(
        """SELECT game_id FROM golgg_games WHERE winner_side IS NOT NULL AND duration_s IS NOT NULL
           AND EXISTS (SELECT 1 FROM golgg_timeline t WHERE t.game_id = golgg_games.game_id)
           ORDER BY game_id""")]
    rng = np.random.default_rng(11)
    test = set(rng.choice(np.array(gids), size=int(len(gids) * holdout), replace=False).tolist())
    Xtr, ytr = [], []
    test_states = []   # (features, y, t_s)
    for gid in gids:
        d = _game_data(conn, gid)
        if not d:
            continue
        g, gold, ev = d
        won = 1.0 if g["winner_side"] == "blue" else 0.0
        times = sorted(set(list(range(0, g["duration_s"] + 1, sample_every_s)) +
                           [e["time_s"] for e in ev if e["time_s"] is not None]))
        j = 0
        for t_s in times:
            while j < len(ev) and (ev[j]["time_s"] or 0) <= t_s:
                j += 1
            x = _features(g, gold, ev, j, t_s)
            if gid in test:
                test_states.append((x, won, t_s))
            else:
                Xtr.append(x); ytr.append(won)
    Xtr = np.array(Xtr); ytr = np.array(ytr)
    beta = np.zeros(Xtr.shape[1]); reg = np.ones(Xtr.shape[1]); reg[0] = 0
    for _ in range(25):
        p = _sigmoid(Xtr @ beta); W = p * (1 - p) + 1e-9
        step = np.linalg.solve((Xtr * W[:, None]).T @ Xtr + np.diag(reg), Xtr.T @ (p - ytr) + reg * beta)
        beta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    Xte = np.array([x for x, _, _ in test_states]); yte = np.array([y for _, y, _ in test_states])
    pte = _sigmoid(Xte @ beta)
    ts = np.array([t for _, _, t in test_states])
    out = {"games_train": len(gids) - len(test), "games_test": len(test),
           "overall": _bins(list(zip(pte.tolist(), yte.tolist())), n_bins), "by_phase": {}}
    for name, lo, hi in (("early", 0, 900), ("mid", 900, 1500), ("late", 1500, 10 ** 9)):
        m = (ts >= lo) & (ts < hi)
        out["by_phase"][name] = _bins(list(zip(pte[m].tolist(), yte[m].tolist())), n_bins)
    return out


def compare_market(conn, n_bins=10):
    """Market odds vs model WP at the same event instants in aligned games."""
    rows = conn.execute("""
        SELECT w.game_id, w.seq, w.time_s, w.wp_before AS model_p,
               o.platform, o.p_before AS mkt_p, a.team_side, g.winner_side
        FROM event_wpa w
        JOIN event_odds o ON o.game_id = w.game_id AND o.seq = w.seq
        JOIN game_alignment a ON a.game_id = o.game_id AND a.platform = o.platform
                             AND a.market_id = o.market_id
        JOIN golgg_games g ON g.game_id = w.game_id
        WHERE w.wp_before IS NOT NULL AND o.p_before IS NOT NULL AND w.time_s > 60
          AND g.winner_side IS NOT NULL""").fetchall()
    # orient market prob to blue; average the two team markets of one platform
    acc = {}
    for r in rows:
        pm = r["mkt_p"] if r["team_side"] == "blue" else 1 - r["mkt_p"]
        key = (r["game_id"], r["seq"], r["platform"])
        a = acc.setdefault(key, {"mkt": [], "model": r["model_p"],
                                 "y": 1.0 if r["winner_side"] == "blue" else 0.0, "t": r["time_s"]})
        a["mkt"].append(pm)
    per_plat = {}
    for (gid, seq, plat), a in acc.items():
        d = per_plat.setdefault(plat, {"mkt": [], "model": []})
        pm = sum(a["mkt"]) / len(a["mkt"])
        d["mkt"].append((pm, a["y"])); d["model"].append((a["model"], a["y"]))
    out = {}
    for plat, d in per_plat.items():
        out[plat] = {"market": _bins(d["mkt"], n_bins), "model_same_points": _bins(d["model"], n_bins),
                     "games": len({k[0] for k in acc if k[2] == plat})}
    return out
