"""Fair market-vs-model comparison at fixed minute marks (not event-anchored).

For each aligned game/platform, minute m maps to wall time via the alignment
(start + m*60 + pauses before m); the market price at that wall time (blue
oriented, averaged over the platform's team markets) is compared with the
model's out-of-fold WP at the same minute state.
"""
import json
import logging
import os
import numpy as np
from lol_ticker import align, config

log = logging.getLogger("wpx_fair")
OUT_DIR = os.path.join(config.REPO_ROOT, "data", "wpx")


def run(conn, oof_name="logit_rich_v3_xt", min_quality=0.4, limit=None, lag_s=0):
    d = np.load(os.path.join(OUT_DIR, "states.npz"), allow_pickle=True)
    gid, t, seq, y = d["gid"], d["t"], d["seq"], d["y"]
    oof = np.load(os.path.join(OUT_DIR, "oof_%s.npy" % oof_name))
    minute_idx = {}
    for i in np.where(seq == -1)[0]:
        minute_idx[(int(gid[i]), int(t[i]))] = i
    rows = conn.execute("""
        SELECT a.game_id, a.platform, a.market_id, a.team_side, a.start_wall, a.end_wall,
               a.pauses, a.duration_s, g.winner_side
        FROM game_alignment a JOIN golgg_games g ON g.game_id = a.game_id
        WHERE a.quality >= %s ORDER BY a.game_id, a.platform""" + (" LIMIT %d" % limit if limit else ""),
        (min_quality,)).fetchall()
    # group by (game, platform)
    groups = {}
    for r in rows:
        groups.setdefault((r["game_id"], r["platform"]), []).append(r)
    res = {"polymarket": {"mkt": [], "mdl": [], "y": [], "t": []},
           "kalshi": {"mkt": [], "mdl": [], "y": [], "t": []}}
    n_done = 0
    for (g_id, plat), al in groups.items():
        base = al[0]
        pauses = base["pauses"] if isinstance(base["pauses"], list) else json.loads(base["pauses"] or "[]")
        series = {}
        for a in al:
            s = align.odds_series(conn, plat, a["market_id"], a["start_wall"] - 600, a["end_wall"] + 300)
            if len(s) >= 10:
                series[a["market_id"]] = (s, a["team_side"])
        if not series:
            continue
        y_blue = 1.0 if base["winner_side"] == "blue" else 0.0
        for m in range(1, int(base["duration_s"] // 60) + 1):
            t_s = m * 60
            i = minute_idx.get((g_id, t_s))
            if i is None:
                continue
            off = sum(p["length_s"] for p in pauses if p["game_time_s"] <= t_s)
            wall = base["start_wall"] + t_s + off - lag_s
            ps = []
            for s, side in series.values():
                p = align._price_at(s, wall, "before")
                if p is not None:
                    ps.append(p if side == "blue" else 1 - p)
            if not ps:
                continue
            r = res[plat]
            r["mkt"].append(float(np.mean(ps))); r["mdl"].append(float(oof[i])); r["y"].append(y_blue); r["t"].append(t_s)
        n_done += 1
        if n_done % 200 == 0:
            log.info("wpx_fair: %d game-platforms", n_done)
    out = {}
    for plat, r in res.items():
        if not r["y"]:
            continue
        mk, md, yy, tt = map(np.array, (r["mkt"], r["mdl"], r["y"], r["t"]))
        def brier(p, m=None):
            if m is None: m = np.ones(len(p), bool)
            return float(np.mean((p[m] - yy[m]) ** 2)) if m.any() else None
        out[plat] = {"n": int(len(yy)), "market": brier(mk), "model": brier(md),
                     "market_early": brier(mk, tt < 900), "model_early": brier(md, tt < 900),
                     "market_mid": brier(mk, (tt >= 900) & (tt < 1500)), "model_mid": brier(md, (tt >= 900) & (tt < 1500)),
                     "market_late": brier(mk, tt >= 1500), "model_late": brier(md, tt >= 1500),
                     "blend50": brier(0.5 * mk + 0.5 * md)}
    return out
