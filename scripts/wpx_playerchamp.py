"""Experiment: add PLAYER x CHAMPION (and player) prior features to the live
model and test in the same game-level 5-fold CV + fair minute-mark comparison.

Features (blue minus red, leak-free: only games strictly before this one):
  pc_wr    sum over 5 players of [shrunk win rate on this champ - shrunk overall win rate]  (OE, 27k games)
  pc_exp   sum log1p(#prior games on this champ)                                            (OE)
  pc_new   #players with 0 prior pro games on the champ                                     (OE)
  pc_gd15  sum [shrunk mean GD@15 on this champ - shrunk mean GD@15 overall] / 1000          (gol.gg)
  p_gd15   sum shrunk mean prior GD@15 overall / 1000 (player laning strength)               (gol.gg)
"""
import json, logging, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import defaultdict
from lol_ticker import db, wpx, wpx_fair
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("pc")
conn = db.connect()

# ---------------- OE: player x champion win-rate mastery -----------------
games = {r["game_id"]: r for r in conn.execute("SELECT game_id, date_utc, blue_team, red_team, winner FROM oe_games WHERE date_utc IS NOT NULL AND winner IS NOT NULL")}
pl = defaultdict(dict)   # (game, team) -> {position: player}
for r in conn.execute("SELECT game_id, team, position, player_id, player FROM oe_players"):
    pl[(r["game_id"], r["team"])][r["position"]] = r["player_id"] or r["player"]
pk = defaultdict(dict)   # (game, team) -> {position: champion}
for r in conn.execute("SELECT game_id, team, position, champion FROM oe_picks WHERE position IS NOT NULL"):
    pk[(r["game_id"], r["team"])][r["position"]] = r["champion"]
n_pc, w_pc, n_p, w_p = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
K_PC, K_P = 5.0, 10.0
oe_feat = {}
for gid, g in sorted(games.items(), key=lambda kv: (kv[1]["date_utc"], kv[0])):
    side_vals = {}
    for team in (g["blue_team"], g["red_team"]):
        players, champs = pl.get((gid, team), {}), pk.get((gid, team), {})
        wr_sum = exp_sum = new = 0.0
        for pos, p in players.items():
            c = champs.get(pos)
            if not p or not c: continue
            wrp = (w_p[p] + K_P * 0.5) / (n_p[p] + K_P)
            wrpc = (w_pc[(p, c)] + K_PC * wrp) / (n_pc[(p, c)] + K_PC)
            wr_sum += wrpc - wrp; exp_sum += np.log1p(n_pc[(p, c)]); new += (n_pc[(p, c)] == 0)
        side_vals[team] = (wr_sum, exp_sum, new)
    b, r_ = side_vals.get(g["blue_team"], (0, 0, 0)), side_vals.get(g["red_team"], (0, 0, 0))
    oe_feat[gid] = (b[0] - r_[0], b[1] - r_[1], b[2] - r_[2])
    # update AFTER recording (leak-free)
    for team in (g["blue_team"], g["red_team"]):
        won = 1 if g["winner"] == team else 0
        players, champs = pl.get((gid, team), {}), pk.get((gid, team), {})
        for pos, p in players.items():
            c = champs.get(pos)
            if not p or not c: continue
            n_p[p] += 1; w_p[p] += won; n_pc[(p, c)] += 1; w_pc[(p, c)] += won
log.info("OE player-champ features for %d games", len(oe_feat))

# ---------------- gol.gg: player x champion GD@15 mastery ----------------
gg = {r["game_id"]: r for r in conn.execute("SELECT game_id, date, oe_game_id FROM golgg_games WHERE date IS NOT NULL")}
rows = conn.execute("""SELECT game_id, player_id, side, champion, (stats->>'GD@15')::float gd15
                       FROM golgg_players WHERE stats IS NOT NULL AND stats->>'GD@15' ~ '^-?[0-9]+'""").fetchall()
by_game = defaultdict(list)
for r in rows: by_game[r["game_id"]].append(r)
s_pc, c_pc, s_p, c_p = defaultdict(float), defaultdict(int), defaultdict(float), defaultdict(int)
K2_PC, K2_P = 4.0, 8.0
gg_feat = {}
for gid in sorted(by_game, key=lambda g: (gg[g]["date"] if g in gg else None, g) if g in gg else (None, g)):
    if gid not in gg: continue
    vals = {"blue": [0.0, 0.0], "red": [0.0, 0.0]}
    for r in by_game[gid]:
        p, c = r["player_id"], r["champion"]
        mp = s_p[p] / (c_p[p] + K2_P)                       # shrunk toward 0
        mpc = (s_pc[(p, c)] + K2_PC * mp) / (c_pc[(p, c)] + K2_PC)
        vals[r["side"]][0] += (mpc - mp) / 1000.0; vals[r["side"]][1] += mp / 1000.0
    gg_feat[gid] = (vals["blue"][0] - vals["red"][0], vals["blue"][1] - vals["red"][1])
    for r in by_game[gid]:
        p, c = r["player_id"], r["champion"]
        s_p[p] += r["gd15"]; c_p[p] += 1; s_pc[(p, c)] += r["gd15"]; c_pc[(p, c)] += 1
log.info("gol.gg player-champ GD@15 features for %d games", len(gg_feat))

# ---------------- attach to states ----------------
d = np.load(os.path.join(wpx.OUT_DIR, "states.npz"), allow_pickle=True)
X, y, gid_s, t, pm, ks, C = d["X"], d["y"], d["gid"], d["t"], d["pm"], d["ks"], d["C"]
names = list(d["names"]); n_champs = len(d["champ_names"])
oe_of = {g: r["oe_game_id"] for g, r in gg.items()}
F = np.zeros((len(y), 5), dtype=np.float32)
cov_oe = cov_gg = 0
ug = np.unique(gid_s)
for g in ug:
    m = gid_s == g
    o = oe_feat.get(oe_of.get(int(g)))
    if o: F[m, 0:3] = o; cov_oe += 1
    q = gg_feat.get(int(g))
    if q: F[m, 3:5] = q; cov_gg += 1
log.info("coverage: OE features %d/%d games, gol.gg features %d/%d games", cov_oe, len(ug), cov_gg, len(ug))
new_names = ["pc_wr", "pc_exp", "pc_new", "pc_gd15", "p_gd15"]
for j, n in enumerate(new_names):
    v = F[:, j]; log.info("  %-8s mean %+.3f sd %.3f  corr(y) %+.3f", n, v.mean(), v.std(), np.corrcoef(v, y)[0, 1])

folds = wpx._folds(gid_s, 5)
has_pm, has_ks = ~np.isnan(pm), ~np.isnan(ks)
def cv(Xm, nm, label):
    zoo = wpx.model_zoo(nm, C, n_champs)
    fp = zoo["champscale_reg(l2=200,cap25)"]
    oof = np.zeros(len(y)); t0 = time.time()
    for f in range(5):
        tr, te = folds != f, folds == f
        oof[te] = fp(Xm[tr], y[tr], Xm[te], np.where(tr)[0], np.where(te)[0])
    b = lambda mk: float(np.mean((oof[mk] - y[mk]) ** 2))
    late = has_pm & (t >= 1500); early = has_pm & (t < 900)
    log.info("%-34s Brier all %.4f  @PM %.4f  @KS %.4f | PM early %.4f late %.4f  (%.0fs)", label, b(np.ones(len(y), bool)), b(has_pm), b(has_ks), b(early), b(late), time.time() - t0)
    return oof
res = {}
res["base"] = cv(X, names, "shipped (elo + pelo + champs)")
sets = {"+pc_wr/exp/new (OE)": [0, 1, 2], "+pc_gd15/p_gd15 (gol.gg)": [3, 4], "+all five": [0, 1, 2, 3, 4], "+p_gd15 only (player, no champ)": [4], "+pc_gd15 only (player x champ)": [3]}
for label, cols in sets.items():
    Xe = np.hstack([X, F[:, cols]]); ne = names + [new_names[c] for c in cols]
    res[label] = cv(Xe, ne, label)
np.save(os.path.join(wpx.OUT_DIR, "oof_pc_all.npy"), res["+all five"])
np.save(os.path.join(wpx.OUT_DIR, "oof_pc_base.npy"), res["base"])
for nm in ("pc_base", "pc_all"):
    for lag in (0, -45):
        r = wpx_fair.run(conn, oof_name=nm, lag_s=lag)
        log.info("FAIR %s lag=%d: %s", nm, lag, {p: {k: round(v, 4) for k, v in vv.items() if k in ("n", "market", "model", "model_late", "market_late")} for p, vv in r.items()})
# also: how much do the new features move the SHFT-KC-style state? (report coefficient signs via full fit)
np.save(os.path.join(wpx.OUT_DIR, "pc_features.npy"), F)
