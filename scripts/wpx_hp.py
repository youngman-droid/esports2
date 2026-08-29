"""HP features for the WP model, from backfilled feed minutes.

Per state (nearest feed minute within 90 s of game clock):
  hp_pool   sum of HP fractions, blue - red (dead = 0)
  hp_low_b/r  players below 30% HP (incl. dead), per side
  lvl_k     (sum levels blue - red) / 5
  has_hp    coverage flag (1 if a feed minute matched)
Time-forward eval: train < 2026-05-25, test after.
"""
import datetime as dt, json, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from lol_ticker import db, wpx
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("hp")
conn = db.connect()
d = np.load(os.path.join(wpx.OUT_DIR, "states.npz"), allow_pickle=True)
X, y, gid, t, pm, ks, C = d["X"], d["y"], d["gid"], d["t"], d["pm"], d["ks"], d["C"]
fn = list(d["names"]); cn = list(d["champ_names"])
g2e = {r["golgg_game_id"]: r["esports_game_id"] for r in conn.execute(
    "SELECT golgg_game_id, esports_game_id FROM feed_games WHERE status='scraped' AND golgg_game_id IS NOT NULL")}
fm = {}
for r in conn.execute("SELECT esports_game_id, minute, data FROM feed_minutes"):
    fm.setdefault(r["esports_game_id"], {})[r["minute"]] = r["data"]
log.info("feed minutes for %d games", len(fm))
F = np.zeros((len(y), 5), dtype=np.float32)   # hp_pool, hp_low_b, hp_low_r, lvl_k, has_hp
cov_states = 0
for i in range(len(y)):
    eg = g2e.get(int(gid[i]))
    if not eg or eg not in fm: continue
    m = int(round(t[i] / 60.0))
    row = fm[eg].get(m) or fm[eg].get(m - 1) or fm[eg].get(m + 1)
    if not row: continue
    hpb = [x for x in (row.get("hpb") or []) if x is not None]
    hpr = [x for x in (row.get("hpr") or []) if x is not None]
    lvb, lvr = row.get("lvb") or [], row.get("lvr") or []
    if len(hpb) != 5 or len(hpr) != 5: continue
    F[i, 0] = sum(hpb) - sum(hpr)
    F[i, 1] = sum(1 for x in hpb if x < 0.3)
    F[i, 2] = sum(1 for x in hpr if x < 0.3)
    F[i, 3] = (sum(lvb) - sum(lvr)) / 5.0 if len(lvb) == 5 and len(lvr) == 5 else 0.0
    F[i, 4] = 1.0
    cov_states += 1
log.info("HP joined to %d/%d states (%.0f%%)", cov_states, len(y), 100.0 * cov_states / len(y))
for j, nm in enumerate(("hp_pool", "hp_low_b", "hp_low_r", "lvl_k")):
    m_ = F[:, 4] > 0
    log.info("  %-8s mean %+.3f sd %.3f corr(y|covered) %+.3f", nm, F[m_, j].mean(), F[m_, j].std(), np.corrcoef(F[m_, j], y[m_])[0, 1])
dates = {r["game_id"]: r["date"] for r in conn.execute("select game_id, date from golgg_games")}
cut = np.quantile([dates[int(g)].toordinal() for g in np.unique(gid) if int(g) in dates], 0.8)
tr = np.array([dates.get(int(g), dt.date(2000, 1, 1)).toordinal() < cut for g in gid]); te = ~tr
log.info("coverage in future test window: %.0f%%", 100.0 * F[te, 4].mean())
idx = {n: i for i, n in enumerate(fn)}
rich_cols = [i for i, n in enumerate(fn) if n != "draft"]
t_col = idx["t"]; xcols = [i for i in rich_cols if fn[i] not in wpx.ALREADY_INTERACTED and fn[i] != "bias"]
def expand(Xm, ii, Fm=None):
    B = Xm[:, rich_cols].astype(np.float32)
    t_raw = Xm[:, t_col:t_col + 1].astype(np.float32); tt = np.minimum(t_raw, 0.5)
    S = np.zeros((len(Xm), len(cn)), dtype=np.float32); Cm = C[ii]
    for k in range(10):
        sgn = 1.0 if k < 5 else -1.0
        ids = Cm[:, k]; ok = ids >= 0
        S[np.where(ok)[0], ids[ok]] += sgn * tt[ok, 0]
    parts = [B, Xm[:, xcols].astype(np.float32) * t_raw, S]
    if Fm is not None:
        parts.append(Fm); parts.append(Fm[:, :4] * t_raw)   # HP features also x time
    return np.hstack(parts)
def irls(A, yv, reg, chunk=65536):
    beta = np.zeros(A.shape[1])
    for it in range(25):
        H = np.diag(reg).astype(np.float64).copy(); g = (reg * beta).copy()
        for lo in range(0, len(A), chunk):
            Ai = A[lo:lo + chunk].astype(np.float64); z = Ai @ beta
            p_ = 1 / (1 + np.exp(-z)); W = p_ * (1 - p_) + 1e-9
            H += (Ai * W[:, None]).T @ Ai; g += Ai.T @ (p_ - yv[lo:lo + chunk])
        step = np.linalg.solve(H, g); beta -= step
        if np.max(np.abs(step)) < 1e-5: break
    return beta
hp_, hk_ = (~np.isnan(pm))[te], (~np.isnan(ks))[te]
def report(tag, p):
    log.info("%-32s future: all %.4f  @PM %.4f  @KS %.4f", tag, np.mean((p - y[te]) ** 2), np.mean((p[hp_] - y[te][hp_]) ** 2), np.mean((p[hk_] - y[te][hk_]) ** 2))
    cm = F[te, 4] > 0
    log.info("%-32s   covered-states only: %.4f (n=%d)", "", np.mean((p[cm] - y[te][cm]) ** 2), cm.sum())
A0t, A0e = expand(X[tr], np.where(tr)[0]), expand(X[te], np.where(te)[0])
reg0 = np.full(A0t.shape[1], 1.0); reg0[0] = 0; reg0[-len(cn):] = 800.0
b0 = irls(A0t, y[tr], reg0)
report("base", 1 / (1 + np.exp(-(A0e.astype(np.float64) @ b0))))
A1t, A1e = expand(X[tr], np.where(tr)[0], F[tr]), expand(X[te], np.where(te)[0], F[te])
for l2h in (1.0, 10.0):
    reg1 = np.concatenate([reg0, np.full(9, l2h)])
    b1 = irls(A1t, y[tr], reg1)
    p1 = 1 / (1 + np.exp(-(A1e.astype(np.float64) @ b1)))
    report("base + HP (l2=%g)" % l2h, p1)
    if l2h == 1.0:
        names_hp = ["hp_pool", "hp_low_b", "hp_low_r", "lvl_k", "has_hp", "hp_pool_xt", "hp_low_b_xt", "hp_low_r_xt", "lvl_k_xt"]
        log.info("HP coefs: %s", {n: round(float(v), 3) for n, v in zip(names_hp, b1[-9:])})
        np.save(os.path.join(wpx.OUT_DIR, "hp_features.npy"), F)
