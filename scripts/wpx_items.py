"""Experiment: score SPECIFIC ITEM CHOICES in the live WP model.

Per completed item (>=2000g, >=200 purchases) K, per state: signed ownership
(blue-owned minus red-owned) plus interactions with the OWNER'S ENEMY profile
(physical-damage share, magic share, healing, crit propensity - all empirical
per champion from gol.gg data). The fit learns coef_K(enemy) = a_K + b_K*phys
+ c_K*magic + d_K*heal + e_K*crit, so e.g. Randuin's into a no-crit comp is
worth less than into a crit comp - without hand-tagging any item.
Items enter BEYOND items_done/item_gold_k, which stay in the base state.
Validation: time-forward (train < 80th pct date, test after).
"""
import json, logging, os, sys, time, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import defaultdict
from lol_ticker import db, wpx
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("items")
conn = db.connect()

d = np.load(os.path.join(wpx.OUT_DIR, "states.npz"), allow_pickle=True)
X, y, gid, t, pm, ks, C = d["X"], d["y"], d["gid"], d["t"], d["pm"], d["ks"], d["C"]
fn = list(d["names"]); cn = list(d["champ_names"])
n = len(y); ugids = [int(g) for g in np.unique(gid)]
log.info("%d states / %d games", n, len(ugids))

# ---------------- item vocab ----------------
vocab_rows = conn.execute("""SELECT b.item_id::int iid, i.name, count(*) cnt FROM golgg_builds b
    JOIN golgg_items i ON i.item_id = b.item_id::int
    WHERE b.event = 'ITEM_PURCHASED' AND i.gold >= 2000
    GROUP BY 1, 2 HAVING count(*) >= 200 ORDER BY cnt DESC""").fetchall()
vocab = [r["iid"] for r in vocab_rows]; iname = {r["iid"]: r["name"] for r in vocab_rows}
V = len(vocab); vidx = {iid: k for k, iid in enumerate(vocab)}
log.info("item vocab: %d items", V)
CRIT_NAMES = {"Infinity Edge", "Lord Dominik's Regards", "Mortal Reminder", "Phantom Dancer",
              "Rapid Firecannon", "Runaan's Hurricane", "Statikk Shiv", "The Collector",
              "Essence Reaver", "Navori Flickerblade", "Yun Tal Wildarrows", "Stormrazor"}
crit_ids = {iid for iid, nm in iname.items() if nm in CRIT_NAMES}
log.info("crit items found: %s", sorted(iname[i] for i in crit_ids))

# ---------------- champion profiles (empirical) ----------------
prof_rows = conn.execute("""SELECT champion,
        avg(nullif(regexp_replace(stats->>'Physical Damage', '[^0-9]', '', 'g'), '')::float) phys,
        avg(nullif(regexp_replace(stats->>'Magic Damage', '[^0-9]', '', 'g'), '')::float) magic,
        avg(nullif(regexp_replace(stats->>'True Damage', '[^0-9]', '', 'g'), '')::float) tru,
        avg(nullif(regexp_replace(stats->>'Total heal', '[^0-9]', '', 'g'), '')::float) heal,
        count(*) cnt
    FROM golgg_players WHERE stats IS NOT NULL GROUP BY champion HAVING count(*) >= 20""").fetchall()
_crit_q = """SELECT p.champion, avg((EXISTS (SELECT 1 FROM golgg_builds b WHERE b.game_id = p.game_id
        AND b.player_id = p.player_id AND b.event = 'ITEM_PURCHASED'
        AND b.item_id = ANY(%s)))::int) frac
    FROM golgg_players p GROUP BY p.champion"""
crit_by_champ = {r["champion"]: float(r["frac"]) for r in conn.execute(_crit_q, (list(crit_ids),))}
prof = {}
for r in prof_rows:
    tot = (r["phys"] or 0) + (r["magic"] or 0) + (r["tru"] or 0)
    if tot <= 0: continue
    prof[r["champion"]] = ((r["phys"] or 0) / tot, (r["magic"] or 0) / tot,
                           (r["heal"] or 0), float(crit_by_champ.get(r["champion"], 0.0)))
heals = np.array([v[2] for v in prof.values()]); hm, hs = heals.mean(), heals.std()
prof = {c: (p, m, (h - hm) / hs, cr) for c, (p, m, h, cr) in prof.items()}
log.info("profiles for %d champions; e.g. %s", len(prof),
         {c: tuple(round(x, 2) for x in prof[c]) for c in ("Jinx", "Ahri", "Milio", "Ornn") if c in prof})

# ---------------- per-game per-player item timelines ----------------
gset = set(ugids)
side_ch = defaultdict(dict)   # game -> player_id -> (side, champion)
for r in conn.execute("SELECT game_id, player_id, side, champion FROM golgg_players"):
    if r["game_id"] in gset:
        side_ch[r["game_id"]][str(r["player_id"])] = (r["side"], r["champion"])
tl = defaultdict(list)        # (game, player) -> [(time, vocab_k, +1/-1)]
qr = conn.execute("""SELECT game_id::int g, player_id, build_time, event, item_id::int iid
                     FROM golgg_builds WHERE event IN ('ITEM_PURCHASED','ITEM_SOLD','ITEM_UNDO')""")
n_ev = 0
for r in qr:
    if r["g"] not in gset or r["iid"] not in vidx: continue
    tl[(r["g"], str(r["player_id"]))].append((r["build_time"] or 0, vidx[r["iid"]], 1 if r["event"] == 'ITEM_PURCHASED' else -1))
    n_ev += 1
log.info("timelines: %d events over %d (game,player)", n_ev, len(tl))

# ---------------- per-state ownership + enemy-profile interactions ----------------
_cache_path = os.path.join(wpx.OUT_DIR, "item_states.npz")
_c = np.load(_cache_path, allow_pickle=True) if os.path.exists(_cache_path) else None
if _c is not None and list(_c["vocab"]) == vocab and _c["D"].shape == (n, V):
    D, INT = _c["D"], _c["INT"]
    log.info("ownership loaded from cache")
else:
    _c = None
D = np.zeros((n, V), dtype=np.float32) if _c is None else D        # blue-owned - red-owned
INT = np.zeros((n, V, 4), dtype=np.float32) if _c is None else INT   # x enemy phys, magic, heal(z), crit
SKIP_BUILD = _c is not None
order = np.argsort(gid, kind="stable")
prof_default = (0.55, 0.35, 0.0, 0.2)
i0 = 0
rows_by_gid = defaultdict(list)
for i in order: rows_by_gid[int(gid[i])].append(i)
for g, rows in ({} if SKIP_BUILD else rows_by_gid).items():
    pls = side_ch.get(g, {})
    team_prof = {"blue": [], "red": []}
    for pid, (side, ch) in pls.items():
        team_prof[side].append(prof.get(ch, prof_default))
    ep = {s: tuple(np.mean([p[k] for p in team_prof[s]] or [prof_default[k]]) for k in range(4)) for s in ("blue", "red")}
    per_pl = []
    for pid, (side, ch) in pls.items():
        ev = sorted(tl.get((g, pid), []))
        if ev: per_pl.append((side, ev))
    ts_rows = sorted(rows, key=lambda i: t[i])
    for side, ev in per_pl:
        sgn = 1.0 if side == "blue" else -1.0
        e_ = ep["red" if side == "blue" else "blue"]
        owned = np.zeros(V, dtype=np.int8)
        j = 0
        for i in ts_rows:
            while j < len(ev) and ev[j][0] <= t[i]:
                owned[ev[j][1]] += ev[j][2]; j += 1
            ow = np.clip(owned, 0, 2)
            nz = np.nonzero(ow)[0]
            if len(nz):
                D[i, nz] += sgn * ow[nz]
                for k in range(4):
                    INT[i, nz, k] += sgn * ow[nz] * e_[k]
        # reset per player: recompute owned per player fresh (j restarts each player) - done by loop structure
log.info("ownership %s: mean |D| per state %.2f", "cached" if SKIP_BUILD else "built", np.abs(D).sum(axis=1).mean())
if not SKIP_BUILD: np.savez_compressed(os.path.join(wpx.OUT_DIR, "item_states.npz"), D=D, INT=INT,
                    vocab=np.array(vocab), names=np.array([iname[i] for i in vocab]))

# ---------------- time-forward eval ----------------
dates = {r["game_id"]: r["date"] for r in conn.execute("select game_id, date from golgg_games")}
cut = np.quantile([dates[g].toordinal() for g in ugids if g in dates], 0.8)
tr = np.array([dates.get(int(g), dt.date(2000, 1, 1)).toordinal() < cut for g in gid]); te = ~tr
idx = {nm: i for i, nm in enumerate(fn)}
rich_cols = [i for i, nm in enumerate(fn) if nm != "draft"]
t_col = idx["t"]; xcols = [i for i in rich_cols if fn[i] not in wpx.ALREADY_INTERACTED and fn[i] != "bias"]
def base_expand(Xm, ii):
    B = Xm[:, rich_cols].astype(np.float32)
    t_raw = Xm[:, t_col:t_col + 1].astype(np.float32); tt = np.minimum(t_raw, 0.5)
    S = np.zeros((len(Xm), len(cn)), dtype=np.float32); Cm = C[ii]
    for k in range(10):
        sgn = 1.0 if k < 5 else -1.0
        ids = Cm[:, k]; ok = ids >= 0
        S[np.where(ok)[0], ids[ok]] += sgn * tt[ok, 0]
    return np.hstack([B, Xm[:, xcols].astype(np.float32) * t_raw, S])
def irls(A, yv, reg, chunk=65536):
    """Newton/IRLS with float64 accumulation of H and the gradient in chunks.
    A stays float32 (memory); float32 H accumulation at ~500k rows carries
    ~1e-3 relative error, which makes Newton oscillate instead of converging."""
    n_, p_ = A.shape
    beta = np.zeros(p_)
    for it in range(25):
        H = np.diag(reg).astype(np.float64).copy()
        gvec = (reg * beta).copy()
        for lo in range(0, n_, chunk):
            Ai = A[lo:lo + chunk].astype(np.float64)
            z = Ai @ beta
            pi = 1 / (1 + np.exp(-z)); Wi = pi * (1 - pi) + 1e-9
            H += (Ai * Wi[:, None]).T @ Ai
            gvec += Ai.T @ (pi - yv[lo:lo + chunk])
        step = np.linalg.solve(H, gvec)
        beta -= step
        if np.max(np.abs(step)) < 1e-5: break
    return beta
hp, hk = (~np.isnan(pm))[te], (~np.isnan(ks))[te]
def report(label, p):
    log.info("%-40s future: all %.4f  @PM %.4f  @KS %.4f", label,
             np.mean((p - y[te]) ** 2), np.mean((p[hp] - y[te][hp]) ** 2), np.mean((p[hk] - y[te][hk]) ** 2))
A_tr, A_te = base_expand(X[tr], np.where(tr)[0]), base_expand(X[te], np.where(te)[0])
nb = A_tr.shape[1]
reg0 = np.full(nb, 1.0); reg0[0] = 0; reg0[-len(cn):] = 800.0
beta0 = irls(A_tr, y[tr], reg0)
report("base (champscale l2=800 cap15)", 1 / (1 + np.exp(-(A_te @ beta0))))
for l2i, l2x, tag in ((50.0, 200.0, "items l2=50/int200"), (200.0, 800.0, "items l2=200/int800"), (25.0, 100.0, "items l2=25/int100")):
    At = np.hstack([A_tr, D[tr], INT[tr].reshape(tr.sum(), V * 4)])
    Ae = np.hstack([A_te, D[te], INT[te].reshape(te.sum(), V * 4)])
    reg = np.concatenate([reg0, np.full(V, l2i), np.full(V * 4, l2x)])
    beta = irls(At, y[tr], reg)
    report("base + " + tag, 1 / (1 + np.exp(-(Ae @ beta))))
    np.save(os.path.join(wpx.OUT_DIR, "item_beta_%s.npy" % tag.replace("/", "_").replace("=", "")), beta)
    del At, Ae
# interpretation for the last beta: base coefs + crit interaction for a few items
bD = beta[nb:nb + V]; bI = beta[nb + V:].reshape(V, 4)
top = np.argsort(-bD)[:8]; bot = np.argsort(bD)[:8]
log.info("TOP items (base coef): %s", [(iname[vocab[i]], round(float(bD[i]), 3)) for i in top])
log.info("BOTTOM items (base coef): %s", [(iname[vocab[i]], round(float(bD[i]), 3)) for i in bot])
for want in ("Randuin's Omen", "Thornmail", "Zhonya's Hourglass", "Kaenic Rookern", "Frozen Heart"):
    if want in iname.values():
        k = [i for i, iid in enumerate(vocab) if iname[iid] == want][0]
        log.info("%s: base %+0.3f  x_phys %+0.3f  x_magic %+0.3f  x_heal %+0.3f  x_crit %+0.3f",
                 want, bD[k], bI[k, 0], bI[k, 1], bI[k, 2], bI[k, 3])
