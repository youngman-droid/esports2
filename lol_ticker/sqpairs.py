"""Solo-queue matchup/synergy prior (Lolalytics) as one draft score.

Pair effect = 0.04 * d2 (Lolalytics' pair win rate in pp beyond both champions'
baselines -> logit), shrunk by n/(n+k) with k = 4/tau^2 estimated from the
solo-queue tables alone.  A pro map on patch P is scored from the n-weighted pool
of patches strictly BEFORE P, so no look-ahead and a new patch works on day one.
No professional outcome ever enters the score.  Gated 2026-09-17:
docs/sq-pair-prior-2026-09-17.md.
"""
import json
import logging
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request

from . import config

log = logging.getLogger(__name__)

ROOT = os.path.join(config.REPO_ROOT, "data", "sq", "lolalytics")
TABLES_PATH = os.path.join(config.REPO_ROOT, "data", "sq", "pair_tables.npz")
LANES = ["top", "jungle", "middle", "bottom", "support"]
OE2L = dict(top="top", jng="jungle", mid="middle", bot="bottom", sup="support")
SPECIAL = {"nunuwillump": "nunu", "renataglasc": "renata", "monkeyking": "wukong"}
UA = "Mozilla/5.0 (research; esports2 private analysis)"
PP_TO_LOGIT = 0.04
# The score enters regressions divided by this (about its sd over pro drafts) so
# a unit ridge penalty means the same as for the 0/1 draft indicators.
SCALE = 0.25
MIN_PRO_PICKS = 3
REFRESH_AGE_S = 6 * 86400
PAGES_PER_COMBO_MIN = 2


def key(name):
    k = re.sub("[^a-z0-9]", "", unicodedata.normalize("NFKD", name or "")
               .encode("ascii", "ignore").decode().lower())
    return SPECIAL.get(k, k)


def pnum(patch):
    return tuple(int(x) for x in str(patch).split("."))


def canon(patch):
    """'16.09' -> '16.9' (Lolalytics spelling); '' -> None."""
    try:
        return ".".join(str(x) for x in pnum(patch))
    except ValueError:
        return None


# ------------------------------------------------------------------ scrape

def _fetch(query):
    req = urllib.request.Request(
        "https://a1.lolalytics.com/mega/?%s&tier=emerald_plus&queue=ranked&region=all" % query,
        headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _combos(conn, patches):
    oe = sorted({p for q in patches for p in ("%d.%02d" % pnum(q), canon(q))})
    rows = conn.execute(
        "SELECT p.champion, p.position, count(*) AS n FROM oe_picks p JOIN oe_games g USING (game_id) "
        "WHERE g.patch = ANY(%s) GROUP BY 1,2 HAVING count(*) >= %s", (oe, MIN_PRO_PICKS)).fetchall()
    return sorted({(key(r["champion"]), OE2L[r["position"]]) for r in rows if r["position"] in OE2L})


def _jobs(patch, combos):
    for k, lane in combos:
        yield lane, k, "team", "ep=build-team&v=1&patch=%s&c=%s&lane=%s" % (patch, k, lane)
        for vs in LANES[LANES.index(lane):]:     # reverse direction is 1 - wr
            yield lane, k, "vs-" + vs, "ep=counter&v=1&patch=%s&c=%s&lane=%s&vslane=%s" % (patch, k, lane, vs)


def patch_available(patch):
    try:
        return "stats" in json.loads(_fetch("ep=counter&v=1&patch=%s&c=aatrox&lane=top" % patch))
    except Exception:
        return False


def scrape_patch(patch, combos, refetch_older_than=None, delay=1.0):
    """Resumable. With refetch_older_than (seconds) stale files are re-fetched --
    used while a patch is still live and its counts keep growing."""
    pdir = os.path.join(ROOT, patch)
    os.makedirs(pdir, exist_ok=True)
    fetched = errors = 0
    for lane, k, what, query in _jobs(patch, combos):
        path = os.path.join(pdir, "%s-%s-%s.json" % (lane, k, what))
        if os.path.exists(path) and (refetch_older_than is None
                                     or time.time() - os.path.getmtime(path) < refetch_older_than):
            continue
        try:
            body = _fetch(query)
            json.loads(body)
            with open(path + ".tmp", "wb") as f:
                f.write(body)
            os.replace(path + ".tmp", path)
            fetched += 1
            errors = 0
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):        # blocked: stop, never work around it
                raise RuntimeError("lolalytics returned %d; stopping" % e.code)
            errors += 1
        except Exception as e:
            errors += 1
            log.warning("sqpairs fetch failed %s: %r", path, e)
        if errors >= 10:
            raise RuntimeError("10 consecutive fetch errors; stopping")
        time.sleep(delay)
    return fetched


def refresh(conn, delay=1.0):
    """Nightly entry point: keep the newest pro patch and the next live patch
    current (weekly re-fetch while a patch is not final), then rebuild tables."""
    newest = conn.execute(
        "SELECT patch FROM oe_games WHERE patch ~ '^[0-9]+\\.[0-9]+$' AND date_utc IS NOT NULL "
        "ORDER BY date_utc DESC LIMIT 1").fetchone()
    if not newest:
        return {"fetched": 0}
    major, minor = pnum(newest["patch"])
    have = sorted((p for p in os.listdir(ROOT) if re.match(r"^\d+\.\d+$", p)), key=pnum) if os.path.isdir(ROOT) else []
    candidates = ["%d.%d" % (major, minor + i) for i in (0, 1, 2)]
    live = [p for p in candidates if patch_available(p)]
    total = 0
    for p in live:
        final = any(pnum(q) > pnum(p) for q in live)       # a newer patch exists -> counts are frozen
        marker = os.path.join(ROOT, p, ".final")
        if os.path.exists(marker):
            continue
        recent = [q for q in have + [p] if pnum(q) >= (major, minor - 2)]
        n = scrape_patch(p, _combos(conn, recent), refetch_older_than=REFRESH_AGE_S, delay=delay)
        total += n
        if final:
            open(marker, "w").close()
        log.info("sqpairs: patch %s fetched %d pages%s", p, n, " (final)" if final else "")
    build_tables()
    return {"fetched": total, "patches": live}


# ------------------------------------------------------------------ tables

def _read_patch(pdir):
    match, syn, cids = {}, {}, {}
    names = sorted(os.listdir(pdir))
    for name in names:
        if "-vs-" not in name or not name.endswith(".json"):
            continue
        with open(os.path.join(pdir, name)) as f:
            j = json.load(f)
        s = j.get("stats")
        if not s:
            continue
        cids[name.split("-")[1]] = s["cid"]
        a = (LANES.index(s["lane"]), s["cid"])
        for r in j.get("counters") or []:
            if r["n"] <= 0:
                continue
            b = (LANES.index(s["vsLane"]), r["cid"])
            v = PP_TO_LOGIT * r["d2"]
            for k, val in (((a, b), v), ((b, a), -v)):
                # same-lane pairs are reported from both pages: average them
                match[k] = ((match[k][0] + val) / 2, max(match[k][1], r["n"])) if k in match else (val, r["n"])
    for name in names:
        if not name.endswith("-team.json"):
            continue
        lane, k = name.split("-")[:2]
        if k not in cids:
            continue
        with open(os.path.join(pdir, name)) as f:
            j = json.load(f)
        if "team" not in j:
            continue
        a = (LANES.index(lane), cids[k])
        h = j["team_h"]
        for mate_lane, rows in j["team"].items():
            for r in rows:
                r = dict(zip(h, r))
                if r["n"] <= 0:
                    continue
                kk = tuple(sorted((a, (LANES.index(mate_lane), r["id"]))))
                val = PP_TO_LOGIT * r["d2"]
                syn[kk] = ((syn[kk][0] + val) / 2, max(syn[kk][1], r["n"])) if kk in syn else (val, r["n"])
    return match, syn, cids


def _pool(tables):
    acc = {}
    for t in tables:
        for k, (v, n) in t.items():
            e = acc.setdefault(k, [0.0, 0.0])
            e[0] += v * n
            e[1] += n
    return {k: (s / n, n) for k, (s, n) in acc.items()}


def _shrink_k(table):
    import numpy as np
    v = np.array([x[0] for x in table.values()])
    n = np.array([x[1] for x in table.values()], dtype=float)
    w = n / n.sum()
    tau2 = max(float(np.sum(w * v * v) - np.sum(w * 4.0 / n)), 1e-5)   # var of a logit from n games ~ 4/n
    return 4.0 / tau2


def build_tables(path=TABLES_PATH):
    """Compact every scraped patch into one artifact (entries + cid map + k)."""
    import numpy as np
    patches = sorted((p for p in os.listdir(ROOT) if re.match(r"^\d+\.\d+$", p)), key=pnum)
    recs = {0: [], 1: []}
    cids, per = {}, {}
    for pi, p in enumerate(patches):
        match, syn, c = _read_patch(os.path.join(ROOT, p))
        cids.update(c)
        per[p] = (match, syn)
        for kind, t in ((0, match), (1, syn)):
            for (a, b), (v, n) in t.items():
                if kind == 0 and a > b:
                    continue            # antisymmetric: store one direction
                recs[kind].append((pi, a[0], a[1], b[0], b[1], v, n))
    k_match = _shrink_k(_pool([per[p][0] for p in patches]))
    k_syn = _shrink_k(_pool([per[p][1] for p in patches]))
    np.savez_compressed(path, patches=np.array(patches), match=np.array(recs[0], dtype=np.float64),
                        syn=np.array(recs[1], dtype=np.float64), keys=np.array(sorted(cids)),
                        cids=np.array([cids[k] for k in sorted(cids)]), k_match=k_match, k_syn=k_syn)
    log.info("sqpairs: tables built for %s (k_match=%.0f, k_syn=%.0f)", patches, k_match, k_syn)
    return {"patches": patches, "k_match": k_match, "k_syn": k_syn}


class Scorer:
    """score(patch, blue, red): champion names in top/jng/mid/bot/sup order."""

    def __init__(self, path=TABLES_PATH):
        import numpy as np
        d = np.load(path, allow_pickle=False)
        self.patches = [str(p) for p in d["patches"]]
        self.cid = dict(zip((str(k) for k in d["keys"]), (int(c) for c in d["cids"])))
        self.k = (float(d["k_match"]), float(d["k_syn"]))
        self._raw = (d["match"], d["syn"])
        self._pools = {}

    def _pooled(self, upto):
        """Shrunk pair tables from patches strictly before index ``upto``."""
        if upto not in self._pools:
            out = []
            for kind, raw in enumerate(self._raw):
                acc = {}
                for pi, la, ca, lb, cb, v, n in raw[raw[:, 0] < upto]:
                    e = acc.setdefault((int(la), int(ca), int(lb), int(cb)), [0.0, 0.0])
                    e[0] += v * n
                    e[1] += n
                k = self.k[kind]
                out.append({key_: s / (n + k) for key_, (s, n) in acc.items()})   # (s/n) * n/(n+k)
            self._pools[upto] = out
        return self._pools[upto]

    def score(self, patch, blue, red):
        p = canon(patch)
        if p is None:
            return None
        upto = sum(1 for q in self.patches if pnum(q) < pnum(p))
        if upto == 0:
            return None
        try:
            b = [(i, self.cid[key(c)]) for i, c in enumerate(blue)]
            r = [(i, self.cid[key(c)]) for i, c in enumerate(red)]
        except KeyError:
            return None
        if len(b) != 5 or len(r) != 5:
            return None
        match, syn = self._pooled(upto)
        m = s = 0.0
        found = 0
        for x in b:
            for y in r:
                if x + y in match:
                    m += match[x + y]; found += 1
                elif y + x in match:
                    m -= match[y + x]; found += 1
        for sign, side in ((1.0, b), (-1.0, r)):
            for i in range(5):
                for j in range(i + 1, 5):
                    kk = tuple(sorted((side[i], side[j])))
                    v = syn.get(kk[0] + kk[1])
                    if v is not None:
                        s += sign * v; found += 1
        return {"matchup": m, "synergy": s, "score": m + s, "coverage": found / 45.0}


_scorer = None


def scorer():
    """Process-wide scorer, or None when no tables have been built."""
    global _scorer
    if _scorer is None and os.path.exists(TABLES_PATH):
        _scorer = Scorer()
    return _scorer


def draft_score(patch, blue, red):
    """Scaled model input; 0.0 whenever the score is unavailable."""
    s = scorer()
    r = s.score(patch, blue, red) if s else None
    return r["score"] / SCALE if r else 0.0
