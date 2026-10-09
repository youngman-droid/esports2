"""A fourth isolated prior: solo-queue pairs, fading monotonically to zero.

Uses the shipped pair scorer, without modifying its production data or model.
The fitted early prior is nonnegative and nonincreasing through 20 minutes;
missing coverage and every later minute preserve the baseline exactly.
"""
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import sqpairs, wpgam

KIND = "sq_early_prior_v1"
TIME_KNOTS = np.array([0., 5., 10., 15., 20.])
INPUT_CONTRACT = dict(name=KIND, source="pinned shipped sqpairs.Scorer",
    pooling="pair cells from strictly earlier patches; table shrink constants are pinned",
    minimum_pair_coverage=.8, missing="zero additional logit, exact baseline fallback",
    time="own nonnegative nonincreasing curve; exactly zero at and after minute 20",
    role_order=["top", "jng", "mid", "bot", "sup"], production_dispatch=False)
_ROLE = {"top": 0, "jungle": 1, "jng": 1, "mid": 2, "bottom": 3, "bot": 3,
         "support": 4, "sup": 4}
_CAPTURE_SCORERS = {}


def _capture_scorer(path):
    """Cache a pinned table generation and hash; detect nightly replacement."""
    stat = path.stat()
    identity = (stat.st_mtime_ns, stat.st_size)
    key = str(path.resolve())
    cached = _CAPTURE_SCORERS.get(key)
    if cached is None or cached[0] != identity:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        scorer = sqpairs.Scorer(str(path))
        if (path.stat().st_mtime_ns, path.stat().st_size) != identity:
            raise ValueError("SQ tables changed while loading")
        scorer._early_table_sha256 = digest
        _CAPTURE_SCORERS[key] = (identity, scorer)
    return _CAPTURE_SCORERS[key][1]


def capture(patch, blue, red, *, scorer=None, table_path=None):
    """Return a patch-bound score with true availability, including score zero."""
    # Riot's live patchVersion includes build/release suffixes. Passing 16.16.1
    # directly to tuple ordering would incorrectly admit same-patch 16.16.
    canonical = sqpairs.canon(patch)
    components = sqpairs.pnum(canonical) if canonical else ()
    canonical = ".".join(map(str, components[:2])) if len(components) >= 2 else None
    out = dict(kind=KIND, available=False, reason="missing_pair_tables", pair_score=0.,
               coverage=0., patch=canonical, source_patches=[],
               input_contract=dict(INPUT_CONTRACT), production_applied=False)
    if (not isinstance(blue, (list, tuple)) or not isinstance(red, (list, tuple))
            or len(blue) != 5 or len(red) != 5
            or any(not isinstance(c, str) or not c for c in list(blue)+list(red))
            or len({sqpairs.key(c) for c in list(blue)+list(red)}) != 10):
        out["reason"] = "invalid_completed_draft"
        return out
    path = Path(table_path or sqpairs.TABLES_PATH)
    if scorer is None:
        if not path.is_file():
            return out
        scorer = _capture_scorer(path)
    if scorer is None:
        return out
    out.update(table_sha256=getattr(scorer, "_early_table_sha256", None),
               shrink_constants=[float(k) for k in getattr(scorer, "k", [])])
    score = scorer.score(canonical, blue, red)
    out["source_patches"] = [p for p in scorer.patches
                            if out["patch"] and sqpairs.pnum(p) < sqpairs.pnum(out["patch"])]
    if score is None:
        out["reason"] = "missing_prior_patch_or_champion"
        return out
    value, coverage = score.get("score"), score.get("coverage")
    if (not isinstance(value, (int, float)) or not math.isfinite(value)
            or not isinstance(coverage, (int, float)) or not math.isfinite(coverage)
            or not 0 <= coverage <= 1):
        out["reason"] = "invalid_pair_score"
        return out
    out["coverage"] = float(coverage)
    if coverage < INPUT_CONTRACT["minimum_pair_coverage"]:
        out["reason"] = "insufficient_pair_coverage"
        return out
    out.update(available=True, reason="prior_patch_pair_score", pair_score=float(value),
               matchup=float(score["matchup"]), synergy=float(score["synergy"]))
    return out


def live_capture(metadata, *, table_path=None):
    """Join completed champions through roles; verify distinct side identities."""
    picks = []
    if isinstance(metadata, Mapping):
        for side, ids in (("blue", set(range(1, 6))), ("red", set(range(6, 11)))):
            players = (metadata.get(side+"TeamMetadata") or {}).get("participantMetadata")
            if not isinstance(players, list) or len(players) != 5:
                break
            roles, found = {}, set()
            for p in players:
                if not isinstance(p, Mapping):
                    break
                pid, role, champion = p.get("participantId"), p.get("role"), p.get("championId")
                if (not isinstance(pid, int) or isinstance(pid, bool) or pid not in ids or pid in found
                        or role not in _ROLE or _ROLE[role] in roles or not isinstance(champion, str)):
                    break
                roles[_ROLE[role]], found = champion, found | {pid}
            if len(roles) != 5 or found != ids:
                break
            picks.append([roles[i] for i in range(5)])
    if len(picks) != 2:
        out = capture(None, [], [])
        out["reason"] = "unverified_live_draft_roles"
        return out
    return capture(metadata.get("patchVersion"), *picks, table_path=table_path)


def _basis(t):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("Finite nonnegative minute clocks required")
    basis = np.zeros((len(t), len(TIME_KNOTS)-1))
    for j in range(len(TIME_KNOTS)-1):
        lo, hi = TIME_KNOTS[max(0, j-1)], TIME_KNOTS[j+1]
        if j:
            left = (t >= lo) & (t <= TIME_KNOTS[j])
            basis[left, j] = (t[left]-lo)/(TIME_KNOTS[j]-lo)
        right = (t >= TIME_KNOTS[j]) & (t < hi)
        basis[right, j] = (hi-t[right])/(hi-TIME_KNOTS[j])
    return basis


def _rows(pair_score, available, baseline_p, t):
    score, p, t = (np.asarray(x, dtype=float) for x in (pair_score, baseline_p, t))
    available = np.asarray(available)
    if available.dtype != bool or available.shape != t.shape:
        raise ValueError("Explicit boolean source coverage required")
    if score.shape != t.shape or p.shape != t.shape or not np.isfinite(score).all():
        raise ValueError("Complete finite row-aligned scores required")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite baseline probabilities required")
    return score, available, p, t, _basis(t)


def fit(pair_score, available, baseline_p, y, gids, t, *, l2=300., smooth=70., source=None):
    score, available, p, t, basis = _rows(pair_score, available, baseline_p, t)
    y, gids = np.asarray(y), np.asarray(gids)
    if not len(t) or y.shape != t.shape or gids.shape != t.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Nonempty binary outcomes and game IDs must match rows")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (l2, smooth)):
        raise ValueError("Positive fixed ridge and smooth penalties required")
    weights = wpgam._game_balanced_weights(gids)
    design = basis * np.where(available, score/.25, 0.)[:, None]
    active = np.any(design != 0, axis=1)
    supported_games = len(np.unique(gids[available & (t < 20)]))
    vector = np.zeros(len(TIME_KNOTS)-1)
    iterations = 0
    if active.any():
        z, w = design[active], weights[active]
        pc = np.clip(p[active], 1e-12, 1-1e-12)
        offset = np.log(pc)-np.log1p(-pc)
        target = y[active]
        def objective(v):
            eta = offset+np.einsum("ij,j->i", z, v)
            difference = np.diff(np.r_[v, 0.])
            grad = l2*v
            grad[:-1] -= smooth*difference[:-1]
            grad[1:] += smooth*difference[:-1]
            grad[-1] -= smooth*difference[-1]
            loss = np.dot(w, np.logaddexp(0., eta)-target*eta)
            loss += .5*l2*np.sum(v*v)+.5*smooth*np.sum(difference*difference)
            return float(loss), grad+np.einsum("ij,i->j", z, w*(wpgam._sigmoid(eta)-target))
        result = minimize(objective, vector, jac=True, bounds=[(0., 2.5)]*len(vector),
            constraints=[dict(type="ineq", fun=lambda v: v[:-1]-v[1:],
                              jac=lambda v: np.eye(len(v))[:-1]-np.eye(len(v))[1:])],
            method="SLSQP", options=dict(maxiter=300, ftol=1e-9))
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError("SQ early optimizer failed: "+str(result.message))
        # Remove numerical constraint tolerance without changing the shape.
        vector = np.minimum.accumulate(np.clip(result.x, 0., 2.5))
        iterations = int(result.nit)
    model = dict(kind=KIND, input_contract=dict(INPUT_CONTRACT), time_knots=TIME_KNOTS.tolist(),
        slopes=np.r_[vector, 0.], l2=float(l2), smooth=float(smooth), score_scale=.25,
        converged=True, iterations=iterations, supported_training_games=supported_games,
        source=dict(source or {}))
    validate(model)
    return model


def validate(model):
    values = np.asarray(model.get("slopes"), dtype=float)
    if (model.get("kind") != KIND or model.get("input_contract") != INPUT_CONTRACT
            or model.get("time_knots") != TIME_KNOTS.tolist() or model.get("converged") is not True
            or model.get("score_scale") != .25 or values.shape != TIME_KNOTS.shape
            or not np.isfinite(values).all() or np.any(values < -1e-8)
            or np.any(values > 2.5+1e-8) or np.any(np.diff(values) > 1e-8) or values[-1] != 0.):
        raise ValueError("SQ early model violates the monotone decay contract")
    if not all(isinstance(model.get(k), (int, float)) and math.isfinite(model[k]) and model[k] > 0
               for k in ("l2", "smooth")):
        raise ValueError("Invalid SQ early penalties")


def correction(model, pair_score, available, t):
    validate(model)
    score, available, _, t, basis = _rows(pair_score, available, np.full(len(t), .5), t)
    return np.where(available, score/model["score_scale"], 0.)*np.einsum("ij,j->i", basis, np.asarray(model["slopes"][:-1]))


def predict(model, pair_score, available, baseline_p, t):
    score, available, p, t, _ = _rows(pair_score, available, baseline_p, t)
    delta = correction(model, score, available, t)
    clipped = np.clip(p, 1e-12, 1-1e-12)
    result = wpgam._sigmoid(np.log(clipped)-np.log1p(-clipped)+delta)
    result[delta == 0] = p[delta == 0]
    return result


def save(model, path):
    validate(model)
    meta = {k: v for k, v in model.items() if k != "slopes"}
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(meta, sort_keys=True)), slopes=model["slopes"])


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        model = json.loads(str(archive["metadata"].item()))
        model["slopes"] = archive["slopes"]
    validate(model)
    return model
