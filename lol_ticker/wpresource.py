"""Pooled role/champion resource curves with physical gold monotonicity.

Absolute per-player gold is transformed identically on blue and red, with a
positive sign for blue and negative for red. Resource slopes are nonnegative;
champion slopes shrink toward a shared role slope. Thus a champion can have a
negative *deviation* from its role without a negative total gold response.
This experimental module has no production dispatch or promotion side effects.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from . import wpgam


KIND = "pooled_resource_gam_v1"
INPUT_NAMES = wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES
ROLE_NAMES = {"gold_" + r for r in wpgam.GOLD_ROLES}
CORE_INDICES = np.array([i for i, n in enumerate(INPUT_NAMES) if n not in ROLE_NAMES])
CORE_NAMES = [INPUT_NAMES[i] for i in CORE_INDICES]


def _groups(C, gids, minimum):
    support = defaultdict(set)
    for slot in range(10):
        for champ, gid in zip(C[:, slot], gids):
            if champ >= 0:
                support[(slot % 5, int(champ))].add(int(gid))
    keys = [(r, -1) for r in range(5)]
    keys += sorted(key for key, games in support.items() if len(games) >= minimum)
    return np.asarray(keys, dtype=np.int32)


def _resource_design(model, gold, C, t):
    gold, C = np.asarray(gold, dtype=float), np.asarray(C, dtype=int)
    if gold.shape != (len(t), 10) or C.shape != gold.shape:
        raise ValueError("Expected ten ordered champion/gold slots")
    if not np.isfinite(gold).all() or np.any(gold < 0):
        raise ValueError("Complete nonnegative per-player total gold is required")
    keys = model["groups"]
    lookup = {tuple(map(int, key)): i for i, key in enumerate(keys)}
    mode, nk = model["mode"], len(wpgam.TIME_KNOTS)
    basis = wpgam.time_basis(t)
    rows, knots = np.nonzero(basis)
    bval = basis[rows, knots]
    rr, cc, vv = [], [], []
    ng = len(keys)
    for slot in range(10):
        role = slot % 5
        group = np.fromiter((lookup.get((role, int(ch)), role) for ch in C[:, slot]),
                            dtype=np.int32, count=len(t))
        value = np.clip(gold[:, slot], 0, model["gold_cap"][role]) / model["gold_scale"][role]
        value *= 1 if slot < 5 else -1
        if mode == "constant":
            # slope(champion, t) = a_champion + h_role(t), all a,h >= 0.
            # The role baseline is a_role + h_role(t); a_champion-a_role is
            # an unrestricted constant deviation subject to total positivity.
            rr.extend([np.arange(len(t)), rows])
            cc.extend([group, ng + role * nk + knots])
            vv.extend([value, value[rows] * bval])
        else:
            rr.append(rows); cc.append(group[rows] * nk + knots)
            vv.append(value[rows] * bval)
    width = ng + 5 * nk if mode == "constant" else ng * nk
    return sparse.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                             shape=(len(t), width))


def _design(model, raw, gold, C, t):
    raw = np.asarray(raw, dtype=float)
    if np.asarray(t).ndim != 1 or not np.isfinite(t).all() or np.any(np.asarray(t) < 0):
        raise ValueError("Nonnegative finite game clocks are required")
    if raw.shape != (len(t), len(INPUT_NAMES)) or not np.isfinite(raw).all():
        raise ValueError("Invalid core input contract")
    z = wpgam._scale_apply(raw[:, CORE_INDICES], *model["scale"])
    z = np.column_stack([np.ones(len(raw)), z])
    basis = wpgam.time_basis(t)
    rows, knots = np.nonzero(basis)
    values = basis[rows, knots]
    nk = basis.shape[1]
    core = sparse.csr_matrix((np.concatenate([z[rows, j] * values for j in range(z.shape[1])]),
        (np.tile(rows, z.shape[1]), np.concatenate([knots + j * nk for j in range(z.shape[1])]))),
        shape=(len(t), z.shape[1] * nk))
    return sparse.hstack([core, _resource_design(model, gold, C, t)], format="csr")


def _penalty(model, vector):
    nk = len(wpgam.TIME_KNOTS)
    nc = (len(CORE_NAMES) + 1) * nk
    core = vector[:nc].reshape(-1, nk)
    resource = vector[nc:]
    l2, smooth, shrink = (model["spec"][k] for k in ("l2", "smooth", "shrink"))
    grad_core = np.zeros_like(core)
    loss = .5 * l2 * np.sum(core[1:] ** 2)
    grad_core[1:] = l2 * core[1:]

    def smooth_penalty(value, grad, strength):
        delta = np.diff(value, axis=1)
        grad[:, :-1] -= strength * delta
        grad[:, 1:] += strength * delta
        return .5 * strength * np.sum(delta ** 2)

    loss += smooth_penalty(core, grad_core, smooth)
    parents = model["groups"][5:, 0]
    if model["mode"] == "constant":
        ng = len(model["groups"])
        a, h = resource[:ng], resource[ng:].reshape(5, nk)
        ga, gh = np.zeros_like(a), np.zeros_like(h)
        baseline = a[:5, None] + h
        loss += .5 * l2 * np.sum(baseline ** 2)
        ga[:5] += l2 * baseline.sum(axis=1)
        gh += l2 * baseline
        loss += smooth_penalty(h, gh, smooth)
        delta = a[5:] - a[parents]
        loss += .5 * shrink * nk * np.dot(delta, delta)
        ga[5:] += shrink * nk * delta
        np.add.at(ga, parents, -shrink * nk * delta)
        grad_resource = np.r_[ga, gh.ravel()]
    else:
        a = resource.reshape(-1, nk)
        ga = np.zeros_like(a)
        loss += .5 * l2 * np.sum(a[:5] ** 2)
        ga[:5] += l2 * a[:5]
        loss += smooth_penalty(a[:5], ga[:5], smooth)
        delta = a[5:] - a[parents]
        loss += .5 * shrink * np.sum(delta ** 2)
        ga[5:] += shrink * delta
        np.add.at(ga, parents, -shrink * delta)
        # Smooth deviations, not just the common role curve.
        dg = np.zeros_like(delta)
        loss += smooth_penalty(delta, dg, smooth)
        ga[5:] += dg
        np.add.at(ga, parents, -dg)
        grad_resource = ga.ravel()
    return float(loss), np.r_[grad_core.ravel(), grad_resource]


def fit(raw, gold, C, y, gids, t, spec=None):
    spec = dict(dict(mode="smooth", min_games=20, l2=24., smooth=70., shrink=800.), **(spec or {}))
    if spec["mode"] not in {"role", "constant", "smooth"}:
        raise ValueError("Unsupported pooling mode")
    if any(float(spec[k]) < 0 for k in ("l2", "smooth", "shrink")):
        raise ValueError("Penalties must be nonnegative")
    gold, C, y, gids, t = map(np.asarray, (gold, C, y, gids, t))
    groups = (_groups(C, gids, spec["min_games"]) if spec["mode"] != "role"
              else np.asarray([(r, -1) for r in range(5)], dtype=np.int32))
    # Fit-only, shared blue/red transforms. Individual clipping remains
    # monotone and cannot expose a negative residual slope after saturation.
    cap = np.array([max(1., np.quantile(gold[:, [r, r + 5]], .995)) for r in range(5)])
    scale = np.maximum(np.std(gold[:, :5] - gold[:, 5:], axis=0), 1000.)
    model = dict(kind=KIND, mode=spec["mode"], groups=groups, gold_cap=cap,
                 gold_scale=scale, scale=wpgam._scale_fit(np.asarray(raw)[:, CORE_INDICES],
                 feature_names=CORE_NAMES), spec=spec)
    design = _design(model, raw, gold, C, t)
    weights = wpgam._game_balanced_weights(gids)
    if y.shape != (len(t),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Binary labels must match input rows")
    nk = len(wpgam.TIME_KNOTS)
    nc = (len(CORE_NAMES) + 1) * nk
    bounds = [(-10., 10.)] * nk
    for name in CORE_NAMES:
        bounds.extend([(0., 10.) if name in wpgam.MONOTONE_FEATURES else (-10., 10.)] * nk)
    bounds += [(0., 10.)] * (design.shape[1] - nc)
    initial = np.zeros(design.shape[1])
    rate = np.clip(np.average(y, weights=weights), 1e-5, 1 - 1e-5)
    initial[:nk] = np.log(rate / (1 - rate))

    def objective(vector):
        eta = design @ vector
        penalty, gradient = _penalty(model, vector)
        loss = np.dot(weights, np.logaddexp(0, eta) - y * eta) + penalty
        gradient += design.T @ (weights * (wpgam._sigmoid(eta) - y))
        return float(loss), gradient

    result = minimize(objective, initial, jac=True, bounds=bounds, method="L-BFGS-B",
                      options=dict(maxiter=600, ftol=1e-9, gtol=1e-4, maxcor=20))
    if not result.success:
        result = minimize(objective, result.x, jac=True, bounds=bounds, method="L-BFGS-B",
                          options=dict(maxiter=1000, ftol=1e-9, gtol=1e-4, maxcor=30))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("Resource GAM optimizer failed: %s" % result.message)
    model.update(coefficients=result.x, converged=True, iterations=int(result.nit),
                 objective=float(result.fun), calibration=dict(intercept=0., slope=1.),
                 input_names=list(INPUT_NAMES), time_knots=wpgam.TIME_KNOTS.tolist())
    validate(model)
    return model


def predict(model, raw, gold, C, t, calibrated=True):
    eta = _design(model, raw, gold, C, np.asarray(t)) @ model["coefficients"]
    if calibrated:
        calibration = model.get("calibration", dict(intercept=0., slope=1.))
        if calibration["slope"] <= 0:
            raise ValueError("Calibration must preserve monotonicity")
        eta = calibration["intercept"] + calibration["slope"] * eta
    return wpgam._sigmoid(eta)


def save(model, path):
    validate(model)
    payload = {k: model[k] for k in ("groups", "gold_cap", "gold_scale", "coefficients")}
    payload.update({"scale_" + str(i): value for i, value in enumerate(model["scale"])})
    meta = {k: model[k] for k in ("kind", "mode", "spec", "converged", "iterations", "objective", "calibration", "input_names", "time_knots")}
    payload["metadata"] = np.array(json.dumps(meta, sort_keys=True))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        model = json.loads(str(archive["metadata"].item()))
        if model["kind"] != KIND:
            raise ValueError("Unsupported resource artifact")
        model.update({k: archive[k] for k in ("groups", "gold_cap", "gold_scale", "coefficients")})
        model["scale"] = tuple(archive["scale_" + str(i)] for i in range(4))
    validate(model)
    return model


def validate(model):
    """Reject artifacts that cannot preserve the declared inference contract."""
    if (model.get("kind") != KIND or model.get("mode") not in {"role", "constant", "smooth"}
            or model.get("input_names") != list(INPUT_NAMES)
            or model.get("time_knots") != wpgam.TIME_KNOTS.tolist()
            or model.get("converged") is not True):
        raise ValueError("Resource artifact contract or convergence mismatch")
    groups = np.asarray(model["groups"])
    if (groups.ndim != 2 or groups.shape[1] != 2 or len(groups) < 5
            or not np.array_equal(groups[:5], [[r, -1] for r in range(5)])
            or len({tuple(row) for row in groups}) != len(groups)
            or np.any((groups[:, 0] < 0) | (groups[:, 0] > 4))
            or np.any(groups[5:, 1] < 0)):
        raise ValueError("Invalid pooled champion groups")
    for name in ("gold_cap", "gold_scale"):
        value = np.asarray(model[name])
        if value.shape != (5,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("Invalid resource transform")
    if any(np.asarray(v).shape != (len(CORE_NAMES),) or not np.isfinite(v).all() for v in model["scale"]):
        raise ValueError("Invalid core scale")
    if np.any(model["scale"][1] <= 0) or np.any(model["scale"][2] > model["scale"][3]):
        raise ValueError("Invalid core scale bounds")
    nk = len(wpgam.TIME_KNOTS)
    nc = (len(CORE_NAMES) + 1) * nk
    nr = len(groups) + 5 * nk if model["mode"] == "constant" else len(groups) * nk
    vector = np.asarray(model["coefficients"])
    if vector.shape != (nc + nr,) or not np.isfinite(vector).all() or np.any(np.abs(vector) > 10.):
        raise ValueError("Invalid coefficient dimensions, values or bounds")
    core = vector[:nc].reshape(-1, nk)
    if np.any(vector[nc:] < 0) or any(np.any(core[i + 1] < 0)
            for i, name in enumerate(CORE_NAMES) if name in wpgam.MONOTONE_FEATURES):
        raise ValueError("Artifact violates physical monotonicity constraints")
    cal = model["calibration"]
    if not np.isfinite([cal["intercept"], cal["slope"]]).all() or cal["slope"] <= 0:
        raise ValueError("Invalid monotone calibration")
