"""Arbitrary-width research residuals on immutable baseline probabilities.

This module is deliberately independent of deployed model dispatch. Candidate
modules certify their own causal input contract, names and coefficient signs.
Missing feature families must be zero; there is no intercept or availability
coefficient. Scaling is learned from training games only without centering.
"""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from . import wpgam


KIND = "frozen_core_time_residual_v1"


def _knots(values):
    out = np.asarray(wpgam.TIME_KNOTS if values is None else values, dtype=float)
    if (out.ndim != 1 or len(out) < 2 or not np.isfinite(out).all()
            or out[0] != 0 or np.any(np.diff(out) <= 0)):
        raise ValueError("Time knots must start at zero and increase strictly")
    return out


def _basis(t, knots):
    t = np.clip(t, knots[0], knots[-1])
    left = np.minimum(np.searchsorted(knots, t, side="right") - 1, len(knots) - 2)
    fraction = (t - knots[left]) / (knots[left + 1] - knots[left])
    out = np.zeros((len(t), len(knots)))
    out[np.arange(len(t)), left] = 1. - fraction
    out[np.arange(len(t)), left + 1] = fraction
    return out


def _rows(extra, baseline_p, t, width):
    extra, baseline_p, t = (np.asarray(v, dtype=float) for v in (extra, baseline_p, t))
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("Nonnegative finite game clocks in minutes are required")
    if extra.shape != (len(t), width) or not np.isfinite(extra).all():
        raise ValueError("Finite feature rows must match names; unknown features must be zero")
    if (baseline_p.shape != (len(t),) or not np.isfinite(baseline_p).all()
            or np.any((baseline_p < 0) | (baseline_p > 1))):
        raise ValueError("Finite baseline probabilities in [0,1] must match rows")
    return extra, baseline_p, t


def _logit(p, endpoints=False):
    safe = np.clip(p, 1e-12, 1. - 1e-12)
    result = np.log(safe) - np.log1p(-safe)
    if endpoints:
        result[p == 0] = -np.inf
        result[p == 1] = np.inf
    return result


def _design(extra, t, scale, knots):
    basis = _basis(t, knots)
    rows, nodes = np.nonzero(basis)
    z = extra / scale
    return sparse.csr_matrix(
        (np.concatenate([z[rows, j] * basis[rows, nodes] for j in range(z.shape[1])]),
         (np.tile(rows, z.shape[1]),
          np.concatenate([nodes + j * len(knots) for j in range(z.shape[1])]))),
        shape=(len(t), z.shape[1] * len(knots)))


def _bounds(value, width):
    out = np.asarray(value, dtype=float)
    if out.shape == (2,):
        out = np.tile(out, (width, 1))
    if (out.shape != (width, 2) or not np.isfinite(out).all()
            or np.any(out[:, 0] > 0) or np.any(out[:, 1] < 0)
            or np.any(out[:, 0] >= out[:, 1])):
        raise ValueError("Each finite coefficient interval must contain zero and have positive width")
    return out


def fit(extra, baseline_p, y, gids, t, *, feature_names, candidate_kind,
        input_contract, spec=None, coefficient_bounds=(0., 10.),
        time_knots=None, zero_after_min=None):
    """Fit a game-balanced time-varying residual, keeping the baseline fixed.

    ``coefficient_bounds`` is one pair or one pair per feature. A supplied
    ``zero_after_min`` must be a knot and fixes every coefficient from that
    knot onward at zero, giving an exact late-game baseline fallback.
    """
    names = list(feature_names)
    if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("Unique nonempty feature names are required")
    if not isinstance(candidate_kind, str) or not candidate_kind or not isinstance(input_contract, dict):
        raise ValueError("A named candidate and explicit input contract are required")
    # Ensure provenance can be persisted without nonfinite JSON values.
    json.dumps(input_contract, allow_nan=False, sort_keys=True)
    knobs = dict(dict(l2=800., smooth=70.), **(spec or {}))
    if (set(knobs) != {"l2", "smooth"}
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not np.isfinite(v) or v <= 0 for v in knobs.values())):
        raise ValueError("Positive finite l2 and smooth penalties are required")
    knots, bounds = _knots(time_knots), _bounds(coefficient_bounds, len(names))
    if zero_after_min is not None:
        if (isinstance(zero_after_min, bool) or not isinstance(zero_after_min, (int, float))
                or zero_after_min <= 0 or not np.any(knots == zero_after_min)):
            raise ValueError("zero_after_min must be a positive time knot")
        zero_after_min = float(zero_after_min)
    extra, baseline_p, t = _rows(extra, baseline_p, t, len(names))
    y, gids = np.asarray(y), np.asarray(gids)
    if not len(t) or y.shape != t.shape or not np.isin(y, [0, 1]).all() or gids.shape != t.shape:
        raise ValueError("Nonempty binary labels and game IDs must match rows")
    weights = wpgam._game_balanced_weights(gids)
    scale = np.maximum(np.sqrt(np.average(extra ** 2, axis=0, weights=weights)), 1.)
    active = np.any(extra != 0, axis=1)
    if zero_after_min is not None:
        active &= t < zero_after_min
    offset = _logit(baseline_p)
    constant = float(np.dot(weights[~active],
                            np.logaddexp(0., offset[~active]) - y[~active] * offset[~active]))
    design = _design(extra[active], t[active], scale, knots)
    aw, ay, ao = weights[active], y[active], offset[active]
    flat_bounds = [tuple(bounds[j]) if zero_after_min is None or knot < zero_after_min else (0., 0.)
                   for j in range(len(names)) for knot in knots]

    def objective(vector):
        eta = ao + design @ vector
        coefficients = vector.reshape(len(names), len(knots))
        difference = np.diff(coefficients, axis=1)
        gradient = knobs["l2"] * coefficients
        gradient[:, :-1] -= knobs["smooth"] * difference
        gradient[:, 1:] += knobs["smooth"] * difference
        loss = (constant + np.dot(aw, np.logaddexp(0., eta) - ay * eta)
                + .5 * knobs["l2"] * np.sum(coefficients ** 2)
                + .5 * knobs["smooth"] * np.sum(difference ** 2))
        gradient = gradient.ravel() + design.T @ (aw * (wpgam._sigmoid(eta) - ay))
        return float(loss), gradient

    result = minimize(objective, np.zeros(design.shape[1]), jac=True,
                      bounds=flat_bounds, method="L-BFGS-B",
                      options=dict(maxiter=600, ftol=1e-10, gtol=1e-5, maxcor=20))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("Residual optimizer failed: %s" % result.message)
    model = dict(kind=KIND, candidate_kind=candidate_kind, feature_names=names,
                 input_contract=input_contract, spec={k: float(v) for k, v in knobs.items()},
                 time_knots=knots.tolist(), coefficient_bounds=bounds.tolist(),
                 zero_after_min=zero_after_min, scale=scale, coefficients=result.x,
                 converged=True, iterations=int(result.nit), objective=float(result.fun),
                 training_rows=len(t), active_rows=int(active.sum()))
    validate(model)
    return model


def validate(model, *, candidate_kind=None, feature_names=None, input_contract=None):
    if not isinstance(model, dict) or model.get("kind") != KIND or model.get("converged") is not True:
        raise ValueError("Residual artifact kind or convergence mismatch")
    names = model.get("feature_names")
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)
            or not isinstance(model.get("candidate_kind"), str) or not model["candidate_kind"]
            or not isinstance(model.get("input_contract"), dict)):
        raise ValueError("Invalid candidate contract")
    if not isinstance(model.get("time_knots"), list):
        raise ValueError("Artifact time knots must be explicit")
    for expected, actual in ((candidate_kind, model["candidate_kind"]),
                             (feature_names, names), (input_contract, model["input_contract"])):
        if expected is not None and expected != actual:
            raise ValueError("Candidate artifact contract mismatch")
    json.dumps(model["input_contract"], allow_nan=False, sort_keys=True)
    knots, bounds = _knots(model.get("time_knots")), _bounds(model.get("coefficient_bounds"), len(names))
    scale, coefficients = np.asarray(model.get("scale")), np.asarray(model.get("coefficients"))
    if (scale.shape != (len(names),) or not np.isfinite(scale).all() or np.any(scale < 1)
            or coefficients.shape != (len(names) * len(knots),) or not np.isfinite(coefficients).all()):
        raise ValueError("Invalid residual coefficients or RMS scales")
    grid = coefficients.reshape(len(names), len(knots))
    if np.any(grid < bounds[:, :1]) or np.any(grid > bounds[:, 1:]):
        raise ValueError("Coefficients violate candidate bounds")
    cutoff = model.get("zero_after_min")
    if cutoff is not None:
        if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool) or cutoff <= 0 or not np.any(knots == cutoff):
            raise ValueError("Invalid terminal decay contract")
        if np.any(grid[:, knots >= cutoff] != 0):
            raise ValueError("Late coefficients violate exact fallback")
    spec = model.get("spec")
    if (not isinstance(spec, dict) or set(spec) != {"l2", "smooth"}
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not np.isfinite(v) or v <= 0 for v in spec.values())):
        raise ValueError("Invalid residual penalty contract")
    if (not isinstance(model.get("iterations"), int) or model["iterations"] < 0
            or not isinstance(model.get("objective"), (int, float))
            or not np.isfinite(model["objective"]) or model["objective"] < 0):
        raise ValueError("Invalid optimizer metadata")


def correction(model, extra, t):
    validate(model)
    extra, _, t = _rows(extra, np.full(len(t), .5), t, len(model["feature_names"]))
    active = np.any(extra != 0, axis=1)
    if model["zero_after_min"] is not None:
        active &= t < model["zero_after_min"]
    out = np.zeros(len(t))
    out[active] = _design(extra[active], t[active], model["scale"],
                          np.asarray(model["time_knots"])) @ model["coefficients"]
    return out


def predict(model, extra, baseline_p, t):
    extra, baseline_p, t = _rows(extra, baseline_p, t, len(model["feature_names"]))
    delta = correction(model, extra, t)
    out = wpgam._sigmoid(_logit(baseline_p, endpoints=True) + delta)
    out[delta == 0] = baseline_p[delta == 0]
    return out


def save(model, path):
    validate(model)
    meta = {k: v for k, v in model.items() if k not in {"scale", "coefficients"}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(meta, allow_nan=False, sort_keys=True)),
                            scale=model["scale"], coefficients=model["coefficients"])


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        model = json.loads(str(archive["metadata"].item()))
        model.update(scale=archive["scale"], coefficients=archive["coefficients"])
    validate(model)
    return model
