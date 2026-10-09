"""Joint research residuals on a frozen core, with independent family curves.

The core probability is an immutable offset. Missing families are zero, with
no intercept or coverage coefficient. RMS scales are learned only from fitting
rows. An early SQ block instead uses its declared fixed score scale and a
nonnegative, monotonically decreasing curve ending at an exact zero knot.
This module is deliberately absent from production model dispatch.
"""
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import wpgam, wpresidual


KIND = "joint_frozen_core_family_residual_v1"
_SPEC_KEYS = {"feature_names", "bounds", "coefficient_bounds", "l2", "smooth",
              "time_knots", "zero_after_min", "monotone_decay", "scale", "input_contract"}


def _spec(name, value):
    if not isinstance(name, str) or not name or not isinstance(value, Mapping):
        raise ValueError("Named feature-family specifications are required")
    if set(value) - _SPEC_KEYS or ("bounds" in value and "coefficient_bounds" in value):
        raise ValueError("Unknown or duplicate family specification fields")
    names = value.get("feature_names")
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)):
        raise ValueError("Unique nonempty names are required within each family")
    penalties = {k: value.get(k, v) for k, v in (("l2", 800.), ("smooth", 70.))}
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not np.isfinite(v) or v <= 0 for v in penalties.values()):
        raise ValueError("Positive finite family penalties are required")
    knots = wpresidual._knots(value.get("time_knots"))
    bounds = wpresidual._bounds(value.get("bounds", value.get("coefficient_bounds", (0., 10.))), len(names))
    cutoff = value.get("zero_after_min")
    if cutoff is not None and (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float))
                               or cutoff <= 0 or not np.any(knots == cutoff)):
        raise ValueError("zero_after_min must be a positive time knot")
    monotone = value.get("monotone_decay", False)
    if not isinstance(monotone, bool):
        raise ValueError("monotone_decay must be explicit boolean metadata")
    if monotone and (len(names) != 1 or cutoff != knots[-1] or bounds[0, 0] != 0):
        raise ValueError("Monotone decay requires one nonnegative feature and a terminal zero knot")
    fixed = value.get("scale")
    if fixed is not None:
        fixed = np.asarray(fixed, dtype=float)
        if fixed.ndim == 0:
            fixed = np.repeat(fixed, len(names))
        if fixed.shape != (len(names),) or not np.isfinite(fixed).all() or np.any(fixed <= 0):
            raise ValueError("Fixed feature scales must be positive and finite")
        fixed = fixed.tolist()
    contract = value.get("input_contract", {})
    if not isinstance(contract, dict):
        raise ValueError("Family input contracts must be dictionaries")
    json.dumps(contract, allow_nan=False, sort_keys=True)
    return dict(name=name, feature_names=list(names), time_knots=knots.tolist(),
                coefficient_bounds=bounds.tolist(), l2=float(penalties["l2"]),
                smooth=float(penalties["smooth"]), zero_after_min=cutoff,
                monotone_decay=monotone, fixed_scale=fixed, input_contract=contract)


def _inputs(blocks, p, t, families):
    t, p = np.asarray(t, dtype=float), np.asarray(p, dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("Finite nonnegative clocks in minutes are required")
    if p.shape != t.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite baseline probabilities in [0,1] must match rows")
    if not isinstance(blocks, Mapping) or set(blocks) != {f["name"] for f in families}:
        raise ValueError("Feature families must match the model contract exactly")
    values = {}
    for family in families:
        name = family["name"]
        extra = np.asarray(blocks[name], dtype=float)
        if extra.shape != (len(t), len(family["feature_names"])) or not np.isfinite(extra).all():
            raise ValueError("Finite row-aligned features are required; unknown features must be zero")
        values[name] = extra
    return values, p, t


class _Design:
    """Compact interpolation partitions, avoiding a dense feature x knot tensor.

    Entirely zero columns and rows are omitted. The remaining matrix is stored
    once across disjoint time partitions; forward and adjoint operations use
    small einsums, avoiding allocation of a full row-by-coefficient design.
    """
    def __init__(self, extra, t, scale, family):
        self.shape = (len(family["feature_names"]), len(family["time_knots"]))
        knots = np.asarray(family["time_knots"])
        active = np.any(extra != 0, axis=1)
        if family["zero_after_min"] is not None:
            active &= t < family["zero_after_min"]
        # Checking columns on the source avoids copying all active rows just
        # to identify zero features; excluded late-only columns are harmless.
        self.columns = np.flatnonzero(np.any(extra != 0, axis=0)) if active.any() else np.array([], dtype=int)
        self.active = active
        self.parts = []
        if not len(self.columns):
            return
        clock = np.clip(t, knots[0], knots[-1])
        left = np.minimum(np.searchsorted(knots, clock, side="right") - 1, len(knots)-2)
        for node in range(len(knots)-1):
            rows = np.flatnonzero(active & (left == node))
            if len(rows):
                z = np.ascontiguousarray(extra[np.ix_(rows, self.columns)] / np.asarray(scale)[self.columns])
                fraction = (clock[rows]-knots[node])/(knots[node+1]-knots[node])
                self.parts.append((node, rows, z, fraction))

    def add(self, coefficients, out):
        grid = coefficients.reshape(self.shape)[self.columns]
        for node, rows, z, fraction in self.parts:
            lo = np.einsum("ij,j->i", z, grid[:, node])
            hi = np.einsum("ij,j->i", z, grid[:, node+1])
            out[rows] += lo*(1.-fraction) + hi*fraction

    def gradient(self, residual):
        grid = np.zeros(self.shape)
        for node, rows, z, fraction in self.parts:
            r = residual[rows]
            grid[self.columns, node] += np.einsum("ij,i->j", z, r*(1.-fraction))
            grid[self.columns, node+1] += np.einsum("ij,i->j", z, r*fraction)
        return grid


def _coefficients(vector, family):
    width, nodes = len(family["feature_names"]), len(family["time_knots"])
    if family["monotone_decay"]:
        return np.r_[np.cumsum(vector[::-1])[::-1], 0.].reshape(1, nodes)
    return vector.reshape(width, nodes)


def _parameter_gradient(gradient, family):
    # slope[k] = sum(increment[k:]); hence dL/dincrement[j] = sum(dL/dslope[:j+1]).
    if family["monotone_decay"]:
        return np.cumsum(gradient[0, :-1])
    return gradient.ravel()


def fit(blocks, baseline_p, y, gids, t, *, specs):
    """Jointly fit declared family curves, retaining the immutable core offset.

    ``blocks`` maps family names to finite N-by-width arrays, with unknown rows
    zero. ``specs`` has matching names and declares feature_names, bounds, l2,
    smooth, and time_knots. A monotone_decay block also requires a final
    zero_after_min knot. ``scale`` pins its scaling (SQ uses raw scores / .25);
    absent scale learns a game-balanced training RMS with a floor of one.
    """
    if not isinstance(specs, Mapping) or not specs:
        raise ValueError("At least one feature family is required")
    families = [_spec(name, spec) for name, spec in sorted(specs.items())]
    values, p, t = _inputs(blocks, baseline_p, t, families)
    y, gids = np.asarray(y), np.asarray(gids)
    if not len(t) or y.shape != t.shape or gids.shape != t.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Nonempty binary labels and game IDs must match rows")
    weights = wpgam._game_balanced_weights(gids)
    designs, slices, flat_bounds, monotone_slices = [], [], [], []
    active = np.zeros(len(t), dtype=bool)
    start = 0
    for family in families:
        extra = values[family["name"]]
        if family["fixed_scale"] is None:
            scale = np.maximum(np.sqrt(np.einsum("ij,ij,i->j", extra, extra, weights)/weights.sum()), 1.)
        else:
            scale = np.asarray(family["fixed_scale"])
        family["scale"] = scale
        design = _Design(extra, t, scale, family)
        designs.append(design)
        active |= design.active
        bounds = np.asarray(family["coefficient_bounds"])
        if family["monotone_decay"]:
            count = len(family["time_knots"])-1
            flat_bounds.extend([(0., float(bounds[0, 1]))]*count)
            monotone_slices.append((slice(start, start+count), float(bounds[0, 1])))
        else:
            count = len(family["feature_names"])*len(family["time_knots"])
            flat_bounds.extend([tuple(bounds[j]) if family["zero_after_min"] is None
                                or knot < family["zero_after_min"] else (0., 0.)
                                for j in range(len(bounds)) for knot in family["time_knots"]])
        slices.append(slice(start, start+count))
        start += count
    # Designs own compact scaled copies; release full float64 conversions before optimizing.
    del values, extra
    offset = wpresidual._logit(p)
    def objective(vector):
        eta = offset.copy()
        grids = [_coefficients(vector[part], family) for family, part in zip(families, slices)]
        for design, grid in zip(designs, grids):
            design.add(grid, eta)
        residual = weights*(wpgam._sigmoid(eta)-y)
        loss = np.dot(weights, np.logaddexp(0., eta)-y*eta)
        gradients = []
        for family, design, grid in zip(families, designs, grids):
            difference = np.diff(grid, axis=1)
            gradient = family["l2"]*grid
            gradient[:, :-1] -= family["smooth"]*difference
            gradient[:, 1:] += family["smooth"]*difference
            gradient += design.gradient(residual)
            loss += .5*family["l2"]*np.sum(grid*grid) + .5*family["smooth"]*np.sum(difference*difference)
            gradients.append(_parameter_gradient(gradient, family))
        return float(loss), np.concatenate(gradients)

    vector, iterations, method = np.zeros(start), 0, "L-BFGS-B"
    result = None
    if active.any():
        result = minimize(objective, vector, jac=True, bounds=flat_bounds, method=method,
                          options=dict(maxiter=600, ftol=1e-10, gtol=1e-5, maxcor=20))
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError("Joint residual optimizer failed: %s" % result.message)
        vector = result.x
        iterations = int(result.nit)
        # Nonnegative increments solve the convex monotonicity constraint. The
        # slope cap adds one sum constraint; pay for SLSQP only if it is active.
        if any(np.cumsum(vector[part][::-1])[-1] > cap for part, cap in monotone_slices):
            initial = vector.copy()
            constraints = []
            for part, cap in monotone_slices:
                if initial[part].sum() > cap:
                    initial[part] *= cap/initial[part].sum()
                jac = np.zeros(start); jac[part] = -1.
                constraints.append(dict(type="ineq", fun=lambda v, s=part, c=cap: c-v[s].sum(),
                                        jac=lambda v, j=jac: j))
            method = "L-BFGS-B then SLSQP"
            result = minimize(objective, initial, jac=True, bounds=flat_bounds,
                              constraints=constraints, method="SLSQP",
                              options=dict(maxiter=600, ftol=1e-9))
            if not result.success or not np.isfinite(result.x).all():
                raise RuntimeError("Capped joint residual optimizer failed: %s" % result.message)
            vector = result.x
            iterations += int(result.nit)
            # Remove numerical constraint tolerance without weakening the cap.
            for part, cap in monotone_slices:
                peak = np.cumsum(vector[part][::-1])[-1]
                if peak > cap:
                    vector[part] *= np.nextafter(cap, 0.)/peak
    for family, part in zip(families, slices):
        family["coefficients"] = _coefficients(vector[part], family).ravel()
    model = dict(kind=KIND, blocks=families, converged=True, optimizer=method,
                 iterations=iterations, objective=objective(vector)[0],
                 training_rows=len(t), active_rows=int(active.sum()))
    validate(model)
    return model


def validate(model, *, specs=None):
    if (not isinstance(model, dict) or model.get("kind") != KIND or model.get("converged") is not True
            or not isinstance(model.get("blocks"), list) or not model["blocks"]):
        raise ValueError("Invalid joint residual kind or convergence contract")
    names = [f.get("name") for f in model["blocks"] if isinstance(f, dict)]
    if (len(names) != len(model["blocks"])
            or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)):
        raise ValueError("Invalid or duplicate family names")
    if specs is not None and (not isinstance(specs, Mapping) or set(specs) != set(names)):
        raise ValueError("Feature-family contract mismatch")
    for family in model["blocks"]:
        required = {"name", "feature_names", "time_knots", "coefficient_bounds", "l2", "smooth",
                    "zero_after_min", "monotone_decay", "fixed_scale", "input_contract", "scale", "coefficients"}
        if (set(family) != required or not isinstance(family["time_knots"], list)
                or not isinstance(family["coefficient_bounds"], list)
                or any(not isinstance(b, list) for b in family["coefficient_bounds"])
                or (family["fixed_scale"] is not None and not isinstance(family["fixed_scale"], list))):
            raise ValueError("Family metadata must be explicit")
        spec = {k: v for k, v in family.items() if k in _SPEC_KEYS}
        spec["scale"] = family["fixed_scale"]
        canonical = _spec(family["name"], spec)
        for key in canonical:
            if canonical[key] != family[key]:
                raise ValueError("Family metadata violates its canonical contract")
        if specs is not None and canonical != _spec(family["name"], specs[family["name"]]):
            raise ValueError("Feature-family contract mismatch")
        width, nodes = len(family["feature_names"]), len(family["time_knots"])
        scale, coefficients = np.asarray(family["scale"]), np.asarray(family["coefficients"])
        if scale.dtype.kind not in "fiu" or coefficients.dtype.kind not in "fiu":
            raise ValueError("Family coefficient arrays must be numeric")
        if (scale.shape != (width,) or not np.isfinite(scale).all() or np.any(scale <= 0)
                or (family["fixed_scale"] is None and np.any(scale < 1))
                or (family["fixed_scale"] is not None and not np.array_equal(scale, family["fixed_scale"]))
                or coefficients.shape != (width*nodes,) or not np.isfinite(coefficients).all()):
            raise ValueError("Invalid family coefficients or scales")
        grid = coefficients.reshape(width, nodes)
        bounds = np.asarray(family["coefficient_bounds"])
        if np.any(grid < bounds[:, :1]) or np.any(grid > bounds[:, 1:]):
            raise ValueError("Family coefficients violate bounds")
        cutoff = family["zero_after_min"]
        if cutoff is not None and np.any(grid[:, np.asarray(family["time_knots"]) >= cutoff] != 0):
            raise ValueError("Late coefficients violate exact fallback")
        if family["monotone_decay"] and np.any(np.diff(grid[0]) > 0):
            raise ValueError("Early curve violates monotone decay")
    if (not isinstance(model.get("iterations"), int) or model["iterations"] < 0
            or not isinstance(model.get("objective"), (int, float))
            or not np.isfinite(model["objective"]) or model["objective"] < 0
            or model.get("optimizer") not in {"L-BFGS-B", "L-BFGS-B then SLSQP"}
            or not isinstance(model.get("training_rows"), int) or model["training_rows"] <= 0
            or not isinstance(model.get("active_rows"), int)
            or not 0 <= model["active_rows"] <= model["training_rows"]):
        raise ValueError("Invalid joint optimizer metadata")


def correction(model, blocks, t):
    validate(model)
    values, _, t = _inputs(blocks, np.full(len(t), .5), t, model["blocks"])
    out = np.zeros(len(t))
    for family in model["blocks"]:
        _Design(values[family["name"]], t, family["scale"], family).add(family["coefficients"], out)
    return out


def predict(model, blocks, baseline_p, t):
    validate(model)
    values, p, t = _inputs(blocks, baseline_p, t, model["blocks"])
    delta = correction(model, values, t)
    out = wpgam._sigmoid(wpresidual._logit(p, endpoints=True)+delta)
    out[delta == 0] = p[delta == 0]
    return out


def save(model, path):
    validate(model)
    meta = {k: v for k, v in model.items() if k != "blocks"}
    meta["blocks"] = [{k: v for k, v in f.items() if k not in {"scale", "coefficients"}} for f in model["blocks"]]
    arrays = {}
    for i, family in enumerate(model["blocks"]):
        arrays["scale_%d" % i], arrays["coefficients_%d" % i] = family["scale"], family["coefficients"]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(meta, allow_nan=False, sort_keys=True)), **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        model = json.loads(str(archive["metadata"].item()))
        if (not isinstance(model, dict) or not isinstance(model.get("blocks"), list)
                or any(not isinstance(f, dict) for f in model["blocks"])):
            raise ValueError("Invalid joint artifact metadata")
        expected = {"metadata"} | {"%s_%d" % (kind, i) for i in range(len(model["blocks"]))
                                    for kind in ("scale", "coefficients")}
        if set(archive.files) != expected:
            raise ValueError("Invalid joint artifact array contract")
        for i, family in enumerate(model["blocks"]):
            family.update(scale=archive["scale_%d" % i], coefficients=archive["coefficients_%d" % i])
    validate(model)
    return model
