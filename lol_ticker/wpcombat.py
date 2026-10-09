"""Conservative role-specific combat readiness and a frozen-core residual.

Historical vectors must already be in verified blue/red role order, and their
clock provenance must be certified by the caller. ``live_observation`` performs
the participant-ID/role join for official window frames rather than trusting
array order. Missing, stale or invalid health contributes exactly zero.

The experimental residual has no intercept and does not refit the core model.
Every coefficient is nonnegative at every time knot. Living resources use
``alive * resource`` without centering or gold-share normalization, preserving
nonnegative physical gold, health and revival partial effects. Item inputs are
excluded: the corrected historical contract ablates inventory because archived
undo events lack the item identity needed for safe reconstruction.
"""
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from . import wpgam


KIND = "role_combat_residual_v1"
ROLES = ("top", "jng", "mid", "bot", "sup")
CHANNELS = ("alive", "hp", "living_gold_k", "living_level")
FEATURE_NAMES = (["alive_" + r for r in ROLES]
                 + ["hp_" + r for r in ROLES]
                 + ["living_gold_" + r + "_k" for r in ROLES]
                 + ["living_level_" + r for r in ROLES])
INPUT_CONTRACT = {
    "name": KIND,
    "role_order": list(ROLES),
    "health": "complete ten-player finite fractions in [0,1]; alive iff hp > 0",
    "gold": "complete ten-player finite nonnegative totalGold; independently gated",
    "level": "complete ten-player integer levels in [1,18]; independently gated",
    "orientation": "blue minus red, with shared role scaling and no centering",
    "missing": "zero features; no availability intercept or age coefficient",
    "clock": "caller certifies observation age upper bound; maximum 90 seconds by default",
    "items": "excluded: ambiguous archived item undo identity; items ablated in corrected core contract",
}
_ROLE_ALIASES = dict(top="top", jungle="jng", jng="jng", mid="mid",
                     bottom="bot", bot="bot", support="sup", sup="sup")


def _number(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _vector(data, key, low, high=None, *, integral=False):
    value = data.get(key)
    if (not isinstance(value, (list, tuple, np.ndarray))
            or (isinstance(value, np.ndarray) and value.ndim != 1) or len(value) != 5):
        return None
    if any(not _number(x) or x < low or (high is not None and x > high)
           or (integral and x != int(x)) for x in value):
        return None
    return np.asarray(value, dtype=float)


def _empty_features(reason, age_upper_s, max_age_s):
    return dict(available=False, reason=reason, features=[0.] * len(FEATURE_NAMES),
                names=list(FEATURE_NAMES), known=[False] * len(FEATURE_NAMES),
                raw={channel: [0.] * 10 for channel in CHANNELS},
                channel_available={channel: False for channel in CHANNELS},
                channel_reasons={channel: reason for channel in CHANNELS},
                age_upper_s=age_upper_s, max_age_s=float(max_age_s),
                excluded_inputs=["items"])


def observation_features(data, *, age_upper_s=0., max_age_s=90.):
    """Extract JSON-serializable readiness with independent optional channels.

    A usable observation requires all ten health fractions. Gold and level
    vectors each gate only their own feature family. ``available`` describes
    health availability, while ``known`` describes every fitted feature.
    Age is metadata only; increasing age cannot create a fitted correction.
    """
    if not _number(max_age_s) or max_age_s < 0:
        raise ValueError("max_age_s must be finite and nonnegative")
    if not _number(age_upper_s) or age_upper_s < 0:
        # Invalid input metadata stays JSON safe and is never fitted.
        return _empty_features("invalid_observation_age", None, max_age_s)
    age_upper_s = float(age_upper_s)
    if age_upper_s > max_age_s:
        return _empty_features("stale_observation", age_upper_s, max_age_s)
    if not isinstance(data, Mapping):
        return _empty_features("missing_observation", age_upper_s, max_age_s)
    if data.get("available") is False:
        return _empty_features(str(data.get("reason") or "unavailable_observation"),
                               age_upper_s, max_age_s)
    hpb, hpr = (_vector(data, key, 0., 1.) for key in ("hpb", "hpr"))
    if hpb is None or hpr is None:
        reason = "missing_health" if any(data.get(k) is None for k in ("hpb", "hpr")) else "invalid_health"
        return _empty_features(reason, age_upper_s, max_age_s)
    hp = np.r_[hpb, hpr]
    alive = (hp > 0.).astype(float)
    channels = {"alive": alive, "hp": hp}
    available = {"alive": True, "hp": True}
    reasons = {"alive": "complete_health", "hp": "complete_health"}
    for channel, keys, low, high, integral, factor in (
            ("living_gold_k", ("gdb", "gdr"), 0., None, False, .001),
            ("living_level", ("lvb", "lvr"), 1., 18., True, 1.)):
        left, right = (_vector(data, key, low, high, integral=integral) for key in keys)
        valid = left is not None and right is not None
        available[channel] = valid
        reasons[channel] = ("complete_resources" if valid else
                            "missing_resources" if any(data.get(k) is None for k in keys)
                            else "invalid_resources")
        channels[channel] = alive * np.r_[left, right] * factor if valid else np.zeros(10)
    out = _empty_features("complete_health", age_upper_s, max_age_s)
    out.update(available=True,
               features=np.concatenate([channels[c][:5] - channels[c][5:] for c in CHANNELS]).tolist(),
               known=[available[c] for c in CHANNELS for _ in ROLES],
               raw={c: channels[c].tolist() for c in CHANNELS},
               channel_available=available, channel_reasons=reasons)
    return out


def live_observation(metadata, frame):
    """Join ten official participants by ID and metadata role, never list order.

    Blue participant IDs must be exactly 1..5 and red IDs 6..10. Both metadata
    and the frame must provide exactly one entry per participant and per role.
    A metadata role permutation is supported, but unknown or duplicate roles
    and participants appearing on the wrong side are rejected.
    """
    def rejected(reason):
        return dict(available=False, reason=reason,
                    **{k + side: [] for k in ("hp", "lv", "gd", "pid") for side in ("b", "r")})
    if not isinstance(metadata, Mapping) or not isinstance(frame, Mapping):
        return rejected("missing_live_observation")
    out = dict(available=True, reason="verified_live_roles", role_order=list(ROLES))
    for side, short, ids in (("blue", "b", set(range(1, 6))), ("red", "r", set(range(6, 11)))):
        team_meta = metadata.get(side + "TeamMetadata")
        team_frame = frame.get(side + "Team")
        if not isinstance(team_meta, Mapping) or not isinstance(team_frame, Mapping):
            return rejected("missing_team_metadata_or_frame")
        players = team_meta.get("participantMetadata")
        observed = team_frame.get("participants")
        if not isinstance(players, (list, tuple)) or len(players) != 5:
            return rejected("incomplete_participant_metadata")
        if not isinstance(observed, (list, tuple)) or len(observed) != 5:
            return rejected("incomplete_live_participants")
        role_ids, frame_by_id = {}, {}
        for p in players:
            if not isinstance(p, Mapping):
                return rejected("invalid_participant_metadata")
            pid, role = p.get("participantId"), p.get("role")
            if not _number(pid) or pid != int(pid) or pid not in ids:
                return rejected("metadata_participant_side_mismatch")
            if not isinstance(role, str) or role not in _ROLE_ALIASES:
                return rejected("unknown_participant_role")
            role = _ROLE_ALIASES[role]
            if role in role_ids or pid in role_ids.values():
                return rejected("duplicate_participant_role_or_id")
            role_ids[role] = int(pid)
        if set(role_ids) != set(ROLES) or set(role_ids.values()) != ids:
            return rejected("incomplete_participant_roles")
        for p in observed:
            if not isinstance(p, Mapping):
                return rejected("invalid_live_participant")
            pid = p.get("participantId")
            if not _number(pid) or pid != int(pid) or pid not in ids:
                return rejected("live_participant_side_mismatch")
            if pid in frame_by_id:
                return rejected("duplicate_live_participant_id")
            frame_by_id[int(pid)] = p
        if set(frame_by_id) != ids:
            return rejected("incomplete_live_participant_ids")
        ordered = [frame_by_id[role_ids[role]] for role in ROLES]
        health = []
        for p in ordered:
            current, maximum = p.get("currentHealth"), p.get("maxHealth")
            if current is None or maximum is None:
                return rejected("missing_health")
            if (not _number(current) or not _number(maximum) or maximum <= 0
                    or current < 0 or current > maximum):
                return rejected("invalid_health")
            health.append(float(current / maximum))
        out["hp" + short] = health
        # Invalid optional resources are kept as missing, never imputed alive
        # player strength; the pure extractor gates the whole resource family.
        out["lv" + short] = [float(p["level"]) if _number(p.get("level")) else None for p in ordered]
        out["gd" + short] = [float(p["totalGold"]) if _number(p.get("totalGold")) else None for p in ordered]
        out["pid" + short] = [role_ids[role] for role in ROLES]
    return out


def _rows(extra, baseline_p, t):
    extra, baseline_p, t = map(lambda a: np.asarray(a, dtype=float), (extra, baseline_p, t))
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("Nonnegative finite game clocks in minutes are required")
    if extra.shape != (len(t), len(FEATURE_NAMES)) or not np.isfinite(extra).all():
        raise ValueError("Expected twenty complete finite combat extras; unavailable features must be zero")
    if (baseline_p.shape != (len(t),) or not np.isfinite(baseline_p).all()
            or np.any((baseline_p < 0) | (baseline_p > 1))):
        raise ValueError("Finite core probabilities in [0,1] must match rows")
    return extra, baseline_p, t


def _logit(p, *, endpoints=False):
    clipped = np.clip(p, 1e-12, 1. - 1e-12)
    result = np.log(clipped) - np.log1p(-clipped)
    if endpoints:
        result[p == 0] = -np.inf
        result[p == 1] = np.inf
    return result


def _design(extra, t, scale):
    basis = wpgam.time_basis(t)
    rows, knots = np.nonzero(basis)
    z = extra / scale
    width = len(FEATURE_NAMES) * len(wpgam.TIME_KNOTS)
    return sparse.csr_matrix((np.concatenate([z[rows, j] * basis[rows, knots] for j in range(z.shape[1])]),
                             (np.tile(rows, z.shape[1]),
                              np.concatenate([knots + j * len(wpgam.TIME_KNOTS) for j in range(z.shape[1])]))),
                            shape=(len(t), width))


def _penalty(coefficients, l2, smooth):
    values = coefficients.reshape(len(FEATURE_NAMES), len(wpgam.TIME_KNOTS))
    delta = np.diff(values, axis=1)
    gradient = l2 * values
    gradient[:, :-1] -= smooth * delta
    gradient[:, 1:] += smooth * delta
    return (.5 * l2 * np.sum(values ** 2) + .5 * smooth * np.sum(delta ** 2), gradient.ravel())


def fit(extra, baseline_p, y, gids, t, spec=None):
    """Fit positive smooth extras on a fixed core-logit offset, train rows only.

    Scales are game-weighted RMS with a unit floor, shared across blue/red and
    learned only from these training rows. No centering, calibration fitting,
    intercept, or baseline parameter update occurs. Hyperparameters should be
    fixed before scoring the final temporal holdout.
    """
    spec = dict(dict(l2=800., smooth=70.), **(spec or {}))
    if set(spec) != {"l2", "smooth"} or any(not _number(spec[k]) or spec[k] <= 0 for k in spec):
        raise ValueError("Combat residual needs positive fixed l2 and smooth penalties")
    extra, baseline_p, t = _rows(extra, baseline_p, t)
    y, gids = np.asarray(y), np.asarray(gids)
    if not len(t) or y.shape != (len(t),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Nonempty binary labels must match input rows")
    if gids.shape != (len(t),):
        raise ValueError("Game IDs must match input rows")
    weights = wpgam._game_balanced_weights(gids)
    scale = np.maximum(np.sqrt(np.average(extra ** 2, axis=0, weights=weights)), 1.)
    offset = _logit(baseline_p)
    active = np.any(extra != 0., axis=1)
    # Unknown health is common in the conservative historical archive. Those
    # rows retain their original full-population game weights and affect RMS
    # scaling, but have zero gradient and need no sparse design allocation.
    constant_loss = float(np.dot(weights[~active],
                                np.logaddexp(0., offset[~active]) - y[~active] * offset[~active]))
    design = _design(extra[active], t[active], scale)
    active_offset, active_y, active_weights = offset[active], y[active], weights[active]

    def objective(vector):
        eta = active_offset + design @ vector
        penalty, gradient = _penalty(vector, spec["l2"], spec["smooth"])
        loss = constant_loss + np.dot(active_weights, np.logaddexp(0., eta) - active_y * eta) + penalty
        gradient += design.T @ (active_weights * (wpgam._sigmoid(eta) - active_y))
        return float(loss), gradient

    result = minimize(objective, np.zeros(design.shape[1]), jac=True, bounds=[(0., 10.)] * design.shape[1],
                      method="L-BFGS-B", options=dict(maxiter=600, ftol=1e-10, gtol=1e-5, maxcor=20))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("Combat residual optimizer failed: %s" % result.message)
    model = dict(kind=KIND, feature_names=list(FEATURE_NAMES), time_knots=wpgam.TIME_KNOTS.tolist(),
                 input_contract=dict(INPUT_CONTRACT), spec={k: float(v) for k, v in spec.items()},
                 scale=scale, coefficients=result.x, converged=True, iterations=int(result.nit),
                 objective=float(result.fun), training_rows=len(t), active_rows=int(active.sum()))
    validate(model)
    return model


def correction(model, extra, t):
    """Return a signed logit correction without modifying the core model."""
    validate(model)
    extra, _, t = _rows(extra, np.full(len(t), .5), t)
    active = np.any(extra != 0., axis=1)
    out = np.zeros(len(t))
    out[active] = _design(extra[active], t[active], model["scale"]) @ model["coefficients"]
    return out


def predict(model, extra, baseline_p, t):
    """Apply readiness to the frozen core; zero correction returns it exactly."""
    extra, baseline_p, t = _rows(extra, baseline_p, t)
    delta = correction(model, extra, t)
    out = wpgam._sigmoid(_logit(baseline_p, endpoints=True) + delta)
    # Preserve the caller's probability bit-for-bit, including its endpoints.
    unchanged = delta == 0.
    out[unchanged] = baseline_p[unchanged]
    return out


def validate(model):
    """Reject artifacts that violate the extraction or monotonicity contract."""
    if (model.get("kind") != KIND or model.get("feature_names") != list(FEATURE_NAMES)
            or model.get("time_knots") != wpgam.TIME_KNOTS.tolist()
            or model.get("input_contract") != INPUT_CONTRACT or model.get("converged") is not True):
        raise ValueError("Combat artifact contract or convergence mismatch")
    scale = np.asarray(model.get("scale"))
    coefficients = np.asarray(model.get("coefficients"))
    if scale.shape != (len(FEATURE_NAMES),) or not np.isfinite(scale).all() or np.any(scale < 1.):
        raise ValueError("Invalid combat RMS scales")
    if (coefficients.shape != (len(FEATURE_NAMES) * len(wpgam.TIME_KNOTS),)
            or not np.isfinite(coefficients).all() or np.any((coefficients < 0) | (coefficients > 10))):
        raise ValueError("Combat coefficients violate monotonicity bounds")
    spec = model.get("spec")
    if not isinstance(spec, dict) or set(spec) != {"l2", "smooth"} or any(not _number(v) or v <= 0 for v in spec.values()):
        raise ValueError("Invalid combat shrinkage contract")
    if (not _number(model.get("objective")) or model["objective"] < 0
            or not isinstance(model.get("iterations"), int) or model["iterations"] < 0):
        raise ValueError("Invalid combat optimizer metadata")


def save(model, path):
    validate(model)
    meta = {k: v for k, v in model.items() if k not in {"scale", "coefficients"}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(meta, sort_keys=True)),
                            scale=model["scale"], coefficients=model["coefficients"])


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        model = json.loads(str(archive["metadata"].item()))
        model.update(scale=archive["scale"], coefficients=archive["coefficients"])
    validate(model)
    return model
