"""Fail-closed live inputs for the tested objective/trend/composition/SQ mix.

Only channels supported by the historical combination study enter the fitted
residual. In particular, sparse approximate trend anchors, synchronous health,
uncertified objective spawn rules and unreviewed composition traits cannot
activate coefficients. Missing channels remain exact zero inputs.
"""
from collections.abc import Mapping
import math
from numbers import Real
import re

import numpy as np

from . import wpcombined, wpcomposition, wpobjective, wpsqearly, wptrend


SUPPORTED = {
    "objective": frozenset(wpobjective.FEATURE_NAMES[:2]),
    "trend": frozenset(n for n in wptrend.FEATURE_NAMES
                       if ("_120s" in n or "_300s" in n)
                       and not n.startswith(("hp_change_", "alive_change_"))),
    "composition": frozenset(wpcomposition.STATIC_NAMES),
    "sq": frozenset({"prior_patch_pair_score"}),
}
_STATE_KEYS = {"objective": "objective_opportunities", "trend": "trajectory",
               "composition": "composition", "sq": "sq_early"}


def _number(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _clock(state):
    value = state.get("clock_s")
    return float(value) if _number(value) and value >= 0 else None


def _ts(state):
    value = state.get("ts")
    if value is None and isinstance(state.get("combat"), Mapping):
        value = state["combat"].get("observation_ts")
    return float(value) if _number(value) else None


def _vector(block):
    """A malformed shape invalidates a block; invalid individual fields are zero."""
    if not isinstance(block, Mapping) or block.get("available") is not True:
        return {}, "unavailable_capture"
    names, values, known = (block.get(k) for k in ("names", "features", "known"))
    if (not isinstance(names, (list, tuple)) or not isinstance(values, (list, tuple, np.ndarray))
            or not isinstance(known, (list, tuple, np.ndarray))
            or isinstance(values, np.ndarray) and values.ndim != 1
            or isinstance(known, np.ndarray) and known.ndim != 1
            or len(names) != len(values) or len(names) != len(known)
            or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)):
        return {}, "invalid_capture_vector"
    observed = {name: float(value) for name, value, flag in zip(names, values, known)
                if isinstance(flag, (bool, np.bool_)) and flag and _number(value)}
    return observed, "captured_inputs"


def exact_trajectory(state, history):
    """Rebuild exact anchors from a trusted, canonical same-frame history.

    This optional input is internal recorder history, not interpolated features.
    Its wrapper declares the game attempt and role join; the final record must
    match the current prediction clock and timestamp. The normal serving path
    can instead supply an already captured ``trajectory_exact`` block.
    """
    if (not isinstance(history, Mapping) or history.get("source") != "same_window_frame"
            or history.get("role_order") != list(wptrend.ROLES)
            or not isinstance(history.get("records"), (list, tuple)) or not history["records"]
            or history.get("attempt_id") is None):
        return None
    attempt = state.get("attempt_id", state.get("game_id"))
    if attempt is not None and str(attempt) != str(history["attempt_id"]):
        return None
    clock, timestamp = _clock(state), _ts(state)
    if clock is None or timestamp is None:
        return None
    previous_clock, previous_ts = None, None
    tracker, result = wptrend.new_tracker(), None
    for record in history["records"]:
        if not isinstance(record, Mapping) or record.get("source") != "same_window_frame":
            return None
        c, ts = record.get("clock_s"), record.get("ts")
        if (not _number(c) or not _number(ts) or c < 0 or c > clock or ts > timestamp
                or previous_clock is not None and c < previous_clock
                or previous_ts is not None and ts <= previous_ts):
            return None
        frame = dict(clock_s=float(c), ts=float(ts), game_id=history["attempt_id"])
        try:
            result = wptrend.capture(frame, tracker, observation=dict(record),
                                    attempt_id=history["attempt_id"], max_gap_s=90., anchor_slack_s=0.)
        except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
            return None
        previous_clock, previous_ts = c, ts
    if previous_clock != clock or previous_ts != timestamp:
        return None
    return result


def _current_dynamic(block, state):
    clock, timestamp = _clock(state), _ts(state)
    return (isinstance(block, Mapping) and clock is not None and timestamp is not None
            and _number(block.get("clock_s")) and block["clock_s"] == clock
            and _number(block.get("observation_ts")) and block["observation_ts"] == timestamp)


def _trend_window(block, state, name):
    window = 120 if "_120s" in name else 300
    windows = block.get("windows")
    info = windows.get(str(window)) if isinstance(windows, Mapping) else None
    gap = block.get("max_gap_s")
    return (isinstance(info, Mapping) and info.get("available") is True
            and info.get("reason") == "causal_window"
            and _number(info.get("actual_window_s")) and info["actual_window_s"] == window
            and _number(info.get("anchor_clock_s"))
            and info["anchor_clock_s"] == _clock(state)-window
            and isinstance(info.get("samples"), int) and not isinstance(info["samples"], bool)
            and info["samples"] >= 2 and _number(gap) and 0 < gap <= 90.)


def _composition_source(block, state, name):
    provenance = block.get("provenance")
    static = provenance.get("static") if isinstance(provenance, Mapping) else None
    timestamp = _ts(state)
    if (not isinstance(static, Mapping) or timestamp is None
            or not _number(static.get("available_from_ts")) or static["available_from_ts"] > timestamp
            or not wpcomposition._primary(static.get("source_url"))
            or not isinstance(static.get("source_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", static["source_sha256"])
            or not wpcomposition._same_patch(block.get("patch"), static.get("version"))):
        return False
    coverage, reasons = block.get("coverage"), block.get("reasons")
    if (not isinstance(coverage, Mapping) or coverage.get(name) != 10
            or not isinstance(reasons, Mapping) or reasons.get(name) != "complete_static_source"):
        return False
    # An explicit release mapping must stay attached to its original feed patch.
    if block.get("feed_patch") is not None and state.get("patch") is not None:
        if block["feed_patch"] != state["patch"]:
            return False
    return True


def _sq_score(block, state, model):
    if (not isinstance(block, Mapping) or block.get("kind") != wpsqearly.KIND
            or block.get("available") is not True or not _number(block.get("pair_score"))
            or not _number(block.get("coverage"))
            or not wpsqearly.INPUT_CONTRACT["minimum_pair_coverage"] <= block["coverage"] <= 1.):
        return None
    clock = _clock(state)
    if clock is None or clock >= 1200.:
        return None
    patch, earlier = block.get("patch"), block.get("source_patches")
    if not isinstance(patch, str) or not re.fullmatch(r"\d+\.\d+", patch):
        return None
    feed_patch = state.get("patch")
    if feed_patch is not None:
        if (not isinstance(feed_patch, str) or not re.fullmatch(r"\d+(?:\.\d+)+", feed_patch)
                or tuple(map(int, feed_patch.split(".")[:2])) != tuple(map(int, patch.split(".")))):
            return None
    if (not isinstance(earlier, (list, tuple)) or not earlier
            or any(not isinstance(p, str) or not re.fullmatch(r"\d+\.\d+", p)
                   or tuple(map(int, p.split("."))) >= tuple(map(int, patch.split("."))) for p in earlier)):
        return None
    bindings = model.get("source_checks", model.get("live_source_bindings", {}))
    expected_hash = bindings.get("sq_table_sha256") if isinstance(bindings, Mapping) else None
    if expected_hash is not None and block.get("table_sha256") != expected_hash:
        return None
    return float(block["pair_score"])


def blocks(state, model):
    """Return ``(one_row_family_arrays, coverage_details)`` for a joint artifact.

    Artifact names determine ordering. Optional ``trained_support`` metadata
    narrows the audited historical support; zero-weight columns always remain
    withheld. Optional ``source_checks.sq_table_sha256`` pins SQ capture.
    Unsupported or malformed captures cannot affect the production residual.
    """
    wpcombined.validate(model)
    state = state if isinstance(state, Mapping) else {}
    features, detail = {}, dict(families={}, available_families=[], clock_s=_clock(state))
    support = model.get("trained_support")
    for family in model["blocks"]:
        family_name, names = family["name"], family["feature_names"]
        out, known = np.zeros((1, len(names))), np.zeros(len(names), dtype=bool)
        reasons = ["unsupported_historical_channel"]*len(names)
        allowed = set(SUPPORTED.get(family_name, ()))
        if support is not None:
            declared = support.get(family_name) if isinstance(support, Mapping) else None
            if (not isinstance(declared, (list, tuple))
                    or any(not isinstance(n, str) for n in declared)):
                allowed.clear()
            else:
                allowed &= set(declared)
        active = np.any(np.asarray(family["coefficients"]).reshape(len(names), -1) != 0, axis=1)
        block = state.get(_STATE_KEYS.get(family_name, ""))
        if family_name == "trend":
            block = state.get("trajectory_exact", block)
            if state.get("trajectory_history") is not None:
                exact = exact_trajectory(state, state["trajectory_history"])
                if exact is not None:
                    block = exact
        values, reason = _vector(block) if family_name != "sq" else ({}, "unavailable_sq_capture")
        for j, name in enumerate(names):
            if name not in allowed:
                continue
            if not active[j]:
                reasons[j] = "untrained_zero_coefficient"
                continue
            if family_name == "sq":
                score = _sq_score(block, state, model)
                if score is not None:
                    out[0, j], known[j], reasons[j] = score, True, "prior_patch_pair_score"
                else:
                    reasons[j] = "unavailable_sq_capture"
                continue
            reasons[j] = reason
            if name not in values:
                continue
            if family_name in {"objective", "trend"} and not _current_dynamic(block, state):
                reasons[j] = "dynamic_frame_mismatch"
                continue
            if family_name == "trend" and not _trend_window(block, state, name):
                reasons[j] = "nonexact_or_incomplete_trajectory_window"
                continue
            if family_name == "objective" and not -1. <= values[name] <= 1.:
                reasons[j] = "invalid_normalized_buff_timer"
                continue
            if family_name == "composition" and not _composition_source(block, state, name):
                reasons[j] = "uncertified_composition_source"
                continue
            out[0, j], known[j], reasons[j] = values[name], True, "supported_capture"
        features[family_name] = out
        detail["families"][family_name] = dict(available=bool(known.any()), names=list(names),
                                              known=known.tolist(), reasons=reasons)
        if known.any():
            detail["available_families"].append(family_name)
    return features, detail
