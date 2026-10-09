"""Hash-bound production dispatch for an explicitly promoted research mixture.

The exact evaluated core and joint residual remain separate immutable files.
One atomic live-stack pointer activates both and their shared calibration.
The prior incumbent artifact remains available for fallback and rollback.
"""
import hashlib
import json
import logging
import math
import re
from pathlib import Path

import numpy as np

from . import config, wpgam, wpcombined, wpbench

log = logging.getLogger(__name__)
STACK_KIND = "wpx_live_combination_v1"
BUNDLE_KIND = "wpx_combination_bundle_v1"
MODEL_KIND = "wpgam_v9_objective_trend_composition_sq_v1"
STACK_PATH = Path(wpgam.OUT_DIR)/"live_stack.json"
_CACHE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    unsigned = {k: v for k, v in value.items() if k != "stack_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _identity(path):
    s = Path(path).stat()
    return (s.st_mtime_ns, s.st_size, s.st_ino)


def rebase(path):
    """Map an absolute path recorded by another checkout onto this one.

    Artifacts store absolute paths from the checkout that wrote them.  After the
    repository moves, ``<old root>/data/...`` is served from this checkout's
    ``data/`` when that file exists here.  Component contents remain SHA-256
    verified, so rebasing cannot substitute different bytes.
    """
    if path is None:
        return None
    path = Path(path)
    root = Path(config.REPO_ROOT)
    if not path.is_absolute() or path == root or root in path.parents:
        return path
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "data":
            candidate = root.joinpath(*parts[i:])
            if candidate.exists():
                return candidate
    return path


def _component(bundle_path, value):
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError("Combination component needs a path and hash")
    path = Path(value["path"])
    if not path.is_absolute():
        path = bundle_path.parent/path
    path = rebase(path)
    digest = value.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Combination component needs a SHA-256")
    return path.resolve(), digest


def load_bundle(path, expected_sha=None):
    path = Path(path).resolve()
    with path.open() as stream:
        bundle = json.load(stream)
    if not isinstance(bundle, dict) or bundle.get("kind") != BUNDLE_KIND:
        raise ValueError("Unsupported combination bundle")
    base_path, base_sha = _component(path, bundle.get("base"))
    joint_path, joint_sha = _component(path, bundle.get("joint"))
    capture = bundle.get("capture") or {}
    table_path, table_sha = _component(path, dict(path=capture.get("sq_table_path"), sha256=capture.get("sq_table_sha256")))
    identity = (_identity(path), _identity(base_path), _identity(joint_path), _identity(table_path), expected_sha)
    cached = _CACHE.get(str(path))
    if cached is not None and cached[0] == identity:
        return cached[1]
    digest = sha(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("Combination bundle hash mismatch")
    if any(sha(p) != h for p, h in ((base_path, base_sha), (joint_path, joint_sha), (table_path, table_sha))):
        raise ValueError("Combination component hash mismatch")
    base, joint = wpgam.load_model(str(base_path)), wpcombined.load(joint_path)
    if (base["kind"] != wpgam.MODEL_KIND
            or list(base["state"]["feature_names"]) != wpgam.PRIOR_INPUTS+wpgam.STATE_FEATURES
            or base["state"]["cal_intercept"] != 0 or base["state"]["cal_slope"] != 1
            or base["state"].get("calibration_probability_clip") is not None):
        raise ValueError("Combination requires the evaluated raw physical-gold core")
    families = [f["name"] for f in joint["blocks"]]
    if set(families) != {"objective", "trend", "composition", "sq"} or set(bundle.get("families", [])) != set(families):
        raise ValueError("Combination family contract mismatch")
    calibration = bundle.get("calibration")
    if (not isinstance(calibration, dict) or calibration.get("probability_clip") != [1e-5, 1-1e-5]
            or any(isinstance(calibration.get(k), bool) or not isinstance(calibration.get(k), (int, float))
                   or not math.isfinite(calibration[k]) for k in ("intercept", "slope"))
            or calibration["slope"] <= 0):
        raise ValueError("Invalid shared combination calibration")
    support = {k: v["supported_features"] for k, v in (bundle.get("support") or {}).items()}
    joint = dict(joint, source_checks={"sq_table_sha256": table_sha})
    if support:
        joint["trained_support"] = support
    loaded = dict(bundle=bundle, path=str(path), sha256=digest, base=base, joint=joint,
                  base_sha256=base_sha, joint_sha256=joint_sha, calibration=calibration,
                  capture=dict(capture, sq_table_path=str(table_path),
                               composition_root=(str(rebase(capture["composition_root"]))
                                                 if capture.get("composition_root") else None)))
    _CACHE[str(path)] = (identity, loaded)
    return loaded


def load_active(path=STACK_PATH):
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as stream:
        stack = json.load(stream)
    if not isinstance(stack, dict) or stack.get("kind") != STACK_KIND or stack.get("deployed") is not True:
        return None
    if stack.get("stack_sha256") != signature(stack):
        raise ValueError("Combination stack checksum mismatch")
    bundle_path, bundle_sha = _component(path, stack.get("bundle"))
    loaded = load_bundle(bundle_path, bundle_sha)
    return dict(loaded, stack=stack, stack_sha256=stack["stack_sha256"])


def capture_context(metadata, context=None):
    """Pin SQ tables and record the explicit gameplay-patch catalog mapping."""
    result = dict(context or {})
    try:
        active = load_active()
    except (OSError, ValueError, KeyError, TypeError) as error:
        log.warning("Production combination capture unavailable: %s", error)
        return result
    if active is None:
        return result
    capture = active["capture"]
    result.setdefault("sq_table_path", capture["sq_table_path"])
    result.setdefault("composition_root", capture.get("composition_root"))
    patch = metadata.get("patchVersion") if isinstance(metadata, dict) else None
    if capture.get("composition_patch_policy") == "gameplay_major_minor" and isinstance(patch, str):
        match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)*", patch)
        if match:
            result.setdefault("composition_patch", match[1]+"."+match[2])
    return result


def predict(loaded, state, blue_champs=(), red_champs=(), *, rounded=True):
    from . import wpcombined_live
    minutes = float(state.get("t_min", 0.) or 0.)
    if not math.isfinite(minutes) or minutes < 0:
        raise ValueError("Production combination needs a finite nonnegative game clock")
    if state.get("clock_s") is not None:
        clock = float(state["clock_s"])
        # Live frames floor their seconds while retaining fractional minutes.
        if not math.isfinite(clock) or clock < 0 or abs(clock-60*minutes) >= 1.+1e-9:
            raise ValueError("Production combination clocks disagree")
    core = wpgam.predict_live_model(loaded["base"], state, blue_champs, red_champs, rounded=False)
    blocks, coverage = wpcombined_live.blocks(state, loaded["joint"])
    # Existing callers pass empty drafts to measure the no-champion forecast.
    # Both added draft channels must obey that same ablation.
    draft_enabled = len(blue_champs) == len(red_champs) == 5
    if not draft_enabled:
        for family in ("composition", "sq"):
            blocks[family][:] = 0
            detail = coverage["families"][family]
            detail.update(available=False, known=[False]*len(detail["names"]), reasons=["champion_ablation"]*len(detail["names"]))
        coverage["available_families"] = [k for k in coverage["available_families"] if k not in {"composition", "sq"}]
    t = np.array([minutes])
    raw = wpcombined.predict(loaded["joint"], blocks, np.array([core["p_blue"]]), t)
    probability = float(wpbench._apply_platt(raw, loaded["calibration"])[0])
    delta = {}
    for block in loaded["joint"]["blocks"]:
        one = dict(loaded["joint"], blocks=[block])
        delta[block["name"]] = float(wpcombined.correction(one, {block["name"]: blocks[block["name"]]}, t)[0])
    out = dict(core)
    slope, intercept = loaded["calibration"]["slope"], loaded["calibration"]["intercept"]
    for key in ("lo_prior", "lo_champ", "lo_state", "lo_time", "lo_champ_state", "lo_deaths", "lo_baron_active", "lo_elder_active"):
        out[key] = round(core.get(key, 0.)*slope, 3)
    out["lo_champ"] = round(out["lo_champ"]+slope*(delta["composition"]+delta["sq"]), 3)
    out["lo_state"] = round(out["lo_state"]+slope*(delta["objective"]+delta["trend"]), 3)
    out["lo_time"] = round(out["lo_time"]+intercept, 3)
    out.update(p_blue=round(probability, 4) if rounded else probability, model_kind=MODEL_KIND,
               production_families=[f["name"] for f in loaded["joint"]["blocks"]],
               combination_coverage=coverage, combination_logit=delta,
               model_sha256=loaded["sha256"], base_sha256=loaded["base_sha256"], joint_sha256=loaded["joint_sha256"],
               stack_sha256=loaded.get("stack_sha256"), stack_reason="explicit user promotion of evaluated joint weights")
    return out
