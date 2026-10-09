"""Independent physical checks for consumed-history combination candidates.

Current-gold interventions keep carried combat observations fixed. A separate
intervention beginning at the archived combat observation may persist into the
prediction: its age must be under 120 seconds, after both historical trajectory
anchors. Health interventions change the same archived sample used by the
core's HP and death inputs, while unavailable current-health trajectories stay
unavailable. These scenarios do not certify fresh objective-readiness coupling.
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpbench, wpcombat, wpgam, wptrend


def _sample(mask, sample_size, rng):
    indices = np.flatnonzero(mask)
    return rng.choice(indices, min(sample_size, len(indices)), replace=False)


def _subset(features, families, indices):
    result = {}
    for family in families:
        block = features[family]
        result[family] = {key: np.asarray(value)[indices].copy()
                          for key, value in block.items()
                          if key in {"extra", "known", "available", "raw", "pair_score", "score"}}
        if "names" in block:
            result[family]["names"] = list(block["names"])
    return result


def _names(block, defaults):
    names = list(block.get("names", defaults))
    if len(names) != block["extra"].shape[1] or len(set(names)) != len(names):
        raise ValueError("Audit feature names do not match the candidate inputs")
    return {name: j for j, name in enumerate(names)}


def _inputs(features):
    """Use the optimizer's array blocks; preserve explicit SQ coverage."""
    inputs = {}
    for family, block in features.items():
        if "extra" in block:
            inputs[family] = np.asarray(block["extra"], dtype=float)
        elif family == "sq":
            key = "pair_score" if "pair_score" in block else "score"
            score = np.asarray(block[key], dtype=float)
            inputs[family] = np.where(block["available"], score, 0.)[:, None]
        else:
            raise ValueError("Audit family requires explicit row-aligned extras")
    return inputs


def _current_gold(core, features, gold, core_names, slot, amount):
    """Increase current player gold, retaining every earlier combat sample."""
    value, changed = core.copy(), deepcopy(features)
    sign, role = (1. if slot < 5 else -1.), wptrend.ROLES[slot % 5]
    delta = sign * amount / 1000.
    for key in ("gold_k", "gold_mom", "gold_" + role):
        value[:, core_names[key]] += delta
    total = (gold.sum(axis=1) + amount) / 1000.
    value[:, core_names["gold_rel"]] = value[:, core_names["gold_k"]] / total
    if "trend" in changed:
        block = changed["trend"]
        names = _names(block, wptrend.FEATURE_NAMES)
        for window in (120, 300):
            for key in ("lead_change_%ds_k" % window,
                        "gold_change_%s_%ds_k" % (role, window)):
                j = names[key]
                block["extra"][block["known"][:, j], j] += delta
    return value, changed


def _observed_gold(features, slot, amount):
    """Change available living gold in the archived combat sample itself."""
    changed = deepcopy(features)
    if "combat" not in changed:
        return changed
    block = changed["combat"]
    names = _names(block, wpcombat.FEATURE_NAMES)
    role = wpcombat.ROLES[slot % 5]
    j = names["living_gold_" + role + "_k"]
    alive = block["raw"][:, slot, 0] > 0
    known = block["known"][:, j]
    sign = 1. if slot < 5 else -1.
    block["extra"][known & alive, j] += sign * amount / 1000.
    return changed


def _observed_hp(core, features, rows_x, row_names, core_names, slot, amount):
    """Improve the archived health sample in both combat and core features."""
    value, changed = core.copy(), deepcopy(features)
    block = changed["combat"]
    original = block["raw"].copy()
    block["raw"][:, slot, 0] = np.minimum(1., original[:, slot, 0] + amount)
    rebuilt = []
    for players in block["raw"]:
        data = {prefix + side: players[start:start+5, channel].tolist()
                for channel, prefix in enumerate(("hp", "gd", "lv"))
                for side, start in (("b", 0), ("r", 5))}
        rebuilt.append(wpcombat.observation_features(data)["features"])
    block["extra"] = np.asarray(rebuilt, dtype=float)
    block["extra"][~block["known"]] = 0.
    sign = 1. if slot < 5 else -1.
    value[:, core_names["hp_pool"]] += sign * (block["raw"][:, slot, 0] - original[:, slot, 0])
    db = (block["raw"][:, :5, 0] <= 0).sum(axis=1)
    dr = (block["raw"][:, 5:, 0] <= 0).sum(axis=1)
    effects = wpgam._death_features(db, dr,
        rows_x[:, row_names["towers_blue"]], rows_x[:, row_names["towers_red"]],
        rows_x[:, row_names["inhib_blue"]], rows_x[:, row_names["inhib_red"]])
    for key, effect in zip(("dead_adv", "dead_adv_sq", "dead_count_sq_adv", "dead_base_pressure"), effects):
        value[:, core_names[key]] = effect
    return value, changed


def _bounded(probability):
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Combined physical audit returned invalid probabilities")


def _movement(after, before, sign, report):
    _bounded(after)
    move = sign * (after - before)
    report["comparisons"] += len(move)
    report["reversals"] += int((move < -1e-10).sum())
    if len(move):
        report["worst_reversal"] = min(report["worst_reversal"], float(move.min()))


def audit(model, core_model, core, calibration, rows, features, gold, mask, *, sample_size=200, seed=20261005):
    """Validate a combination without changing fitted or serving artifacts."""
    from lol_ticker import wpcombined
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size <= 0:
        raise ValueError("A positive audit sample size is required")
    families = [block["name"] for block in model["blocks"]]
    n = len(rows["gid"])
    mask = np.asarray(mask)
    gold = np.asarray(gold, dtype=float)
    if mask.dtype != bool or mask.shape != (n,) or core.shape[0] != n or gold.shape != (n, 10):
        raise ValueError("Audit inputs must share the full corrected-row population")
    if not np.isfinite(gold).all() or np.any(gold < 0):
        raise ValueError("Audit requires complete physical player gold")
    if not set(families).issubset(features):
        raise ValueError("Missing combination audit family inputs")
    names = core_model.get("feature_names", wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES)
    if len(names) != core.shape[1] or len(set(names)) != len(names):
        raise ValueError("Audit core feature names do not match the frozen model")
    core_names = {name: j for j, name in enumerate(names)}
    row_names = {name: j for j, name in enumerate(rows["names"])}
    rng = np.random.default_rng(seed)
    indices = _sample(mask, sample_size, rng)
    result = dict(states=len(indices), zero_fallback_exact=True,
        correction_side_symmetry_max_error=0., sq_late_zero_exact=True,
        current_gold=dict(comparisons=0, reversals=0, worst_reversal=0.),
        archived_gold=dict(comparisons=0, reversals=0, worst_reversal=0.),
        archived_hp=dict(comparisons=0, reversals=0, worst_reversal=0.),
        limitations=["current-gold changes leave aged combat observations fixed",
                     "archived-gold changes begin at the combat sample and persist; age <120s, before prediction but after supported120/300s anchors",
                     "HP checks alter the archived sample used by core and combat; current-health trajectories remain unknown",
                     "unsupported objective readiness/spawn couplings are not physically certified"])
    if not len(indices):
        result.update(comparisons=0, reversals=0)
        return result

    def predict(value, blocks, t):
        baseline = wpgam.predict_state(core_model, value, t)
        return wpbench._apply_platt(wpcombined.predict(model, _inputs(blocks), baseline, t), calibration)

    raw = np.array(core[indices], dtype=float)
    selected = _subset(features, families, indices)
    t = rows["t_min"][indices]
    baseline = wpgam.predict_state(core_model, raw, t)
    zero = deepcopy(selected)
    for block in zero.values():
        if "extra" in block:
            block["extra"][:] = 0.
        for key in ("pair_score", "score"):
            if key in block:
                block[key][:] = 0.
    if not np.array_equal(wpcombined.predict(model, _inputs(zero), baseline, t), baseline):
        raise ValueError("Combined all-zero fallback changed frozen-core probabilities")
    opposite = deepcopy(selected)
    for block in opposite.values():
        if "extra" in block:
            block["extra"] *= -1.
        for key in ("pair_score", "score"):
            if key in block:
                block[key] *= -1.
    delta = wpcombined.correction(model, _inputs(selected), t)
    symmetry = np.max(np.abs(delta + wpcombined.correction(model, _inputs(opposite), t)))
    result["correction_side_symmetry_max_error"] = float(symmetry)
    if symmetry > 1e-12:
        raise ValueError("Combined residual side antisymmetry failed")
    if "sq" in families:
        without_sq = deepcopy(selected)
        for key in ("pair_score", "score"):
            if key in without_sq["sq"]:
                without_sq["sq"][key][:] = 0.
        if "extra" in without_sq["sq"]:
            without_sq["sq"]["extra"][:] = 0.
        late_t = np.maximum(t, 20.)
        if not np.array_equal(wpcombined.correction(model, _inputs(selected), late_t),
                              wpcombined.correction(model, _inputs(without_sq), late_t)):
            raise ValueError("Combined SQ correction remains nonzero at or after minute20")
    ref = predict(raw, selected, t)
    _bounded(ref)
    for slot in range(10):
        sign = 1. if slot < 5 else -1.
        for amount in (100., 1000.):
            value, changed = _current_gold(raw, selected, gold[indices], core_names, slot, amount)
            _movement(predict(value, changed, t), ref, sign, result["current_gold"])

    if "combat" in families and "raw" in features["combat"]:
        valid = mask & np.asarray(features["combat"]["available"], dtype=bool)
        age = np.asarray(rows["hp_age_upper_s"], dtype=float)
        valid &= np.isfinite(age) & (age >= 0) & (age < 120.)
        combat_indices = _sample(valid, sample_size, rng)
        result["combat_states"] = len(combat_indices)
        if len(combat_indices):
            raw = np.array(core[combat_indices], dtype=float)
            selected = _subset(features, families, combat_indices)
            t = rows["t_min"][combat_indices]
            hp = selected["combat"]["raw"][:, :, 0]
            if hp.shape != (len(combat_indices), 10) or not np.isfinite(hp).all() or np.any((hp < 0) | (hp > 1)):
                raise ValueError("Archived combat audit requires complete valid HP")
            # Current-health trajectories and readiness coupling would need a
            # distinct synchronous intervention, not this carried-sample audit.
            if "trend" in selected:
                names = _names(selected["trend"], wptrend.FEATURE_NAMES)
                hp_columns = [j for name, j in names.items() if name.startswith(("hp_change_", "alive_change_"))]
                if selected["trend"]["known"][:, hp_columns].any():
                    raise ValueError("Archived HP audit cannot certify synchronous health trajectories")
            ref = predict(raw, selected, t)
            for slot in range(10):
                sign = 1. if slot < 5 else -1.
                for amount in (100., 1000.):
                    value, changed = _current_gold(raw, selected, gold[combat_indices], core_names, slot, amount)
                    changed = _observed_gold(changed, slot, amount)
                    _movement(predict(value, changed, t), ref, sign, result["archived_gold"])
                for amount in (.1, 1.):
                    value, changed = _observed_hp(raw, selected, rows["X"][combat_indices],
                                                 row_names, core_names, slot, amount)
                    _movement(predict(value, changed, t), ref, sign, result["archived_hp"])
    result["comparisons"] = sum(result[key]["comparisons"] for key in ("current_gold", "archived_gold", "archived_hp"))
    result["reversals"] = sum(result[key]["reversals"] for key in ("current_gold", "archived_gold", "archived_hp"))
    if result["reversals"]:
        raise ValueError("Combined physical audit failed with %d reversals" % result["reversals"])
    return result
