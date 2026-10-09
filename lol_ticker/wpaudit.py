"""Physical-input and preprocessing checks for staged WP models."""
import numpy as np

from . import wpgam


def artifact_shape(model):
    state = model["state"]
    names = list(state["feature_names"])
    failures = []
    if names != wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES:
        failures.append("physical role-gold feature contract required")
    if not np.isfinite(state["theta"]).all() or state["cal_slope"] <= 0:
        failures.append("finite coefficients and positive calibration required")
    for i, name in enumerate(names):
        if name in wpgam.MONOTONE_FEATURES and np.any(state["theta"][i+1] < -1e-10):
            failures.append("negative oriented coefficient: " + name)
        if name in wpgam.DISCRETE_BOUNDS:
            lower, upper = wpgam.DISCRETE_BOUNDS[name]
            if state["lo"][i] > lower or state["hi"][i] < upper:
                failures.append("discrete support clipped: " + name)
    return {"passed": not failures, "failures": failures}


def gold_perturbation(raw, feature_names, total_gold_k, slot, amount_k):
    """Add current gold to one player; earlier gold and all other state stay fixed."""
    names = {str(n): i for i, n in enumerate(feature_names)}
    changed = np.asarray(raw, dtype=float).copy()
    sign = 1.0 if slot < 5 else -1.0
    changed[:, names["gold_k"]] += sign * amount_k
    changed[:, names["gold_mom"]] += sign * amount_k
    for role, name in enumerate(wpgam.GOLD_ROLES):
        if "gold_" + name in names:
            changed[:, names["gold_" + name]] += sign * amount_k * (role == slot % 5)
        elif "gold_" + name + "_alloc" in names:
            changed[:, names["gold_" + name + "_alloc"]] += sign * amount_k * (
                float(role == slot % 5) - .2)
    changed[:, names["gold_rel"]] = changed[:, names["gold_k"]] / (
        np.asarray(total_gold_k) + amount_k)
    return changed


def audit_gold(state_model, raw, t_min, total_gold_k, amounts=(.1, 1.), tolerance=1e-8):
    """Test every player on supplied states, without inspecting any outcomes."""
    raw, t_min = np.asarray(raw), np.asarray(t_min)
    before = wpgam.predict_state(state_model, raw, t_min)
    if not np.isfinite(before).all():
        raise ValueError("non-finite baseline predictions")
    rows = []
    for amount in amounts:
        for slot in range(10):
            changed = gold_perturbation(raw, state_model["feature_names"],
                                        total_gold_k, slot, amount)
            after = wpgam.predict_state(state_model, changed, t_min)
            if not np.isfinite(after).all():
                raise ValueError("non-finite perturbed predictions")
            delta = (1 if slot < 5 else -1) * (after - before)
            rows.append({"slot": slot, "gold_added": float(amount * 1000),
                         "violations": int(np.sum(delta < -tolerance)),
                         "worst_probability_change": float(delta.min())})
    return {"passed": all(r["violations"] == 0 for r in rows),
            "states": len(raw), "comparisons": len(raw) * len(rows),
            "violations": sum(r["violations"] for r in rows), "per_slot": rows}
