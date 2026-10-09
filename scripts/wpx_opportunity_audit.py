"""Audit evaluated objective/trend artifacts with coherent current-player gold.

Uses corrected consumed rows and the hash-verified frozen-core caches. It
changes neither fitted artifacts nor probabilities in production. The audit
does not certify still-unobserved HP/spawn/position feature families.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpgam, wpobjective, wptrend, wpresidual, wpbench
from scripts import wpx_combat as historical


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output, dataset, cache):
    plan = json.loads((output/"plan.json").read_text())
    module = wpobjective if plan["family"] == "objective" else wptrend
    if any(sha(path) != digest for path, digest in plan["sources"].items()):
        raise ValueError("Current scorer does not match evaluated source snapshots")
    rows, _ = historical.load_rows(dataset)
    if sha(dataset) != plan["dataset_sha256"]:
        raise ValueError("Audit dataset mismatch")
    original_plan = json.loads((cache/"plan.json").read_text())
    resource = cache.parent/"resources"/"resources.npz"
    if sha(resource) != original_plan["resources_sha256"]:
        raise ValueError("Audit resource cache mismatch")
    with np.load(resource, allow_pickle=False) as z:
        gold = z["gold"]
    with np.load(output/"features.npz", allow_pickle=False) as z:
        extra, known, available = z["extra"], z["known"], z["available"]
    results = {}
    stages = [("development_"+str(i+1), b) for i, b in enumerate(original_plan["development"])]
    stages.append(("frozen", original_plan["final"]))
    for label, block in stages:
        _, masks, calibration, _, core_model, core = historical.baseline_stage(cache, label, block, rows, original_plan)
        residual_path = output/label/"residual.npz"
        model = module.load(residual_path)
        indices = np.flatnonzero(masks["validation"])
        indices = np.random.default_rng(20261004).choice(indices, min(1000, len(indices)), replace=False)
        raw, additions, t = np.array(core[indices]), extra[indices].astype(float), rows["t_min"][indices]
        names = {name: i for i, name in enumerate(core_model["feature_names"])}
        extra_names = {name: i for i, name in enumerate(module.FEATURE_NAMES)}
        def forecast(value, inputs):
            baseline = wpgam.predict_state(core_model, value, t)
            return wpbench._apply_platt(module.predict(model, inputs, baseline, t), calibration)
        ref = forecast(raw, additions)
        baseline = wpgam.predict_state(core_model, raw, t)
        zero = module.predict(model, np.zeros_like(additions), baseline, t)
        if not np.array_equal(zero, baseline):
            raise ValueError("Unknown-input baseline fallback changed")
        symmetry = np.max(np.abs(wpresidual.correction(model, additions, t)+wpresidual.correction(model, -additions, t)))
        if symmetry > 1e-12:
            raise ValueError("Residual side symmetry failed")
        comparisons = violations = 0
        worst = 0.
        for slot in range(10):
            sign, role = (1. if slot < 5 else -1.), wptrend.ROLES[slot % 5]
            for amount in (100., 1000.):
                value, changed = raw.copy(), additions.copy()
                delta = sign*amount/1000.
                for key in ("gold_k", "gold_mom", "gold_"+role):
                    value[:, names[key]] += delta
                total = gold[indices].sum(axis=1)/1000. + amount/1000.
                value[:, names["gold_rel"]] = value[:, names["gold_k"]]/total
                if plan["family"] == "trend":
                    # Current player gold increases current lead/role growth;
                    # earlier anchors and all event/HP observations stay fixed.
                    for window in wptrend.WINDOWS:
                        for name in ("lead_change_%ds_k" % window, "gold_change_%s_%ds_k" % (role, window)):
                            j = extra_names[name]
                            changed[known[indices, j], j] += delta
                # Objective historical extras contain only remaining time;
                # unsupported readiness coupling stays unknown, never invented.
                after = forecast(value, changed)
                movement = sign*(after-ref)
                if not np.isfinite(after).all() or np.any((after < 0) | (after > 1)):
                    raise ValueError("Invalid composed probability")
                comparisons += len(indices)
                violations += int((movement < -1e-10).sum())
                worst = min(worst, float(movement.min()))
        if violations:
            raise ValueError("Composed physical-gold audit failed")
        results[label] = dict(states=len(indices), coherent_gold_comparisons=comparisons,
                              reversals=violations, worst_reversal=worst, zero_fallback_exact=True,
                              correction_side_symmetry_max_error=float(symmetry), model_sha256=sha(residual_path))
    if any(sha(path) != digest for path, digest in plan["sources"].items()):
        raise ValueError("Evaluated scorer changed during audit")
    report = dict(completed=True, family=plan["family"], stages=results,
                  comparisons=sum(s["coherent_gold_comparisons"] for s in results.values()),
                  reversals=0, audit_source_sha256=sha(__file__), evaluated_plan_sha256=sha(output/"plan.json"),
                  limitation="covers exact historical supported columns and current-gold perturbations; unavailable fresh HP/spawn coupling remains unvalidated")
    (output/"additional_shape_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(dict(family=report["family"], comparisons=report["comparisons"], reversals=0), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(wpgam.OUT_DIR)/"states_inputs_v2_canonical_before_2026-09-03.npz")
    parser.add_argument("--baseline-cache", type=Path, default=Path(wpgam.OUT_DIR)/"action_20260911/resource_study")
    args = parser.parse_args(); main(args.out, args.dataset, args.baseline_cache)
