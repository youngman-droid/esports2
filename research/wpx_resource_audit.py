"""Read-only supplementary checks for a completed corrected resource study.

Checks provenance and saved probability alignment, then probes every frozen
and monthly artifact through the study's inference path. Subgroup comparisons
are descriptive consumed-history evidence and never alter model selection.
Only the explicitly named supplementary report is written.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lol_ticker import wpbench, wpgam
from research import wpx_resource as runner
from research.wpx_major import competition_group


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return runner.sha(path)


def checked(path, expected):
    actual = sha(path)
    require(actual == expected, "SHA-256 mismatch: " + str(path))
    return actual


def probabilities(values):
    values = np.asarray(values)
    require(np.isfinite(values).all() and np.all((0 <= values) & (values <= 1)),
            "nonfinite or out-of-range probability")
    return values


def remove_hp(X, names):
    """Apply the documented missing-observation contract before encoding."""
    changed = np.array(X, copy=True)
    idx = {str(n): i for i, n in enumerate(names)}
    for name in ("hp_pool", "hp_low_b", "hp_low_r", "lvl_k", "has_hp"):
        changed[:, idx[name]] = 0
    return changed


def calibration_audit(p, y, gids, method, saved, validation_p, validation_y, validation_gids):
    """Solve the bounded convex calibration independently by nested root finding.

    The intercept optimum is a monotone scalar root for a fixed slope. The
    envelope's slope derivative is monotone too, giving the global box optimum
    without relying on the study's L-BFGS stopping status.
    """
    from scipy.optimize import brentq
    from scipy.special import expit
    if method == "none":
        return dict(method="identity", optimization_required=False)
    p, y = np.asarray(p), np.asarray(y)
    logits = (np.log(np.clip(p, 1e-5, 1 - 1e-5) / np.clip(1 - p, 1e-5, 1))
              if method == "temperature" else np.log(np.clip(p, 1e-5, 1 - 1e-5) /
                                                       (1 - np.clip(p, 1e-5, 1 - 1e-5))))
    _, inverse, counts = np.unique(gids, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    weights /= weights.mean()
    def derivatives(intercept, slope):
        residual = weights * (expit(intercept + slope * logits) - y)
        return np.array([residual.sum(), np.dot(residual, logits) + .25 * slope])
    def bounded_root(function, lo, hi):
        if function(lo) >= 0:
            return lo
        if function(hi) <= 0:
            return hi
        return brentq(function, lo, hi, xtol=1e-13, rtol=1e-14)
    def intercept_at(slope):
        return 0. if method == "temperature" else bounded_root(lambda a: derivatives(a, slope)[0], -5., 5.)
    slope = bounded_root(lambda b: derivatives(intercept_at(b), b)[1], .05, 5.)
    beta = np.array([intercept_at(slope), slope])
    original = np.array([saved["intercept"], saved["slope"]])
    def objective(value):
        eta = value[0] + value[1] * logits
        return float(np.dot(weights, np.logaddexp(0, eta) - y * eta) + .125 * value[1] ** 2)
    def kkt(value):
        gradient = derivatives(*value)
        lo, hi = np.array([-5., .05]), np.array([5., 5.])
        gradient[(value <= lo + 1e-10) & (gradient > 0)] = 0
        gradient[(value >= hi - 1e-10) & (gradient < 0)] = 0
        if method == "temperature":
            gradient[0] = 0
        return float(np.max(np.abs(gradient)) / weights.sum())
    require(kkt(beta) < 1e-10, "independent calibration root violates normalized KKT condition")
    refit = dict(intercept=float(beta[0]), slope=float(beta[1]))
    before = wpbench._apply_platt(validation_p, saved)
    after = wpbench._apply_platt(validation_p, refit)
    before_metrics = runner.methods.metrics(before, validation_y, validation_gids)
    after_metrics = runner.methods.metrics(after, validation_y, validation_gids)
    return dict(method=method, solver="nested_monotone_derivative_roots", independent_calibration=refit,
                saved_normalized_kkt=kkt(original), refit_normalized_kkt=kkt(beta),
                saved_objective=objective(original), refit_objective=objective(beta),
                objective_improvement=objective(original) - objective(beta),
                max_coefficient_difference=float(np.max(np.abs(original - beta))),
                max_validation_probability_difference=float(np.max(np.abs(after - before))),
                validation_brier_change=after_metrics["brier_game"] - before_metrics["brier_game"],
                validation_logloss_change=after_metrics["logloss_game"] - before_metrics["logloss_game"])


def encode_state(part, X, names, old):
    """Rebuild every state view together; preserve frozen pregame priors."""
    changed = {k: np.array(v, copy=True) for k, v in part.items()}
    nprior = len(wpgam.PRIOR_INPUTS)
    changed["raw"][:, nprior:] = wpgam.state_values_from_matrix(X, names)
    changed["old_raw"][:, nprior:] = old.state_values_from_matrix(X, names)
    changed["rich_raw"][:, :len(runner.wa.CORE_NAMES)] = changed["raw"]
    return changed


def behavioral_audit(model, calibration, part, X, names, old):
    idx = {str(n): i for i, n in enumerate(names)}
    def predict(state):
        return probabilities(wpbench._apply_platt(runner.predict(model, state, old), calibration))
    original = predict(part)
    # Removal is a missing-input perturbation, with no promised direction.
    missing_X = remove_hp(X, names)
    missing_part = encode_state(part, missing_X, names, old)
    missing = predict(missing_part)
    second = predict(encode_state(missing_part, remove_hp(missing_X, names), names, old))
    require(np.array_equal(missing, second), "HP fallback is not idempotent")
    cases = {}
    def compare(label, before_X, after_X, sign):
        before = predict(encode_state(part, before_X, names, old))
        after = predict(encode_state(part, after_X, names, old))
        move = sign * (after - before)
        cases[label] = dict(comparisons=len(move), violations=int(np.sum(move < -1e-10)),
                            worst_reversal=float(min(0., move.min())), finite=True)
    # Favorable Elder grant increments only the Elder count and buff. Elemental
    # dragon/soul channels remain unchanged, matching the corrected contract.
    for side, sign in (("blue", 1), ("red", -1)):
        before = np.array(missing_X, copy=True)
        before[:, idx["elder_buff"]] = 0
        after = np.array(before, copy=True)
        after[:, idx["elder_buff"]] = sign
        after[:, idx["d_elder"]] += sign
        compare("elder_" + side, before, after, sign)
    # Use unavailable HP so death perturbations do not contradict a measured
    # health vector. Re-encoding changes all four nonlinear death channels.
    for side, sign in (("red", 1), ("blue", -1)):
        for count in range(5):
            before = np.array(missing_X, copy=True)
            before[:, idx["dead_" + side]] = count
            after = np.array(before, copy=True)
            after[:, idx["dead_" + side]] = count + 1
            compare("dead_%s_%d_to_%d" % (side, count, count + 1), before, after, sign)
    violations = sum(v["violations"] for v in cases.values())
    required = model["family"] == "core" or model["family"].startswith("pooled_")
    return dict(sampled_states=len(X), original_probability_min=float(original.min()),
                original_probability_max=float(original.max()), cases=cases,
                missing_hp=dict(finite=True, idempotent=True,
                                mean_probability_change=float(np.mean(missing - original)),
                                max_abs_probability_change=float(np.max(np.abs(missing - original))),
                                direction_required=False),
                required_for_candidate=required, favorable_response_violations=violations,
                passed=violations == 0)


def sample_part(base, fixed_X, extra, core, indices, old):
    rows = base["row_game"][indices]
    raw = np.asarray(core[indices])
    X = fixed_X[indices]
    np.testing.assert_allclose(raw[:, len(wpgam.PRIOR_INPUTS):],
                               wpgam.state_values_from_matrix(X, base["names"]), atol=1e-10)
    part = dict(raw=raw, old_raw=np.c_[raw[:, :len(wpgam.PRIOR_INPUTS)],
                                      old.state_values_from_matrix(X, base["names"])],
                rich_raw=np.c_[raw, extra["extra"][indices]], gold=extra["gold"][indices],
                C=base["game_C"][rows], cats=extra["cats"][rows], t=base["t_min"][indices])
    return part, X


def slices(predictions, y, gids, t, X, names, league, selected, series):
    idx = {str(n): i for i, n in enumerate(names)}
    known = {}
    for label in np.unique(league):
        known[str(label)] = competition_group(str(label))
    major = np.array([known[str(label)] is not None for label in league])
    hp = X[:, idx["has_hp"]] > .5
    masks = dict(all=np.ones(len(y), dtype=bool), hp_missing=~hp, hp_available=hp,
                 early=t < 15, mid=(t >= 15) & (t < 30), late=t >= 30,
                 major=major, nonmajor=~major,
                 elder_active=np.abs(X[:, idx["elder_buff"]]) > 0,
                 any_dead=(X[:, idx["dead_blue"]] + X[:, idx["dead_red"]]) > 0)
    out = {}
    for label, mask in masks.items():
        if not mask.any():
            out[label] = dict(states=0, games=0)
            continue
        family_metrics = {family: runner.methods.metrics(predictions[family][mask], y[mask], gids[mask])
                          for family in dict.fromkeys((selected, "core", "v8_reference"))}
        paired = {reference: runner.methods.paired(predictions[selected][mask], predictions[reference][mask],
                                                   y[mask], gids[mask], series)
                  for reference in ("core", "v8_reference")}
        out[label] = dict(states=int(mask.sum()), games=len(np.unique(gids[mask])),
                          metrics=family_metrics, selected_minus_reference=paired)
    return out


def main(dataset, study, resources, output, sample):
    require(sample > 0, "sample must be positive")
    require(not output.exists(), "supplementary output already exists")
    complete = json.loads((study / "completion.json").read_text())
    require(complete.get("completed") is True, "study has not completed")
    plan = json.loads((study / "plan.json").read_text())
    selection = json.loads((study / "selection.json").read_text())
    require(selection["selected"] == complete["selected"], "completion selection differs")
    checked(dataset, plan["dataset_sha256"])
    checked(dataset.with_suffix(dataset.suffix + ".manifest.json"), plan["input_manifest_sha256"])
    checked(resources / "resources.npz", plan["resources_sha256"])
    for file, digest in {**plan["sources"], **plan["protected"]}.items():
        checked(Path(file) if Path(file).is_absolute() else ROOT / file, digest)
    snapshot = {str(p): sha(p) for p in sorted((study / "source_snapshot").iterdir())}
    require(collections.Counter(snapshot.values()) == collections.Counter(plan["sources"].values()),
            "source snapshots do not match registered sources")
    old_path = next(Path(p) for p in plan["sources"] if Path(p).name == "incumbent_source.py")
    old = runner.past_reference(old_path if old_path.is_absolute() else ROOT / old_path)
    base = wpbench._load_base(str(dataset))
    extra = runner.load_extra(base, dataset, resources)
    fixed_X = base["raw_X"][base["raw_fixed"]]
    with np.load(dataset, allow_pickle=False) as archive:
        league = archive["league"][base["raw_fixed"]]
    index = {str(n): i for i, n in enumerate(base["names"])}
    missing = fixed_X[:, index["has_hp"]] <= .5
    hp_columns = [index[n] for n in ("hp_pool", "hp_low_b", "hp_low_r", "lvl_k")]
    require(np.all(fixed_X[missing][:, hp_columns] == 0), "missing-HP corpus row has nonzero telemetry")
    blocks, final = runner.wa.split_blocks(base["game_dates"], base["outer_train"], folds=3)
    stages = [("development_" + str(i + 1), b) for i, b in enumerate(blocks)] + [("frozen", final)]
    dates = base["game_dates"]
    monthly = sorted({date[:7] for date in dates[~base["outer_train"]]})
    for month in monthly:
        start, end = month + "-01", str(np.datetime64(month, "M") + 1) + "-01"
        if start == final["validation_start"]:
            continue
        cutoff = str(np.datetime64(start) - np.timedelta64(28, "D"))
        stages.append(("replay_" + month, dict(fit=dates < cutoff,
            calibration=(dates >= cutoff) & (dates < start),
            validation=(dates >= start) & (dates < end) & ~base["outer_train"])))
    plan_identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    records_out, audits, chosen_predictions, validation_indices, calibrations = {}, {}, {}, {}, {}
    for stage_name, block in stages:
        directory = study / stage_name
        cache = json.loads((directory / "stack_cache.json").read_text())
        expected = dict(plan_sha256=plan_identity,
                        fit_gids_sha256=hashlib.sha256(base["game_gid"][block["fit"]].tobytes()).hexdigest())
        require(cache["identity"] == expected, "prior stack identity differs: " + stage_name)
        checked(directory / "core.npy", cache["core_sha256"])
        checked(directory / "priors.pkl", cache["prior_sha256"])
        core = np.load(directory / "core.npy", mmap_mode="r")
        require(core.shape == (len(base["y"]), len(runner.wa.CORE_NAMES)), "prior stack shape differs")
        records = json.loads((directory / "results.json").read_text())
        selected_stage = not stage_name.startswith("development_")
        expected_keys = ({f + "_" + str(c["index"]) for f, c in selection["choices"].items()} if selected_stage
                         else {f + "_" + str(i) for f, specs in plan["specs"].items() for i in range(len(specs))})
        require(set(records) == expected_keys, "incomplete stage families: " + stage_name)
        chosen_predictions[stage_name] = {}
        validation_indices[stage_name] = np.flatnonzero(block["validation"][base["row_game"]])
        for key, record in records.items():
            model_path, probability_path = directory / (key + ".pkl"), directory / (key + "_raw.npz")
            checked(model_path, record["model_sha256"])
            checked(probability_path, record["prediction_sha256"])
            model = pickle.loads(model_path.read_bytes())  # Only after matching the frozen hash.
            require(model["family"] == record["family"], "artifact family differs from stage record")
            verified = {}
            with np.load(probability_path, allow_pickle=False) as saved:
                cal_indices = np.flatnonzero(block["calibration"][base["row_game"]])
                val_indices = validation_indices[stage_name]
                calibrations[stage_name + "/" + key] = {
                    method: calibration_audit(saved["calibration"], base["y"][cal_indices], base["gid"][cal_indices],
                        method, result["calibration"], saved["validation"], base["y"][val_indices], base["gid"][val_indices])
                    for method, result in record["methods"].items()}
                for label in ("calibration", "validation"):
                    all_indices = np.flatnonzero(block[label][base["row_game"]])
                    stored = probabilities(saved[label])
                    require(stored.shape == (len(all_indices),), "prediction row count differs")
                    pick = np.sort(np.random.default_rng(20260912).choice(len(all_indices),
                                   min(sample if selected_stage else 128, len(all_indices)), replace=False))
                    part, X = sample_part(base, fixed_X, extra, core, all_indices[pick], old)
                    regenerated = probabilities(runner.predict(model, part, old))
                    np.testing.assert_allclose(regenerated, stored[pick], atol=1e-10, rtol=1e-10)
                    verified[label] = dict(rows=len(stored), reconstructed_rows=len(pick), finite=True)
                    if label == "validation":
                        for method, method_record in record["methods"].items():
                            calibrated = probabilities(wpbench._apply_platt(stored, method_record["calibration"]))
                            actual = runner.methods.metrics(calibrated, base["y"][all_indices], base["gid"][all_indices])
                            for metric, value in actual.items():
                                require(np.isclose(value, method_record["metrics"][metric], atol=1e-12, rtol=1e-10),
                                        "saved metric differs: " + key + "/" + metric)
                        if selected_stage:
                            family = record["family"]
                            method = selection["choices"][family]["calibration"]
                            calibration = record["methods"][method]["calibration"]
                            chosen_predictions[stage_name][family] = probabilities(wpbench._apply_platt(stored, calibration))
                            audits[stage_name + "/" + family] = behavioral_audit(model, calibration, part, X, base["names"], old)
            records_out[stage_name + "/" + key] = dict(model_sha256=record["model_sha256"],
                prediction_sha256=record["prediction_sha256"], predictions=verified)
        del core
        print("verified stage " + stage_name, flush=True)
    frozen_indices = validation_indices["frozen"]
    rolling = {f: np.full(len(base["y"]), np.nan) for f in plan["specs"]}
    for month in monthly:
        selected_month = (base["game_dates"][base["row_game"]].astype("U7") == month) & ~base["outer_train"][base["row_game"]]
        stage = "frozen" if month + "-01" == final["validation_start"] else "replay_" + month
        positions = validation_indices[stage]
        for family, values in chosen_predictions[stage].items():
            rolling[family][positions[selected_month[positions]]] = values[selected_month[positions]]
    for label, expected in (("frozen", chosen_predictions["frozen"]),
                            ("rolling", {f: probabilities(p[frozen_indices]) for f, p in rolling.items()})):
        with np.load(study / (label + "_predictions.npz"), allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["gid"], base["gid"][frozen_indices])
            np.testing.assert_array_equal(archive["y"], base["y"][frozen_indices])
            for family, p in expected.items():
                np.testing.assert_allclose(archive[family], p, atol=1e-12, rtol=1e-12)
        require(set(expected) == set(plan["specs"]), "missing aggregate prediction family")
    selected = selection["selected"]["family"]
    descriptive = slices({f: p[frozen_indices] for f, p in rolling.items()},
        base["y"][frozen_indices], base["gid"][frozen_indices], base["t_min"][frozen_indices],
        fixed_X[frozen_indices], base["names"], league[frozen_indices], selected, extra["series"])
    for file, digest in {**plan["sources"], **plan["protected"]}.items():
        checked(Path(file) if Path(file).is_absolute() else ROOT / file, digest)
    failures = [name for name, audit in audits.items() if audit["required_for_candidate"] and not audit["passed"]]
    solved = [value for stage in calibrations.values() for value in stage.values()
              if value.get("optimization_required") is not False]
    calibration_summary = dict(independently_solved=len(solved),
        max_abs_validation_brier_change=max(abs(v["validation_brier_change"]) for v in solved),
        max_coefficient_difference=max(v["max_coefficient_difference"] for v in solved),
        max_saved_normalized_kkt=max(v["saved_normalized_kkt"] for v in solved),
        max_refit_normalized_kkt=max(v["refit_normalized_kkt"] for v in solved))
    calibration_failures = [artifact + "/" + method for artifact, methods in calibrations.items()
        for method, value in methods.items() if value.get("optimization_required") is not False
        and (abs(value["validation_brier_change"]) > 1e-7 or value["saved_normalized_kkt"] > 1e-5)]
    calibration_summary.update(failures=calibration_failures,
        numerical_tolerance=dict(max_abs_validation_brier_change=1e-7, max_saved_normalized_kkt=1e-5))
    report = dict(kind="resource_study_supplementary_audit_v1", passed=not failures and not calibration_failures,
        required_family_failures=failures, selected=selected, selection_unchanged=True,
        dataset_sha256=plan["dataset_sha256"], plan_sha256=sha(study / "plan.json"),
        audit_source_sha256=sha(Path(__file__)), source_snapshots=snapshot,
        artifacts=records_out, behavior=audits, calibrations=calibrations, calibration_summary=calibration_summary,
        missing_hp_corpus_rows=int(missing.sum()),
        missing_hp_corpus_nonzero_telemetry=0, slices=descriptive,
        aggregate_prediction_hashes={label: sha(study / (label + "_predictions.npz")) for label in ("frozen", "rolling")},
        limitations=["Behavior probes sample validation states per artifact; coefficient constraints supply broader support.",
                      "V8 and rich controls are diagnostics, excluded from candidate-safety gating.",
                      "Missing HP is encoded upstream as zero health/level channels and has_hp=0; probability movement has no required direction.",
                      "Subgroups overlap and are exploratory consumed-history descriptions, with no reselection or multiplicity adjustment."])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(dict(output=str(output), passed=report["passed"], artifacts=len(records_out),
                          behavioral_artifacts=len(audits), selected=selected, required_family_failures=failures,
                          calibration_failures=calibration_failures), indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=1000)
    args = parser.parse_args()
    sys.exit(main(args.dataset, args.study, args.resources, args.output, args.sample))
