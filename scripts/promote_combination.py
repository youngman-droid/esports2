"""Package the exact tested August combination for an explicit user promotion.

This command only stages immutable artifacts. It never writes the active stack,
production model paths, evaluation registries, or service state. The base stays
uncalibrated because the evaluated Platt transform follows the joint residual.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpgam, wpcombined, wpbench, sqpairs
from scripts import wpx_combat, wpx_combinations


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("objective", "trend", "composition", "sq")
KEY = "+".join(FAMILIES)
LABEL = "replay_2026-08"
KIND = "wpx_combination_bundle_v1"
DEFAULT_DATASET = ROOT / "data/wpx/states_inputs_v2_canonical_before_2026-09-03.npz"
DEFAULT_CACHE = ROOT / "data/wpx/action_20260911/resource_study"
DEFAULT_EXPERIMENT = ROOT / "data/wpx/combinations_2026-10-04"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError("Hash mismatch: " + str(path))
    return actual


def reconstruct_base(priors, fitted, *, metadata=None):
    """Rehydrate cached fitted upstream/core weights without any fitting."""
    if set(priors) != {"train", "pregame", "champ_state_beta"}:
        raise ValueError("Unexpected frozen upstream cache contract")
    expected = wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES
    names = [str(name) for name in fitted.get("feature_names", [])]
    if names != expected or fitted.get("optimizer_success") is not True:
        raise ValueError("An optimized physical-gold core is required")
    pre, state = copy.deepcopy(priors["pregame"]), copy.deepcopy(fitted)
    if (len(pre["mean"]) != len(wpgam.PREGAME_FEATURES)
            or len(pre["champ_names"]) != len(priors["champ_state_beta"])
            or state["theta"].shape != (len(names) + 1, len(state["knots"]))):
        raise ValueError("Frozen upstream/core dimensions disagree")
    # These cache weights precede shared Platt. Calibrating the core here would
    # change the offset the tested joint residual was trained against.
    if state.get("cal_intercept", 0.) != 0. or state.get("cal_slope", 1.) != 1.:
        raise ValueError("The cached core must be uncalibrated")
    if state.get("calibration_probability_clip") is not None:
        raise ValueError("The raw core must have no probability clipping")
    state.update(cal_intercept=0., cal_slope=1., calibration_method="identity_before_joint_residual",
                 calibration_probability_clip=None)
    return dict(kind=wpgam.MODEL_KIND, pregame=pre, state=state,
                champ_state=dict(beta=np.array(priors["champ_state_beta"], copy=True),
                                 cap_min=wpgam.CHAMP_STATE_CAP_MIN, l2=wpgam.CHAMP_STATE_L2),
                meta=dict(metadata or {}))


def assert_match(actual, expected, *, what, tolerance=1e-11):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Reproduction shape/finiteness mismatch: " + what)
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.
    if error > tolerance:
        raise ValueError("Reproduction mismatch for %s: %.17g" % (what, error))
    return error


def _completed_artifact(experiment, completion, relative):
    expected = completion["artifacts"].get(relative)
    if not expected:
        raise ValueError("Experiment completion does not bind " + relative)
    return _bound(experiment / relative, expected)


def stage(output, *, dataset=DEFAULT_DATASET, cache=DEFAULT_CACHE, experiment=DEFAULT_EXPERIMENT, sq_tables=None):
    """Return a self-contained staged bundle path; activation is a separate step."""
    output, dataset, cache, experiment = map(lambda p: Path(p).resolve(), (output, dataset, cache, experiment))
    if output.exists():
        raise ValueError("Use a new staging directory to preserve immutable artifacts")
    completion = json.loads((experiment / "completion.json").read_text())
    if completion.get("completed") is not True:
        raise ValueError("Combination experiment is incomplete")
    dependencies = {str(experiment / "completion.json"): sha(experiment / "completion.json")}
    relatives = ("plan.json", "report.json", LABEL + "/report.json", LABEL + "/" + KEY + ".npz",
                 LABEL + "/" + KEY + "_predictions.npz")
    for relative in relatives:
        dependencies[str(experiment / relative)] = _completed_artifact(experiment, completion, relative)
    plan = json.loads((experiment / "plan.json").read_text())
    experiment_report = json.loads((experiment / "report.json").read_text())
    if experiment_report["selection"]["exploratory_best"] != KEY:
        raise ValueError("The requested bundle must be the reported development winner")
    dependencies[str(dataset)] = _bound(dataset, plan["dataset_sha256"])
    manifest_path = Path(str(dataset) + ".manifest.json")
    dependencies[str(manifest_path)] = _bound(manifest_path, plan["manifest_sha256"])
    original_path = cache / "plan.json"
    dependencies[str(original_path)] = _bound(original_path, plan["baseline_plan_sha256"])
    original = json.loads(original_path.read_text())
    rows, _ = wpx_combat.load_rows(dataset)
    if max(rows["date"]) > "2026-09-02":
        raise ValueError("Packaging must not inspect new outcomes")
    stage_report = json.loads((experiment / LABEL / "report.json").read_text())
    baseline, masks, calibration, cache_hashes, fitted, core = wpx_combat.baseline_stage(
        cache, LABEL, stage_report["block"], rows, original)
    if calibration != stage_report["calibration"] or cache_hashes != stage_report["baseline_hashes"]:
        raise ValueError("The tested core/calibration cache provenance changed")
    for name, digest in cache_hashes.items():
        dependencies[str(cache / LABEL / name)] = digest
    priors = pickle.loads((cache / LABEL / "priors.pkl").read_bytes())
    if not np.array_equal(priors["train"], masks["fit"]):
        raise ValueError("The cached upstream training population differs")
    base = reconstruct_base(priors, fitted, metadata=dict(
        packaged_from=LABEL, immutable_fitting_population=stage_report["block"],
        dataset_sha256=plan["dataset_sha256"], cached_sources=cache_hashes,
        refitted=False, calibration_applied_after_joint=True))
    nonfit = masks["calibration"] | masks["validation"]
    with np.load(dataset, allow_pickle=False) as source:
        C = source["C"][source["seq"] < 0][nonfit]
    upstream = wpgam.prior_values_from_matrix(base, rows["X"][nonfit], rows["names"], C)
    live_core_rows = np.column_stack([upstream, wpgam.state_values_from_matrix(
        rows["X"][nonfit], rows["names"], feature_names=wpgam.STATE_FEATURES)])
    upstream_error = assert_match(live_core_rows, core[nonfit], what="fitted upstream and physical state rows")
    raw_error = assert_match(wpgam.predict_state(base["state"], live_core_rows, rows["t_min"][nonfit]),
                             wpgam.predict_state(fitted, core[nonfit], rows["t_min"][nonfit]),
                             what="raw core probabilities")
    features, feature_sources = wpx_combinations.load_features(dataset.parent, rows, plan["dataset_sha256"])
    dependencies.update(feature_sources)
    joint_source = experiment / LABEL / (KEY + ".npz")
    joint = wpcombined.load(joint_source)
    if {f["name"] for f in joint["blocks"]} != set(FAMILIES):
        raise ValueError("Unexpected requested family set")
    val, fit = masks["validation"], masks["fit"]
    raw = wpcombined.predict(joint, {k: features[k]["extra"][val] for k in FAMILIES},
                             baseline["validation"], rows["t_min"][val])
    reproduced = wpbench._apply_platt(raw, calibration)
    with np.load(experiment / LABEL / (KEY + "_predictions.npz"), allow_pickle=False) as saved:
        if not np.array_equal(saved["gid"], rows["gid"][val]) or not np.array_equal(saved["t"], rows["t"][val]):
            raise ValueError("Tested probability population changed")
        final_error = assert_match(reproduced, saved["candidate"], what="exact tested combined probabilities")
        baseline_error = assert_match(wpbench._apply_platt(baseline["validation"], calibration),
                                      saved["baseline"], what="exact tested calibrated baseline")
    support = {k: dict(feature_names=features[k]["names"],
                       supported_features=[n for n, known in zip(features[k]["names"],
                                                                features[k]["known"][fit].any(axis=0)) if known],
                       nonzero_fit_features=[n for n, active in zip(features[k]["names"],
                                                                   np.any(features[k]["extra"][fit] != 0, axis=0)) if active])
               for k in FAMILIES}
    sq_plan_path = dataset.parent / "sqearly_2026-10-04_final/plan.json"
    sq_source = json.loads(sq_plan_path.read_text())["source"]
    sq_table = Path(sq_tables or sq_source["tables_path"]).resolve()
    dependencies[str(sq_table)] = _bound(sq_table, sq_source["tables_sha256"])
    # Preserving the score formula is as important as pinning its table cells.
    dependencies[str(Path(sqpairs.__file__).resolve())] = _bound(
        sqpairs.__file__, sq_source["scorer_source_sha256"])
    for path, digest in dependencies.items():
        _bound(path, digest)
    output.mkdir(parents=True)
    base_path, joint_path, pinned_sq = output / "model_base.npz", output / "model_joint.npz", output / "pair_tables.npz"
    wpgam.save_model(base, str(base_path), meta=base["meta"])
    shutil.copyfile(joint_source, joint_path)
    shutil.copyfile(sq_table, pinned_sq)
    restored = wpgam.load_model(str(base_path))
    serialization_error = assert_match(wpgam.predict_state(restored["state"], live_core_rows, rows["t_min"][nonfit]),
                                       wpgam.predict_state(base["state"], live_core_rows, rows["t_min"][nonfit]),
                                       what="serialized base artifact")
    candidate = stage_report["candidates"][KEY]
    bundle = dict(kind=KIND, staged=True, created_at=datetime.now(timezone.utc).isoformat(),
        base=dict(path=str(base_path), sha256=sha(base_path), kind=wpgam.MODEL_KIND),
        joint=dict(path=str(joint_path), sha256=sha(joint_path), kind=wpcombined.KIND),
        calibration=dict(intercept=calibration["intercept"], slope=calibration["slope"],
                         probability_clip=[1e-5, 1. - 1e-5], order="raw_core_then_joint_then_platt"),
        families=list(FAMILIES), support=support,
        capture=dict(sq_table_path=str(pinned_sq), sq_table_sha256=sha(pinned_sq),
                     sq_evaluated_origin=sq_source["tables_path"], sq_recovered_from=str(sq_table),
                     composition_root=str(ROOT / "data/composition"), composition_patch_policy="gameplay_major_minor"),
        provenance=dict(experiment=str(experiment), stage=LABEL, refitted=False, dependencies=dependencies,
                        runtime_sources={str(p): sha(p) for p in (Path(__file__).resolve(),
                            Path(wpgam.__file__).resolve(), Path(wpcombined.__file__).resolve(), Path(wpbench.__file__).resolve())}),
        user_override=dict(explicit_request="Set those as the production weights.",
                           statistical_gate_passed=False, automatic_selection=experiment_report["selected"],
                           reason="User explicitly chose the reported exploratory development winner after the inconclusive gain was disclosed"),
        evaluation=dict(development_winner=KEY, august_metrics=candidate["metrics"],
                        august_baseline=stage_report["baseline"], paired=candidate["paired"],
                        shape_audit=candidate["shape_audit"], limitations=plan["limitations"]),
        reproducibility=dict(nonfit_states=int(nonfit.sum()), august_states=int(val.sum()),
            upstream_max_abs_error=upstream_error, raw_core_max_abs_error=raw_error,
            calibrated_baseline_max_abs_error=baseline_error, combined_max_abs_error=final_error,
            serialization_max_abs_error=serialization_error, tolerance=1e-11))
    path = output / "bundle.json"
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sq-tables", type=Path, help="Recover the exact evaluated SQ generation from a hash-matched local backup")
    args = parser.parse_args()
    print(stage(args.output, sq_tables=args.sq_tables))
