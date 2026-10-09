"""Fit and verify the frozen resource-study selection on consumed history.

No deployment or registration side effects. The latest 28 days before the
exclusive cutoff calibrate the exact earlier fitted stack; upstream learners
and the state model never see those calibration outcomes during fitting.
"""
import argparse
import ast
import copy
import datetime as dt
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpadapt, wpbench, wpcandidate as candidate, wpexposure, wpgam, wpresource, wpaudit
from research.wpx_adapt import load_extra

log = logging.getLogger("fit_candidate")
ALLOWED_FAMILIES = {"core", "pooled_role", "pooled_constant", "pooled_smooth"}
MAX_CUTOFF = "2026-09-03"


def upstream_source_fingerprint(source):
    """Only these three state/serialization functions are outside prior fitting."""
    tree = ast.parse(source)
    excluded = {"predict_state", "save_model", "load_model"}
    tree.body = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 or node.name not in excluded]
    return candidate.digest(ast.dump(tree, include_attributes=False))


def validate_prior_cache_identity(saved, current, output):
    if saved == current:
        return
    prior = output/"source_snapshot"/"wpgam_before_native_calibration.py"
    adjusted = copy.deepcopy(saved)
    adjusted["sources"]["wpgam.py"] = current["sources"]["wpgam.py"]
    if (adjusted != current or not prior.exists() or candidate.sha256(prior) != saved["sources"]["wpgam.py"]
            or upstream_source_fingerprint(prior.read_text()) != upstream_source_fingerprint(Path(wpgam.__file__).read_text())):
        raise ValueError("Prior cache input or upstream fitting source mismatch")
    # Preserve the original cache manifest. The explicit attestation permits
    # state-only native calibration support without refitting unchanged
    # nuisance learners or pretending they ran under the newer file hash.
    dump(output/"prior_source_transition.json", dict(
        original_wpgam_sha256=saved["sources"]["wpgam.py"], current_wpgam_sha256=current["sources"]["wpgam.py"],
        unchanged_upstream_ast_sha256=upstream_source_fingerprint(prior.read_text()),
        excluded_functions=["predict_state", "save_model", "load_model"],
        reason="these state prediction/serialization functions are not called while building chronological nuisance priors"))


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_inputs(dataset, study, cutoff):
    dataset, study = Path(dataset), Path(study)
    plan = json.loads((study / "plan.json").read_text())
    manifest = json.loads(Path(str(dataset) + ".manifest.json").read_text())
    sha = candidate.sha256(dataset)
    if not (plan["dataset_sha256"] == manifest["sha256"] == sha
            and manifest.get("zero_future_timestamp_joins") is True
            and manifest["input_contract"] == plan["input_contract"]):
        raise ValueError("Training data and frozen study causal provenance mismatch")
    if str(cutoff) > MAX_CUTOFF:
        raise ValueError("Final trainer accepts only previously consumed dates before 2026-09-03")
    with np.load(dataset, allow_pickle=False) as archive:
        contract = json.loads(str(archive["input_contract"].item()))
        if (contract != manifest["input_contract"] or
                str(archive["input_contract_sha256"].item()) != manifest["input_contract_sha256"]):
            raise ValueError("Dataset input contract differs from audited manifest")
    base = wpbench._load_base(str(dataset))
    dates = base["game_dates"]
    if base["split_method"] != "date" or np.any(dates >= cutoff):
        raise ValueError("Dataset contains dates outside the consumed training cutoff")
    inventory = wpexposure.migrate(base["game_gid"], dates, write=False)
    if wpexposure.fresh_mask(inventory, base["game_gid"], dates).any():
        raise ValueError("Final training cannot consume unexposed outcomes")
    if max(dates) > plan["consumed_through"]:
        raise ValueError("Dataset extends past the study's frozen exposure boundary")
    calibration_start = str(np.datetime64(cutoff)-np.timedelta64(28, "D"))
    fit = dates < calibration_start
    calibration = (dates >= calibration_start) & (dates < cutoff)
    if not fit.any() or not calibration.any() or max(dates[fit]) >= min(dates[calibration]):
        raise ValueError("Cannot form the required earlier fit and latest 28-day calibration blocks")
    return base, plan, manifest, fit, calibration, calibration_start


def prepare_priors(base, fit, output, cutoff, manifest):
    """Reusable, hash-verified nuisance fit independent of selected state family."""
    output.mkdir(parents=True, exist_ok=True)
    raw_path, nuisance_path, identity_path = [output/name for name in
                                             ("stacked_raw.npy", "nuisance.npz", "prior_cache.json")]
    identity = {"dataset_sha256": manifest["sha256"], "input_contract_sha256": manifest["input_contract_sha256"],
                "cutoff_exclusive": cutoff, "fit_game_ids_sha256": candidate.digest(base["game_gid"][fit].tolist()),
                "sources": {Path(module.__file__).name: candidate.sha256(module.__file__) for module in (wpgam, wpbench)}}
    if any(path.exists() for path in (raw_path, nuisance_path, identity_path)):
        if not all(path.exists() for path in (raw_path, nuisance_path, identity_path)):
            raise ValueError("Incomplete prior cache; finish its live preparation process before resuming")
        saved = json.loads(identity_path.read_text())
        validate_prior_cache_identity(saved["identity"], identity, output)
        if (saved["raw_sha256"] != candidate.sha256(raw_path) or saved["nuisance_sha256"] != candidate.sha256(nuisance_path)):
            raise ValueError("Prior cache input, code or artifact hash mismatch")
    else:
        log.info("Fit chronological nuisance stack on %d games through %s", int(fit.sum()), max(base["game_dates"][fit]))
        stack = wpbench._stacked_arrays(base, fit)
        np.save(raw_path, stack["raw"])
        np.savez_compressed(nuisance_path, champ_state_beta=stack["champ_state_beta"],
                            **{"pre_"+key: value for key, value in stack["pregame"].items()})
        dump(identity_path, dict(identity=identity, raw_sha256=candidate.sha256(raw_path),
                                  nuisance_sha256=candidate.sha256(nuisance_path)))
    raw = np.load(raw_path, mmap_mode="r")
    if raw.shape != (len(base["y"]), len(wpresource.INPUT_NAMES)) or not np.isfinite(raw).all():
        raise ValueError("Invalid stacked input cache")
    with np.load(nuisance_path, allow_pickle=False) as archive:
        nuisance = {"pregame": {key[4:]: archive[key] for key in archive.files if key.startswith("pre_")},
                    "champ_state": {"beta": archive["champ_state_beta"], "cap_min": wpgam.CHAMP_STATE_CAP_MIN,
                                    "l2": wpgam.CHAMP_STATE_L2}}
        nuisance["pregame"]["intercept"] = float(nuisance["pregame"]["intercept"])
    return raw, nuisance


def selected_settings(study, plan):
    selection = json.loads((study/"selection.json").read_text())
    selected = selection["selected"]
    family = selected["family"]
    if family not in ALLOWED_FAMILIES:
        raise ValueError("Selected family is not eligible for prospective export")
    core = selection["choices"]["core"]
    for choice in (selected, core):
        if choice["calibration"] not in {"none", "temperature", "platt"}:
            raise ValueError("Unsupported frozen calibration selection")
        index = int(choice["index"])
        if index < 0 or index >= len(plan["specs"][choice["family"]]):
            raise ValueError("Frozen selection index is outside the study plan")
    return selection, dict(plan["specs"][family][selected["index"]])


def core_model(nuisance, raw, base, fit_rows, calibration_rows, calibration_method):
    state = wpgam.fit_state_model(raw[fit_rows], base["y"][fit_rows], base["gid"][fit_rows], base["t_min"][fit_rows])
    raw_probability = wpgam.predict_state(state, raw[calibration_rows], base["t_min"][calibration_rows])
    calibration = wpadapt.calibrate(raw_probability, base["y"][calibration_rows], base["gid"][calibration_rows], calibration_method)
    # Native state inference reproduces the study's exact probability clip
    # before applying these held-out coefficients. Earlier artifacts omit
    # this optional field and preserve their previous inference semantics.
    state.update(cal_intercept=float(calibration["intercept"]), cal_slope=float(calibration["slope"]),
                 calibration_probability_clip=[1e-5, 1-1e-5],
                 calibration_method="native_frozen_28day_"+calibration_method)
    model = dict(copy.deepcopy(nuisance), kind=wpgam.MODEL_KIND, state=state)
    return model, calibration


def live_state(row, names, gold, t):
    v = dict(zip(map(str, names), map(float, row)))
    state = {key: v.get(key, 0.) for key in wpgam.PREGAME_FEATURES}
    state.update(t_min=float(t), clock_s=float(t)*60, gold_players=gold.tolist(),
                 gold_blue=float(sum(gold[:5])), gold_red=float(sum(gold[5:])),
                 gold_diff_k=v["gold_k"], gold_diff_prev_k=v["gold_k"]-v["gold_mom"],
                 gold_role=[v["gold_"+role] for role in wpgam.GOLD_ROLES],
                 cs_diff_k=v["cs_k"], kills=v["d_kill"], towers=v["d_tower"],
                 dragons=v["d_dragon"], barons=v["d_baron"], inhibs=v["d_inhib"],
                 elders=v["d_elder"], elder_active=v["elder_buff"],
                 t_since_kill_min=v["t_since_kill"])
    for key in ("baron_active", "drag_blue", "drag_red", "towers_blue", "towers_red", "inhib_blue", "inhib_red",
                "dead_blue", "dead_red", "hp_pool", "lvl_k", "has_hp"):
        state[key] = v[key]
    return state


def verify(model, loaded, raw, base, extra, calibration_rows, sample=2000):
    rng = np.random.default_rng(20260912)
    eligible = np.flatnonzero(calibration_rows)
    ix = rng.choice(eligible, min(sample, len(eligible)), replace=False)
    r, gold = np.asarray(raw[ix]), extra["gold"][ix]
    C, t = base["game_C"][base["row_game"]][ix], base["t_min"][ix]
    before = candidate.predict_candidate_arrays(model, r, gold, C, t)
    after = candidate.predict_candidate_arrays(loaded, r, gold, C, t)
    difference = float(np.max(np.abs(before-after)))
    if difference > 1e-12 or not np.isfinite(after).all():
        raise ValueError("Candidate save/load prediction mismatch")
    violations, comparisons, worst = 0, 0, 0.
    for slot in range(10):
        for amount in (.1, 1., 30.):
            changed = wpaudit.gold_perturbation(r, wpresource.INPUT_NAMES, gold.sum(axis=1)/1000, slot, amount)
            changed_gold = gold.copy(); changed_gold[:, slot] += amount*1000
            p = candidate.predict_candidate_arrays(loaded, changed, changed_gold, C, t)
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError("Invalid physical-gold audit probability")
            movement = (1 if slot < 5 else -1)*(p-after)
            violations += int(np.sum(movement < -1e-10)); comparisons += len(p)
            worst = min(worst, float(movement.min()))
    if violations:
        raise ValueError("Final candidate fails physical-gold monotonicity")
    original = base["raw_X"][base["raw_fixed"]]
    max_live_delta, max_features_delta = 0., 0.
    for j in range(min(100, len(ix))):
        index = ix[j]
        state = live_state(original[index], base["names"], gold[j], t[j])
        blue, red = ([str(base["champ_names"][c]) if c >= 0 else "" for c in C[j, :5]],
                     [str(base["champ_names"][c]) if c >= 0 else "" for c in C[j, 5:]])
        predicted = candidate.predict_candidate_model(loaded, state, blue, red)["p_blue"]
        max_live_delta = max(max_live_delta, abs(predicted-after[j]))
        max_features_delta = max(max_features_delta,
            float(np.max(np.abs(wpgam.state_values_from_live(state)-r[j, len(wpgam.PRIOR_INPUTS):]))))
        if loaded.get("kind") == candidate.BUNDLE_KIND:
            missing = dict(state, gold_players=None)
            fallback = candidate.predict_candidate_model(loaded, missing, blue, red)
            reference = candidate.predict_candidate_model(loaded["core"], missing, blue, red)
            if not fallback["fallback"] or fallback["p_blue"] != reference["p_blue"]:
                raise ValueError("Pooled missing-gold fallback differs from its frozen core")
    if max_live_delta > 2e-6 or max_features_delta > 5e-5:
        raise ValueError("Historical/live adapter parity failure: probability=%g features=%g" % (max_live_delta, max_features_delta))
    return {"passed": True, "sampled_states": len(ix), "roundtrip_max_abs_delta": difference,
            "live_max_abs_probability_delta": max_live_delta, "live_max_abs_feature_delta": max_features_delta,
            "physical_gold": {"comparisons": comparisons, "violations": violations, "worst_reversal": worst,
                              "amounts": [100, 1000, 30000], "all_ten_slots": True},
            "missing_gold_fallback": "frozen core verified" if loaded.get("kind") == candidate.BUNDLE_KIND else "core model"}


def main(dataset, study, resources, output, cutoff=MAX_CUTOFF, prepare_only=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    dataset, study, resources = Path(dataset), Path(study), Path(resources)
    base, study_plan, manifest, fit, calibration_games, calibration_start = validate_inputs(dataset, study, cutoff)
    raw, nuisance = prepare_priors(base, fit, output, cutoff, manifest)
    if prepare_only:
        log.info("Prior preparation complete; frozen selection can be exported when ready")
        return
    selection, spec = selected_settings(study, study_plan)
    selected = selection["selected"]
    target = output/"candidate.npz"
    if target.exists():
        raise ValueError("Candidate artifact already exists; final fitting never overwrites an artifact")
    extra = load_extra(base, dataset, resources)
    if (not extra["present"].all() or candidate.sha256(resources/"resources.npz") != study_plan["resources_sha256"]):
        raise ValueError("Resource rows differ from the frozen study")
    fit_rows, calibration_rows = fit[base["row_game"]], calibration_games[base["row_game"]]
    protected = {path: candidate.sha256(path) for path in study_plan["protected"]}
    final_plan = dict(kind="selected_candidate_final_fit_v1", dataset_sha256=manifest["sha256"],
                     study_plan_sha256=candidate.sha256(study/"plan.json"),
                     selection_sha256=candidate.sha256(study/"selection.json"), selected=selected, spec=spec,
                     cutoff_exclusive=cutoff, calibration_start=calibration_start,
                     sources={str(Path(module.__file__).resolve()): candidate.sha256(module.__file__)
                              for module in (wpgam, wpbench, wpresource, wpadapt, candidate)},
                     trainer_source_sha256=candidate.sha256(__file__), protected=protected,
                     prior_cache_sha256=candidate.sha256(output/"prior_cache.json"),
                     prior_source_transition=(json.loads((output/"prior_source_transition.json").read_text())
                                              if (output/"prior_source_transition.json").exists() else None))
    dump(output/"final_plan.json", final_plan)
    log.info("Fit core/fallback on %d games, calibrate %d later games using %s", int(fit.sum()),
             int(calibration_games.sum()), selection["choices"]["core"]["calibration"])
    core, core_calibration = core_model(nuisance, raw, base, fit_rows, calibration_rows, selection["choices"]["core"]["calibration"])
    meta = dict(training_cutoff=cutoff+"T00:00:00+00:00", cutoff_exclusive=True,
                input_contract=manifest["input_contract"], dataset_sha256=manifest["sha256"],
                final_plan_sha256=candidate.digest(final_plan), selection=selected, selected_spec=spec,
                fit_games=int(fit.sum()), calibration_games=int(calibration_games.sum()),
                fit_states=int(fit_rows.sum()), calibration_states=int(calibration_rows.sum()),
                fit_max_date=str(max(base["game_dates"][fit])), calibration_start=calibration_start,
                calibration_max_date=str(max(base["game_dates"][calibration_games])),
                native_calibration=dict(core_calibration, probability_clip=[1e-5, 1-1e-5]),
                historical_evidence="consumed development only; no prospective accuracy claim",
                nuisance_fit="five chronological out-of-fold pregame/champion-state priors; full nuisance fit ends before calibration",
                fitted_at=dt.datetime.now(dt.timezone.utc).isoformat())
    core["meta"] = dict(meta, selection=selection["choices"]["core"], selected_spec={})
    if selected["family"] == "core":
        model = core
        wpgam.save_model(core, str(target), core["meta"])
    else:
        log.info("Fit selected %s with %s", selected["family"], spec)
        C = base["game_C"][base["row_game"]]
        fitted = wpresource.fit(raw[fit_rows], extra["gold"][fit_rows], C[fit_rows], base["y"][fit_rows],
                                base["gid"][fit_rows], base["t_min"][fit_rows], spec)
        p = wpresource.predict(fitted, raw[calibration_rows], extra["gold"][calibration_rows], C[calibration_rows],
                                base["t_min"][calibration_rows], calibrated=False)
        meta["live_calibration"] = wpadapt.calibrate(p, base["y"][calibration_rows], base["gid"][calibration_rows], selected["calibration"])
        meta["production_adapter_required"] = "pooled_bundle_v1"
        meta["staged_only"] = True
        model = dict(kind=candidate.BUNDLE_KIND, core=core, resource=fitted, meta=meta)
        candidate.save_bundle(core, fitted, meta, str(target))
    loaded = candidate.load_candidate_model(str(target))
    report = verify(model, loaded, raw, base, extra, calibration_rows)
    for path, sha in protected.items():
        if candidate.sha256(path) != sha:
            raise ValueError("Protected production artifact changed during final fit: "+path)
    for path, sha in final_plan["sources"].items():
        if candidate.sha256(path) != sha:
            raise ValueError("Final-fit source changed during fitting: "+path)
    report.update(artifact=str(target.resolve()), artifact_sha256=candidate.sha256(target),
                  selected=selected, input_contract=candidate.input_contract(loaded),
                  training_cutoff=loaded["meta"]["training_cutoff"],
                  final_plan_sha256=candidate.digest(final_plan), production_changed=False)
    dump(output/"verification.json", report)
    os.chmod(target, 0o444)
    log.info("Verified candidate %s; artifact remains staged, unregistered and undeployed", target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", default=MAX_CUTOFF)
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main(arguments.dataset, arguments.study, arguments.resources, arguments.output,
         arguments.cutoff, arguments.prepare_only)
