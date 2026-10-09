"""Corrected-input reference replay and a narrow pooled-resource GAM search.

Only already-exposed historical dates are allowed. Models, selection, source
snapshots and predictions are isolated in a new output directory. This runner
does not deploy or register candidates; its chosen artifact must pass the
separate prospective capture protocol.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpadapt as wa, wpbench, wpgam, wpresource
from research import wpx_methods_v9 as methods
from research.wpx_adapt import load_extra


log = logging.getLogger("pooled_resource")
METHODS = ("none", "temperature", "platt")
SPECS = {
    "core": [{}], "v8_reference": [{}],
    "rich_control": [dict(features="rich", context_l2=800., l2=24., smooth=70.),
                     dict(features="rich", context_l2=2000., l2=24., smooth=70.)],
    "pooled_role": [dict(mode="role")],
    "pooled_constant": [dict(mode="constant", shrink=v) for v in (800., 2000.)],
    "pooled_smooth": [dict(mode="smooth", shrink=v) for v in (800., 2000.)],
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    methods.dump(Path(path), value)


def past_reference(path):
    spec = importlib.util.spec_from_file_location("lol_ticker._resource_v8_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.MODEL_KIND != wpgam.LEGACY_MODEL_KIND:
        raise ValueError("Expected recorded v8 reference source")
    return module


def prepare(base, extra, block, directory, old):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "core.npy"
    prior_path = directory / "priors.pkl"
    cache_path = directory / "stack_cache.json"
    identity = dict(plan_sha256=base["cache_plan_sha256"],
                    fit_gids_sha256=hashlib.sha256(base["game_gid"][block["fit"]].tobytes()).hexdigest())
    if any(p.exists() for p in (path, prior_path, cache_path)):
        if not all(p.exists() for p in (path, prior_path, cache_path)):
            raise ValueError("Incomplete prior cache; remove that incomplete stage cache before resuming")
        cached = json.loads(cache_path.read_text())
        if (cached["identity"] != identity or cached["core_sha256"] != sha(path)
                or cached["prior_sha256"] != sha(prior_path)):
            raise ValueError("Prior cache provenance/content mismatch")
    else:
        log.info("Rebuild complete past-only prior stack before %s", block["fit_end"])
        stack = wpbench._stacked_arrays(base, block["fit"])
        np.save(path, stack.pop("raw"))
        prior_path.write_bytes(pickle.dumps(stack, protocol=4))
        dump(cache_path, dict(identity=identity, core_sha256=sha(path), prior_sha256=sha(prior_path)))
    core = np.load(path, mmap_mode="r")
    if core.shape != (len(base["y"]), len(wa.CORE_NAMES)) or not np.isfinite(core).all():
        raise ValueError("Prior cache shape/value mismatch")
    original = base["raw_X"][base["raw_fixed"]]
    legacy = old.state_values_from_matrix(original, base["names"])
    result = {}
    for label in ("fit", "calibration", "validation"):
        mask = block[label][base["row_game"]]
        result[label] = dict(raw=np.asarray(core[mask]),
            old_raw=np.c_[core[mask, :len(wpgam.PRIOR_INPUTS)], legacy[mask]],
            rich_raw=np.c_[core[mask], extra["extra"][mask]],
            gold=extra["gold"][mask], C=base["game_C"][base["row_game"]][mask],
            cats=extra["cats"][base["row_game"]][mask],
            t=base["t_min"][mask], y=base["y"][mask], gid=base["gid"][mask], mask=mask)
    return result


def fit(family, spec, part, old):
    if family == "core":
        fitted = wpgam.fit_state_model(part["raw"], part["y"], part["gid"], part["t"])
    elif family == "v8_reference":
        fitted = old.fit_state_model(part["old_raw"], part["y"], part["gid"], part["t"])
    elif family == "rich_control":
        fitted = wa.fit_gam(part["rich_raw"], part["cats"], part["y"], part["gid"], part["t"],
                            wpgam._game_balanced_weights(part["gid"]), spec)
    else:
        fitted = wpresource.fit(part["raw"], part["gold"], part["C"], part["y"],
                                part["gid"], part["t"], spec)
    return dict(family=family, fitted=fitted)


def predict(model, part, old):
    family, fitted = model["family"], model["fitted"]
    if family == "core":
        return wpgam.predict_state(fitted, part["raw"], part["t"])
    if family == "v8_reference":
        return old.predict_state(fitted, part["old_raw"], part["t"])
    if family == "rich_control":
        return wa.predict(fitted, part["rich_raw"], part["cats"], part["t"])
    return wpresource.predict(fitted, part["raw"], part["gold"], part["C"], part["t"], calibrated=False)


def gold_audit(model, calibration, part, old, sample=2000):
    rng = np.random.default_rng(904)
    ix = rng.choice(len(part["y"]), min(sample, len(part["y"])), replace=False)
    small = {k: v[ix].copy() for k, v in part.items() if k != "mask"}
    baseline = wpbench._apply_platt(predict(model, small, old), calibration)
    if not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 1)):
        raise ValueError("Invalid baseline prediction in gold audit")
    comparisons = violations = 0
    worst = 0.
    names = {n: i for i, n in enumerate(wa.CORE_NAMES)}
    total = small["gold"].sum(axis=1) / 1000.
    # Physical changes move all engineered views coherently, including the
    # legacy allocation representation and rich signed individual gold.
    for slot in range(10):
        sign, role = (1. if slot < 5 else -1.), slot % 5
        for amount in (100., 1000.):
            changed = {k: v.copy() for k, v in small.items()}
            changed["gold"][:, slot] += amount
            delta = sign * amount / 1000.
            raw = changed["raw"]
            raw[:, names["gold_k"]] += delta
            raw[:, names["gold_mom"]] += delta
            raw[:, names["gold_" + wpgam.GOLD_ROLES[role]]] += delta
            raw[:, names["gold_rel"]] = raw[:, names["gold_k"]] / (total + amount / 1000.)
            changed["rich_raw"][:, :len(names)] = raw
            changed["rich_raw"][:, len(names) + slot] += sign * amount / 10000.
            oldnames = {n: i for i, n in enumerate(wpgam.PRIOR_INPUTS + list(old.STATE_FEATURES))}
            for n in ("gold_k", "gold_mom", "gold_rel"):
                changed["old_raw"][:, oldnames[n]] = raw[:, names[n]]
            for r in range(4):
                changed["old_raw"][:, oldnames["gold_" + wpgam.GOLD_ROLES[r] + "_alloc"]] += delta * ((r == role) - .2)
            after = wpbench._apply_platt(predict(model, changed, old), calibration)
            if not np.isfinite(after).all() or np.any((after < 0) | (after > 1)):
                raise ValueError("Invalid perturbed prediction in gold audit")
            movement = sign * (after - baseline)
            comparisons += len(movement)
            violations += int(np.sum(movement < -1e-10))
            worst = min(worst, float(movement.min()))
    return dict(comparisons=comparisons, violations=violations, worst_reversal=worst,
                sampled_states=len(ix), seed=904, finite=True)


def stage(base, extra, block, directory, old, choices=None):
    parts = prepare(base, extra, block, directory, old)
    record_path = directory / "results.json"
    records = json.loads(record_path.read_text()) if record_path.exists() else {}
    predictions, fitted = {}, {}
    for family, specs in SPECS.items():
        indices = [choices[family]["index"]] if choices else range(len(specs))
        for i in indices:
            key = family + "_" + str(i)
            model_path, pred_path = directory / (key + ".pkl"), directory / (key + "_raw.npz")
            if key in records and model_path.exists() and pred_path.exists():
                if sha(model_path) != records[key]["model_sha256"] or sha(pred_path) != records[key]["prediction_sha256"]:
                    raise ValueError("Cached artifact hash mismatch")
                model = pickle.loads(model_path.read_bytes())
                with np.load(pred_path) as p:
                    cp, vp = p["calibration"], p["validation"]
            else:
                tick = time.time()
                model = fit(family, specs[i], parts["fit"], old)
                cp, vp = [predict(model, parts[label], old) for label in ("calibration", "validation")]
                model_path.write_bytes(pickle.dumps(model, protocol=4))
                np.savez_compressed(pred_path, calibration=cp, validation=vp)
                records[key] = dict(family=family, index=i, spec=specs[i], methods={},
                    fit_seconds=time.time() - tick, model_sha256=sha(model_path), prediction_sha256=sha(pred_path))
            methods_to_try = [choices[family]["calibration"]] if choices else METHODS
            for method in methods_to_try:
                calibration = wa.calibrate(cp, parts["calibration"]["y"], parts["calibration"]["gid"], method)
                p = wpbench._apply_platt(vp, calibration)
                records[key]["methods"][method] = dict(calibration=calibration,
                    metrics=methods.metrics(p, parts["validation"]["y"], parts["validation"]["gid"]))
                predictions[family if choices else key + "_" + method] = p
                if choices:
                    fitted[family] = (model, calibration)
                    records[key]["gold_audit"] = gold_audit(model, calibration, parts["validation"], old)
            dump(record_path, records)
            log.info("%s %s Brier %s", directory.name, key,
                     {m: round(records[key]["methods"][m]["metrics"]["brier_game"], 6) for m in methods_to_try})
            del cp, vp
            gc.collect()
    return records, predictions, fitted, parts


def summarize(predictions, y, gid, series):
    return {label: dict(metrics=methods.metrics(p, y, gid),
                       paired=methods.paired(p, predictions["core"], y, gid, series),
                       calibration=wpbench._calibration_diagnostics(p, y, gid))
            for label, p in predictions.items()}


def robust_selection(choices, records, blocks, base, series, output):
    """Development-only selection with equal block weight and series resampling."""
    checks = {}
    for family in ("pooled_role", "pooled_constant", "pooled_smooth"):
        deltas, bootstraps = [], []
        for i, block in enumerate(blocks):
            mask = block["validation"][base["row_game"]]
            probabilities = {}
            for label in ("core", family):
                choice = choices[label]
                key = label + "_" + str(choice["index"])
                with np.load(output / ("development_" + str(i + 1)) / (key + "_raw.npz")) as saved:
                    raw = saved["validation"]
                cal = records[i][key]["methods"][choice["calibration"]]["calibration"]
                probabilities[label] = wpbench._apply_platt(raw, cal)
            gids, losses = methods.per_game((probabilities[family] - base["y"][mask]) ** 2
                                            - (probabilities["core"] - base["y"][mask]) ** 2,
                                            base["gid"][mask])
            _, inverse = np.unique([series[int(g)] for g in gids], return_inverse=True)
            sums, counts = np.bincount(inverse, weights=losses), np.bincount(inverse)
            draws = np.random.default_rng(20260912 + i).integers(len(sums), size=(5000, len(sums)))
            bootstraps.append(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
            deltas.append(float(losses.mean()))
        interval = np.quantile(np.mean(bootstraps, axis=0), [.025, .975]).tolist()
        without = [float(np.mean([d for j, d in enumerate(deltas) if j != i])) for i in range(len(deltas))]
        checks[family] = dict(block_delta= deltas, mean_delta=float(np.mean(deltas)),
                             leave_one_block_delta=without, ci95=interval,
                             eligible=bool(interval[1] < 0 and max(without) < 0))
    eligible = [choices["core"]] + [choices[f] for f, check in checks.items() if check["eligible"]]
    selected = min(eligible, key=lambda c: (c["mean_brier"], c["family"] != "core"))
    return selected, checks


def main(dataset, output, resources, old_source):
    output.mkdir(parents=True, exist_ok=True); resources.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset.with_suffix(dataset.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("sha256") != sha(dataset) or manifest.get("zero_future_timestamp_joins") is not True:
        raise ValueError("Corrected dataset requires a matching successful causal audit manifest")
    with np.load(dataset) as archive:
        if (str(archive["input_contract_sha256"].item()) != manifest["input_contract_sha256"]
                or json.loads(str(archive["input_contract"].item())) != manifest["input_contract"]):
            raise ValueError("Dataset input contract disagrees with audited manifest")
    old = past_reference(old_source)
    base = wpbench._load_base(str(dataset))
    registry = Path(wpgam.OUT_DIR) / "evaluation_registry.json"
    consumed = json.loads(registry.read_text())["consumed_through"]
    if base["split_method"] != "date" or max(base["game_dates"]) > consumed:
        raise ValueError("Unexposed dates are forbidden in this diagnostic")
    extra = load_extra(base, dataset, resources)
    if not extra["present"].all():
        raise ValueError("Resource comparison requires complete same-row player gold")
    blocks, final = wa.split_blocks(base["game_dates"], base["outer_train"], folds=3)
    source_paths = [Path(m.__file__) for m in (wpgam, wpbench, wa, wpresource, methods)]
    source_paths += [Path(sys.modules[load_extra.__module__].__file__), Path(__file__), old_source]
    protected = [Path(wpgam.MODEL_PATH), Path(wpgam.OUT_DIR) / "live_stack.json", registry]
    plan = dict(kind="corrected_pooled_resource_experiment_v1", specs=SPECS,
        dataset_sha256=sha(dataset), resources_sha256=sha(resources / "resources.npz"),
        input_manifest_sha256=sha(manifest_path), input_contract=manifest["input_contract"],
        sources={str(p): sha(p) for p in source_paths}, protected={str(p): sha(p) for p in protected},
        consumed_through=consumed, calibration_methods=METHODS,
        development=[{k: v for k, v in b.items() if isinstance(v, str)} for b in blocks],
        final={k: v for k, v in final.items() if isinstance(v, str)},
        v8_reference="recorded v8 state architecture with shared corrected, chronological upstream priors and identical calibration blocks",
        selection="mean development game-balanced Brier; pooled candidate must improve every leave-one-development-block mean and have an equal-block-weight series bootstrap upper bound below zero; otherwise core; rich/v8 diagnostic controls excluded",
        uncertainty="paired date/match clusters; exploratory consumed history",
        gold_audit="every frozen and monthly artifact; 2000 states x 10 slots x 2 amounts")
    plan = json.loads(json.dumps(plan))
    if (output / "plan.json").exists() and json.loads((output / "plan.json").read_text()) != plan:
        raise ValueError("Plan/source/data changed; use a new output directory")
    dump(output / "plan.json", plan)
    base["cache_plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    for i, p in enumerate(source_paths):
        destination = output / "source_snapshot" / (str(i) + "_" + p.name)
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(p.read_bytes())
    records = []
    for i, block in enumerate(blocks):
        record, _, _, _ = stage(base, extra, block, output / ("development_" + str(i + 1)), old)
        records.append(record)
    choices, ranking = methods.select(records, SPECS)
    selected, robustness = robust_selection(choices, records, blocks, base, extra["series"], output)
    selection = dict(choices=choices, ranking=ranking, selected=selected, robustness=robustness)
    dump(output / "selection.json", selection)
    log.info("Frozen development selection: %s", selected)
    record, predictions, fitted, parts = stage(base, extra, final, output / "frozen", old, choices)
    val = parts["validation"]
    frozen = summarize(predictions, val["y"], val["gid"], extra["series"])
    for name, result in frozen.items():
        result["gold_audit"] = record[name + "_" + str(choices[name]["index"])]["gold_audit"]
    dump(output / "frozen.json", frozen)
    np.savez_compressed(output / "frozen_predictions.npz", gid=val["gid"], y=val["y"], **predictions)
    mask = val["mask"]
    del fitted, parts; gc.collect()
    replay = {name: np.full(len(base["y"]), np.nan) for name in SPECS}
    months = []
    dates = base["game_dates"]
    for month in sorted({d[:7] for d in dates[~base["outer_train"]]}):
        start, end = month + "-01", str(np.datetime64(month, "M") + 1) + "-01"
        cutoff = str(np.datetime64(start) - np.timedelta64(28, "D"))
        block = dict(fit=dates < cutoff, calibration=(dates >= cutoff) & (dates < start),
            validation=(dates >= start) & (dates < end) & ~base["outer_train"],
            fit_end=cutoff, validation_start=start, validation_end=end)
        mm = block["validation"][base["row_game"]]
        if start == final["validation_start"]:
            with np.load(output / "frozen_predictions.npz") as saved:
                pred = {f: saved[f][mm[mask]] for f in SPECS}
            audits = {f: frozen[f]["gold_audit"] for f in SPECS}
        else:
            rr, pred, _, _ = stage(base, extra, block, output / ("replay_" + month), old, choices)
            audits = {f: rr[f + "_" + str(choices[f]["index"])]["gold_audit"] for f in SPECS}
        for name, p in pred.items():
            replay[name][mm] = p
        months.append(dict(month=month, summary=summarize(pred, base["y"][mm], base["gid"][mm], extra["series"]), gold_audits=audits))
        dump(output / "months.json", months)
        gc.collect()
    replay = {k: v[mask] for k, v in replay.items()}
    if not all(np.isfinite(v).all() for v in replay.values()):
        raise ValueError("Incomplete replay")
    summary = summarize(replay, base["y"][mask], base["gid"][mask], extra["series"])
    dump(output / "rolling.json", summary)
    np.savez_compressed(output / "rolling_predictions.npz", gid=base["gid"][mask], y=base["y"][mask], **replay)
    for path, digest in plan["protected"].items():
        if sha(path) != digest:
            raise ValueError("Protected production artifact changed: " + path)
    dump(output / "completion.json", dict(completed=True, selected=selected, production_changed=False,
                                           candidate_artifact_pending=True))
    log.info("Completed historical experiment; selected candidate awaits full fit and prospective registration")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--v8-source", type=Path, default=Path("data/wpx/repairs_v9/incumbent_source.py"))
    args = parser.parse_args()
    main(args.dataset, args.output, args.resources, args.v8_source)
