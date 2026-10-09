"""Retest the repaired live contract with/without 180-day state-weight decay.

Three earlier development blocks select calibration and the candidate; a
frozen later diagnostic and monthly replay never reselect settings. Only
already-consumed dates are read. All outputs are isolated from deployment.
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpaudit, wpadapt as wa, wpbench, wpgam
from research.wpx_adapt import load_extra

log = logging.getLogger("recency")
SPECS = {"repaired": dict(features="core", l2=24., smooth=70., half_life=None),
         "recency180": dict(features="core", l2=24., smooth=70., half_life=180.)}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def state_model(model, calibration):
    mean, std, lo, hi = model["scale"]
    return dict(theta=model["theta"], feature_names=np.array(wa.CORE_NAMES),
                mean=mean, std=std, lo=lo, hi=hi, knots=wpgam.TIME_KNOTS,
                cal_intercept=calibration["intercept"], cal_slope=calibration["slope"],
                optimizer_success=model["converged"], optimizer_message="converged",
                calibration_method="strict_earlier_28_days", l2=24., smooth=70.)


def stage(base, block, choices=None):
    log.info("Fit before %s; validate %s..%s", block["fit_end"],
             block["validation_start"], block["validation_end"])
    stack = wpbench._stacked_arrays(base, block["fit"])
    raw = stack["raw"]
    masks = {k: block[k][base["row_game"]] for k in ("fit", "calibration", "validation")}
    fit, cal, val = (masks[k] for k in ("fit", "calibration", "validation"))
    cats = np.zeros((len(raw), len(wa.CAT_NAMES)), dtype=int)
    results, predictions, models = {}, {}, {}
    for label, spec in SPECS.items():
        _, weights = wa.temporal_weights(base["gid"][fit],
            base["game_dates"][base["row_game"]][fit], block["fit_end"],
            half_life=spec["half_life"])
        model = wa.fit_gam(raw[fit], cats[fit], base["y"][fit], base["gid"][fit],
                           base["t_min"][fit], weights, spec)
        cp = wa.predict(model, raw[cal], cats[cal], base["t_min"][cal])
        vp = wa.predict(model, raw[val], cats[val], base["t_min"][val])
        methods = (choices[label],) if choices else ("none", "temperature", "platt")
        results[label], predictions[label] = {}, {}
        for method in methods:
            calibration = wa.calibrate(cp, base["y"][cal], base["gid"][cal], method)
            p = wpbench._apply_platt(vp, calibration)
            if not np.isfinite(p).all():
                raise ValueError("non-finite validation probabilities")
            results[label][method] = dict(
                metrics=wpbench._basic_metrics(p, base["y"][val], base["gid"][val]),
                calibration=calibration)
            predictions[label][method] = p
            if choices:
                models[label] = dict(kind=wpgam.MODEL_KIND, pregame=stack["pregame"],
                    champ_state=dict(beta=stack["champ_state_beta"], cap_min=15., l2=800.),
                    state=state_model(model, calibration))
        log.info("%s %s", label, {k: round(v["metrics"]["brier_game"], 6)
                                  for k, v in results[label].items()})
    return results, predictions, models, raw[val], val


def summarize(predictions, reference, base, mask, series):
    y, gids = base["y"][mask], base["gid"][mask]
    return {label: dict(metrics=wpbench._basic_metrics(p, y, gids),
        paired=wa.series_interval(p, reference, y, gids, series))
        for label, p in predictions.items()}


def main(dataset, output, resource_cache, incumbent_source=None):
    output.mkdir(parents=True, exist_ok=True)
    base = wpbench._load_base(str(dataset))
    registry = json.loads((Path(wpgam.OUT_DIR) / "evaluation_registry.json").read_text())
    if base["split_method"] != "date" or max(base["game_dates"]) > registry["consumed_through"]:
        raise ValueError("Fresh outcomes are reserved for prospective promotion")
    blocks, final = wa.split_blocks(base["game_dates"], base["outer_train"], folds=3)
    extra = load_extra(base, dataset, resource_cache)
    code = b"".join(Path(m.__file__).read_bytes() for m in (wa, wpgam, wpbench, wpaudit))
    plan = dict(dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(code + Path(__file__).read_bytes()).hexdigest(),
        specs=SPECS, calibration_days=28, seed=904, consumed_through=registry["consumed_through"],
        development=[{k:v for k,v in b.items() if isinstance(v,str)} for b in blocks],
        final={k:v for k,v in final.items() if isinstance(v,str)},
        selection="mean development game Brier; settings frozen before later diagnostics",
        production_changed=False)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Changed plan; use a new output directory")
    dump(plan_path, plan)
    records = []
    for i, block in enumerate(blocks):
        record, _, _, _, _ = stage(base, block)
        records.append(record)
        dump(output / f"development_{i+1}.json", record)
    ranking = []
    for label in SPECS:
        for method in ("none", "temperature", "platt"):
            mean = np.mean([r[label][method]["metrics"]["brier_game"] for r in records])
            ranking.append(dict(label=label, calibration=method, mean_brier=float(mean)))
    ranking.sort(key=lambda r: (r["mean_brier"], r["label"] != "repaired"))
    choices = {label: next(r["calibration"] for r in ranking if r["label"] == label)
               for label in SPECS}
    selection = dict(selected=ranking[0], choices=choices, ranking=ranking)
    dump(output / "selection.json", selection)
    result, pp, models, raw, mask = stage(base, final, choices)
    predictions = {k:v[choices[k]] for k,v in pp.items()}
    frozen = summarize(predictions, predictions["repaired"], base, mask, extra["series"])
    sample = np.random.default_rng(904).choice(len(raw), min(20000,len(raw)), replace=False)
    total = extra["gold"][mask][sample].sum(axis=1) / 1000
    for label, model in models.items():
        frozen[label]["artifact_audit"] = wpaudit.artifact_shape(model)
        frozen[label]["gold_audit"] = wpaudit.audit_gold(model["state"], raw[sample],
                                                         base["t_min"][mask][sample], total)
        wpgam.save_model(model, str(output / (label + "_frozen.npz")), plan)
    if incumbent_source:
        spec = importlib.util.spec_from_file_location("lol_ticker._v8_reference", incumbent_source)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        if old.MODEL_KIND != wpgam.LEGACY_MODEL_KIND:
            raise ValueError("Expected the recorded v8 implementation")
        with np.load(dataset, allow_pickle=True) as d:
            incumbent = old.fit_arrays(d["X"], d["y"], d["gid"], d["t"], d["seq"], d["C"],
                list(d["names"]), list(d["champ_names"]), train_games=base["outer_train"], dates=d["date"])
        fixed_X = base["raw_X"][base["raw_fixed"]][mask]
        fixed_C = base["raw_C"][base["raw_fixed"]][mask]
        old_raw = np.column_stack([old.prior_values_from_matrix(incumbent, fixed_X, base["names"], fixed_C),
                                  old.state_values_from_matrix(fixed_X, base["names"])])
        old_p = old.predict_state(incumbent["state"], old_raw, base["t_min"][mask])
        frozen["versus_v8"] = summarize(predictions, old_p, base, mask, extra["series"])
        frozen["v8_metrics"] = wpbench._basic_metrics(old_p, base["y"][mask], base["gid"][mask])
        plan["incumbent_source_sha256"] = hashlib.sha256(incumbent_source.read_bytes()).hexdigest()
        predictions["v8"] = old_p
    dump(output / "frozen.json", frozen)
    np.savez_compressed(output / "frozen_predictions.npz", gid=base["gid"][mask],
                        y=base["y"][mask], **predictions)
    replay = {k:np.full(len(base["y"]), np.nan) for k in SPECS}
    months = []
    dates = base["game_dates"]
    for month in sorted(set(d[:7] for d in dates[~base["outer_train"]])):
        start = month + "-01"; end = str(np.datetime64(month, "M") + 1) + "-01"
        cutoff = str(np.datetime64(start) - np.timedelta64(28, "D"))
        block = dict(fit=dates < cutoff, calibration=(dates >= cutoff)&(dates < start),
                     validation=(dates >= start)&(dates < end)&~base["outer_train"],
                     fit_end=cutoff, validation_start=start, validation_end=end)
        rr, pp, _, _, mm = stage(base, block, choices)
        for label in SPECS: replay[label][mm] = pp[label][choices[label]]
        months.append(dict(month=month, results=rr))
        dump(output / "months.json", months)
    replay = {k:v[mask] for k,v in replay.items()}
    if not all(np.isfinite(v).all() for v in replay.values()):
        raise ValueError("Incomplete monthly replay")
    rolling = summarize(replay, replay["repaired"], base, mask, extra["series"])
    dump(output / "rolling.json", rolling)
    np.savez_compressed(output / "rolling_predictions.npz", gid=base["gid"][mask],
                        y=base["y"][mask], **replay)
    # Stage a current candidate using the frozen choice, still without promotion.
    end = str(np.datetime64(max(dates)) + np.timedelta64(1, "D"))
    cutoff = str(np.datetime64(end) - np.timedelta64(28, "D"))
    current = dict(fit=dates < cutoff, calibration=dates >= cutoff,
                   validation=dates >= cutoff, fit_end=cutoff,
                   validation_start=cutoff, validation_end=end)
    # These final validation predictions are used for input audits only, never
    # for accuracy reporting or selection (the calibration block overlaps).
    _, _, current_models, audit_raw, audit_mask = stage(base, current, choices)
    selected = selection["selected"]["label"]
    candidate = current_models[selected]
    audit = wpaudit.audit_gold(candidate["state"], audit_raw, base["t_min"][audit_mask],
                               extra["gold"][audit_mask].sum(axis=1)/1000)
    from lol_ticker import wpdeploy
    shape = wpdeploy.audit_live_shape(predict_gam=lambda s: wpgam.predict_live_model(
        candidate, s, rounded=False)["p_blue"])
    status = dict(selected=selection["selected"], artifact=wpaudit.artifact_shape(candidate),
                  gold=audit, live_shape=shape, promotion="requires fresh prospective evidence",
                  calibration_through=max(dates), production_changed=False)
    wpgam.save_model(candidate, str(output / "candidate.npz"), dict(plan, candidate_status=status))
    restored = wpgam.load_model(str(output / "candidate.npz"))
    np.testing.assert_allclose(wpgam.predict_state(candidate["state"], audit_raw[:100], base["t_min"][audit_mask][:100]),
        wpgam.predict_state(restored["state"], audit_raw[:100], base["t_min"][audit_mask][:100]), rtol=0, atol=1e-12)
    dump(output / "candidate_status.json", status)
    log.info("Completed: %s", {k:v["metrics"]["brier_game"] for k,v in rolling.items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(wpgam.OUT_DIR)/"states.npz")
    parser.add_argument("--output", type=Path, default=Path(wpgam.OUT_DIR)/"repairs_v9")
    parser.add_argument("--resource-cache", type=Path, required=True)
    parser.add_argument("--incumbent-source", type=Path)
    args = parser.parse_args()
    main(args.dataset, args.output, args.resource_cache, args.incumbent_source)
