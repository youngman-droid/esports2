"""Exhaustive joint ablations of supported research families on consumed history.

Five families give 31 challengers. June/July select before August is scored;
September's 19 maps are diagnostic only. No registry or production writes.
"""
import argparse
import gc
import hashlib
import itertools
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpbench, wpgam, wpcombat, wpobjective, wptrend, wpcomposition, wpsqearly
from research import wpx_combat, wpx_methods_v9 as methods

log = logging.getLogger("combinations")
FAMILIES = ("combat", "objective", "trend", "composition", "sq")
FEATURE_CACHES = dict(
    combat="combat_2026-10-04_verified/features.npz",
    objective="objective_2026-10-04_screen/features.npz",
    trend="trend_2026-10-04_screen/features.npz",
    composition="combination_composition_2026-10-04/features.npz",
    sq="sqearly_2026-10-04_final/scores.npz")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combinations():
    return [tuple(s) for size in range(1, len(FAMILIES)+1)
            for s in itertools.combinations(FAMILIES, size)]


def load_features(root, rows, dataset_sha256=None):
    features, sources = {}, {}
    for family, relative in FEATURE_CACHES.items():
        path = root/relative
        digest = sha(path)
        completion = path.parent/"completion.json"
        if family != "composition":
            recorded = json.loads(completion.read_text())
            if recorded.get("completed") is not True or recorded["artifacts"].get(path.name) != digest:
                raise ValueError("Family cache is not hash-matched to a completed experiment: "+family)
            sources[str(completion)] = sha(completion)
            family_plan_path = path.parent/"plan.json"
            if recorded["artifacts"].get("plan.json") != sha(family_plan_path):
                raise ValueError("Family plan changed: "+family)
            if dataset_sha256 is not None and json.loads(family_plan_path.read_text())["dataset_sha256"] != dataset_sha256:
                raise ValueError("Family dataset mismatch: "+family)
            sources[str(family_plan_path)] = sha(family_plan_path)
        else:
            provenance = path.parent/"provenance.json"
            recorded = json.loads(provenance.read_text())
            if recorded.get("utc_day_gate") is not True:
                raise ValueError("Composition cache lacks its historical source gate")
            if dataset_sha256 is not None and recorded["dataset_sha256"] != dataset_sha256:
                raise ValueError("Composition dataset mismatch")
            for source_path, source_hash in recorded["sources"].items():
                if sha(source_path) != source_hash:
                    raise ValueError("Composition source changed: "+source_path)
                sources[source_path] = source_hash
            sources[str(provenance)] = sha(provenance)
        with np.load(path, allow_pickle=False) as z:
            if any(not np.array_equal(z[k], rows[k]) for k in ("gid", "t")):
                raise ValueError("Feature row identity mismatch: "+family)
            if family == "sq":
                available = z["available"]
                value = np.where(available, z["pair_score"], 0.)[:, None]
                block = dict(extra=value, known=available[:, None], available=available,
                             names=["prior_patch_pair_score"])
            else:
                block = {k: z[k] for k in ("extra", "known", "available")}
                block["names"] = z["names"].tolist()
                if family == "combat":
                    block["raw"] = z["raw"]
        value, known = block["extra"], block["known"]
        if (value.shape != known.shape or value.shape != (len(rows["gid"]), len(block["names"]))
                or known.dtype != bool or block["available"].dtype != bool
                or block["available"].shape != (len(rows["gid"]),)
                or not np.array_equal(block["available"], known.any(axis=1))
                or not np.isfinite(value).all() or np.any(value[~known] != 0)):
            raise ValueError("Invalid coverage/zero-fallback cache: "+family)
        features[family], sources[str(path)] = block, digest
    return features, sources


def specifications(features):
    out = {}
    contracts = dict(combat=wpcombat.INPUT_CONTRACT, objective=wpobjective.INPUT_CONTRACT,
                     trend=wptrend.INPUT_CONTRACT, composition=wpcomposition.INPUT_CONTRACT,
                     sq=wpsqearly.INPUT_CONTRACT)
    for family, block in features.items():
        out[family] = dict(feature_names=block["names"], bounds=[-10., 10.] if family == "composition" else [0., 10.],
                           l2=800., smooth=70., time_knots=wpgam.TIME_KNOTS.tolist(), input_contract=contracts[family])
    out["sq"].update(bounds=[0., 2.5], l2=300., time_knots=[0., 5., 10., 15., 20.],
                     monotone_decay=True, zero_after_min=20., scale=.25)
    return out


def joint_intervals(predictions, ref, y, gids, series, draws=4000):
    """Shared cluster draws and simultaneous max-t bands across all challengers."""
    names = list(predictions)
    games, row_game = np.unique(gids, return_inverse=True)
    n = np.bincount(row_game)
    differences = np.column_stack([np.bincount(row_game, weights=(predictions[k]-y)**2-(ref-y)**2)/n for k in names])
    _, cluster = np.unique([series[int(g)] for g in games], return_inverse=True)
    counts = np.bincount(cluster)
    sums = np.column_stack([np.bincount(cluster, weights=differences[:, j]) for j in range(len(names))])
    rng = np.random.default_rng(240304873)
    boot = np.empty((draws, len(names)))
    for start in range(0, draws, 256):
        end = min(draws, start+256)
        ix = rng.integers(len(counts), size=(end-start, len(counts)))
        boot[start:end] = sums[ix].sum(axis=1)/counts[ix].sum(axis=1)[:, None]
    delta = differences.mean(axis=0)
    error = boot.std(axis=0, ddof=1)
    student = np.divide(boot-delta, error, out=np.zeros_like(boot), where=error > 1e-15)
    critical = float(np.quantile(np.max(np.abs(student), axis=1), .95))
    return {k: dict(delta_brier=float(delta[j]), ci95=np.quantile(boot[:, j], [.025, .975]).tolist(),
                    simultaneous_ci95=[float(delta[j]-critical*error[j]), float(delta[j]+critical*error[j])],
                    games=len(games), clusters=len(counts), draws=draws, family_size=len(names),
                    max_t_critical=critical) for j, k in enumerate(names)}


def slices(p, ref, rows, mask, features, series):
    out = {}
    for name, local in (("early_under20", rows["t_min"][mask] < 20),
                        ("late_20plus", rows["t_min"][mask] >= 20),
                        ("combat_covered", features["combat"]["available"][mask]),
                        ("sq_covered_early", features["sq"]["available"][mask] & (rows["t_min"][mask] < 20))):
        if local.any():
            y, g = rows["y"][mask][local], rows["gid"][mask][local]
            out[name] = dict(metrics=methods.metrics(p[local], y, g), baseline=methods.metrics(ref[local], y, g),
                             paired=methods.paired(p[local], ref[local], y, g, series))
        else:
            out[name] = None
    return out


def stage(output, month, rows, features, specs, gold, series, cache, original, *, resume=False):
    from lol_ticker import wpcombined
    from research.wpx_combination_audit import audit
    label = "replay_"+month
    directory = output/label
    directory.mkdir(exist_ok=resume)
    start = month+"-01"
    block = dict(fit_end=str(np.datetime64(start)-np.timedelta64(28, "D")), validation_start=start,
                 validation_end=str(np.datetime64(month, "M")+1)+"-01")
    baseline, masks, calibration, hashes, core_model, core = wpx_combat.baseline_stage(cache, label, block, rows, original)
    fit, val = masks["fit"], masks["validation"]
    ref = wpbench._apply_platt(baseline["validation"], calibration)
    fit_blocks = {k: v["extra"][fit] for k, v in features.items()}
    val_blocks = {k: v["extra"][val] for k, v in features.items()}
    report = dict(block=block, baseline_hashes=hashes, calibration=calibration,
                  baseline=methods.metrics(ref, rows["y"][val], rows["gid"][val]), candidates={},
                  coverage={k: {part: dict(states=int((v["available"] & m).sum()),
                             games=len(np.unique(rows["gid"][v["available"] & m])))
                             for part, m in (("fit", fit), ("validation", val))} for k, v in features.items()})
    predictions = {}
    for subset in combinations():
        key = "+".join(subset)
        model_path, prediction_path = directory/(key+".npz"), directory/(key+"_predictions.npz")
        report_path = directory/(key+".json")
        if resume and report_path.exists():
            saved = json.loads(report_path.read_text())
            if sha(model_path) != saved["model_sha256"] or sha(prediction_path) != saved["predictions_sha256"]:
                raise ValueError("Resume artifact changed: "+key)
            with np.load(prediction_path, allow_pickle=False) as z:
                if not np.array_equal(z["gid"], rows["gid"][val]) or not np.array_equal(z["baseline"], ref):
                    raise ValueError("Resume scoring population changed")
                p = z["candidate"]
            report["candidates"][key], predictions[key] = saved, p
            continue
        began = time.monotonic()
        selected_specs = {k: specs[k] for k in subset}
        model = wpcombined.fit({k: fit_blocks[k] for k in subset}, baseline["fit"], rows["y"][fit],
                               rows["gid"][fit], rows["t_min"][fit], specs=selected_specs)
        wpcombined.save(model, model_path)
        loaded = wpcombined.load(model_path)
        raw = wpcombined.predict(loaded, {k: val_blocks[k] for k in subset}, baseline["validation"], rows["t_min"][val])
        p = wpbench._apply_platt(raw, calibration)
        shape = audit(loaded, core_model, core, calibration, rows, features, gold, val)
        saved = dict(families=list(subset), metrics=methods.metrics(p, rows["y"][val], rows["gid"][val]),
                     slices=slices(p, ref, rows, val, features, series), shape_audit=shape,
                     elapsed_seconds=time.monotonic()-began, model_sha256=sha(model_path))
        np.savez_compressed(prediction_path, gid=rows["gid"][val], t=rows["t"][val], y=rows["y"][val], baseline=ref, candidate=p)
        saved["predictions_sha256"] = sha(prediction_path)
        methods.dump(report_path, saved)
        report["candidates"][key], predictions[key] = saved, p
        log.info("%s %s Brier delta=%+.7f %.1fs", month, key,
                 saved["metrics"]["brier_game"]-report["baseline"]["brier_game"], saved["elapsed_seconds"])
        del model, loaded
        gc.collect()
    intervals = joint_intervals(predictions, ref, rows["y"][val], rows["gid"][val], series)
    for key in predictions:
        report["candidates"][key]["paired"] = intervals[key]
    methods.dump(directory/"report.json", report)
    return report, dict(predictions=predictions, baseline=ref, y=rows["y"][val], gid=rows["gid"][val])


def pool(parts, reports, series):
    names = list(parts[0]["predictions"])
    p = {k: np.concatenate([v["predictions"][k] for v in parts]) for k in names}
    ref, y, gid = (np.concatenate([v[k] for v in parts]) for k in ("baseline", "y", "gid"))
    intervals = joint_intervals(p, ref, y, gid, series)
    results = {k: dict(metrics=methods.metrics(p[k], y, gid), paired=intervals[k],
                monthly_deltas=[r["candidates"][k]["paired"]["delta_brier"] for r in reports]) for k in names}
    baseline = methods.metrics(ref, y, gid)
    eligible = [k for k in names if intervals[k]["simultaneous_ci95"][1] < 0
                and results[k]["metrics"]["logloss_game"] <= baseline["logloss_game"]
                and max(results[k]["monthly_deltas"]) <= 0]
    rank = sorted(names, key=lambda k: results[k]["metrics"]["brier_game"])
    return dict(baseline=baseline, candidates=results, ranking=rank, eligible=eligible,
                selected=min(eligible, key=lambda k: results[k]["metrics"]["brier_game"]) if eligible else "baseline",
                exploratory_best=rank[0])


def main(dataset, cache, output, resume=False):
    from lol_ticker import wpcombined
    rows, manifest = wpx_combat.load_rows(dataset)
    if max(rows["date"]) > "2026-09-02":
        raise ValueError("Only consumed history through September 2 is allowed")
    original = json.loads((cache/"plan.json").read_text())
    if sha(dataset) != original["dataset_sha256"]:
        raise ValueError("Baseline dataset mismatch")
    resources = cache.parent/"resources/resources.npz"
    if sha(resources) != original["resources_sha256"]:
        raise ValueError("Resource cache mismatch")
    with np.load(resources, allow_pickle=False) as z:
        gold = z["gold"]
        series = dict(zip(z["series_gid"].astype(int), z["series_key"].astype(str)))
    features, source = load_features(Path(wpgam.OUT_DIR), rows, sha(dataset))
    specs = specifications(features)
    protected_paths = [Path(wpgam.OUT_DIR)/n for n in ("model_live_gam.npz", "model_live.npz", "live_stack.json",
                       "evaluation_registry.json", "historical_evaluation_registry.json")]
    protected = {str(p): sha(p) for p in protected_paths}
    source_paths = [Path(__file__), Path(wpcombined.__file__), Path(wpx_combat.__file__), Path(methods.__file__),
                    Path(wpgam.__file__), Path(wpbench.__file__), Path("research/wpx_combination_audit.py"),
                    Path("research/wpx_combination_composition.py")]
    source.update({str(p): sha(p) for p in source_paths})
    plan = dict(kind="joint_research_family_factorial_v1", families=list(FAMILIES),
        combinations=[list(s) for s in combinations()], specs=specs,
        development=["2026-06", "2026-07"], later="2026-08", small_diagnostic="2026-09",
        dataset_sha256=sha(dataset), manifest_sha256=sha(Path(str(dataset)+".manifest.json")),
        baseline_plan_sha256=sha(cache/"plan.json"), resources_sha256=sha(resources), sources=source, protected=protected,
        selection="Pooled June/July game Brier: simultaneous max-t 95% upper bound <0, no game log-loss increase, each monthly Brier delta <=0; else baseline",
        calibration="Unchanged earlier-block baseline Platt shared by baseline and every challenger",
        limitations=["All outcomes already consumed; exploratory historical comparisons, not prospective promotion evidence",
                     "June SQ training covers only 45 games; later training coverage increases",
                     "No certified confidence or Fearless history; no rich composition annotations",
                     "September has 19 games; six date/match clusters",
                     "SQ prior cells precede patch, but SQ-only global shrink nuisance constants include later scraped patches",
                     "Residual fitting uses in-sample frozen-core training offsets; validation predictions remain strictly chronological",
                     manifest["identity_limitation"], manifest["death_limitation"]])
    if output.exists():
        if not resume or json.loads((output/"plan.json").read_text()) != plan:
            raise ValueError("Use a new output directory or exact-plan --resume")
    else:
        output.mkdir(parents=True)
        methods.dump(output/"plan.json", plan)
        snapshots = output/"source_snapshot"; snapshots.mkdir()
        for p in source_paths:
            (snapshots/p.name).write_bytes(p.read_bytes())
    development, parts = [], []
    for month in plan["development"]:
        report, part = stage(output, month, rows, features, specs, gold, series, cache, original, resume=resume)
        development.append(report); parts.append(part)
    selection = pool(parts, development, series)
    methods.dump(output/"selection.json", selection)
    log.info("Development selected %s; exploratory best %s", selection["selected"], selection["exploratory_best"])
    later, _ = stage(output, plan["later"], rows, features, specs, gold, series, cache, original, resume=resume)
    diagnostic, _ = stage(output, plan["small_diagnostic"], rows, features, specs, gold, series, cache, original, resume=resume)
    methods.dump(output/"report.json", dict(completed=True, selected=selection["selected"], selection=selection,
        development=development, later=later, small_diagnostic=diagnostic,
        production_changed=False, prospective_registration=False))
    if any(sha(p) != h for p, h in {**source, **protected}.items()):
        raise ValueError("Pinned input/source or protected artifact changed during experiment")
    artifacts = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and p.name != "completion.json"}
    methods.dump(output/"completion.json", dict(completed=True, artifacts=artifacts, production_changed=False,
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path(wpgam.OUT_DIR)/"states_inputs_v2_canonical_before_2026-09-03.npz")
    ap.add_argument("--baseline-cache", type=Path, default=Path(wpgam.OUT_DIR)/"action_20260911/resource_study")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(); main(a.dataset, a.baseline_cache, a.out, a.resume)
