"""Consumed-history screen of the isolated, monotonically decaying SQ prior.

Reuses audited corrected-core caches. No database, network, registry writes,
production fit, deployment or candidate registration. August selects baseline
versus one prespecified curve; September's 19 historical games are diagnostic.
"""
import argparse
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import sqpairs, wpsqearly, wpgam, wpbench
from scripts import wpx_combat, wpx_methods_v9 as methods

log = logging.getLogger("sq_early")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_scores(dataset, rows, table_path):
    scorer = sqpairs.Scorer(str(table_path))
    with np.load(dataset, allow_pickle=False) as z:
        first = wpgam._game_rows(z["gid"])
        games, patches, champions = z["gid"][first], z["patch"][first], z["C"][first]
        names = z["champ_names"].tolist()
    by_game, coverage, source_patches = {}, {}, set()
    for gid, patch, slots in zip(games, patches, champions):
        picks = [str(names[int(c)]) if 0 <= c < len(names) else "" for c in slots]
        capture = wpsqearly.capture(str(patch), picks[:5], picks[5:], scorer=scorer)
        by_game[int(gid)] = (capture["pair_score"], capture["available"])
        coverage[int(gid)] = capture["coverage"]
        source_patches.update(capture["source_patches"])
    score = np.array([by_game[int(g)][0] for g in rows["gid"]])
    available = np.array([by_game[int(g)][1] for g in rows["gid"]], dtype=bool)
    return score, available, dict(covered_games=sum(v[1] for v in by_game.values()),
        total_games=len(by_game), covered_states=int(available.sum()), total_states=len(available),
        pair_coverage_by_game=coverage, source_patches=sorted(source_patches, key=sqpairs.pnum))


def summarize(p, baseline, rows, mask, available, series):
    report = {}
    for label, keep in (("all", mask), ("covered", mask & available),
                        ("covered_early", mask & available & (rows["t_min"] < 20)),
                        ("late", mask & (rows["t_min"] >= 20))):
        local = keep[mask]
        if not local.any():
            report[label] = None
            continue
        y, gids = rows["y"][keep], rows["gid"][keep]
        report[label] = dict(metrics=methods.metrics(p[local], y, gids),
            baseline=methods.metrics(baseline[local], y, gids),
            paired=methods.paired(p[local], baseline[local], y, gids, series))
    return report


def stage(output, label, block, rows, score, available, series, cache, plan, source):
    directory = output/label; directory.mkdir()
    baseline, masks, calibration, hashes, _, _ = wpx_combat.baseline_stage(cache, label, block, rows, plan)
    fit, val = masks["fit"], masks["validation"]
    model = wpsqearly.fit(score[fit], available[fit], baseline["fit"], rows["y"][fit],
                        rows["gid"][fit], rows["t_min"][fit], source=source)
    wpsqearly.save(model, directory/"candidate.npz")
    loaded = wpsqearly.load(directory/"candidate.npz")
    raw = wpsqearly.predict(loaded, score[val], available[val], baseline["validation"], rows["t_min"][val])
    ref, p = [wpbench._apply_platt(v, calibration) for v in (baseline["validation"], raw)]
    fallback = ~available[val] | (rows["t_min"][val] >= 20)
    if not np.array_equal(p[fallback], ref[fallback]):
        raise ValueError("SQ missing/late forecasts must match the baseline exactly")
    report = dict(block=block, baseline_hashes=hashes, calibration=calibration,
        own_knot_slopes_on_raw_score=(model["slopes"]/.25).tolist(),
        supported_training_games=model["supported_training_games"],
        covered_validation_games=len(np.unique(rows["gid"][val & available])),
        fallback_exact=True, scores=summarize(p, ref, rows, val, available, series))
    np.savez_compressed(directory/"predictions.npz", gid=rows["gid"][val], t=rows["t"][val],
                        y=rows["y"][val], baseline=ref, candidate=p, covered=available[val])
    methods.dump(directory/"report.json", report)
    log.info("%s covered training=%s raw slopes=%s delta=%s", label,
             model["supported_training_games"], report["own_knot_slopes_on_raw_score"], report["scores"]["all"]["paired"])
    return report


def main(dataset, cache, table_path, output):
    if output.exists():
        raise FileExistsError("Use a new output directory")
    rows, manifest = wpx_combat.load_rows(dataset)
    if max(rows["date"]) > "2026-09-02":
        raise ValueError("This plan is restricted to historical games through September 2")
    original = json.loads((cache/"plan.json").read_text())
    if original["dataset_sha256"] != sha(dataset):
        raise ValueError("Baseline dataset mismatch")
    resources = cache.parent/"resources/resources.npz"
    if sha(resources) != original["resources_sha256"]:
        raise ValueError("Cluster cache mismatch")
    with np.load(resources, allow_pickle=False) as z:
        series = dict(zip(z["series_gid"].astype(int), z["series_key"].astype(str)))
    protected = {str(p): sha(p) for p in (Path(wpgam.MODEL_PATH), Path(wpgam.OUT_DIR)/"live_stack.json",
                                        Path(wpgam.OUT_DIR)/"evaluation_registry.json")}
    source = dict(tables_path=str(table_path.resolve()), tables_sha256=sha(table_path),
                  scorer_source_sha256=sha(Path(sqpairs.__file__)))
    score, available, coverage = collect_scores(dataset, rows, table_path)
    output.mkdir(parents=True)
    np.savez_compressed(output/"scores.npz", gid=rows["gid"], t=rows["t"], pair_score=score, available=available)
    plan = dict(kind=wpsqearly.KIND, dataset_sha256=sha(dataset), manifest_sha256=sha(Path(str(dataset)+".manifest.json")),
        baseline_plan_sha256=sha(cache/"plan.json"), source=source, fixed_penalties=dict(l2=300., smooth=70.),
        selection="August only: negative paired upper Brier bound and no log-loss regression; otherwise baseline",
        calibration="unchanged earlier-block baseline Platt for both; no refit on scored dates",
        limitations=["consumed historical diagnostic; no promotion evidence",
                     "September contains only 19 games; cannot validate a small gain",
                     "shipped scorer pins globally estimated SQ-only shrink constants, including later scraped patches; prior-patch pair cells only",
                     "frozen-core residual; not a joint refit of four upstream GAM priors"], protected=protected)
    methods.dump(output/"plan.json", plan)
    snapshots = output/"source_snapshot"; snapshots.mkdir()
    sources = [Path(__file__)]
    sources += [Path(m.__file__) for m in (wpsqearly, sqpairs, wpgam, wpbench, wpx_combat, methods)]
    for path in sources:
        (snapshots/path.name).write_bytes(path.read_bytes())
    reports = []
    for month in ("2026-08", "2026-09"):
        start, end = month+"-01", str(np.datetime64(month, "M")+1)+"-01"
        block = dict(fit_end=str(np.datetime64(start)-np.timedelta64(28, "D")),
                     validation_start=start, validation_end=end)
        reports.append(stage(output, "replay_"+month, block, rows, score, available, series, cache, original, source))
    first = reports[0]["scores"]["all"]
    selected = "sq_early" if first["paired"]["ci95"][1] < 0 and first["metrics"]["logloss_game"] <= first["baseline"]["logloss_game"] else "baseline"
    methods.dump(output/"report.json", dict(completed=True, selected=selected, coverage=coverage,
        development=reports[0], later_diagnostic=reports[1], production_changed=False, prospective_registration=False))
    if any(sha(p) != h for p, h in protected.items()) or sha(table_path) != source["tables_sha256"]:
        raise ValueError("Protected artifact or pinned tables changed during the screen")
    artifacts = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}
    baseline_meta = {str(cache/label/name): sha(cache/label/name)
                     for label in ("replay_2026-08", "replay_2026-09") for name in ("stack_cache.json", "results.json")}
    methods.dump(output/"completion.json", dict(completed=True, artifacts=artifacts, baseline_metadata=baseline_meta,
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__), production_changed=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path(wpgam.OUT_DIR)/"states_inputs_v2_canonical_before_2026-09-03.npz")
    ap.add_argument("--baseline-cache", type=Path, default=Path(wpgam.OUT_DIR)/"action_20260911/resource_study")
    ap.add_argument("--tables", type=Path, default=Path(sqpairs.TABLES_PATH))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); main(args.dataset, args.baseline_cache, args.tables, args.out)
