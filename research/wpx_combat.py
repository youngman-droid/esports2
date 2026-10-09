"""Role-specific combat readiness screen on audited, consumed historical inputs.

Adds a small nonnegative residual to saved chronological corrected-core fits.
The baseline and challenger share the baseline's earlier-block calibration so
unavailable telemetry gives exactly the same forecast. No deployment, candidate
registration, outcome-registry update or database writes occur.
"""
import argparse
import hashlib
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import config, wpcombat, wpgam, wpbench
from scripts import wpx_methods_v9 as methods

log = logging.getLogger("combat")
SPECS = {"role_health": {"l2": 800., "smooth": 70.},
         "role_health_resources": {"l2": 800., "smooth": 70.}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize(output, cache):
    """Bind evaluated snapshots, scoring dependencies and completed outputs."""
    import platform
    import scipy
    plan = json.loads((output/"plan.json").read_text())
    snapshots = output/"source_snapshot"
    for path, digest in plan["source"].items():
        if sha(snapshots/Path(path).name) != digest:
            raise ValueError("Evaluated source snapshot changed")
    for module in (wpbench, methods):
        path = Path(module.__file__)
        target = snapshots/path.name
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError("Scoring dependency snapshot changed")
        target.write_bytes(path.read_bytes())
    (snapshots/"baseline_plan.json").write_bytes((cache/"plan.json").read_bytes())
    original = {str(cache/"plan.json"): sha(cache/"plan.json")}
    for label in ("development_1", "development_2", "development_3", "frozen"):
        for name in ("stack_cache.json", "results.json"):
            path = cache/label/name
            original[str(path)] = sha(path)
    artifacts = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name != "completion.json"}
    methods.dump(output/"completion.json", dict(completed=True, artifacts=artifacts,
        baseline_metadata=original, versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        evaluated_sources=plan["source"], finalization_source_sha256=sha(Path(__file__)), production_changed=False))


def load_rows(dataset):
    manifest = json.loads(Path(str(dataset) + ".manifest.json").read_text())
    if manifest.get("sha256") != sha(dataset) or manifest.get("zero_future_timestamp_joins") is not True:
        raise ValueError("A hash-matched, causally audited corrected dataset is required")
    with np.load(dataset, allow_pickle=False) as z:
        if str(z["input_contract_sha256"].item()) != manifest["input_contract_sha256"]:
            raise ValueError("Input contract mismatch")
        fixed = z["seq"] < 0
        names = z["names"].tolist()
        X = z["X"][fixed]
        first = wpgam._game_rows(z["gid"])
        rows = {k: z[k][fixed] for k in ("gid", "y", "date", "t", "prediction_clock_s",
            "hp_clock_lo_s", "hp_clock_hi_s", "hp_age_upper_s", "hp_observation_ts", "hp_origin_ts")}
        rows.update(X=X, names=names, game_gid=z["gid"][first], game_dates=z["date"][first].astype(str))
    rows["date"] = rows["date"].astype(str)
    rows["t_min"] = rows["t"] / 60.
    rows["used"] = X[:, names.index("has_hp")] > 0
    registry = json.loads((Path(wpgam.OUT_DIR) / "evaluation_registry.json").read_text())
    if max(rows["date"]) > registry["consumed_through"]:
        raise ValueError("This diagnostic accepts only previously consumed dates")
    used = rows["used"]
    lo, hi, clock = (rows[k][used] for k in ("hp_clock_lo_s", "hp_clock_hi_s", "prediction_clock_s"))
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()
            and np.all(0 <= lo) and np.all(lo <= hi) and np.all(hi <= clock)
            and np.all(clock - lo <= 90)
            and np.allclose(clock - lo, rows["hp_age_upper_s"][used])
            and np.all(rows["hp_observation_ts"][used] + 1 - rows["hp_origin_ts"][used] <= clock)):
        raise ValueError("Historical combat observations violate the causal clock bounds")
    return rows, manifest


def source_records(conn, gids):
    """Fetch only input telemetry for the already-audited historical game IDs."""
    records = {}
    for r in conn.execute("""SELECT fg.golgg_game_id gid, fm.ts, fm.data
        FROM feed_games fg JOIN feed_minutes fm USING(esports_game_id)
        WHERE fg.status='scraped' AND fg.golgg_game_id=ANY(%s)""", (list(map(int, gids)),)):
        key = (int(r["gid"]), int(r["ts"]))
        if key in records and records[key] != r["data"]:
            raise ValueError("Contradictory telemetry for the same game/timestamp")
        records[key] = r["data"]
    return records


def extract(rows, records, gold):
    extra = np.zeros((len(rows["gid"]), len(wpcombat.FEATURE_NAMES)))
    available = np.zeros(len(extra), dtype=bool)
    death = np.zeros(len(extra), dtype=bool)
    known = np.zeros_like(extra, dtype=bool)
    raw = np.full((len(extra), 10, 3), np.nan)
    idx = {n: i for i, n in enumerate(rows["names"])}
    if gold.shape != (len(extra), 10) or not np.isfinite(gold).all() or np.any(gold < 0):
        raise ValueError("Complete hash-verified player gold is required to audit combat joins")
    timelines = {}
    for gid in np.unique(rows["gid"][rows["used"]]):
        indices = np.flatnonzero(rows["gid"] == gid)
        timelines[int(gid)] = (rows["prediction_clock_s"][indices], gold[indices])
    checked = set()
    for i in np.flatnonzero(rows["used"]):
        key = (int(rows["gid"][i]), int(rows["hp_observation_ts"][i]))
        if key not in records:
            raise ValueError("Audited observation is missing from its historical source")
        data = records[key]
        interval = (key, float(rows["hp_clock_lo_s"][i]), float(rows["hp_clock_hi_s"][i]))
        if interval not in checked:
            observed = np.asarray(list(data.get("gdb") or []) + list(data.get("gdr") or []), dtype=float)
            clocks, values = timelines[key[0]]
            lower = values[clocks == interval[1]]
            prefix = values[clocks <= interval[2]]
            if (observed.shape != (10,) or not np.isfinite(observed).all() or np.any(observed < 0)
                    or len(lower) != 1 or not np.all(lower[0] <= observed) or not np.any(lower[0] < observed)
                    or not len(prefix) or np.any(np.diff(prefix, axis=0) < 0)
                    or not np.any(np.all(prefix >= observed, axis=1) & np.any(prefix > observed, axis=1))):
                raise ValueError("Raw player gold violates the audited observation clock bracket")
            checked.add(interval)
        h = list(data.get("hpb") or []) + list(data.get("hpr") or [])
        if len(h) != 10:
            raise ValueError("Incomplete audited health vector")
        expected = {"hp_pool": sum(h[:5])-sum(h[5:]),
                    "hp_low_b": sum(v < .3 for v in h[:5]), "hp_low_r": sum(v < .3 for v in h[5:]),
                    "dead_blue": sum(v <= 0 for v in h[:5]), "dead_red": sum(v <= 0 for v in h[5:])}
        if any(abs(rows["X"][i, idx[n]] - v) > 2e-5 for n, v in expected.items()):
            raise ValueError("Per-player telemetry disagrees with the audited baseline row")
        for suffix, pids in (("b", list(range(1, 6))), ("r", list(range(6, 11)))):
            if data.get("pid"+suffix) is not None and data["pid"+suffix] != pids:
                raise ValueError("Stored participant ordering disagrees with the audited role slots")
        features = wpcombat.observation_features(data, age_upper_s=float(rows["hp_age_upper_s"][i]))
        if not features["available"]:
            raise ValueError("An audited health observation was rejected: " + features["reason"])
        extra[i] = features["features"]
        known[i] = features["known"]
        available[i] = True
        death[i] = any(v <= 0 for v in h)
        for channel, prefix in enumerate(("hp", "gd", "lv")):
            values = list(data.get(prefix+"b") or []) + list(data.get(prefix+"r") or [])
            if len(values) == 10:
                raw[i, :, channel] = [np.nan if v is None else v for v in values]
    return dict(extra=extra, known=known, available=available, death=death, raw=raw)


def coverage(rows, features, mask):
    mask = np.asarray(mask, dtype=bool)
    present = mask & features["available"]
    dead = mask & features["death"]
    return dict(states=int(mask.sum()), games=int(len(np.unique(rows["gid"][mask]))),
                available_states=int(present.sum()), available_games=int(len(np.unique(rows["gid"][present]))),
                death_states=int(dead.sum()), death_games=int(len(np.unique(rows["gid"][dead]))),
                feature_known_states={n: int((mask & features["known"][:, j]).sum())
                                      for j, n in enumerate(wpcombat.FEATURE_NAMES)})


def variant(extra, name):
    value = np.asarray(extra).copy()
    if name == "role_health":
        value[:, 10:] = 0
    return value


def baseline_stage(cache, label, block, rows, plan):
    directory = cache / label
    provenance = json.loads((directory / "stack_cache.json").read_text())
    recorded = json.loads((directory / "results.json").read_text())["core_0"]
    game_fit = rows["game_dates"] < block["fit_end"]
    identity = dict(plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                    fit_gids_sha256=hashlib.sha256(rows["game_gid"][game_fit].tobytes()).hexdigest())
    if provenance["identity"] != identity:
        raise ValueError("Baseline cache fitting population/plan mismatch")
    paths = {"core.npy": provenance["core_sha256"], "priors.pkl": provenance["prior_sha256"],
             "core_0.pkl": recorded["model_sha256"], "core_0_raw.npz": recorded["prediction_sha256"]}
    if any(sha(directory / n) != h for n, h in paths.items()):
        raise ValueError("Baseline cache content mismatch")
    core = np.load(directory / "core.npy", mmap_mode="r")
    model = pickle.loads((directory / "core_0.pkl").read_bytes())["fitted"]
    if core.shape != (len(rows["gid"]), len(wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES)):
        raise ValueError("Baseline cache shape mismatch")
    dates = rows["date"]
    masks = dict(fit=dates < block["fit_end"], calibration=(dates >= block["fit_end"]) & (dates < block["validation_start"]),
                 validation=(dates >= block["validation_start"]) & (dates < block["validation_end"]))
    if label == "frozen":
        masks["validation"] = dates >= block["validation_start"]  # last consumed date is inclusive
    predictions = {k: wpgam.predict_state(model, core[m], rows["t_min"][m]) for k, m in masks.items()}
    with np.load(directory / "core_0_raw.npz", allow_pickle=False) as saved:
        for part in ("calibration", "validation"):
            if predictions[part].shape != saved[part].shape or not np.allclose(predictions[part], saved[part], atol=1e-11, rtol=0):
                raise ValueError("Current scorer does not reproduce the cached baseline")
    calibration = recorded["methods"]["platt"]["calibration"]
    if not np.isfinite([calibration["intercept"], calibration["slope"]]).all() or calibration["slope"] <= 0:
        raise ValueError("Baseline calibration must preserve monotonicity")
    return predictions, masks, calibration, paths, model, core


def shape_audit(residual, core_model, core, calibration, rows, features, gold, mask, name):
    """Perturb physical players coherently in both baseline and readiness."""
    indices = np.flatnonzero(mask & features["available"])
    indices = np.random.default_rng(20261004).choice(indices, min(500, len(indices)), replace=False)
    if not len(indices):
        return dict(states=0, comparisons=0, violations=0, worst_reversal=0.)
    raw = np.array(core[indices])
    obs = features["raw"][indices].copy()  # player x health/gold/level
    t = rows["t_min"][indices]
    names = {n: i for i, n in enumerate(wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES)}
    old_idx = {n: i for i, n in enumerate(rows["names"])}

    def extras(values):
        result = []
        for v in values:
            data = {prefix+side: v[start:start+5, ch].tolist()
                    for ch, prefix in enumerate(("hp", "gd", "lv")) for side, start in (("b", 0), ("r", 5))}
            result.append(wpcombat.observation_features(data)["features"])
        return variant(result, name)

    def forecast(value, observations):
        baseline = wpgam.predict_state(core_model, value, t)
        return wpbench._apply_platt(wpcombat.predict(residual, extras(observations), baseline, t), calibration)

    ref = forecast(raw, obs)
    comparisons = violations = 0
    worst = 0.
    for slot in range(10):
        sign, role = (1. if slot < 5 else -1.), wpcombat.ROLES[slot % 5]
        for channel, amount in (("gold", 100.), ("gold", 1000.), ("hp", .1), ("hp", 1.), ("level", 1.)):
            value, changed = raw.copy(), obs.copy()
            if channel == "gold":
                changed[:, slot, 1] += amount
                delta = sign * amount / 1000.
                for key in ("gold_k", "gold_mom", "gold_"+role):
                    value[:, names[key]] += delta
                total = gold[indices].sum(axis=1)/1000. + amount/1000.
                value[:, names["gold_rel"]] = value[:, names["gold_k"]]/total
            elif channel == "hp":
                changed[:, slot, 0] = np.minimum(1., changed[:, slot, 0]+amount)
                value[:, names["hp_pool"]] += sign*(changed[:, slot, 0]-obs[:, slot, 0])
                db, dr = (changed[:, :5, 0] <= 0).sum(axis=1), (changed[:, 5:, 0] <= 0).sum(axis=1)
                old = rows["X"][indices]
                effects = wpgam._death_features(db, dr, old[:, old_idx["towers_blue"]], old[:, old_idx["towers_red"]],
                                              old[:, old_idx["inhib_blue"]], old[:, old_idx["inhib_red"]])
                for key, effect in zip(("dead_adv", "dead_adv_sq", "dead_count_sq_adv", "dead_base_pressure"), effects):
                    value[:, names[key]] = effect
            else:
                valid_level = features["known"][indices, 15+slot % 5]
                changed[valid_level, slot, 2] = np.minimum(18., changed[valid_level, slot, 2]+amount)
                delta = np.nan_to_num(changed[:, slot, 2]-obs[:, slot, 2])
                value[:, names["lvl_k"]] += sign*delta/5.
            after = forecast(value, changed)
            if not np.isfinite(after).all() or np.any((after < 0) | (after > 1)):
                raise ValueError("Invalid composed combat prediction")
            movement = sign*(after-ref)
            comparisons += len(movement)
            violations += int((movement < -1e-10).sum())
            worst = min(worst, float(movement.min()))
    if violations:
        raise ValueError("Composed combat model violates physical monotonicity: %s reversals, worst %s" % (violations, worst))
    return dict(states=len(indices), comparisons=comparisons, violations=violations, worst_reversal=worst)


def scores(p, baseline, rows, mask, features, series):
    result = {}
    for label, keep in (("all", mask), ("telemetry", mask & features["available"]),
                        ("deaths", mask & features["death"]),
                        ("fresh_30s", mask & features["available"] & (rows["hp_age_upper_s"] <= 30))):
        local = keep[mask]
        if not local.any():
            result[label] = None
            continue
        y, gid = rows["y"][keep], rows["gid"][keep]
        result[label] = dict(metrics=methods.metrics(p[local], y, gid),
                             baseline=methods.metrics(baseline[local], y, gid),
                             paired=methods.paired(p[local], baseline[local], y, gid, series))
    result["by_phase"] = {}
    for label, lo, hi in (("early", 0, 15), ("mid", 15, 25), ("late", 25, 1000)):
        keep = mask & (rows["t_min"] >= lo) & (rows["t_min"] < hi)
        local = keep[mask]
        if local.any():
            result["by_phase"][label] = methods.paired(p[local], baseline[local], rows["y"][keep], rows["gid"][keep], series)
    return result


def stage(output, label, block, rows, features, gold, series, cache, plan):
    directory = output / label
    directory.mkdir()
    baseline, masks, calibration, hashes, core_model, core = baseline_stage(cache, label, block, rows, plan)
    fit, val = masks["fit"], masks["validation"]
    ref = wpbench._apply_platt(baseline["validation"], calibration)
    result = dict(block=block, baseline_hashes=hashes, calibration=calibration,
                  fit_coverage=coverage(rows, features, fit), validation_coverage=coverage(rows, features, val), variants={})
    result["training_telemetry_supported"] = result["fit_coverage"]["available_states"] > 0
    for name, spec in SPECS.items():
        extra = variant(features["extra"], name)
        model = wpcombat.fit(extra[fit], baseline["fit"], rows["y"][fit], rows["gid"][fit], rows["t_min"][fit], spec=spec)
        wpcombat.save(model, directory / (name + ".npz"))
        loaded = wpcombat.load(directory / (name + ".npz"))
        raw_p = wpcombat.predict(loaded, extra[val], baseline["validation"], rows["t_min"][val])
        p = wpbench._apply_platt(raw_p, calibration)
        missing = ~features["available"][val]
        if not np.array_equal(p[missing], ref[missing]):
            raise ValueError("Missing-telemetry forecasts changed from baseline")
        result["variants"][name] = scores(p, ref, rows, val, features, series)
        result["variants"][name]["missing_fallback_exact"] = True
        result["variants"][name]["shape_audit"] = shape_audit(loaded, core_model, core, calibration, rows, features, gold, val, name)
        np.savez_compressed(directory / (name + "_predictions.npz"), gid=rows["gid"][val], t=rows["t"][val],
                            y=rows["y"][val], baseline=ref, candidate=p)
        log.info("%s %s delta Brier %.7f", label, name, result["variants"][name]["all"]["paired"]["delta_brier"])
    methods.dump(directory / "report.json", result)
    return result


def main(dataset, cache, output):
    if output.exists():
        raise FileExistsError("Use a new experiment directory: " + str(output))
    rows, manifest = load_rows(dataset)
    original_plan = json.loads((cache / "plan.json").read_text())
    if original_plan["dataset_sha256"] != sha(dataset):
        raise ValueError("Baseline study used a different dataset")
    protected = [Path(wpgam.MODEL_PATH), Path(wpgam.OUT_DIR) / "model_live.npz", Path(wpgam.OUT_DIR) / "live_stack.json",
                 Path(wpgam.OUT_DIR) / "evaluation_registry.json", Path(wpgam.OUT_DIR) / "historical_evaluation_registry.json"]
    original_hashes = {str(p): sha(p) for p in protected}
    with psycopg.connect(config.PG_DSN, row_factory=dict_row, options="-c default_transaction_read_only=on -c statement_timeout=60000") as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        records = source_records(conn, np.unique(rows["gid"][rows["used"]]))
    resource_path = cache.parent / "resources" / "resources.npz"
    if sha(resource_path) != original_plan["resources_sha256"]:
        raise ValueError("Series-cluster resource cache changed")
    with np.load(resource_path, allow_pickle=False) as z:
        series = dict(zip(z["series_gid"].astype(int), z["series_key"].astype(str)))
        gold = z["gold"]
    features = extract(rows, records, gold)
    output.mkdir(parents=True)
    np.savez_compressed(output / "features.npz", gid=rows["gid"], t=rows["t"], **features,
                        names=np.array(wpcombat.FEATURE_NAMES), age_upper_s=rows["hp_age_upper_s"])
    plan = dict(kind="role_combat_residual_v1", specs=SPECS, dataset_sha256=sha(dataset),
        manifest_sha256=sha(Path(str(dataset)+".manifest.json")), baseline_plan_sha256=sha(cache/"plan.json"),
        features_sha256=sha(output/"features.npz"), protected=original_hashes,
        source_observations_sha256=hashlib.sha256(json.dumps(
            [(g, t, records[(g, t)]) for g, t in sorted({(int(rows["gid"][i]), int(rows["hp_observation_ts"][i]))
                for i in np.flatnonzero(rows["used"])})], sort_keys=True).encode()).hexdigest(),
        source={str(p): sha(p) for p in (Path(__file__), Path(wpcombat.__file__), Path(wpgam.__file__))},
        selection="most recent pre-May development block; choose a residual only if paired upper Brier bound is negative and log loss does not regress; otherwise baseline",
        calibration="same heldout baseline Platt transform for both models; no calibration or residual refit on scored dates",
        limitations=[manifest["identity_limitation"], manifest["death_limitation"],
                      "archived health fractions rounded to three decimals; extremely low positive health can resemble death",
                      "no item inputs: ambiguous historical undo transitions", "consumed historical diagnostic, not promotion evidence"],
        coverage=coverage(rows, features, np.ones(len(rows["gid"]), dtype=bool)))
    methods.dump(output/"plan.json", plan)
    for path in plan["source"]:
        target = output/"source_snapshot"/Path(path).name
        target.parent.mkdir(exist_ok=True); target.write_bytes(Path(path).read_bytes())
    development = []
    for i, block in enumerate(original_plan["development"]):
        development.append(stage(output, "development_"+str(i+1), block, rows, features, gold, series, cache, original_plan))
    latest = development[-1]["variants"]
    eligible = [n for n, s in latest.items() if s["all"]["paired"]["ci95"][1] < 0
                and s["all"]["metrics"]["logloss_game"] <= s["all"]["baseline"]["logloss_game"]]
    selected = min(eligible, key=lambda n: latest[n]["all"]["metrics"]["brier_game"]) if eligible else "baseline"
    methods.dump(output/"selection.json", dict(selected=selected, eligible=eligible, stage="development_3"))
    frozen = stage(output, "frozen", original_plan["final"], rows, features, gold, series, cache, original_plan)
    if any(sha(p) != h for p, h in original_hashes.items()):
        raise ValueError("Protected production artifacts changed during the screen")
    methods.dump(output/"report.json", dict(completed=True, selected=selected, production_changed=False,
        development=development, frozen=frozen, coverage=plan["coverage"], prospective_registration=False))
    finalize(output, cache)
    log.info("Completed combat screen; development selected %s", selected)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(wpgam.OUT_DIR)/"states_inputs_v2_canonical_before_2026-09-03.npz")
    parser.add_argument("--baseline-cache", type=Path, default=Path(wpgam.OUT_DIR)/"action_20260911/resource_study")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.baseline_cache, args.out)
