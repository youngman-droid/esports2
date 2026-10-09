"""Development diagnostics and exact-artifact prospective deployment gates.

Legacy component/blend searches use only globally exposed historical outcomes.
New and same-kind fits stage until a frozen candidate has a direct paired
series/date comparison with the incumbent, bound to exact artifact provenance.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import tempfile
import time

import numpy as np

from . import util, wpaudit, wpbench, wpgam, wpx, wpexposure


RESULT_PATH = os.path.join(wpgam.OUT_DIR, "live_stack_benchmark.json")
REGISTRY_PATH = os.path.join(wpgam.OUT_DIR, "evaluation_registry.json")
WEIGHT_GRID = np.linspace(0.0, 1.0, 21)
MIN_FRESH_GAMES = 100
SHAPE_TIMES = (0.0, 2.0, 5.0, 15.0, 30.0, 45.0, 60.0, 75.0)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def blend_predictions(gam, legacy, w_gam):
    return wpgam._sigmoid(float(w_gam) * _logit(gam)
                          + (1.0 - float(w_gam)) * _logit(legacy))


def _game_score(p, y, gids):
    _, loss = wpbench._per_game((np.asarray(p) - np.asarray(y)) ** 2, gids)
    return float(loss.mean())


def select_weight(gam, legacy, y, gids, grid=WEIGHT_GRID):
    curve = []
    for weight in np.asarray(grid, dtype=np.float64):
        score = _game_score(blend_predictions(gam, legacy, weight), y, gids)
        curve.append({"w_gam": float(weight), "brier_game": score})
    # Prefer the simpler GAM when scores are numerically tied.
    best = min(curve, key=lambda row: (row["brier_game"], -row["w_gam"]))
    return float(best["w_gam"]), curve


def paired_gate(gam, candidate, y, gids, bootstrap=5000, seed=83,
                min_improvement=0.0, clusters=None):
    result = wpexposure.paired_interval(candidate, gam, y, gids, clusters,
                                        bootstrap=bootstrap, seed=seed)
    delta, ci = result["candidate_minus_incumbent"], result["ci95"]
    result.update(passed=bool(ci is not None and ci[1] < -float(min_improvement)),
        candidate_minus_gam=delta, gam_minus_candidate=-delta,
        min_improvement=float(min_improvement),
        bootstrap_unit="game" if clusters is None else "series_date")
    return result


def _neutral_live_state(t_min):
    team_gold = 2500.0 + 1500.0 * float(t_min)
    return {
        "t_min": float(t_min),
        "gold_blue": team_gold, "gold_red": team_gold,
        "gold_diff_k": 0.0, "gold_diff_prev_k": 0.0,
        "gold_role": [0.0] * 5, "t_since_kill_min": 10.0,
        "has_hp": 1.0,
        "elo_oe": 0.0, "pelo_oe": 0.0, "elo_gg": 0.0,
        "form_diff": 0.0,
    }


def _shape_sweep(case, t_min):
    """Yield coherent live states ordered from red to blue advantage."""
    if case in {"gold", "gold_momentum", "cs", "kills",
                "dead_advantage", "hp_pool", "level", "team_prior"}:
        values = np.linspace(-4.0, 4.0, 9)
    elif case == "towers":
        maximum = max(1, min(4, int(float(t_min) // 4)))
        values = np.arange(-maximum, maximum + 1)
    elif case == "dragons":
        maximum = min(4, max(0, int((float(t_min) - 5.0) // 5.0) + 1))
        if maximum <= 0:
            return
        values = np.arange(-maximum, maximum + 1)
    elif case == "barons":
        maximum = min(3, max(0, int((float(t_min) - 20.0) // 6.0) + 1))
        if maximum <= 0:
            return
        values = np.arange(-maximum, maximum + 1)
    elif case == "inhibitors":
        if float(t_min) < 10.0:
            return
        values = np.arange(-3, 4)
    elif case == "elders":
        maximum = min(3, max(0, int((float(t_min) - 25.0) // 6.0) + 1))
        if maximum <= 0:
            return
        values = np.arange(-maximum, maximum + 1)
    elif case == "base_death_pressure":
        if float(t_min) < 15.0:
            return
        values = np.arange(-4, 5)
    elif case == "baron_active" and float(t_min) >= 20.0:
        values = np.asarray([-1.0, 0.0, 1.0])
    elif case == "elder_active" and float(t_min) >= 25.0:
        values = np.asarray([-1.0, 0.0, 1.0])
    elif case in {"baron_active", "elder_active"}:
        return
    else:
        raise ValueError("unknown shape-audit case %s" % case)
    for raw in values:
        value = float(raw)
        state = _neutral_live_state(t_min)
        if case == "gold":
            limit = max(1.0, min(8.0, 0.35 * float(t_min)))
            value = float(raw) / 4.0 * limit
            team_gold = state["gold_blue"]
            state.update({
                "gold_diff_k": value, "gold_diff_prev_k": value,
                "gold_blue": team_gold + 500.0 * value,
                "gold_red": team_gold - 500.0 * value,
                "gold_role": [value / 5.0] * 5,
            })
        elif case == "gold_momentum":
            state["gold_diff_prev_k"] = -value * min(1.0, t_min / 15.0)
        elif case == "cs":
            state["cs_diff_k"] = value * min(1.0, t_min / 20.0)
        elif case == "kills":
            state["kills"] = value * min(1.0, t_min / 10.0)
        elif case == "towers":
            state.update({"towers": value,
                          "towers_blue": max(value, 0.0),
                          "towers_red": max(-value, 0.0)})
        elif case == "dragons":
            state.update({"dragons": value,
                          "drag_blue": max(int(round(value)), 0),
                          "drag_red": max(int(round(-value)), 0)})
        elif case == "barons":
            state["barons"] = value
        elif case == "inhibitors":
            state.update({"inhibs": value,
                          "inhib_blue": max(value, 0.0),
                          "inhib_red": max(-value, 0.0)})
        elif case == "elders":
            state["elders"] = value
        elif case == "baron_active":
            state["baron_active"] = value
        elif case == "elder_active":
            state["elder_active"] = value
        elif case == "dead_advantage":
            state.update({"dead_red": max(value, 0.0),
                          "dead_blue": max(-value, 0.0)})
        elif case == "base_death_pressure":
            state.update({"dead_red": max(value, 0.0),
                          "dead_blue": max(-value, 0.0),
                          "towers_blue": 8.0 if value > 0.0 else 0.0,
                          "towers_red": 8.0 if value < 0.0 else 0.0})
        elif case == "hp_pool":
            state["hp_pool"] = value
        elif case == "level":
            state["lvl_k"] = value
        elif case == "team_prior":
            # Move every interchangeable pregame strength source together so
            # the audit remains meaningful when one source is clipped/missing.
            prior = value / 5.0
            state.update({"elo_oe": prior, "pelo_oe": prior,
                          "elo_gg": prior, "form_diff": prior})
        yield state


def audit_live_shape(gam_path=wpgam.MODEL_PATH,
                     legacy_path=wpx.LEGACY_LIVE_MODEL_PATH, w_gam=1.0,
                     tolerance=1e-8, predict_gam=None, predict_legacy=None):
    """Audit the actual live stack over ordered, physically coherent sweeps.

    The constrained GAM is monotone by construction.  A promoted blend must
    also be monotone after live feature construction, clipping, rounding, and
    interpolation with the unconstrained comparator.
    """
    weight = float(w_gam)
    if predict_gam is None:
        loaded = wpgam.load_model(gam_path)
        predict_gam = lambda state: wpgam.predict_live_model(
            loaded, state, rounded=False)["p_blue"]
    if predict_legacy is None and weight < 1.0:
        predict_legacy = lambda state: wpx._predict_live_legacy(
            state, path=legacy_path)["p_blue"]
    cases = ("gold", "gold_momentum", "cs", "kills", "towers",
             "dragons", "barons", "inhibitors", "elders",
             "baron_active", "elder_active", "dead_advantage",
             "base_death_pressure", "hp_pool", "level", "team_prior")
    failures = []
    comparisons = 0
    worst_step = 0.0
    for t_min in SHAPE_TIMES:
        for case in cases:
            probabilities = []
            for state in _shape_sweep(case, t_min):
                gam_p = float(predict_gam(state))
                if weight >= 1.0:
                    candidate = gam_p
                else:
                    legacy_p = float(predict_legacy(state))
                    candidate = float(blend_predictions(
                        [gam_p], [legacy_p], weight)[0])
                probabilities.append(candidate)
            probabilities = np.asarray(probabilities, dtype=np.float64)
            if not np.isfinite(probabilities).all():
                failures.append({"case": case, "t_min": float(t_min),
                                 "reason": "non_finite_prediction"})
                continue
            steps = np.diff(probabilities)
            comparisons += int(len(steps))
            if len(steps):
                worst_step = min(worst_step, float(steps.min()))
            bad = np.where(steps < -float(tolerance))[0]
            if len(bad):
                failures.append({
                    "case": case, "t_min": float(t_min),
                    "minimum_probability_step": float(steps[bad].min()),
                    "violations": int(len(bad)),
                })
    # Individual role changes, including early asymmetric/clipped states.
    # Uniform allocation sweeps alone cancel the old role-allocation terms.
    for t_min in SHAPE_TIMES:
        for lead in (-.35, 0., .35):
            base = _neutral_live_state(t_min)
            total = base["gold_blue"] + base["gold_red"]
            base.update(gold_blue=total * (1 + lead) / 2,
                        gold_red=total * (1 - lead) / 2,
                        gold_diff_k=total * lead / 1000,
                        gold_diff_prev_k=total * lead / 1000,
                        gold_role=[total * lead / 5000] * 5)
            for amount in (100., 1000.):
                for slot in range(10):
                    sign = 1 if slot < 5 else -1
                    changed = dict(base, gold_role=list(base["gold_role"]))
                    side = "gold_blue" if sign > 0 else "gold_red"
                    changed[side] += amount
                    changed["gold_diff_k"] += sign * amount / 1000
                    changed["gold_role"][slot % 5] += sign * amount / 1000
                    pair = np.array([predict_gam(s) for s in (base, changed)])
                    if weight < 1.0:
                        pair = blend_predictions(pair, [predict_legacy(s)
                                                        for s in (base, changed)], weight)
                    step = float(sign * (pair[1] - pair[0]))
                    comparisons += 1
                    worst_step = min(worst_step, step)
                    if not np.isfinite(pair).all() or step < -tolerance:
                        failures.append({"case": "player_gold", "t_min": t_min,
                                         "slot": slot, "added_gold": amount,
                                         "minimum_probability_step": step})
    return {
        "passed": not failures, "audited": True,
        "weight_w_gam": weight, "times_min": list(SHAPE_TIMES),
        "cases": len(cases) + 1, "comparisons": comparisons,
        "tolerance": float(tolerance), "worst_probability_step": worst_step,
        "failures": failures,
    }


def _apply_shape_gate(benchmark, gam_path, legacy_path):
    """Attach live shape audits and make them a mandatory promotion gate."""
    selected_weight = float(benchmark.get("selection", {}).get(
        "w_gam", benchmark.get("deployed_w_gam", 1.0)))
    artifact_audit = wpaudit.artifact_shape(wpgam.load_model(gam_path))
    if not artifact_audit["passed"]:
        raise RuntimeError("staged GAM failed feature audit: %s" % artifact_audit["failures"])
    gam_shape = audit_live_shape(gam_path, legacy_path, 1.0)
    if not gam_shape["passed"]:
        raise RuntimeError("staged constrained GAM failed live shape audit: %s" %
                           gam_shape["failures"])
    candidate_shape = (gam_shape if selected_weight >= 1.0 else
                       audit_live_shape(gam_path, legacy_path, selected_weight))
    benchmark.setdefault("gate", {})["gam_shape_audit"] = gam_shape
    benchmark["gate"]["artifact_shape_audit"] = artifact_audit
    benchmark["gate"]["shape_audit"] = candidate_shape
    benchmark["gate"]["shape_passed"] = bool(candidate_shape["passed"])
    outcome_passed = bool(benchmark["gate"].get(
        "outcome_gate_passed", benchmark["gate"].get("passed", False)))
    deployed = bool(outcome_passed and candidate_shape["passed"]
                    and selected_weight < 1.0)
    benchmark["deployed"] = deployed
    benchmark["deployed_w_gam"] = selected_weight if deployed else 1.0
    benchmark["gate"]["passed"] = deployed
    if outcome_passed and not candidate_shape["passed"]:
        benchmark["gate"]["rejection_reason"] = (
            "candidate failed live-contract monotonicity audit")
    return benchmark


def _row_game_index(d, first):
    gids = np.asarray(d["gid"])
    game_gid = gids[first]
    lookup = {int(g): i for i, g in enumerate(game_gid)}
    return np.fromiter((lookup[int(g)] for g in gids), dtype=np.int32,
                       count=len(gids))


def _gam_predict(d, model, mask):
    names = list(d["names"])
    prior = wpgam.prior_values_from_matrix(model, d["X"][mask], names,
                                            d["C"][mask])
    raw = np.column_stack([prior,
                           wpgam.state_values_from_matrix(d["X"][mask], names)])
    return wpgam.predict_state(model["state"], raw, d["t"][mask] / 60.0)


def _legacy_fit(d, game_mask, row_game):
    rows = np.where((np.asarray(d["seq"]) < 0) & game_mask[row_game])[0]
    return wpx.fit_legacy_arrays(
        d["X"], d["y"], d["gid"], d["C"], list(d["names"]),
        list(d["champ_names"]), rows=rows)


def _block_predictions(d, game_fit, game_score, row_game):
    gam = wpgam.fit_arrays(
        d["X"], d["y"], d["gid"], d["t"], d["seq"], d["C"],
        list(d["names"]), list(d["champ_names"]), train_games=game_fit,
        dates=d["date"] if "date" in d.files else None)
    legacy = _legacy_fit(d, game_fit, row_game)
    score_rows = (np.asarray(d["seq"]) < 0) & game_score[row_game]
    return {
        "gam": _gam_predict(d, gam, score_rows),
        "legacy": wpx.predict_legacy_arrays(
            legacy, d["X"][score_rows], d["C"][score_rows]),
        "y": np.asarray(d["y"])[score_rows],
        "gid": np.asarray(d["gid"])[score_rows],
        "t_min": np.asarray(d["t"])[score_rows] / 60.0,
        "legacy_converged": bool(legacy["converged"]),
    }


def _load_registry(path, game_dates, experiment="live_stack"):
    """Load the append-only holdout ledger, quarantining pre-existing games."""
    if os.path.exists(path):
        with open(path) as fh:
            registry = json.load(fh)
        if registry.get("kind") != "wpx_evaluation_registry_v1":
            raise ValueError("unsupported evaluation registry")
        registered = registry.get("experiment")
        if registered is not None and registered != experiment:
            raise ValueError("evaluation registry belongs to %s, not %s" %
                             (registered, experiment))
        if registered is None:
            registry["experiment"] = str(experiment)
            return registry, True
        return registry, False
    # Migration rule: this repository already used its current newest-date
    # block during v5-v7 development.  Treat every game present at adoption as
    # consumed rather than relabeling it as a fresh holdout.
    return {
        "kind": "wpx_evaluation_registry_v1",
        "experiment": str(experiment),
        "consumed_through": str(max(np.asarray(game_dates).astype(str).tolist())),
        "minimum_fresh_games": MIN_FRESH_GAMES,
        "history": [{"action": "migration_quarantine",
                     "reason": "all outcomes present when registry was introduced were already inspectable"}],
    }, True


def _write_registry(path, registry):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _development_split(gids, dates, registry_path, exposure_path=None):
    """Diagnostic runners cannot consume a new outcome via another ledger."""
    directory = os.path.dirname(os.path.abspath(registry_path))
    exposure_path = exposure_path or os.path.join(directory, "outcome_exposure.json")
    paths = {registry_path, *(os.path.join(directory, name) for name in wpexposure.LEGACY_NAMES)}
    inventory = wpexposure.migrate(gids, dates, path=exposure_path,
                                   legacy_paths=sorted(paths))
    fresh = wpexposure.fresh_mask(inventory, gids, dates)
    consumed = ~fresh
    train, test, _ = wpbench._date_blocks(dates, consumed)
    return train, test, fresh, inventory


def _row_clusters(d, mask):
    matches = d["match_id"][mask] if "match_id" in d.files else None
    return wpexposure.cluster_labels(d["gid"][mask], d["date"][mask], matches)


def run(dataset_path=None, output_path=RESULT_PATH, registry_path=REGISTRY_PATH,
        bootstrap=5000, min_improvement=0.0,
        minimum_fresh_games=MIN_FRESH_GAMES, exposure_path=None):
    dataset_path = dataset_path or os.path.join(wpgam.OUT_DIR, "states.npz")
    started = time.time()
    d = np.load(dataset_path, allow_pickle=True)
    first, diagnostic_train, split_method = wpgam._date_split(d)
    game_dates = np.asarray(d["date"])[first].astype(str)
    outer_train, outer_test, fresh_test, inventory = _development_split(
        np.asarray(d["gid"])[first], game_dates, registry_path, exposure_path)
    consumed_through = inventory["consumed_through"]
    # These fitted-component comparisons remain useful development evidence.
    # Only immutable, forward candidate-vs-incumbent forecasts can promote.
    fresh_eligible = False
    inner_train, validation, validation_cutoff = wpbench._date_blocks(
        game_dates, outer_train)
    row_game = _row_game_index(d, first)

    validation_pred = _block_predictions(
        d, inner_train, validation, row_game)
    weight, curve = select_weight(
        validation_pred["gam"], validation_pred["legacy"],
        validation_pred["y"], validation_pred["gid"])

    test_pred = _block_predictions(
        d, outer_train, outer_test, row_game)
    candidate = blend_predictions(
        test_pred["gam"], test_pred["legacy"], weight)
    gate = paired_gate(
        test_pred["gam"], candidate, test_pred["y"], test_pred["gid"],
        bootstrap=bootstrap, min_improvement=min_improvement,
        clusters=_row_clusters(d, (np.asarray(d["seq"]) < 0) & outer_test[row_game]))
    # A selected weight of one is the standalone GAM, not a deployed blend.
    gate["statistical_passed"] = bool(gate["passed"])
    gate["fresh_holdout_eligible"] = bool(fresh_eligible)
    gate["fresh_games_available"] = int(fresh_test.sum())
    gate["minimum_fresh_games"] = int(minimum_fresh_games)
    gate["consumed_through_before_run"] = consumed_through
    gate["outcome_gate_passed"] = bool(
        gate["passed"] and fresh_eligible and weight < 1.0
        and validation_pred["legacy_converged"]
        and test_pred["legacy_converged"])
    gate["passed"] = False  # Final promotion also requires the staged shape audit.
    result = {
        "kind": "wpx_live_stack_benchmark_v1",
        "dataset": os.path.basename(dataset_path),
        "provenance": {"dataset_sha256": _sha256(dataset_path),
                       "source_revision": _source_revision(),
                       "comparison": "refitted_components_diagnostic"},
        "split": {
            "method": split_method,
            "inner_train_games": int(inner_train.sum()),
            "validation_games": int(validation.sum()),
            "outer_train_games": int(outer_train.sum()),
            "test_games": int(outer_test.sum()),
            "validation_start": str(validation_cutoff),
            "test_start": str(np.sort(game_dates[outer_test])[0]),
            "holdout_status": ("fresh_forward_block" if fresh_eligible
                               else "diagnostic_reused_block"),
        },
        "contract": {
            "gam_kind": wpgam.MODEL_KIND,
            "legacy_kind": wpx.LEGACY_LIVE_CONTRACT,
            "causal_fixed_minutes": True,
            "game_balanced": True,
            "legacy_excluded_features": sorted(wpx.LEGACY_LIVE_EXCLUDED),
        },
        "selection": {"metric": "validation game-balanced Brier",
                      "w_gam": weight, "curve": curve},
        "test": {
            "gam": wpbench._summary(
                test_pred["gam"], test_pred["y"], test_pred["gid"],
                test_pred["t_min"], bootstrap=min(bootstrap, 2000)),
            "legacy": wpbench._summary(
                test_pred["legacy"], test_pred["y"], test_pred["gid"],
                test_pred["t_min"], bootstrap=min(bootstrap, 2000)),
            "candidate": wpbench._summary(
                candidate, test_pred["y"], test_pred["gid"],
                test_pred["t_min"], bootstrap=min(bootstrap, 2000)),
        },
        "gate": gate,
        "deployed_w_gam": 1.0,
        "deployed": False,
        "total_seconds": round(time.time() - started, 2),
    }
    # Legacy histories are preserved byte-for-byte; the canonical union was
    # persisted before fitting. Fresh dates were excluded from all masks.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    d.close()
    return result


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(benchmark, gam_path, legacy_path):
    deployed = bool(benchmark.get("deployed"))
    manifest = {
        "kind": "wpx_live_stack_v1", "deployed": deployed,
        "w_gam": float(benchmark.get("deployed_w_gam", 1.0)),
        "reason": ("paired chronological holdout and live shape gates passed" if deployed
                   else "blend rejected; standalone constrained GAM"),
        "components": {
            "gam_kind": wpgam.MODEL_KIND,
            "gam_sha256": _sha256(gam_path),
            "legacy_kind": wpx.LEGACY_LIVE_CONTRACT,
            "legacy_sha256": _sha256(legacy_path),
        },
        "benchmark": benchmark,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["stack_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def _source_revision():
    return util.source_revision(wpx.config.REPO_ROOT)


def _backup(path):
    if os.path.exists(path):
        shutil.copy2(path, path + ".previous")


def reseal_manifest(path=wpx.LIVE_STACK_PATH, gam_path=wpgam.MODEL_PATH,
                    legacy_path=wpx.LEGACY_LIVE_MODEL_PATH,
                    result_path=RESULT_PATH):
    """Re-audit and update source provenance after code-only changes."""
    with open(path) as fh:
        manifest = json.load(fh)
    if manifest.get("kind") != "wpx_live_stack_v1":
        raise ValueError("unsupported live stack manifest")
    expected = manifest.get("components", {})
    if (_sha256(gam_path) != expected.get("gam_sha256") or
            _sha256(legacy_path) != expected.get("legacy_sha256")):
        raise ValueError("refusing to reseal a manifest with changed components")
    benchmark = _apply_shape_gate(
        dict(manifest.get("benchmark", {})), gam_path, legacy_path)
    manifest["benchmark"] = benchmark
    manifest["deployed"] = bool(benchmark.get("deployed"))
    manifest["w_gam"] = float(benchmark.get("deployed_w_gam", 1.0))
    manifest["reason"] = (
        "paired chronological holdout and live shape gates passed"
        if manifest["deployed"] else
        "blend rejected; standalone constrained GAM")
    manifest["source_revision"] = _source_revision()
    unsigned = {k: v for k, v in manifest.items() if k != "stack_sha256"}
    manifest["stack_sha256"] = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result_tmp = result_path + ".tmp"
    with open(result_tmp, "w") as fh:
        json.dump(benchmark, fh, indent=2, sort_keys=True)
    os.replace(result_tmp, result_path)
    manifest_tmp = path + ".tmp"
    with open(manifest_tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    os.replace(manifest_tmp, path)
    return manifest


def _validate_benchmark_provenance(benchmark, dataset_path, gam_path, legacy_path):
    """A report about other fits/data/source cannot approve these artifacts."""
    if benchmark.get("kind") != "wpx_live_stack_benchmark_v1":
        raise ValueError("invalid precomputed live-stack benchmark")
    expected = {"dataset_sha256": _sha256(dataset_path),
                "gam_sha256": _sha256(gam_path),
                "legacy_sha256": _sha256(legacy_path),
                "source_revision": _source_revision()}
    got = benchmark.get("provenance", {})
    for key, value in expected.items():
        if got.get(key) != value:
            raise ValueError("precomputed benchmark provenance mismatch: %s" % key)


def _direct_evidence_valid(benchmark, candidate_path, incumbent_path, dataset_path):
    """Validate frozen, exact-artifact direct evidence, not a model-kind label.

    This deliberately does not accept the legacy blend-vs-refitted-GAM gate.
    Hash checks also prevent a result being reused after a calibration refit.
    """
    evidence = (benchmark or {}).get("direct_evidence", {})
    reason = "changed model requires frozen exact-hash candidate-versus-incumbent evidence"
    if evidence.get("kind") != "wpx_incumbent_comparison_v1":
        return False, reason
    plan = evidence.get("frozen_plan", {})
    plan_body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if not plan_body or evidence.get("plan_sha256") != wpexposure.digest(plan_body):
        return False, "direct comparison frozen plan hash mismatch"
    if not os.path.exists(incumbent_path):
        return False, "direct comparison has no incumbent artifact"
    sources = plan.get("inference_source")
    source_revision = _source_revision()
    if sources:
        required = {"wpgam.py", "wpx.py", "live.py", "shadow.py", "wpcandidate.py"}
        if not isinstance(sources, dict) or not required.issubset(sources):
            return False, "direct comparison inference source manifest incomplete"
        for name, expected_hash in sources.items():
            if os.path.basename(name) != name:
                return False, "direct comparison source path is not a module filename"
            path = os.path.join(os.path.dirname(__file__), name)
            if not os.path.isfile(path) or _sha256(path) != expected_hash:
                return False, "direct comparison inference source mismatch: %s" % name
        source_revision = wpexposure.digest(sources)
    expected = {"candidate_sha256": _sha256(candidate_path),
                "incumbent_sha256": _sha256(incumbent_path),
                "dataset_sha256": _sha256(dataset_path),
                "source_revision": source_revision}
    provenance = plan.get("provenance", {})
    for key, value in expected.items():
        if evidence.get(key) != value or provenance.get(key) != value:
            return False, "direct comparison provenance mismatch: %s" % key
    if (evidence.get("incumbent_stack_sha256") != provenance.get("incumbent_stack_sha256") or
            not evidence.get("incumbent_stack_sha256")):
        return False, "direct comparison frozen incumbent stack mismatch"
    contract = evidence.get("input_contract_sha256")
    if not contract or contract != provenance.get("input_contract_sha256"):
        return False, "direct comparison input-contract mismatch"
    rule = plan.get("rule", {})
    if (rule.get("metric") != "game_balanced_brier" or
            rule.get("bootstrap_unit") != evidence.get("bootstrap_unit") or
            rule.get("bootstrap_unit") not in {"series_date", "date"} or
            rule.get("endpoint") != "fixed_date" or
            not evidence.get("endpoint_complete") or not evidence.get("game_balanced")):
        return False, "direct comparison does not match a closed frozen evaluation rule"
    try:
        endpoint = datetime.fromisoformat(str(rule["end_date"]).replace("Z", "+00:00"))
        endpoint = endpoint.replace(tzinfo=endpoint.tzinfo or timezone.utc)
    except (KeyError, ValueError):
        return False, "direct comparison has no valid fixed endpoint"
    if datetime.now(timezone.utc) < endpoint:
        return False, "direct comparison endpoint is still open"
    minimum = int(rule.get("minimum_games", 0))
    if minimum < 2 or int(evidence.get("games", 0)) < minimum or int(evidence.get("clusters", 0)) < 2:
        return False, "direct comparison has insufficient games or clusters"
    ci = evidence.get("ci95")
    min_gain = float(rule.get("min_improvement", 0.0))
    if (not isinstance(ci, list) or len(ci) != 2 or not np.isfinite(ci).all() or
            ci[0] > ci[1] or ci[1] >= -min_gain):
        return False, "direct comparison misses frozen improvement threshold"
    if (float(rule.get("precision_target", 0)) <= 0 or
            (ci[1] - ci[0]) / 2 > float(rule["precision_target"])):
        return False, "direct comparison misses frozen precision target"
    limit = float(rule.get("max_logloss_regression", 0.0))
    if float(evidence.get("candidate_minus_incumbent_logloss", float("inf"))) > limit:
        return False, "direct comparison misses supporting log-loss gate"
    calibration_limit = float(rule.get("max_calibration_error_regression", 0.0))
    if float(evidence.get("candidate_minus_incumbent_calibration_error", float("inf"))) > calibration_limit:
        return False, "direct comparison misses supporting calibration gate"
    return True, "frozen exact-hash direct comparison gates passed"


def _require_consumed_training_inputs(dataset_path, exposure_path=wpexposure.PATH):
    """Read identities only; reject unexposed labels before any model fitting.

    A missing shared inventory is empty, not implicit permission to consume an
    arbitrary supplied dataset. Adoption/migration must happen explicitly before
    this training entry point. Neither shared nor legacy ledgers are rewritten.
    """
    with np.load(dataset_path, allow_pickle=False) as archive:
        if not {"gid", "date"}.issubset(archive.files):
            raise ValueError("refresh requires complete game IDs and dates before fitting")
        gids = np.asarray(archive["gid"])
        dates = np.asarray(archive["date"]).astype(str)
    if (gids.ndim != 1 or dates.ndim != 1 or len(gids) != len(dates) or not len(gids)
            or not np.issubdtype(gids.dtype, np.integer) or np.any(gids <= 0)):
        raise ValueError("refresh requires aligned positive integer game IDs and dates")
    for date in np.unique(dates):
        try:
            parsed = datetime.fromisoformat(str(date))
        except ValueError as exc:
            raise ValueError("refresh requires complete ISO game dates") from exc
        if len(date) != 10 or parsed.date().isoformat() != date:
            raise ValueError("refresh requires complete ISO game dates")
    unique, first = np.unique(gids, return_index=True)
    lookup = dict(zip(unique.tolist(), dates[first].tolist()))
    if any(lookup[int(gid)] != date for gid, date in zip(gids, dates)):
        raise ValueError("refresh cannot assign multiple dates to one game")
    inventory = wpexposure.load(exposure_path)
    game_dates = dates[first]
    if wpexposure.fresh_mask(inventory, unique, game_dates).any():
        raise ValueError("refresh accepts exposed development outcomes only; filter the dataset before fitting")
    return {"games": len(unique), "last_game_date": str(max(game_dates)),
            "exposure_path": os.path.abspath(exposure_path), "fresh_games": 0}


def refresh(dataset_path=None, gam_path=wpgam.MODEL_PATH,
            legacy_path=wpx.LEGACY_LIVE_MODEL_PATH,
            stack_path=wpx.LIVE_STACK_PATH, result_path=RESULT_PATH,
            bootstrap=5000, min_improvement=0.0, benchmark_result=None,
            exposure_path=wpexposure.PATH):
    """Fit a candidate; stage it unless exact frozen direct evidence exists.

    Shape checks are required even for staging. This function never scores a
    fresh outcome block. Approved exact candidates are promoted without an
    additional refit; the manifest is replaced last and previous files remain.
    """
    dataset_path = dataset_path or os.path.join(wpgam.OUT_DIR, "states.npz")
    training_exposure = _require_consumed_training_inputs(dataset_path, exposure_path)
    incumbent_kind = (wpgam.load_model(gam_path)["kind"]
                      if os.path.exists(gam_path) else None)
    incumbent_sha256 = _sha256(gam_path) if incumbent_kind is not None else None
    os.makedirs(os.path.dirname(gam_path), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wpx-stage-",
                                     dir=os.path.dirname(gam_path)) as stage:
        stage_gam = os.path.join(stage, "model_live_gam.npz")
        stage_legacy = os.path.join(stage, "model_live.npz")
        stage_result = os.path.join(stage, "live_stack_benchmark.json")
        stage_stack = os.path.join(stage, "live_stack.json")

        wpgam.fit_full(dataset_path=dataset_path, model_path=stage_gam)
        # Artifact load + finite smoke prediction are mandatory before the
        # expensive gate, catching corrupt or incompatible candidates early.
        staged_gam = wpgam.load_model(stage_gam)
        if staged_gam["state"].get("optimizer_success") is not True:
            raise RuntimeError("staged GAM optimizer failed: %s" %
                               staged_gam["state"].get("optimizer_message"))
        smoke = {"t_min": 20.0, "gold_blue": 35000, "gold_red": 34000,
                 "gold_diff_k": 1.0, "gold_diff_prev_k": 0.5}
        if not np.isfinite(wpgam.predict_live(smoke, path=stage_gam)["p_blue"]):
            raise FloatingPointError("staged GAM smoke prediction is not finite")
        direct_valid, direct_reason = _direct_evidence_valid(
            benchmark_result, stage_gam, gam_path, dataset_path)
        if direct_valid:
            with open(stack_path) as fh:
                incumbent_stack = json.load(fh)
            evidence = benchmark_result["direct_evidence"]
            if (not evidence.get("incumbent_stack_sha256") or
                    evidence["incumbent_stack_sha256"] != incumbent_stack.get("stack_sha256")):
                direct_valid, direct_reason = False, "direct comparison incumbent stack hash mismatch"
        if not direct_valid:
            # Every changed fit/calibrator needs evidence against the exact
            # incumbent. A same-kind string does not make a refresh exempt.
            feature_audit = wpaudit.artifact_shape(staged_gam)
            shape_audit = audit_live_shape(stage_gam, stage_legacy, 1.0)
            if not feature_audit["passed"] or not shape_audit["passed"]:
                raise RuntimeError("new GAM contract failed staged input audits")
            candidate_path = gam_path + ".candidate.npz"
            os.replace(stage_gam, candidate_path)
            status = {"kind": "wpx_model_candidate_v2", "deployed": False,
                      "incumbent_kind": incumbent_kind, "incumbent_sha256": incumbent_sha256,
                      "candidate_kind": wpgam.MODEL_KIND,
                      "candidate_path": candidate_path, "candidate_sha256": _sha256(candidate_path),
                      "feature_audit": feature_audit, "shape_audit": shape_audit,
                      "training_exposure": training_exposure,
                      "reason": direct_reason}
            with open(candidate_path + ".json", "w") as fh:
                json.dump(status, fh, indent=2, sort_keys=True)
            wpgam.log.warning("GAM refresh staged only: %s", status["reason"])
            return status

        if benchmark_result.get("kind") == "wpx_live_stack_benchmark_v1":
            _validate_benchmark_provenance(benchmark_result, dataset_path, stage_gam, legacy_path)
        # Direct evidence concerns this exact standalone challenger. A newly
        # refitted legacy blend has a different forecast and needs its own
        # frozen comparison, so preserve the incumbent legacy component.
        return promote_candidate(stage_gam, benchmark_result["direct_evidence"],
            dataset_path, gam_path=gam_path, legacy_path=legacy_path,
            stack_path=stack_path, result_path=result_path)


def promote_candidate(candidate_path, direct_evidence, dataset_path,
                      gam_path=wpgam.MODEL_PATH, legacy_path=wpx.LEGACY_LIVE_MODEL_PATH,
                      stack_path=wpx.LIVE_STACK_PATH, result_path=RESULT_PATH):
    """Promote the exact frozen candidate without refitting after evaluation.

    The incumbent comparison must identify the live stack as well as the GAM;
    otherwise a comparison to a blend could be mistaken for a standalone GAM.
    """
    benchmark = {"direct_evidence": direct_evidence}
    valid, reason = _direct_evidence_valid(benchmark, candidate_path, gam_path, dataset_path)
    if not valid:
        raise ValueError(reason)
    with open(stack_path) as fh:
        incumbent_stack = json.load(fh)
    expected_stack = direct_evidence.get("incumbent_stack_sha256")
    if not expected_stack or incumbent_stack.get("stack_sha256") != expected_stack:
        raise ValueError("direct comparison incumbent stack hash mismatch")
    if _sha256(gam_path) != incumbent_stack.get("components", {}).get("gam_sha256"):
        raise ValueError("incumbent stack component hash mismatch")
    model = wpgam.load_model(candidate_path)
    if model.get("meta", {}).get("staged_only") or model.get("meta", {}).get("production_adapter_required"):
        raise ValueError("candidate requires an unsupported production adapter")
    feature_audit = wpaudit.artifact_shape(model)
    shape_audit = audit_live_shape(candidate_path, legacy_path, 1.0)
    if not feature_audit["passed"] or not shape_audit["passed"]:
        raise ValueError("frozen candidate failed current inference shape audits")
    result = {"kind": "wpx_direct_candidate_promotion_v1", "deployed": False,
        "gam_promoted": True, "deployed_w_gam": 1.0, "direct_evidence": direct_evidence,
        "feature_audit": feature_audit, "shape_audit": shape_audit}
    manifest = build_manifest(result, candidate_path, legacy_path)
    manifest.update(reason="frozen candidate passed direct incumbent comparison and shape audits",
                    source_revision=_source_revision(), dataset_sha256=_sha256(dataset_path))
    unsigned = {k: v for k, v in manifest.items() if k != "stack_sha256"}
    manifest["stack_sha256"] = wpexposure.digest(unsigned)
    with tempfile.TemporaryDirectory(prefix="wpx-promote-", dir=os.path.dirname(gam_path)) as stage:
        staged_gam = os.path.join(stage, "model.npz")
        shutil.copy2(candidate_path, staged_gam)
        staged_manifest, staged_result = os.path.join(stage, "stack.json"), os.path.join(stage, "result.json")
        for path, value in ((staged_manifest, manifest), (staged_result, result)):
            with open(path, "w") as fh:
                json.dump(value, fh, indent=2, sort_keys=True)
        for path in (gam_path, stack_path, result_path):
            _backup(path)
        os.replace(staged_gam, gam_path)
        os.replace(staged_result, result_path)
        os.replace(staged_manifest, stack_path)
    return manifest


def report(result):
    split = result["split"]
    print("live-stack gate: %d validation / %d outer-test games" %
          (split["validation_games"], split["test_games"]))
    for name in ("gam", "legacy", "candidate"):
        m = result["test"][name]
        print("  %-10s Brier(game)=%.6f logloss=%.6f" %
              (name, m["brier_game"], m["logloss_game"]))
    gate = result["gate"]
    if result["deployed"]:
        decision = "DEPLOY BLEND"
    elif gate.get("outcome_gate_passed") and "shape_audit" not in gate:
        decision = "OUTCOME GATE PASSED; STAGED SHAPE AUDIT REQUIRED"
    else:
        decision = "GAM ONLY"
    print("  selected w_gam=%.2f delta=%+.6f CI=%s => %s" %
          (result["selection"]["w_gam"], gate["candidate_minus_gam"],
           gate["ci95"], decision))
