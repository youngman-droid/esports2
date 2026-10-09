"""Historical-odds distillation with market-free live inference.

Past Polymarket/Kalshi probabilities supervise a second constrained state
surface.  A validation block chooses how much of that historical-market
teacher to mix with the outcome GAM. The shared exposure inventory restricts
these diagnostics to consumed history; deployment requires separate frozen
candidate-versus-incumbent forward forecasts. At live inference
both probabilities are functions only of team/draft/game-state inputs; no
quote from the current match is accepted.
"""
import hashlib
import json
import logging
import os
import time

import numpy as np

from . import wpbench, wpgam, wpexposure


log = logging.getLogger("wphist")
KIND = "historical_odds_distill_v1"
ARTIFACT_PATH = os.path.join(wpgam.OUT_DIR, "historical_blend.npz")
RESULT_PATH = os.path.join(wpgam.OUT_DIR, "historical_blend.json")
REGISTRY_PATH = os.path.join(wpgam.OUT_DIR,
                             "historical_evaluation_registry.json")
SPECS = (
    {"l2": 12.0, "smooth": 35.0},
    {"l2": 24.0, "smooth": 70.0},
    {"l2": 48.0, "smooth": 140.0},
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def _historical_target(d):
    """Mean available historical exchange probability for every stored row."""
    pm = np.asarray(d["pm"], dtype=np.float64)
    ks = np.asarray(d["ks"], dtype=np.float64)
    pm_ok, ks_ok = np.isfinite(pm), np.isfinite(ks)
    count = pm_ok.astype(np.int8) + ks_ok.astype(np.int8)
    target = np.full(len(pm), np.nan, dtype=np.float64)
    present = count > 0
    target[present] = (np.where(pm_ok, pm, 0.0)[present]
                       + np.where(ks_ok, ks, 0.0)[present]) / count[present]
    return target, pm_ok, ks_ok


def _raw_rows(d, model, mask):
    """Independent production features and base probabilities for selected rows."""
    names = list(d["names"])
    raw = np.column_stack([
        wpgam.prior_values_from_matrix(model, d["X"][mask], names, d["C"][mask]),
        wpgam.state_values_from_matrix(d["X"][mask], names),
    ])
    t_min = np.asarray(d["t"][mask], dtype=np.float64) / 60.0
    base_p = wpgam.predict_state(model["state"], raw, t_min)
    return raw, t_min, base_p


def _fit_teacher(d, base_model, mask, target, spec):
    raw, t_min, _ = _raw_rows(d, base_model, mask)
    return wpgam.fit_state_model(
        raw, target[mask], d["gid"][mask], t_min,
        l2=spec["l2"], smooth=spec["smooth"])


def _market_mse(pred, target, gids):
    loss = (np.asarray(pred) - np.asarray(target)) ** 2
    _, per_game = wpbench._per_game(loss, gids)
    return {"state": float(loss.mean()), "game": float(per_game.mean())}


def _fit_mix_weight(model_p, teacher_p, y, gids):
    """Outcome-selected log-odds interpolation, constrained to [model, teacher]."""
    from scipy.optimize import minimize_scalar

    lm, lt = _logit(model_p), _logit(teacher_p)
    y, gids = np.asarray(y, dtype=np.float64), np.asarray(gids)
    weights = wpgam._game_balanced_weights(gids)

    def objective(alpha):
        p = wpgam._sigmoid(lm + alpha * (lt - lm))
        return float(np.sum(weights * (p - y) ** 2))

    fit = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded",
                          options={"xatol": 1e-8})
    return float(np.clip(fit.x, 0.0, 1.0))


def _mix(model_p, teacher_p, alpha):
    return wpgam._sigmoid(
        _logit(model_p) + float(alpha) * (_logit(teacher_p) - _logit(model_p)))


def _paired_gate(model_p, blend_p, y, gids, bootstrap=2000, seed=173, clusters=None):
    result = wpexposure.paired_interval(blend_p, model_p, y, gids, clusters,
                                        bootstrap=bootstrap, seed=seed)
    result["blend_minus_model"] = result["candidate_minus_incumbent"]
    result["bootstrap_unit"] = "game" if clusters is None else "series_date"
    return result


def _save_artifact(teacher, alpha, deployed, meta, path=ARTIFACT_PATH,
                   base_path=wpgam.MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        kind=np.asarray(KIND),
        base_model_kind=np.asarray(wpgam.MODEL_KIND),
        base_model_sha256=np.asarray(_sha256(base_path)),
        deployed=np.asarray(bool(deployed)),
        alpha=np.asarray(float(alpha)),
        feature_names=teacher["feature_names"],
        mean=teacher["mean"], std=teacher["std"],
        lo=teacher["lo"], hi=teacher["hi"],
        theta=teacher["theta"], knots=teacher["knots"],
        meta=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    return path


def load_artifact(path=ARTIFACT_PATH):
    with np.load(path, allow_pickle=False) as d:
        kind = str(d["kind"].item())
        if kind != KIND:
            raise ValueError("unsupported historical blend kind %s" % kind)
        base_kind = str(d["base_model_kind"].item())
        if base_kind not in wpgam.SUPPORTED_MODEL_KINDS:
            raise ValueError("historical blend base-model contract mismatch")
        return {
            "kind": kind,
            "base_model_kind": base_kind,
            "base_model_sha256": str(d["base_model_sha256"].item()),
            "deployed": bool(d["deployed"].item()),
            "alpha": float(d["alpha"]),
            "teacher": {
                "feature_names": d["feature_names"],
                "mean": d["mean"], "std": d["std"],
                "lo": d["lo"], "hi": d["hi"],
                "theta": d["theta"], "knots": d["knots"],
                "cal_intercept": 0.0, "cal_slope": 1.0,
            },
            "meta": json.loads(str(d["meta"].item())),
        }


def status(path=ARTIFACT_PATH):
    if not os.path.exists(path):
        return {"available": False, "deployed": False}
    artifact = load_artifact(path)
    return {"available": True, "deployed": artifact["deployed"],
            "kind": artifact["kind"], "alpha": artifact["alpha"],
            "meta": artifact["meta"]}


def _live_raw(base, state, blue_champs, red_champs):
    pre = base["pregame"]
    C, unknown = wpgam._live_champ_row(
        pre["champ_names"], blue_champs, red_champs)
    pre_raw = wpgam.pregame_values_from_live(state)
    _, team, champ = wpgam.predict_pregame(pre, pre_raw, C)
    champ_state = float(wpgam.champ_state_scores(base["champ_state"]["beta"], C)[0])
    raw = np.concatenate([
        [pre["intercept"] + float(team[0]), float(champ[0]), champ_state],
        wpgam.state_values_from_live(
            state, list(base["state"]["feature_names"])[len(wpgam.PRIOR_INPUTS):]),
    ])[None, :]
    return raw, unknown


def predict_live(model_p, state, blue_champs=(), red_champs=(),
                 path=ARTIFACT_PATH, base_path=wpgam.MODEL_PATH):
    """Return a deployed historical blend without accepting a live quote."""
    from . import wpcombined_prod
    if state.get("model_kind") == wpcombined_prod.MODEL_KIND or (base_path == wpgam.MODEL_PATH and wpcombined_prod.load_active() is not None):
        raise ValueError("historical blend is not calibrated for the production combination")
    artifact = load_artifact(path)
    if not artifact["deployed"]:
        return None
    actual_sha = _sha256(base_path)
    if actual_sha != artifact["base_model_sha256"]:
        raise ValueError("historical blend is stale for the current base model")
    base = wpgam.load_model(base_path)
    if base["kind"] != artifact["base_model_kind"]:
        raise ValueError("historical blend base-model contract mismatch")
    raw, unknown = _live_raw(base, state, blue_champs, red_champs)
    t_min = float(state.get("t_min", 0.0) or 0.0)
    teacher_p = float(wpgam.predict_state(
        artifact["teacher"], raw, [t_min])[0])
    p = float(_mix([float(model_p)], [teacher_p], artifact["alpha"])[0])
    return {
        "p_blue": p,
        "model_p": float(model_p),
        "historical_teacher_p": teacher_p,
        "alpha": artifact["alpha"],
        "source": "model+historical-odds",
        "kind": artifact["kind"],
        "uses_live_odds": False,
        "unknown_champions": unknown,
    }


def run(dataset_path=None, output_path=RESULT_PATH, artifact_path=ARTIFACT_PATH,
        base_path=wpgam.MODEL_PATH, bootstrap=2000,
        registry_path=REGISTRY_PATH, exposure_path=None):
    base_full = wpgam.load_model(base_path)
    if (base_full["kind"] != wpgam.MODEL_KIND or
            list(base_full["state"]["feature_names"]) !=
            wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES):
        raise ValueError(
            "current source can train a historical teacher only for a %s base "
            "with the current feature contract; base artifact is %s" %
            (wpgam.MODEL_KIND, base_full["kind"]))
    dataset_path = dataset_path or os.path.join(wpgam.OUT_DIR, "states.npz")
    started = time.time()
    d = np.load(dataset_path, allow_pickle=True)
    first, diagnostic_train, split_method = wpgam._date_split(d)
    game_gid = np.asarray(d["gid"])[first]
    game_dates = np.asarray(d["date"])[first].astype(str)
    from . import wpdeploy
    outer_train, outer_test, fresh_test, inventory = wpdeploy._development_split(
        game_gid, game_dates, registry_path, exposure_path)
    fresh_eligible = False  # new frozen candidates are evaluated prospectively
    inner_train, validation, validation_cutoff = wpbench._date_blocks(
        game_dates, outer_train)
    gid_index = {int(g): i for i, g in enumerate(game_gid)}
    row_game = np.fromiter(
        (gid_index[int(g)] for g in d["gid"]), dtype=np.int32,
        count=len(d["gid"]))
    fixed = np.asarray(d["seq"]) < 0
    event = ~fixed
    target, pm_ok, ks_ok = _historical_target(d)
    quoted = np.isfinite(target)

    log.info("historical blend: fitting causal inner/outer base models")
    inner_model = wpgam.fit_arrays(
        d["X"], d["y"], d["gid"], d["t"], d["seq"], d["C"],
        list(d["names"]), list(d["champ_names"]), train_games=inner_train,
        dates=d["date"])
    outer_model = wpgam.fit_arrays(
        d["X"], d["y"], d["gid"], d["t"], d["seq"], d["C"],
        list(d["names"]), list(d["champ_names"]), train_games=outer_train,
        dates=d["date"])

    inner_quotes = event & quoted & inner_train[row_game]
    validation_quotes = event & quoted & validation[row_game]
    candidates = []
    best_spec = None
    best_score = np.inf
    validation_raw, validation_quote_t, _ = _raw_rows(
        d, inner_model, validation_quotes)
    for spec in SPECS:
        teacher = _fit_teacher(d, inner_model, inner_quotes, target, spec)
        pred = wpgam.predict_state(teacher, validation_raw, validation_quote_t)
        metric = _market_mse(pred, target[validation_quotes],
                             d["gid"][validation_quotes])
        candidates.append({"spec": spec, "historical_market_mse": metric})
        if metric["game"] < best_score:
            best_score, best_spec = metric["game"], dict(spec)

    validation_fixed = fixed & validation[row_game]
    val_raw, val_t, val_model = _raw_rows(d, inner_model, validation_fixed)
    inner_teacher = _fit_teacher(
        d, inner_model, inner_quotes, target, best_spec)
    val_teacher = wpgam.predict_state(inner_teacher, val_raw, val_t)
    alpha = _fit_mix_weight(
        val_model, val_teacher, d["y"][validation_fixed],
        d["gid"][validation_fixed])

    outer_quotes = event & quoted & outer_train[row_game]
    outer_teacher = _fit_teacher(
        d, outer_model, outer_quotes, target, best_spec)
    test_fixed = fixed & outer_test[row_game]
    test_raw, test_t, test_model = _raw_rows(d, outer_model, test_fixed)
    test_teacher = wpgam.predict_state(outer_teacher, test_raw, test_t)
    test_blend = _mix(test_model, test_teacher, alpha)
    y_test, gid_test = d["y"][test_fixed], d["gid"][test_fixed]
    model_summary = wpbench._summary(
        test_model, y_test, gid_test, test_t, bootstrap=bootstrap, seed=181)
    teacher_summary = wpbench._summary(
        test_teacher, y_test, gid_test, test_t, bootstrap=bootstrap, seed=183)
    blend_summary = wpbench._summary(
        test_blend, y_test, gid_test, test_t, bootstrap=bootstrap, seed=185)
    paired = _paired_gate(
        test_model, test_blend, y_test, gid_test,
        bootstrap=max(bootstrap, 2000), clusters=wpdeploy._row_clusters(d, test_fixed))
    statistical_passed = bool(
        blend_summary["brier_game"] < model_summary["brier_game"]
        and alpha > 1e-6 and paired["ci95"] is not None and paired["ci95"][1] < 0.0)
    deployed = bool(statistical_passed and fresh_eligible)

    # Refit only on information available for actual deployment.  The teacher
    # sees all stored historical quotes; live inference still sees none.
    all_quotes = event & quoted & (~fresh_test)[row_game]
    full_teacher = _fit_teacher(d, base_full, all_quotes, target, best_spec)
    report = {
        "kind": KIND,
        "dataset": os.path.basename(dataset_path),
        "split_method": split_method,
        "split": {
            "inner_train_games": int(inner_train.sum()),
            "validation_games": int(validation.sum()),
            "outer_train_games": int(outer_train.sum()),
            "test_games": int(outer_test.sum()),
            "validation_start": str(validation_cutoff),
            "test_start": str(np.sort(game_dates[outer_test])[0]),
            "holdout_status": ("fresh_forward_block" if fresh_eligible
                               else "diagnostic_reused_block"),
        },
        "historical_quotes": {
            "rows": int(all_quotes.sum()),
            "games": int(len(np.unique(d["gid"][all_quotes]))),
            "polymarket_rows": int((event & pm_ok).sum()),
            "kalshi_rows": int((event & ks_ok).sum()),
        },
        "selection": {"candidates": candidates, "selected": best_spec,
                      "outcome_mix_alpha": alpha},
        "test": {"model": model_summary, "historical_teacher": teacher_summary,
                 "historical_blend": blend_summary, "paired": paired},
        "deployed": deployed,
        "deployment_rule": "diagnostic only; immutable challenger-versus-incumbent forward evidence required",
        "gate": {"statistical_passed": statistical_passed,
                 "fresh_holdout_eligible": fresh_eligible,
                 "fresh_games_available": int(fresh_test.sum()),
                 "consumed_through_before_run": inventory["consumed_through"]},
        "uses_live_odds": False,
        "seconds": round(time.time() - started, 2),
    }
    meta = {
        "dataset": report["dataset"], "fitted_at": int(time.time()),
        "historical_quotes": report["historical_quotes"],
        "selection": report["selection"], "test": report["test"],
        "deployment_rule": report["deployment_rule"],
        "uses_live_odds": False,
    }
    # Canonical exposure union is persisted before fitting. These diagnostics
    # never advance either legacy registry or expose an unreserved fresh block.
    _save_artifact(full_teacher, alpha, deployed, meta, artifact_path, base_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    d.close()
    return report


def report(result):
    test = result["test"]
    print("historical-odds distillation: %d quote rows / %d games; alpha=%.4f" %
          (result["historical_quotes"]["rows"],
           result["historical_quotes"]["games"],
           result["selection"]["outcome_mix_alpha"]))
    for name in ("model", "historical_teacher", "historical_blend"):
        row = test[name]
        print("  %-20s Brier(game)=%.6f  Brier(state)=%.6f" %
              (name, row["brier_game"], row["brier_state"]))
    paired = test["paired"]
    print("  blend-model=%+.6f  95%% CI %+.6f..%+.6f" %
          (paired["blend_minus_model"], paired["ci95"][0], paired["ci95"][1]))
    print("  deployment:", "ENABLED" if result["deployed"] else "REJECTED")
