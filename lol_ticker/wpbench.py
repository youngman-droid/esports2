"""Fair model-family benchmark for the causal win-probability dataset.

The outer test set is the newest 20% of games by date.  Hyperparameters and
ensemble weights are selected on the newest block inside the remaining 80%,
so the outer test is not used for tuning.  Every state model sees the same
fixed-minute rows, the same stacked pregame inputs, and game-balanced weights.
"""
import json
import logging
import os
import time

import numpy as np

from . import wpgam, wpexposure


log = logging.getLogger("wpbench")
RESULT_PATH = os.path.join(wpgam.OUT_DIR, "method_benchmark.json")
FAMILIES = (
    "ridge_logit",
    "unconstrained_gam",
    "constrained_gam",
    "hist_boost",
    "monotone_hist_boost",
)


def _game_balanced_weights(gids):
    return wpgam._game_balanced_weights(np.asarray(gids))


def _date_blocks(dates, outer_train, validation_fraction=0.2):
    """Split the outer training games again, keeping complete dates together."""
    dates = np.asarray(dates).astype(str)
    candidates = dates[outer_train]
    n_validation = max(1, int(round(len(candidates) * validation_fraction)))
    cutoff = np.sort(candidates)[-n_validation]
    inner_train = outer_train & (dates < cutoff)
    validation = outer_train & ~inner_train
    if not inner_train.any() or not validation.any():
        raise ValueError("could not construct chronological validation block")
    return inner_train, validation, cutoff


def _load_base(dataset_path):
    with np.load(dataset_path, allow_pickle=True) as d:
        first, outer_train, split_method = wpgam._date_split(d)
        all_gid = np.asarray(d["gid"])
        game_gid = all_gid[first]
        game_dates = np.asarray(d["date"])[first].astype(str)
        inner_train, validation, validation_cutoff = _date_blocks(game_dates, outer_train)
        fixed = np.asarray(d["seq"]) < 0
        fixed_gid = all_gid[fixed]
        gid_index = {int(g): i for i, g in enumerate(game_gid)}
        row_game = np.fromiter((gid_index[int(g)] for g in fixed_gid), dtype=np.int32,
                               count=len(fixed_gid))
        base = {
            "dataset_path": dataset_path,
            "split_method": split_method,
            "game_gid": game_gid,
            "game_dates": game_dates,
            "game_y": np.asarray(d["y"])[first].astype(np.float64),
            "game_C": np.asarray(d["C"])[first],
            "game_pre": wpgam.pregame_values_from_matrix(
                np.asarray(d["X"])[first], list(d["names"])),
            "champ_names": list(d["champ_names"]),
            "row_game": row_game,
            "gid": fixed_gid,
            "y": np.asarray(d["y"])[fixed].astype(np.float64),
            "t_min": np.asarray(d["t"])[fixed].astype(np.float64) / 60.0,
            "state": wpgam.state_values_from_matrix(
                np.asarray(d["X"])[fixed], list(d["names"])),
            "names": list(d["names"]),
            # full-row copies for the champion-state channel: its coefficient
            # fit uses event-anchored rows too when CHAMP_STATE_ALL_ROWS is set
            "raw_X": np.asarray(d["X"]),
            "raw_C": np.asarray(d["C"]),
            "raw_y": np.asarray(d["y"]).astype(np.float64),
            "raw_gid": all_gid,
            "raw_fixed": fixed,
            "outer_train": outer_train,
            "inner_train": inner_train,
            "validation": validation,
            "validation_cutoff": validation_cutoff,
        }
    return base


def _stacked_arrays(base, game_train):
    """Create leak-free stacked priors for one training-game mask."""
    game_train = np.asarray(game_train, dtype=bool)
    pre = wpgam.fit_pregame(
        base["game_pre"][game_train], base["game_C"][game_train],
        base["game_y"][game_train], base["champ_names"])
    team_oof, champ_oof = wpgam.oof_pregame(
        base["game_pre"][game_train], base["game_C"][game_train],
        base["game_y"][game_train], base["game_gid"][game_train],
        base["champ_names"], k=5, dates=base["game_dates"][game_train])
    prior = np.empty((len(game_train), 3), dtype=np.float64)
    prior[game_train, 0] = team_oof
    prior[game_train, 1] = champ_oof
    row_train = game_train[base["row_game"]]
    train_gids = base["game_gid"][game_train]
    train_set = set(train_gids.tolist())
    n_champs = len(base["champ_names"])
    in_train_all = np.asarray([g in train_set for g in base["raw_gid"]])
    cs_rows = np.where(in_train_all if wpgam.CHAMP_STATE_ALL_ROWS
                       else (in_train_all & base["raw_fixed"]))[0]
    cs_beta = wpgam.fit_champ_state(
        base["raw_X"], base["names"], base["raw_C"], base["raw_y"],
        cs_rows, n_champs)
    cs_by_gid = wpgam.oof_champ_state(
        base["raw_X"], base["names"], base["raw_C"], base["raw_y"],
        base["raw_gid"], cs_rows, train_gids, n_champs,
        k=wpgam.CHAMP_STATE_FOLDS, dates=base["game_dates"][game_train])
    prior[game_train, 2] = np.asarray([cs_by_gid[g] for g in train_gids])
    other = ~game_train
    if other.any():
        _, team, champ = wpgam.predict_pregame(
            pre, base["game_pre"][other], base["game_C"][other])
        prior[other, 0] = pre["intercept"] + team
        prior[other, 1] = champ
        prior[other, 2] = wpgam.champ_state_scores(cs_beta, base["game_C"][other])
    raw = np.column_stack([prior[base["row_game"]], base["state"]])
    return {
        "raw": raw,
        "train": row_train,
        "pregame": pre,
        "champ_state_beta": cs_beta,
    }


def _ridge_design_fit(raw, t_min):
    names = wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES
    mean, std, lo, hi = wpgam._scale_fit(
        raw, feature_names=names if raw.shape[1] == len(names) else None)
    z = wpgam._scale_apply(raw, mean, std, lo, hi)
    # Drop the first time-basis column because the classifier has an intercept.
    design = np.column_stack([wpgam.time_basis(t_min)[:, 1:], z])
    return design, (mean, std, lo, hi)


def _ridge_design_apply(raw, t_min, scale):
    z = wpgam._scale_apply(raw, *scale)
    return np.column_stack([wpgam.time_basis(t_min)[:, 1:], z])


def _tree_design_fit(raw, t_min):
    names = wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES
    mean, std, lo, hi = wpgam._scale_fit(
        raw, feature_names=names if raw.shape[1] == len(names) else None)
    z = wpgam._scale_apply(raw, mean, std, lo, hi)
    return np.column_stack([np.clip(t_min, 0.0, 60.0) / 45.0, z]), (mean, std, lo, hi)


def _tree_design_apply(raw, t_min, scale):
    z = wpgam._scale_apply(raw, *scale)
    return np.column_stack([np.clip(t_min, 0.0, 60.0) / 45.0, z])


def _fit_method(family, spec, raw, y, gids, t_min):
    weights = _game_balanced_weights(gids)
    if family == "ridge_logit":
        design, scale = _ridge_design_fit(raw, t_min)
        design = np.column_stack([np.ones(len(design)), design])
        reg = np.concatenate([[0.0], np.full(design.shape[1] - 1,
                                             1.0 / float(spec["C"]))])
        beta = wpgam._fit_static_logit(
            design, y, reg, bounds=[(-20.0, 20.0)] * design.shape[1],
            weights=weights, maxiter=400)
        return {"family": family, "scale": scale, "beta": beta}
    if family in ("unconstrained_gam", "constrained_gam"):
        monotone = wpgam.MONOTONE_FEATURES if family == "constrained_gam" else ()
        model = wpgam.fit_state_model(
            raw, y, gids, t_min, l2=float(spec["l2"]),
            smooth=float(spec["smooth"]), maxiter=300,
            monotone_features=monotone)
        return {"family": family, "model": model}
    if family in ("hist_boost", "monotone_hist_boost"):
        # Some macOS/Python combinations cannot query physical cores through
        # sysctl inside a sandbox; give joblib an explicit, conservative cap.
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(8, os.cpu_count() or 1)))
        from sklearn.ensemble import HistGradientBoostingClassifier
        design, scale = _tree_design_fit(raw, t_min)
        monotone = None
        if family == "monotone_hist_boost":
            feature_names = wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES
            monotone = [0] + [1 if n in wpgam.MONOTONE_FEATURES else 0
                              for n in feature_names]
        estimator = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=float(spec.get("learning_rate", 0.05)),
            max_iter=int(spec.get("max_iter", 220)),
            max_leaf_nodes=int(spec["max_leaf_nodes"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            l2_regularization=float(spec["l2"]), early_stopping=False,
            monotonic_cst=monotone, random_state=43)
        estimator.fit(design, y, sample_weight=weights)
        return {"family": family, "scale": scale, "estimator": estimator}
    raise ValueError("unknown family %s" % family)


def _predict_method(model, raw, t_min):
    family = model["family"]
    if family == "ridge_logit":
        design = _ridge_design_apply(raw, t_min, model["scale"])
        design = np.column_stack([np.ones(len(design)), design])
        eta = np.einsum("ij,j->i", design, model["beta"], optimize=True)
        return wpgam._sigmoid(eta)
    if family in ("unconstrained_gam", "constrained_gam"):
        return wpgam.predict_state(model["model"], raw, t_min)
    if family in ("hist_boost", "monotone_hist_boost"):
        design = _tree_design_apply(raw, t_min, model["scale"])
        return model["estimator"].predict_proba(design)[:, 1]
    raise ValueError("unknown family %s" % family)


def _per_game(values, gids):
    values, gids = np.asarray(values), np.asarray(gids)
    ug = np.unique(gids)
    return ug, np.asarray([values[gids == g].mean() for g in ug])


def _basic_metrics(p, y, gids):
    p, y = np.asarray(p), np.asarray(y)
    loss = (p - y) ** 2
    _, game_loss = _per_game(loss, gids)
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    ll = -(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))
    _, game_ll = _per_game(ll, gids)
    return {
        "brier_state": float(loss.mean()),
        "brier_game": float(game_loss.mean()),
        "logloss_state": float(ll.mean()),
        "logloss_game": float(game_ll.mean()),
    }


def _calibration_diagnostics(p, y, gids):
    p = np.clip(np.asarray(p), 1e-5, 1.0 - 1e-5)
    y, gids = np.asarray(y), np.asarray(gids)
    logit = np.log(p / (1.0 - p))
    design = np.column_stack([np.ones(len(p)), logit])
    beta = wpgam._fit_static_logit(
        design, y, np.array([0.0, 0.25]),
        bounds=[(-5.0, 5.0), (0.05, 5.0)],
        weights=_game_balanced_weights(gids))
    # Equal-frequency, game-balanced expected calibration error.
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    weights = _game_balanced_weights(gids)
    ece = 0.0
    total = weights.sum()
    for idx in bins:
        bw = weights[idx]
        ece += bw.sum() / total * abs(np.average(p[idx], weights=bw) -
                                      np.average(y[idx], weights=bw))
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "ece10": float(ece)}


def _apply_platt(p, calibration):
    p = np.clip(np.asarray(p), 1e-5, 1.0 - 1e-5)
    logit = np.log(p / (1.0 - p))
    return wpgam._sigmoid(calibration["intercept"] +
                          calibration["slope"] * logit)


def _summary(p, y, gids, t_min, bootstrap=1000, seed=61):
    result = _basic_metrics(p, y, gids)
    loss = (np.asarray(p) - np.asarray(y)) ** 2
    _, game_loss = _per_game(loss, gids)
    rng = np.random.default_rng(seed)
    draws = rng.choice(game_loss, size=(bootstrap, len(game_loss)), replace=True).mean(axis=1)
    result["brier_game_ci95"] = [float(np.quantile(draws, 0.025)),
                                  float(np.quantile(draws, 0.975))]
    result["calibration"] = _calibration_diagnostics(p, y, gids)
    result["states"] = int(len(y))
    result["games"] = int(len(np.unique(gids)))
    result["by_phase"] = {}
    for name, lo, hi in (("early", 0.0, 15.0), ("mid", 15.0, 25.0),
                         ("late", 25.0, 1e9)):
        mask = (t_min >= lo) & (t_min < hi)
        if mask.any():
            result["by_phase"][name] = _basic_metrics(
                np.asarray(p)[mask], np.asarray(y)[mask], np.asarray(gids)[mask])
    return result


def _specs(quick=False):
    specs = {
        "ridge_logit": [{"C": c} for c in (0.03, 0.1, 0.3)],
        "unconstrained_gam": [
            {"l2": 6.0, "smooth": 20.0},
            {"l2": 12.0, "smooth": 35.0},
            {"l2": 24.0, "smooth": 70.0},
        ],
        "constrained_gam": [
            {"l2": 6.0, "smooth": 20.0},
            {"l2": 12.0, "smooth": 35.0},
            {"l2": 24.0, "smooth": 70.0},
        ],
        "hist_boost": [
            {"max_leaf_nodes": 15, "min_samples_leaf": 150, "l2": 10.0},
            {"max_leaf_nodes": 31, "min_samples_leaf": 200, "l2": 15.0},
            {"max_leaf_nodes": 31, "min_samples_leaf": 400, "l2": 30.0},
        ],
        "monotone_hist_boost": [
            {"max_leaf_nodes": 15, "min_samples_leaf": 150, "l2": 10.0},
            {"max_leaf_nodes": 31, "min_samples_leaf": 200, "l2": 15.0},
            {"max_leaf_nodes": 31, "min_samples_leaf": 400, "l2": 30.0},
        ],
    }
    if quick:
        return {k: v[:1] for k, v in specs.items()}
    return specs


def _fit_blend(predictions, y, gids, members):
    from scipy.optimize import minimize
    matrix = np.column_stack([predictions[m] for m in members])
    weights = _game_balanced_weights(gids)

    def objective(alpha):
        resid = np.einsum("ij,j->i", matrix, alpha, optimize=True) - y
        loss = np.sum(weights * resid * resid) / weights.sum()
        grad = 2.0 * np.einsum("ij,i->j", matrix, weights * resid,
                               optimize=True) / weights.sum()
        return float(loss), grad

    start = np.full(len(members), 1.0 / len(members))
    result = minimize(objective, start, jac=True, method="SLSQP",
                      bounds=[(0.0, 1.0)] * len(members),
                      constraints={"type": "eq", "fun": lambda a: a.sum() - 1.0,
                                   "jac": lambda a: np.ones_like(a)},
                      options={"ftol": 1e-12, "maxiter": 200})
    alpha = np.clip(result.x, 0.0, 1.0)
    alpha /= alpha.sum()
    return {m: float(a) for m, a in zip(members, alpha)}


def _blend(predictions, weights):
    return sum(weights[m] * predictions[m] for m in weights)


def _paired_differences(predictions, y, gids, best, bootstrap=2000, seed=67):
    rng = np.random.default_rng(seed)
    best_loss = (predictions[best] - y) ** 2
    ug, best_game = _per_game(best_loss, gids)
    out = {}
    for name, p in predictions.items():
        _, game = _per_game((p - y) ** 2, gids)
        delta = game - best_game
        draws = rng.choice(delta, size=(bootstrap, len(ug)), replace=True).mean(axis=1)
        out[name] = {
            "delta_brier_game_vs_best": float(delta.mean()),
            "delta_ci95": [float(np.quantile(draws, 0.025)),
                           float(np.quantile(draws, 0.975))],
        }
    return out


def _nested_calibration_stack(base, game_train):
    """Rebuild the full stacked pipeline strictly before calibration dates."""
    fit, calibration, cutoff = _date_blocks(base["game_dates"], game_train)
    stacked = _stacked_arrays(base, fit)
    calibration_rows = calibration[base["row_game"]]
    return stacked, calibration_rows, {"fit_end": str(max(base["game_dates"][fit])),
        "calibration_start": str(cutoff),
        "calibration_end": str(max(base["game_dates"][calibration])),
        "fit_games": int(fit.sum()), "calibration_games": int(calibration.sum())}


def run(dataset_path=None, output_path=RESULT_PATH, bootstrap=1000, quick=False,
        exposure_path=wpexposure.PATH):
    dataset_path = dataset_path or os.path.join(wpgam.OUT_DIR, "states.npz")
    started = time.time()
    base = _load_base(dataset_path)
    inventory = wpexposure.migrate(base["game_gid"], base["game_dates"], path=exposure_path)
    if wpexposure.fresh_mask(inventory, base["game_gid"], base["game_dates"]).any():
        raise ValueError("method search accepts consumed development outcomes only; filter the dataset first")
    inner, inner_calibration, inner_provenance = _nested_calibration_stack(
        base, base["inner_train"])
    inner_fit = inner["train"]
    validation = base["validation"][base["row_game"]]
    selected = {}
    calibrations = {}
    validation_predictions = {}
    validation_report = {}
    specs = _specs(quick)

    for family in FAMILIES:
        candidates = []
        best_score = np.inf
        best_model = None
        for spec in specs[family]:
            tick = time.time()
            model = _fit_method(
                family, spec, inner["raw"][inner_fit], base["y"][inner_fit],
                base["gid"][inner_fit], base["t_min"][inner_fit])
            calibration_p = _predict_method(model, inner["raw"][inner_calibration],
                                             base["t_min"][inner_calibration])
            coefficients = _calibration_diagnostics(calibration_p,
                base["y"][inner_calibration], base["gid"][inner_calibration])
            p = _apply_platt(_predict_method(model, inner["raw"][validation],
                                base["t_min"][validation]), coefficients)
            metrics = _basic_metrics(p, base["y"][validation],
                                     base["gid"][validation])
            candidates.append({"spec": spec, "metrics": metrics,
                               "seconds": round(time.time() - tick, 2)})
            log.info("validation %-22s %s Brier(game)=%.6f", family, spec,
                     metrics["brier_game"])
            if metrics["brier_game"] < best_score:
                best_score, best_model = metrics["brier_game"], model
                selected[family] = spec
                calibrations[family] = coefficients
        validation_predictions[family] = _predict_method(
            best_model, inner["raw"][validation], base["t_min"][validation])
        validation_report[family] = {"selected": selected[family],
                                     "candidates": candidates}

    validation_predictions["pregame"] = wpgam._sigmoid(
        inner["raw"][validation, 0] + inner["raw"][validation, 1])
    calibrations["pregame"] = _calibration_diagnostics(wpgam._sigmoid(
        inner["raw"][inner_calibration, 0] + inner["raw"][inner_calibration, 1]),
        base["y"][inner_calibration], base["gid"][inner_calibration])
    calibrated_validation = {
        name: _apply_platt(p, calibrations[name])
        for name, p in validation_predictions.items()
    }
    all_members = list(FAMILIES)
    monotone_members = ["constrained_gam", "monotone_hist_boost"]
    blend_weights = {
        "convex_ensemble": _fit_blend(
            calibrated_validation, base["y"][validation],
            base["gid"][validation], all_members),
        "monotone_ensemble": _fit_blend(
            calibrated_validation, base["y"][validation],
            base["gid"][validation], monotone_members),
    }

    # The outer test is touched only after all family settings and blend
    # weights have been selected on the validation date block.
    outer, outer_calibration, outer_provenance = _nested_calibration_stack(
        base, base["outer_train"])
    outer_fit = outer["train"]
    outer_test = (~base["outer_train"])[base["row_game"]]
    raw_test_predictions = {
        "pregame": wpgam._sigmoid(
            outer["raw"][outer_test, 0] + outer["raw"][outer_test, 1])
    }
    raw_calibration_predictions = {
        "pregame": wpgam._sigmoid(
            outer["raw"][outer_calibration, 0] + outer["raw"][outer_calibration, 1])
    }
    fitted_seconds = {}
    for family in FAMILIES:
        tick = time.time()
        model = _fit_method(
            family, selected[family], outer["raw"][outer_fit],
            base["y"][outer_fit], base["gid"][outer_fit],
            base["t_min"][outer_fit])
        raw_test_predictions[family] = _predict_method(
            model, outer["raw"][outer_test], base["t_min"][outer_test])
        raw_calibration_predictions[family] = _predict_method(
            model, outer["raw"][outer_calibration], base["t_min"][outer_calibration])
        fitted_seconds[family] = round(time.time() - tick, 2)
    # Both coefficients are fitted only on later predictions of an entirely
    # earlier pipeline. Never recenter the intercept on fitted training rows.
    final_calibrations = {name: _calibration_diagnostics(
        p, base["y"][outer_calibration], base["gid"][outer_calibration])
        for name, p in raw_calibration_predictions.items()}
    test_predictions = {
        name: _apply_platt(p, final_calibrations[name])
        for name, p in raw_test_predictions.items()
    }
    for name, weights in blend_weights.items():
        test_predictions[name] = _blend(test_predictions, weights)

    test_report = {
        name: _summary(p, base["y"][outer_test], base["gid"][outer_test],
                       base["t_min"][outer_test], bootstrap=bootstrap)
        for name, p in test_predictions.items()
    }
    ranking = sorted(test_report, key=lambda n: (test_report[n]["brier_game"],
                                                  test_report[n]["logloss_game"]))
    paired = _paired_differences(
        test_predictions, base["y"][outer_test], base["gid"][outer_test],
        ranking[0], bootstrap=max(bootstrap, 2000))
    for name in test_report:
        test_report[name].update(paired[name])

    result = {
        "kind": "wpx_method_benchmark_nested_calibration_v2",
        "dataset_sha256": wpexposure.sha256(dataset_path),
        "holdout_status": "consumed_development_only",
        "calibration_protocol": "fit < calibration < validation/test; full pipeline rebuilt; both coefficients held out",
        "calibration_blocks": {"selection": inner_provenance, "final": outer_provenance},
        "dataset": os.path.basename(dataset_path),
        "selection_metric": "validation game-balanced Brier",
        "split": {
            "method": base["split_method"],
            "inner_train_games": int(base["inner_train"].sum()),
            "validation_games": int(base["validation"].sum()),
            "outer_train_games": int(base["outer_train"].sum()),
            "test_games": int((~base["outer_train"]).sum()),
            "validation_start": str(base["validation_cutoff"]),
            "test_start": str(sorted(base["game_dates"][~base["outer_train"]])[0]),
        },
        "validation": validation_report,
        "validation_calibration": calibrations,
        "final_calibration": final_calibrations,
        "blend_weights": blend_weights,
        "test_raw": {
            name: _basic_metrics(p, base["y"][outer_test], base["gid"][outer_test])
            for name, p in raw_test_predictions.items()
        },
        "test": test_report,
        "ranking": ranking,
        "fit_seconds": fitted_seconds,
        "total_seconds": round(time.time() - started, 2),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    return result


def report(result):
    split = result["split"]
    print("method benchmark: %d inner train / %d validation / %d outer test games" %
          (split["inner_train_games"], split["validation_games"],
           split["test_games"]))
    print("%-24s %10s %10s %10s %8s %8s %18s" %
          ("method", "Brier(g)", "Brier(s)", "logloss", "cal", "ECE", "delta vs best (95% CI)"))
    for name in result["ranking"]:
        m = result["test"][name]
        ci = m["delta_ci95"]
        print("%-24s %10.5f %10.5f %10.5f %8.3f %8.4f %+7.5f [%+.5f,%+.5f]" %
              (name, m["brier_game"], m["brier_state"], m["logloss_game"],
               m["calibration"]["slope"], m["calibration"]["ece10"],
               m["delta_brier_game_vs_best"], ci[0], ci[1]))
    print("validation-selected blend weights:")
    for name, weights in result["blend_weights"].items():
        print("  %s: %s" % (name, ", ".join("%s=%.3f" % x for x in weights.items())))
    print("elapsed %.1fs" % result["total_seconds"])
