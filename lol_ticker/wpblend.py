"""Chronological optimization of market/model probability blends.

Polymarket blend families are tuned inside the middle date block and evaluated
on the then-held-out outer diagnostic.  Kalshi begins later in the dataset,
so its
covered games receive their own 40/20/40 forward split.  Event rows and games
are identical for every method within a platform comparison.
"""
import json
import logging
import os
import time

import numpy as np

from . import wpbench, wpgam


log = logging.getLogger("wpblend")
RESULT_PATH = os.path.join(wpgam.OUT_DIR, "blend_benchmark.json")
MODEL_PATH = os.path.join(wpgam.OUT_DIR, "blend_model.json")
LATENCY_RESULT_PATH = os.path.join(wpgam.OUT_DIR, "blend_latency45.json")
LATENCY_MODEL_PATH = os.path.join(wpgam.OUT_DIR, "blend_model_live.json")
TIME_KNOTS = wpgam.TIME_KNOTS
POST_DRAFT_MARKET_OFFSET_S = 120


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def _predict_rows(d, model, mask):
    raw = np.column_stack([
        wpgam.prior_values_from_matrix(model, d["X"][mask], list(d["names"]),
                                       d["C"][mask]),
        wpgam.state_values_from_matrix(d["X"][mask], list(d["names"])),
    ])
    return wpgam.predict_state(model["state"], raw, d["t"][mask] / 60.0)


def _weights(gids):
    return wpgam._game_balanced_weights(np.asarray(gids))


def _data(market, model, y, gids, t_min, dates, extra=None):
    out = {
        "market": np.asarray(market, dtype=np.float64),
        "model": np.asarray(model, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "gid": np.asarray(gids),
        "t_min": np.asarray(t_min, dtype=np.float64),
        "date": np.asarray(dates).astype(str),
    }
    if extra is not None:
        out["extra"] = np.asarray(extra, dtype=np.float64)
    return out


def _subset(data, mask):
    return {k: np.asarray(v)[mask] for k, v in data.items()}


def _concat(*datasets):
    return {k: np.concatenate([np.asarray(d[k]) for d in datasets])
            for k in datasets[0]}


def _chron_split(data, holdout=0.4):
    gids, dates = data["gid"], data["date"]
    ug, first = np.unique(gids, return_index=True)
    game_dates = dates[first]
    n_test = max(1, int(round(len(ug) * holdout)))
    cutoff = np.sort(game_dates)[-n_test]
    train = dates < cutoff
    if not train.any() or train.all():
        raise ValueError("could not construct chronological blend split")
    return _subset(data, train), _subset(data, ~train), str(cutoff)


def _fit_probability_blend(data, spec):
    m, x, y = data["market"], data["model"], data["y"]
    w = _weights(data["gid"])
    delta = m - x
    denom = np.sum(w * delta * delta)
    alpha = np.sum(w * delta * (y - x)) / denom if denom > 1e-12 else 0.5
    return {"family": "probability_blend", "market_weight": float(np.clip(alpha, 0.0, 1.0))}


def _fit_logit_blend(data, spec):
    from scipy.optimize import minimize
    lm, lx, y = _logit(data["market"]), _logit(data["model"]), data["y"]
    sw = _weights(data["gid"])

    def objective(a):
        eta = a[0] * lm + (1.0 - a[0]) * lx
        p = wpgam._sigmoid(eta)
        resid = p - y
        loss = np.sum(sw * resid * resid)
        grad = 2.0 * np.sum(sw * resid * p * (1.0 - p) * (lm - lx))
        return float(loss), np.array([grad])

    fit = minimize(objective, np.array([0.5]), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, 1.0)], options={"ftol": 1e-12})
    return {"family": "logit_blend", "market_weight": float(fit.x[0])}


def _fit_logit_design(design, y, gids, reg, bounds, loss="logloss"):
    if loss == "logloss":
        return wpgam._fit_static_logit(
            design, y, reg, bounds=bounds, weights=_weights(gids))
    if loss != "brier":
        raise ValueError("unknown stack loss %s" % loss)
    from scipy.optimize import minimize
    design = np.asarray(design, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg = np.asarray(reg, dtype=np.float64)
    sw = _weights(gids)

    def objective(beta):
        eta = np.einsum("ij,j->i", design, beta, optimize=True)
        p = wpgam._sigmoid(eta)
        resid = p - y
        loss_value = np.sum(sw * resid * resid) + 0.5 * np.sum(reg * beta * beta)
        grad = 2.0 * np.einsum(
            "ij,i->j", design, sw * resid * p * (1.0 - p), optimize=True)
        grad += reg * beta
        return float(loss_value), grad

    fit = minimize(objective, np.zeros(design.shape[1]), jac=True,
                   method="L-BFGS-B", bounds=bounds,
                   options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 300})
    if not fit.success:
        log.warning("Brier stack optimizer: %s", fit.message)
    return fit.x


def _fit_market_platt(data, spec):
    design = np.column_stack([np.ones(len(data["y"])), _logit(data["market"])])
    beta = _fit_logit_design(
        design, data["y"], data["gid"],
        np.array([0.0, float(spec["l2"])]),
        bounds=[(-5.0, 5.0), (0.05, 5.0)], loss=spec.get("loss", "logloss"))
    return {"family": "market_platt", "beta": beta}


def _fit_logit_stack(data, spec):
    design = np.column_stack([
        np.ones(len(data["y"])), _logit(data["market"]), _logit(data["model"])
    ])
    l2 = float(spec["l2"])
    beta = _fit_logit_design(
        design, data["y"], data["gid"], np.array([0.0, l2, l2]),
        bounds=[(-5.0, 5.0), (0.0, 5.0), (0.0, 5.0)],
        loss=spec.get("loss", "logloss"))
    return {"family": "logit_stack", "beta": beta}


def _fit_time_probability(data, spec):
    from scipy.optimize import minimize
    basis = wpgam.time_basis(data["t_min"], TIME_KNOTS)
    m, x, y = data["market"], data["model"], data["y"]
    sw = _weights(data["gid"])
    smooth = float(spec["smooth"])

    def objective(theta):
        alpha = np.einsum("ik,k->i", basis, theta, optimize=True)
        p = x + alpha * (m - x)
        resid = p - y
        loss = np.sum(sw * resid * resid)
        grad = 2.0 * np.einsum("ik,i->k", basis, sw * resid * (m - x), optimize=True)
        delta = theta[1:] - theta[:-1]
        loss += 0.5 * smooth * np.sum(delta * delta)
        grad[:-1] -= smooth * delta
        grad[1:] += smooth * delta
        return float(loss), grad

    fit = minimize(objective, np.full(len(TIME_KNOTS), 0.5), jac=True,
                   method="L-BFGS-B", bounds=[(0.0, 1.0)] * len(TIME_KNOTS),
                   options={"ftol": 1e-12, "maxiter": 300})
    return {"family": "time_probability", "weights": fit.x,
            "knots": TIME_KNOTS.copy()}


def _fit_time_logit_residual(data, spec):
    from scipy.optimize import minimize
    basis = wpgam.time_basis(data["t_min"], TIME_KNOTS)
    lm, lx, y = _logit(data["market"]), _logit(data["model"]), data["y"]
    sw = _weights(data["gid"])
    smooth = float(spec["smooth"])
    ridge = float(spec.get("ridge", 10.0))
    nk = len(TIME_KNOTS)

    def objective(flat):
        intercept, alpha = flat[:nk], flat[nk:]
        a = np.einsum("ik,k->i", basis, intercept, optimize=True)
        w = np.einsum("ik,k->i", basis, alpha, optimize=True)
        eta = lm + a + w * (lx - lm)
        p = wpgam._sigmoid(eta)
        if spec.get("loss", "logloss") == "brier":
            resid = 2.0 * sw * (p - y) * p * (1.0 - p)
            loss = np.sum(sw * (p - y) ** 2)
        else:
            resid = sw * (p - y)
            loss = np.sum(sw * (np.logaddexp(0.0, eta) - y * eta))
        grad_a = np.einsum("ik,i->k", basis, resid, optimize=True)
        grad_w = np.einsum("ik,i->k", basis, resid * (lx - lm), optimize=True)
        da, dw = np.diff(intercept), np.diff(alpha)
        loss += 0.5 * smooth * (np.sum(da * da) + np.sum(dw * dw))
        grad_a[:-1] -= smooth * da
        grad_a[1:] += smooth * da
        grad_w[:-1] -= smooth * dw
        grad_w[1:] += smooth * dw
        loss += 0.5 * ridge * np.sum(intercept * intercept)
        grad_a += ridge * intercept
        return float(loss), np.concatenate([grad_a, grad_w])

    initial = np.concatenate([np.zeros(nk), np.full(nk, 0.25)])
    bounds = [(-2.0, 2.0)] * nk + [(0.0, 1.0)] * nk
    fit = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"ftol": 1e-11, "maxiter": 400})
    return {"family": "time_logit_residual", "intercepts": fit.x[:nk],
            "weights": fit.x[nk:], "knots": TIME_KNOTS.copy()}


def _fit_candidate(family, data, spec):
    if family in ("market", "model"):
        return {"family": family}
    return {
        "probability_blend": _fit_probability_blend,
        "logit_blend": _fit_logit_blend,
        "market_platt": _fit_market_platt,
        "logit_stack": _fit_logit_stack,
        "time_probability": _fit_time_probability,
        "time_logit_residual": _fit_time_logit_residual,
    }[family](data, spec)


def _predict_candidate(model, data):
    family = model["family"]
    m, x = data["market"], data["model"]
    if family == "market":
        return m.copy()
    if family == "model":
        return x.copy()
    if family == "probability_blend":
        a = float(model["market_weight"])
        return a * m + (1.0 - a) * x
    if family == "logit_blend":
        a = float(model["market_weight"])
        return wpgam._sigmoid(a * _logit(m) + (1.0 - a) * _logit(x))
    if family == "market_platt":
        beta = np.asarray(model["beta"])
        return wpgam._sigmoid(beta[0] + beta[1] * _logit(m))
    if family == "logit_stack":
        beta = np.asarray(model["beta"])
        return wpgam._sigmoid(beta[0] + beta[1] * _logit(m) + beta[2] * _logit(x))
    basis = wpgam.time_basis(data["t_min"], np.asarray(model["knots"]))
    if family == "time_probability":
        a = np.einsum("ik,k->i", basis, np.asarray(model["weights"]), optimize=True)
        return a * m + (1.0 - a) * x
    if family == "time_logit_residual":
        a = np.einsum("ik,k->i", basis, np.asarray(model["intercepts"]), optimize=True)
        w = np.einsum("ik,k->i", basis, np.asarray(model["weights"]), optimize=True)
        return wpgam._sigmoid(_logit(m) + a + w * (_logit(x) - _logit(m)))
    raise ValueError("unknown blend family %s" % family)


def _specs():
    static_specs = [
        {"l2": x, "loss": loss}
        for loss in ("logloss", "brier")
        for x in (0.1, 1.0, 10.0, 100.0)
    ]
    return {
        "market": [{}],
        "model": [{}],
        "probability_blend": [{}],
        "logit_blend": [{}],
        "market_platt": static_specs,
        "logit_stack": static_specs,
        "time_probability": [{"smooth": x} for x in (0.0, 10.0, 100.0, 1000.0)],
        "time_logit_residual": [
            {"smooth": x, "ridge": r, "loss": loss}
            for loss in ("logloss", "brier")
            for x, r in ((1.0, 0.1), (10.0, 1.0), (100.0, 10.0),
                         (1000.0, 100.0))
        ],
    }


def _encode(model):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in model.items()}


def load_model(path=LATENCY_MODEL_PATH):
    with open(path) as fh:
        artifact = json.load(fh)
    if artifact.get("base_model_kind") != wpgam.MODEL_KIND:
        raise ValueError("blend artifact expects %s, live model is %s" %
                         (artifact.get("base_model_kind"), wpgam.MODEL_KIND))
    return artifact


def predict_live(model_p, t_min, markets, path=LATENCY_MODEL_PATH,
                 after_draft=False):
    """Blend the latest model state with available blue-oriented market quotes."""
    artifact = load_model(path)
    models = (artifact.get("after_draft") if after_draft else None) or artifact
    available = {p: float(v) for p, v in markets.items()
                 if p in ("polymarket", "kalshi") and v is not None}
    lead = (artifact.get("after_draft_market_offset_s", POST_DRAFT_MARKET_OFFSET_S)
            if after_draft else artifact.get("market_lead_s"))
    out = {"market_lead_s": lead,
           "after_draft": bool(after_draft), "platforms": {}, "sources": {}}
    for platform, market_p in available.items():
        data = _data([market_p], [model_p], [0.0], [0], [t_min], [""])
        out["platforms"][platform] = float(
            _predict_candidate(models[platform], data)[0])
        family = models[platform]["family"]
        out["sources"][platform] = (
            "model" if family == "model" else
            platform if family in ("market", "market_platt") else
            "model+" + platform)
    recommended = None
    if "polymarket" in available and "kalshi" in available:
        both = models["both"]
        uses = both.get("uses", "both")
        if uses == "both" and both["family"] in (
                "probability_simplex", "logit_simplex", "three_way_stack"):
            data = _data([available["polymarket"]], [model_p], [0.0], [0],
                         [t_min], [""], extra=[available["kalshi"]])
            recommended = float(_predict_three(both, data)[0])
            source = "model+polymarket+kalshi"
        else:
            platform = uses if uses in available else "kalshi"
            recommended = out["platforms"][platform]
            source = out["sources"][platform]
    elif available:
        platform = next(iter(available))
        recommended = out["platforms"][platform]
        source = out["sources"][platform]
    else:
        source = "model"
        recommended = float(model_p)
    out["p_blue"] = recommended
    out["source"] = source
    return out


def _experiment(dev, test, bootstrap=1000):
    fit, selection, selection_start = _chron_split(dev, holdout=0.4)
    selected, validation = {}, {}
    for family, specs in _specs().items():
        rows, best_score, best_spec = [], np.inf, None
        for spec in specs:
            model = _fit_candidate(family, fit, spec)
            p = _predict_candidate(model, selection)
            metrics = wpbench._basic_metrics(p, selection["y"], selection["gid"])
            rows.append({"spec": spec, "metrics": metrics})
            if metrics["brier_game"] < best_score:
                best_score, best_spec = metrics["brier_game"], spec
        selected[family] = best_spec
        validation[family] = {"selected": best_spec, "candidates": rows}

    predictions, fitted = {}, {}
    for family in _specs():
        model = _fit_candidate(family, dev, selected[family])
        fitted[family] = model
        predictions[family] = _predict_candidate(model, test)
    summaries = {
        name: wpbench._summary(p, test["y"], test["gid"], test["t_min"],
                               bootstrap=bootstrap)
        for name, p in predictions.items()
    }
    ranking = sorted(summaries, key=lambda n: (summaries[n]["brier_game"],
                                                summaries[n]["logloss_game"]))
    paired = wpbench._paired_differences(
        predictions, test["y"], test["gid"], ranking[0],
        bootstrap=max(bootstrap, 2000), seed=83)
    for name in summaries:
        summaries[name].update(paired[name])
    winner = ranking[0]
    production = _fit_candidate(winner, _concat(dev, test), selected[winner])
    return {
        "split": {
            "fit_games": int(len(np.unique(fit["gid"]))),
            "selection_games": int(len(np.unique(selection["gid"]))),
            "test_games": int(len(np.unique(test["gid"]))),
            "selection_start": selection_start,
            "test_start": str(sorted(test["date"])[0]),
        },
        "selection": validation,
        "test": summaries,
        "ranking": ranking,
        "winner": winner,
        "winner_model": _encode(fitted[winner]),
        "production_model": _encode(production),
    }


def _fit_three_way(data, l2=10.0, loss="logloss"):
    design = np.column_stack([
        np.ones(len(data["y"])), _logit(data["model"]),
        _logit(data["market"]), _logit(data["extra"]),
    ])
    beta = _fit_logit_design(
        design, data["y"], data["gid"], np.array([0.0, l2, l2, l2]),
        bounds=[(-5.0, 5.0)] + [(0.0, 5.0)] * 3, loss=loss)
    return beta


def _fit_three_simplex(data, logits=False):
    from scipy.optimize import minimize
    matrix = np.column_stack([data["model"], data["market"], data["extra"]])
    if logits:
        matrix = _logit(matrix)
    y, sw = data["y"], _weights(data["gid"])

    def objective(alpha):
        linear = np.einsum("ij,j->i", matrix, alpha, optimize=True)
        p = wpgam._sigmoid(linear) if logits else linear
        resid = p - y
        loss = np.sum(sw * resid * resid)
        factor = p * (1.0 - p) if logits else np.ones(len(p))
        grad = 2.0 * np.einsum("ij,i->j", matrix, sw * resid * factor,
                               optimize=True)
        return float(loss), grad

    fit = minimize(
        objective, np.full(3, 1.0 / 3.0), jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints={"type": "eq", "fun": lambda a: a.sum() - 1.0,
                     "jac": lambda a: np.ones_like(a)},
        options={"ftol": 1e-12, "maxiter": 300})
    alpha = np.clip(fit.x, 0.0, 1.0)
    alpha /= alpha.sum()
    return alpha


def _predict_three(model, data):
    family = model["family"]
    matrix = np.column_stack([data["model"], data["market"], data["extra"]])
    if family in ("probability_simplex", "logit_simplex"):
        if family == "logit_simplex":
            matrix = _logit(matrix)
        linear = np.einsum("ij,j->i", matrix, np.asarray(model["weights"]),
                           optimize=True)
        return wpgam._sigmoid(linear) if family == "logit_simplex" else linear
    beta = np.asarray(model["beta"])
    return wpgam._sigmoid(beta[0] + beta[1] * _logit(data["model"]) +
                          beta[2] * _logit(data["market"]) +
                          beta[3] * _logit(data["extra"]))


def _fit_three_candidate(family, data, spec):
    if family == "probability_simplex":
        return {"family": family, "weights": _fit_three_simplex(data, logits=False)}
    if family == "logit_simplex":
        return {"family": family, "weights": _fit_three_simplex(data, logits=True)}
    return {"family": family,
            "beta": _fit_three_way(data, float(spec["l2"]),
                                    loss=spec.get("loss", "logloss"))}


def _three_way_experiment(data, polymarket_model, kalshi_model, bootstrap=1000,
                          polymarket_production=None, kalshi_production=None):
    dev, test, test_start = _chron_split(data, holdout=0.4)
    fit, selection, selection_start = _chron_split(dev, holdout=1.0 / 3.0)
    specs = {
        "probability_simplex": [{}],
        "logit_simplex": [{}],
        "three_way_stack": [
            {"l2": x, "loss": loss}
            for loss in ("logloss", "brier")
            for x in (0.1, 1.0, 10.0, 100.0)
        ],
    }
    selected, validation = {}, {}
    fitted = {}
    predictions = {}
    for family, family_specs in specs.items():
        candidates, best = [], None
        for spec in family_specs:
            model = _fit_three_candidate(family, fit, spec)
            p = _predict_three(model, selection)
            metrics = wpbench._basic_metrics(p, selection["y"], selection["gid"])
            candidates.append({"spec": spec, "metrics": metrics})
            if best is None or metrics["brier_game"] < best[0]:
                best = (metrics["brier_game"], spec)
        selected[family] = best[1]
        validation[family] = candidates
        fitted[family] = _fit_three_candidate(family, dev, best[1])
        predictions[family] = _predict_three(fitted[family], test)
    pm_data = dict(test, market=test["market"])
    ks_data = dict(test, market=test["extra"])
    predictions = {
        **predictions,
        "polymarket_two_way": _predict_candidate(polymarket_model, pm_data),
        "kalshi_two_way": _predict_candidate(kalshi_model, ks_data),
        "polymarket": test["market"],
        "kalshi": test["extra"],
        "model": test["model"],
    }
    summaries = {n: wpbench._summary(p, test["y"], test["gid"], test["t_min"],
                                     bootstrap=bootstrap)
                 for n, p in predictions.items()}
    ranking = sorted(summaries, key=lambda n: summaries[n]["brier_game"])
    paired = wpbench._paired_differences(
        predictions, test["y"], test["gid"], ranking[0],
        bootstrap=max(bootstrap, 2000), seed=89)
    for name in summaries:
        summaries[name].update(paired[name])
    winner = ranking[0]
    if winner in specs:
        production = _fit_three_candidate(winner, data, selected[winner])
        production_uses = "both"
    elif winner == "polymarket_two_way":
        production = polymarket_production or polymarket_model
        production_uses = "polymarket"
    elif winner == "kalshi_two_way":
        production = kalshi_production or kalshi_model
        production_uses = "kalshi"
    else:
        production = {"family": winner}
        production_uses = winner
    return {
        "split": {"fit_games": int(len(np.unique(fit["gid"]))),
                  "selection_games": int(len(np.unique(selection["gid"]))),
                  "test_games": int(len(np.unique(test["gid"]))),
                  "selection_start": selection_start, "test_start": test_start},
        "selection": validation, "selected": selected, "test": summaries,
        "ranking": ranking, "winner": winner,
        "winner_model": _encode(fitted[winner]) if winner in fitted else _encode(production),
        "production_model": _encode(production),
        "production_uses": production_uses,
        "source_order": ["model", "polymarket", "kalshi"],
    }


def run(dataset_path=None, output_path=RESULT_PATH, model_path=MODEL_PATH,
        bootstrap=1000):
    dataset_path = dataset_path or os.path.join(wpgam.OUT_DIR, "states.npz")
    started = time.time()
    d = np.load(dataset_path, allow_pickle=True)
    first, outer_train, split_method = wpgam._date_split(d)
    game_gid = d["gid"][first]
    game_dates = d["date"][first].astype(str)
    inner_train, validation, _ = wpbench._date_blocks(game_dates, outer_train)
    gid_index = {int(g): i for i, g in enumerate(game_gid)}
    row_game = np.fromiter((gid_index[int(g)] for g in d["gid"]), dtype=np.int32,
                           count=len(d["gid"]))
    event = d["seq"] >= 0

    log.info("fitting state model for blend-development predictions")
    inner_model = wpgam.fit_arrays(
        d["X"], d["y"], d["gid"], d["t"], d["seq"], d["C"],
        list(d["names"]), list(d["champ_names"]), train_games=inner_train,
        dates=d["date"])
    log.info("fitting state model for final-test predictions")
    outer_model = wpgam.fit_arrays(
        d["X"], d["y"], d["gid"], d["t"], d["seq"], d["C"],
        list(d["names"]), list(d["champ_names"]), train_games=outer_train,
        dates=d["date"])

    pm_dev_mask = event & validation[row_game] & np.isfinite(d["pm"])
    pm_test_mask = event & (~outer_train)[row_game] & np.isfinite(d["pm"])
    pm_dev = _data(d["pm"][pm_dev_mask], _predict_rows(d, inner_model, pm_dev_mask),
                   d["y"][pm_dev_mask], d["gid"][pm_dev_mask],
                   d["t"][pm_dev_mask] / 60.0, d["date"][pm_dev_mask])
    pm_test = _data(d["pm"][pm_test_mask], _predict_rows(d, outer_model, pm_test_mask),
                    d["y"][pm_test_mask], d["gid"][pm_test_mask],
                    d["t"][pm_test_mask] / 60.0, d["date"][pm_test_mask])
    polymarket = _experiment(pm_dev, pm_test, bootstrap)

    ks_all_mask = event & (~outer_train)[row_game] & np.isfinite(d["ks"])
    ks_all = _data(d["ks"][ks_all_mask], _predict_rows(d, outer_model, ks_all_mask),
                   d["y"][ks_all_mask], d["gid"][ks_all_mask],
                   d["t"][ks_all_mask] / 60.0, d["date"][ks_all_mask])
    ks_dev, ks_test, _ = _chron_split(ks_all, holdout=0.4)
    kalshi = _experiment(ks_dev, ks_test, bootstrap)

    both_mask = ks_all_mask & np.isfinite(d["pm"])
    both = _data(d["pm"][both_mask], _predict_rows(d, outer_model, both_mask),
                 d["y"][both_mask], d["gid"][both_mask],
                 d["t"][both_mask] / 60.0, d["date"][both_mask],
                 extra=d["ks"][both_mask])
    three_way = _three_way_experiment(
        both, polymarket["winner_model"], kalshi["winner_model"], bootstrap,
        polymarket_production=polymarket["production_model"],
        kalshi_production=kalshi["production_model"])

    result = {
        "dataset": os.path.basename(dataset_path), "split_method": split_method,
        "polymarket": polymarket, "kalshi": kalshi, "both": three_way,
        "total_seconds": round(time.time() - started, 2),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    artifact = {
        "kind": "wpblend_v1", "base_model_kind": wpgam.MODEL_KIND,
        "polymarket": polymarket["production_model"],
        "kalshi": kalshi["production_model"],
        "both": {**three_way["production_model"],
                 "source_order": three_way["source_order"],
                 "uses": three_way["production_uses"]},
    }
    with open(model_path, "w") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
    d.close()
    return result


def _fixed_prediction_map(d, model, game_mask, row_game):
    mask = (d["seq"] < 0) & game_mask[row_game]
    p = _predict_rows(d, model, mask)
    return {(int(g), int(t)): (float(pp), float(y), str(date))
            for g, t, pp, y, date in
            zip(d["gid"][mask], d["t"][mask], p, d["y"][mask], d["date"][mask])}


def _records_data(records, extra_records=None):
    keys = sorted(records)
    if extra_records is not None:
        keys = [k for k in keys if k in extra_records]
    market, model, y, gids, t_min, dates, extra = [], [], [], [], [], [], []
    for key in keys:
        r = records[key]
        market.append(r["market"]); model.append(r["model"]); y.append(r["y"])
        gids.append(key[0]); t_min.append(key[1] / 60.0); dates.append(r["date"])
        if extra_records is not None:
            extra.append(extra_records[key]["market"])
    return _data(market, model, y, gids, t_min, dates,
                 extra=extra if extra_records is not None else None)


def _extract_latency_records(conn, d, inner_model, outer_model, inner_train,
                             validation, outer_train, game_gid, game_dates,
                             row_game, lag_s=45, min_quality=0.4):
    """Fixed-minute model state paired with the later executable market quote."""
    from . import align
    val_map = _fixed_prediction_map(d, inner_model, validation, row_game)
    test_map = _fixed_prediction_map(d, outer_model, ~outer_train, row_game)
    val_gids = set(game_gid[validation].tolist())
    test_gids = set(game_gid[~outer_train].tolist())
    wanted = val_gids | test_gids
    rows = conn.execute("""
        SELECT a.game_id, a.platform, a.market_id, a.team_side, a.start_wall,
               a.end_wall, a.pauses, a.duration_s
        FROM game_alignment a
        WHERE a.quality >= %s AND a.platform IN ('polymarket', 'kalshi')
        ORDER BY a.game_id, a.platform""", (min_quality,)).fetchall()
    groups = {}
    for row in rows:
        if row["game_id"] in wanted:
            groups.setdefault((row["game_id"], row["platform"]), []).append(row)
    records = {p: {"dev": {}, "test": {}} for p in ("polymarket", "kalshi")}
    for n, ((gid, platform), aligned) in enumerate(groups.items(), 1):
        block = "dev" if gid in val_gids else "test"
        prediction_map = val_map if block == "dev" else test_map
        base = aligned[0]
        pauses = base["pauses"]
        if not isinstance(pauses, list):
            pauses = json.loads(pauses or "[]")
        series = []
        for row in aligned:
            odds = align.odds_series(
                conn, platform, row["market_id"],
                row["start_wall"] - 600, row["end_wall"] + 300)
            if len(odds) >= 10:
                series.append((odds, row["team_side"]))
        if not series:
            continue
        for minute in range(1, int(base["duration_s"] // 60) + 1):
            t_s = minute * 60
            pred = prediction_map.get((int(gid), t_s))
            if pred is None:
                continue
            pause_s = sum(p["length_s"] for p in pauses if p["game_time_s"] <= t_s)
            wall = base["start_wall"] + t_s + pause_s + lag_s
            prices = []
            for odds, side in series:
                price = align._price_at(odds, wall, "before")
                if price is not None:
                    prices.append(price if side == "blue" else 1.0 - price)
            if prices:
                records[platform][block][(int(gid), t_s)] = {
                    "market": float(np.mean(prices)), "model": pred[0],
                    "y": pred[1], "date": pred[2],
                }
        if n % 300 == 0:
            log.info("latency extraction: %d/%d aligned game-platforms", n, len(groups))
    return records


def _extract_after_draft_records(conn, d, inner_model, outer_model, validation,
                                 outer_train, game_gid, row_game):
    """Pair the causal draft-complete model state with stored post-draft odds."""
    from . import draft
    val_map = _fixed_prediction_map(d, inner_model, validation, row_game)
    test_map = _fixed_prediction_map(d, outer_model, ~outer_train, row_game)
    val_gids = set(int(g) for g in game_gid[validation])
    test_gids = set(int(g) for g in game_gid[~outer_train])
    wanted = val_gids | test_gids
    rows = conn.execute("""
        SELECT g.game_id, g.blue_team, g.red_team, d.platform, d.team, d.post_p
        FROM golgg_games g
        JOIN draft_deltas d ON d.oe_game_id = g.oe_game_id
        WHERE d.post_p IS NOT NULL AND d.platform IN ('polymarket', 'kalshi')
        ORDER BY g.game_id, d.platform""").fetchall()
    grouped = {}
    for row in rows:
        gid = int(row["game_id"])
        if gid not in wanted:
            continue
        team = draft.norm_team(row["team"])
        if team == draft.norm_team(row["blue_team"]):
            p_blue = float(row["post_p"])
        elif team == draft.norm_team(row["red_team"]):
            p_blue = 1.0 - float(row["post_p"])
        else:
            continue
        grouped.setdefault((gid, row["platform"]), []).append(p_blue)
    records = {p: {"dev": {}, "test": {}} for p in ("polymarket", "kalshi")}
    for (gid, platform), prices in grouped.items():
        block = "dev" if gid in val_gids else "test"
        pred = (val_map if block == "dev" else test_map).get((gid, 0))
        if pred is None:
            continue
        records[platform][block][(gid, 0)] = {
            "market": float(np.mean(prices)), "model": pred[0],
            "y": pred[1], "date": pred[2],
        }
    return records


def _run_after_draft(records, bootstrap):
    out = {}
    for platform in ("polymarket", "kalshi"):
        dev = _records_data(records[platform]["dev"])
        test = _records_data(records[platform]["test"])
        if len(np.unique(dev["gid"])) < 20:
            all_data = _concat(dev, test)
            dev, test, _ = _chron_split(all_data, holdout=0.4)
        out[platform] = _experiment(dev, test, bootstrap)
    both_records = {
        block: {**records["polymarket"][block]}
        for block in ("dev", "test")
    }
    both_dev = _records_data(both_records["dev"], records["kalshi"]["dev"])
    both_test = _records_data(both_records["test"], records["kalshi"]["test"])
    both = _concat(both_dev, both_test)
    out["both"] = _three_way_experiment(
        both, out["polymarket"]["winner_model"], out["kalshi"]["winner_model"],
        bootstrap, polymarket_production=out["polymarket"]["production_model"],
        kalshi_production=out["kalshi"]["production_model"])
    out["coverage"] = {
        p: {b: {"states": len(r), "games": len(set(k[0] for k in r))}
            for b, r in blocks.items()}
        for p, blocks in records.items()
    }
    return out


def run_latency(conn, lag_s=45, dataset_path=None,
                output_path=LATENCY_RESULT_PATH, model_path=LATENCY_MODEL_PATH,
                bootstrap=1000):
    dataset_path = dataset_path or os.path.join(wpgam.OUT_DIR, "states.npz")
    if lag_s != 45 and output_path == LATENCY_RESULT_PATH:
        output_path = os.path.join(wpgam.OUT_DIR, "blend_latency%d.json" % lag_s)
    if lag_s != 45 and model_path == LATENCY_MODEL_PATH:
        model_path = os.path.join(wpgam.OUT_DIR, "blend_model_live%d.json" % lag_s)
    started = time.time()
    d = np.load(dataset_path, allow_pickle=True)
    first, outer_train, split_method = wpgam._date_split(d)
    game_gid, game_dates = d["gid"][first], d["date"][first].astype(str)
    inner_train, validation, _ = wpbench._date_blocks(game_dates, outer_train)
    gid_index = {int(g): i for i, g in enumerate(game_gid)}
    row_game = np.fromiter((gid_index[int(g)] for g in d["gid"]), dtype=np.int32,
                           count=len(d["gid"]))
    log.info("fitting state models for +%ds latency blend", lag_s)
    inner_model = wpgam.fit_arrays(
        d["X"], d["y"], d["gid"], d["t"], d["seq"], d["C"],
        list(d["names"]), list(d["champ_names"]), train_games=inner_train,
        dates=d["date"])
    outer_model = wpgam.fit_arrays(
        d["X"], d["y"], d["gid"], d["t"], d["seq"], d["C"],
        list(d["names"]), list(d["champ_names"]), train_games=outer_train,
        dates=d["date"])
    records = _extract_latency_records(
        conn, d, inner_model, outer_model, inner_train, validation, outer_train,
        game_gid, game_dates, row_game, lag_s=lag_s)
    draft_records = _extract_after_draft_records(
        conn, d, inner_model, outer_model, validation, outer_train,
        game_gid, row_game)
    pm_dev = _records_data(records["polymarket"]["dev"])
    pm_test = _records_data(records["polymarket"]["test"])
    polymarket = _experiment(pm_dev, pm_test, bootstrap)
    ks_all = _records_data(records["kalshi"]["test"])
    ks_dev, ks_test, _ = _chron_split(ks_all, holdout=0.4)
    kalshi = _experiment(ks_dev, ks_test, bootstrap)
    both = _records_data(records["polymarket"]["test"],
                         records["kalshi"]["test"])
    three_way = _three_way_experiment(
        both, polymarket["winner_model"], kalshi["winner_model"], bootstrap,
        polymarket_production=polymarket["production_model"],
        kalshi_production=kalshi["production_model"])
    after_draft = _run_after_draft(draft_records, bootstrap)
    after_draft["market_offset_s"] = POST_DRAFT_MARKET_OFFSET_S
    result = {
        "dataset": os.path.basename(dataset_path), "split_method": split_method,
        "market_lead_s": int(lag_s), "polymarket": polymarket,
        "kalshi": kalshi, "both": three_way,
        "after_draft": after_draft,
        "coverage": {
            p: {b: {"states": len(r), "games": len(set(k[0] for k in r))}
                for b, r in blocks.items()}
            for p, blocks in records.items()
        },
        "total_seconds": round(time.time() - started, 2),
    }
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    artifact = {
        "kind": "wpblend_live_v1", "base_model_kind": wpgam.MODEL_KIND,
        "market_lead_s": int(lag_s),
        "after_draft_market_offset_s": POST_DRAFT_MARKET_OFFSET_S,
        "polymarket": polymarket["production_model"],
        "kalshi": kalshi["production_model"],
        "both": {**three_way["production_model"],
                 "source_order": three_way["source_order"],
                 "uses": three_way["production_uses"]},
        "after_draft": {
            "polymarket": after_draft["polymarket"]["production_model"],
            "kalshi": after_draft["kalshi"]["production_model"],
            "both": {**after_draft["both"]["production_model"],
                     "source_order": after_draft["both"]["source_order"],
                     "uses": after_draft["both"]["production_uses"]},
        },
    }
    with open(model_path, "w") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
    d.close()
    return result


def report(result):
    for platform in ("polymarket", "kalshi"):
        r = result[platform]
        print("%s blend: %d fit / %d select / %d test games" %
              (platform, r["split"]["fit_games"], r["split"]["selection_games"],
               r["split"]["test_games"]))
        for name in r["ranking"]:
            m = r["test"][name]
            ci = m["delta_ci95"]
            print("  %-22s Brier(g)=%.5f Brier(s)=%.5f delta=%+.5f [%+.5f,%+.5f]" %
                  (name, m["brier_game"], m["brier_state"],
                   m["delta_brier_game_vs_best"], ci[0], ci[1]))
        print("  winner:", r["winner"], r["winner_model"])
    r = result["both"]
    print("both exchanges: %d test games; winner=%s" %
          (r["split"]["test_games"], r["winner"]))
    for name in r["ranking"]:
        m = r["test"][name]
        print("  %-22s Brier(g)=%.5f Brier(s)=%.5f" %
              (name, m["brier_game"], m["brier_state"]))
    if "after_draft" in result:
        print("post-draft (+%ds market quote):" %
              result["after_draft"]["market_offset_s"])
        for platform in ("polymarket", "kalshi"):
            r = result["after_draft"][platform]
            win, market, model = r["test"][r["winner"]], r["test"]["market"], r["test"]["model"]
            print("  %-10s %s Brier(g)=%.5f vs market %.5f / model %.5f (%d games)" %
                  (platform, r["winner"], win["brier_game"],
                   market["brier_game"], model["brier_game"],
                   r["split"]["test_games"]))
    print("elapsed %.1fs" % result["total_seconds"])
