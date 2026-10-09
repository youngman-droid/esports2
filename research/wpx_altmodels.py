"""Alternative model families plugged into the nested chronological benchmark.

Reuses lol_ticker.wpbench.run (same split, rows, game-balanced weights, Platt
calibration and paired game-block bootstrap) by monkeypatching the family
dispatch.  Nothing in the repo is modified.
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpbench, wpgam  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("alt")
OUT = os.path.join(wpgam.OUT_DIR, "alt_method_benchmark.json")

FEATURE_NAMES = wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES
# Continuous oriented advantages that get a monotone nonlinear shape.
SPLINE_FEATURES = ["gold_k", "gold_mom", "cs_k", "d_kill", "d_tower", "dead_adv",
                   "hp_pool", "lvl_k", "gold_rel", "prior_team_logit"]


# ---------------------------------------------------------------- surface fit
def fit_surface(Z, y, gids, t_min, mono_mask, l2, smooth, knots=wpgam.TIME_KNOTS,
                maxiter=400):
    """Generalized version of wpgam.fit_state_model on pre-standardized columns."""
    from scipy.optimize import minimize
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    B = wpgam.time_basis(t_min, knots)
    w = wpgam._game_balanced_weights(gids)
    nf, nk = Z.shape[1], B.shape[1]
    bounds = [(-10.0, 10.0)] * nk
    for m in mono_mask:
        bounds.extend([(0.0, 10.0) if m else (-10.0, 10.0)] * nk)

    def objective(flat):
        flat = np.clip(flat, -10.0, 10.0)
        theta = flat.reshape(nf + 1, nk)
        coef = np.einsum("ik,fk->if", B, theta[1:], optimize=True)
        eta = B @ theta[0] + np.sum(Z * coef, axis=1)
        resid = w * (wpgam._sigmoid(eta) - y)
        loss = np.sum(w * (np.logaddexp(0.0, eta) - y * eta))
        grad = np.empty_like(theta)
        grad[0] = B.T @ resid
        grad[1:] = np.einsum("if,ik->fk", Z * resid[:, None], B, optimize=True)
        loss += 0.5 * l2 * np.sum(theta[1:] ** 2)
        grad[1:] += l2 * theta[1:]
        delta = theta[:, 1:] - theta[:, :-1]
        loss += 0.5 * smooth * np.sum(delta * delta)
        grad[:, :-1] -= smooth * delta
        grad[:, 1:] += smooth * delta
        return float(loss), grad.reshape(-1)

    init = np.zeros((nf + 1, nk))
    base = np.clip(np.average(y, weights=w), 1e-4, 1 - 1e-4)
    init[0, :] = np.log(base / (1 - base))
    res = minimize(objective, init.reshape(-1), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 2e-5, "maxcor": 20})
    if not res.success:
        res = minimize(objective, res.x, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 800, "ftol": 1e-10, "gtol": 2e-5, "maxcor": 20})
    if not res.success or not np.isfinite(res.x).all():
        raise RuntimeError("surface optimizer: %s" % res.message)
    return {"theta": np.clip(res.x, -10, 10).reshape(nf + 1, nk), "knots": knots}


def predict_surface_logit(model, Z, t_min):
    B = wpgam.time_basis(t_min, model["knots"])
    eff = np.einsum("ik,fk->if", B, model["theta"], optimize=True)
    return eff[:, 0] + np.sum(np.asarray(Z, dtype=np.float64) * eff[:, 1:], axis=1)


# ---------------------------------------------------------------- spline GAM
def _spline_fit(raw, n_seg):
    """Monotone piecewise-linear (degree-1 I-spline) expansion of selected columns."""
    mean, std, lo, hi = wpgam._scale_fit(raw, feature_names=FEATURE_NAMES)
    z = wpgam._scale_apply(raw, mean, std, lo, hi)
    cols, mono, breaks = [], [], {}
    qs = np.linspace(0, 1, n_seg + 1)[1:-1]
    for j, name in enumerate(FEATURE_NAMES):
        if name in SPLINE_FEATURES:
            zj = z[:, j]
            c = np.unique(np.quantile(zj, qs))
            c = np.concatenate([[zj.min()], c, [zj.max()]])
            breaks[j] = c
            for s in range(len(c) - 1):
                cols.append(np.clip(zj - c[s], 0, c[s + 1] - c[s]))
                mono.append(True)
        else:
            cols.append(z[:, j])
            mono.append(name in wpgam.MONOTONE_FEATURES)
    D = np.column_stack(cols)
    dmean, dstd = D.mean(axis=0), D.std(axis=0)
    dstd[dstd < 1e-6] = 1.0
    return (D - dmean) / dstd, {"scale": (mean, std, lo, hi), "breaks": breaks,
                                "dmean": dmean, "dstd": dstd, "mono": mono}


def _spline_apply(raw, info):
    z = wpgam._scale_apply(raw, *info["scale"])
    cols = []
    for j, name in enumerate(FEATURE_NAMES):
        if j in info["breaks"]:
            c = info["breaks"][j]
            for s in range(len(c) - 1):
                cols.append(np.clip(z[:, j] - c[s], 0, c[s + 1] - c[s]))
        else:
            cols.append(z[:, j])
    return (np.column_stack(cols) - info["dmean"]) / info["dstd"]


# ---------------------------------------------------------------- MLP (numpy)
def _mlp_fit(X, y, w, hidden, epochs, wd, seed, lr=1e-3, batch=2048):
    rng = np.random.default_rng(seed)
    sizes = [X.shape[1]] + list(hidden) + [1]
    W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
    b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
    b[-1][:] = np.log(np.average(y, weights=w) / (1 - np.average(y, weights=w)))
    mW = [np.zeros_like(x) for x in W]; vW = [np.zeros_like(x) for x in W]
    mb = [np.zeros_like(x) for x in b]; vb = [np.zeros_like(x) for x in b]
    n = len(y); step = 0
    total_steps = epochs * ((n + batch - 1) // batch)
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb, wb = X[idx], y[idx], w[idx]
            # forward
            acts = [xb]; pre = []
            h = xb
            for i in range(len(W) - 1):
                zz = h @ W[i] + b[i]; pre.append(zz); h = np.maximum(zz, 0); acts.append(h)
            eta = (h @ W[-1] + b[-1])[:, 0]
            p = wpgam._sigmoid(eta)
            # backward (weighted BCE, mean over batch)
            g = (wb * (p - yb) / wb.sum())[:, None]
            grads_W = [None] * len(W); grads_b = [None] * len(W)
            grads_W[-1] = acts[-1].T @ g; grads_b[-1] = g.sum(axis=0)
            gh = g @ W[-1].T
            for i in range(len(W) - 2, -1, -1):
                gz = gh * (pre[i] > 0)
                grads_W[i] = acts[i].T @ gz; grads_b[i] = gz.sum(axis=0)
                if i > 0:
                    gh = gz @ W[i].T
            step += 1
            lr_t = lr * 0.5 * (1 + np.cos(np.pi * step / total_steps))
            for i in range(len(W)):
                gW = grads_W[i] + wd * W[i]
                mW[i] = 0.9 * mW[i] + 0.1 * gW; vW[i] = 0.999 * vW[i] + 0.001 * gW * gW
                mb[i] = 0.9 * mb[i] + 0.1 * grads_b[i]; vb[i] = 0.999 * vb[i] + 0.001 * grads_b[i] ** 2
                mhat = mW[i] / (1 - 0.9 ** step); vhat = vW[i] / (1 - 0.999 ** step)
                W[i] -= lr_t * mhat / (np.sqrt(vhat) + 1e-8)
                mhat = mb[i] / (1 - 0.9 ** step); vhat = vb[i] / (1 - 0.999 ** step)
                b[i] -= lr_t * mhat / (np.sqrt(vhat) + 1e-8)
    return W, b


def _mlp_logit(params, X):
    W, b = params
    h = X
    for i in range(len(W) - 1):
        h = np.maximum(h @ W[i] + b[i], 0)
    return (h @ W[-1] + b[-1])[:, 0]


def _mlp_design(raw, t_min, scale):
    z = wpgam._scale_apply(raw, *scale)
    return np.column_stack([wpgam.time_basis(t_min), z])


# ---------------------------------------------------------------- RFF logit
def _rff_design(raw, t_min, scale, Wr, br):
    z = wpgam._scale_apply(raw, *scale)
    inp = np.column_stack([z, 2.0 * np.clip(t_min, 0, 60) / 45.0])
    phi = np.sqrt(2.0 / Wr.shape[1]) * np.cos(inp @ Wr + br)
    return np.column_stack([np.ones(len(z)), wpgam.time_basis(t_min)[:, 1:], z, phi])


# ---------------------------------------------------------------- dispatch
_orig_fit, _orig_predict = wpbench._fit_method, wpbench._predict_method


def fit_method(family, spec, raw, y, gids, t_min):
    weights = wpgam._game_balanced_weights(gids)
    if family == "spline_gam":
        D, info = _spline_fit(raw, int(spec["segments"]))
        model = fit_surface(D, y, gids, t_min, info["mono"], float(spec["l2"]), float(spec["smooth"]))
        return {"family": family, "info": info, "model": model}
    if family == "gam_boost_stack":
        from sklearn.ensemble import HistGradientBoostingClassifier
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
        ug = np.unique(gids)
        gf = wpgam._fold_ids(ug, 5, seed=31)
        fold_of = dict(zip(ug, gf))
        rf = np.asarray([fold_of[g] for g in gids])
        oof = np.zeros(len(y))
        for f in range(gf.max() + 1):
            tr, te = rf != f, rf == f
            sub = wpgam.fit_state_model(raw[tr], y[tr], gids[tr], t_min[tr], maxiter=250)
            oof[te] = wpgam.predict_state(sub, raw[te], t_min[te], components=True)[1]
        full = wpgam.fit_state_model(raw, y, gids, t_min)
        scale = wpgam._scale_fit(raw, feature_names=FEATURE_NAMES)
        z = wpgam._scale_apply(raw, *scale)
        design = np.column_stack([np.clip(t_min, 0, 60) / 45.0, oof, z])
        mono = [0, 1] + [1 if n in wpgam.MONOTONE_FEATURES else 0 for n in FEATURE_NAMES]
        est = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=float(spec["lr"]), max_iter=int(spec["iters"]),
            max_leaf_nodes=int(spec["leaves"]), min_samples_leaf=int(spec["msl"]),
            l2_regularization=float(spec["l2"]), early_stopping=False,
            monotonic_cst=mono, random_state=43)
        est.fit(design, y, sample_weight=weights)
        return {"family": family, "gam": full, "scale": scale, "estimator": est}
    if family == "mlp":
        scale = wpgam._scale_fit(raw, feature_names=FEATURE_NAMES)
        X = _mlp_design(raw, t_min, scale)
        params = [_mlp_fit(X, y, weights, spec["hidden"], int(spec["epochs"]),
                           float(spec["wd"]), seed=100 + s) for s in range(int(spec["seeds"]))]
        return {"family": family, "scale": scale, "params": params}
    if family == "rff_logit":
        scale = wpgam._scale_fit(raw, feature_names=FEATURE_NAMES)
        rng = np.random.default_rng(7)
        d_in = raw.shape[1] + 1
        Wr = rng.normal(0, float(spec["gamma"]), (d_in, int(spec["D"])))
        br = rng.uniform(0, 2 * np.pi, int(spec["D"]))
        A = _rff_design(raw, t_min, scale, Wr, br)
        n_lin = 1 + (len(wpgam.TIME_KNOTS) - 1) + raw.shape[1]
        reg = np.concatenate([[0.0], np.full(n_lin - 1, 1.0 / 0.03),
                              np.full(int(spec["D"]), 1.0 / float(spec["C"]))])
        beta = wpgam._fit_static_logit(A, y, reg, bounds=[(-20, 20)] * A.shape[1],
                                       weights=weights, maxiter=400)
        return {"family": family, "scale": scale, "Wr": Wr, "br": br, "beta": beta}
    return _orig_fit(family, spec, raw, y, gids, t_min)


def predict_method(model, raw, t_min):
    family = model["family"]
    if family == "spline_gam":
        return wpgam._sigmoid(predict_surface_logit(model["model"], _spline_apply(raw, model["info"]), t_min))
    if family == "gam_boost_stack":
        lg = wpgam.predict_state(model["gam"], raw, t_min, components=True)[1]
        z = wpgam._scale_apply(raw, *model["scale"])
        design = np.column_stack([np.clip(t_min, 0, 60) / 45.0, lg, z])
        return model["estimator"].predict_proba(design)[:, 1]
    if family == "mlp":
        X = _mlp_design(raw, t_min, model["scale"])
        return np.mean([wpgam._sigmoid(_mlp_logit(p, X)) for p in model["params"]], axis=0)
    if family == "rff_logit":
        A = _rff_design(raw, t_min, model["scale"], model["Wr"], model["br"])
        return wpgam._sigmoid(A @ model["beta"])
    return _orig_predict(model, raw, t_min)


def specs(quick=False):
    s = {
        "constrained_gam": [{"l2": 24.0, "smooth": 70.0}],
        "monotone_hist_boost": [{"max_leaf_nodes": 31, "min_samples_leaf": 200, "l2": 15.0}],
        "spline_gam": [
            {"segments": 4, "l2": 24.0, "smooth": 70.0},
            {"segments": 6, "l2": 24.0, "smooth": 70.0},
            {"segments": 6, "l2": 60.0, "smooth": 150.0},
        ],
        "gam_boost_stack": [
            {"lr": 0.03, "iters": 120, "leaves": 15, "msl": 300, "l2": 20.0},
            {"lr": 0.03, "iters": 250, "leaves": 31, "msl": 300, "l2": 20.0},
        ],
        "mlp": [
            {"hidden": (64, 32), "epochs": 10, "wd": 1e-4, "seeds": 3},
            {"hidden": (128, 64), "epochs": 10, "wd": 1e-4, "seeds": 3},
            {"hidden": (64, 32), "epochs": 20, "wd": 1e-3, "seeds": 3},
        ],
        "rff_logit": [
            {"D": 300, "gamma": 0.35, "C": 0.3},
            {"D": 300, "gamma": 0.7, "C": 0.1},
        ],
    }
    if quick:
        return {k: v[:1] for k, v in s.items()}
    return s


if __name__ == "__main__":
    wpbench.FAMILIES = ("constrained_gam", "monotone_hist_boost", "spline_gam",
                        "gam_boost_stack", "mlp", "rff_logit")
    wpbench._specs = specs
    wpbench._fit_method = fit_method
    wpbench._predict_method = predict_method
    quick = "--quick" in sys.argv
    t0 = time.time()
    result = wpbench.run(output_path=OUT, bootstrap=1000, quick=quick)
    wpbench.report(result)
    print("done in %.0fs" % (time.time() - t0))
