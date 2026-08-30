"""Causal, shape-constrained win-probability model.

The model has two stages:

1. A one-row-per-game pregame logistic model combines the live-available team
   priors with strongly regularized signed champion effects.
2. A smooth time-varying logistic model combines that pregame logit with a
   deliberately small set of features shared by the historical and live
   pipelines.  Coefficients for oriented advantages are non-negative at every
   time knot, so obvious state improvements cannot lower win probability.

Only fixed-minute states are used for the in-game fit.  This makes the fitting
distribution match continuous live scoring and avoids giving event-heavy games
hundreds of extra, highly correlated rows.

v6 additions (each live-derivable from existing feed fields): a gol.gg-based
team Elo in the pregame stage (OE ratings go stale when the source CSV lags),
relative gold share, and time since the last kill.

v7 adds the champion-state channel: per-champion coefficients estimated
champscale-style on in-game states (out-of-fold for training games) are
compressed into one signed per-game score that enters the state model as a
third prior input with a non-negative smooth time curve.  This carries the
in-game champion information that the pregame outcome fit cannot see and
closed most of the legacy champscale model's remaining accuracy edge.
"""
import json
import logging
import os
import time

import numpy as np

from . import config


log = logging.getLogger("wpgam")
MODEL_KIND = "wpgam_v7_champ_state"
OUT_DIR = os.path.join(config.REPO_ROOT, "data", "wpx")
MODEL_PATH = os.path.join(OUT_DIR, "model_live_gam.npz")

# The 60-min knot exists for the marathon tail: with the surface clamped at
# 45 the model was badly overconfident past 45 minutes (calibration slope
# 0.59 on that slice); the extra knot lets coefficients keep evolving there.
TIME_KNOTS = np.array([0.0, 10.0, 20.0, 30.0, 45.0, 60.0], dtype=np.float64)
STATE_L2 = 24.0
STATE_SMOOTH = 70.0
# Champion-state channel: per-champion coefficients fit champscale-style on
# in-game states (jointly with the full exploration feature set, entering as
# presence x min(t, cap)); the hyperparameters are the time-forward-validated
# champscale spec.  The per-game signed score enters the state model as a
# third prior input whose non-negative knot curve learns the time ramp.
# The coefficient fit uses ALL training rows (fixed-minute + event-anchored):
# champion effects need the teamfight-dense event samples — restricting the
# fit to fixed minutes cost 0.0005 game Brier on the newest-date holdout.
CHAMP_STATE_L2 = 800.0
CHAMP_STATE_CAP_MIN = 15.0
CHAMP_STATE_ALL_ROWS = True
CHAMP_STATE_FOLDS = 5
PRIOR_INPUTS = ["prior_team_logit", "prior_champ_logit", "prior_champ_state"]
# elo_gg (gol.gg-based team Elo) backs up the Oracle's Elixir ratings, whose
# source CSV can go stale for weeks; gol.gg coverage is ~99.8% of games.
PREGAME_FEATURES = ["elo_oe", "pelo_oe", "form_diff", "elo_gg"]

# Every feature here can be constructed identically from a historical WPX row
# and an official-feed live state.  Differences are oriented toward blue.
STATE_FEATURES = [
    "gold_k", "gold_mom", "cs_k",
    "gold_top_alloc", "gold_jng_alloc", "gold_mid_alloc", "gold_bot_alloc",
    "d_kill", "d_tower", "d_dragon", "d_baron", "baron_active",
    "d_inhib", "d_elder", "elder_active", "soul",
    "dead_adv", "dead_adv_sq", "dead_count_sq_adv", "dead_base_pressure",
    "hp_pool", "lvl_k", "has_hp",
    "gold_rel", "t_since_kill",
]

# A non-negative coefficient at each time knot makes the partial derivative of
# log-odds with respect to these oriented advantages non-negative everywhere.
MONOTONE_FEATURES = {
    "prior_team_logit", "prior_champ_logit", "prior_champ_state",
    "gold_k", "gold_mom", "cs_k", "d_kill", "d_tower",
    "d_dragon", "d_baron", "baron_active", "d_inhib", "d_elder",
    "elder_active", "soul", "dead_adv",
    "dead_adv_sq", "dead_count_sq_adv", "dead_base_pressure",
    "hp_pool", "lvl_k", "gold_rel",
}


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _norm_champ(name):
    return (str(name or "").lower().replace("'", "").replace(" ", "")
            .replace(".", ""))


def time_basis(t_min, knots=TIME_KNOTS):
    """Piecewise-linear non-negative hat basis; rows sum to one."""
    t = np.asarray(t_min, dtype=np.float64).reshape(-1)
    knots = np.asarray(knots, dtype=np.float64)
    if len(knots) < 2 or np.any(np.diff(knots) <= 0):
        raise ValueError("time knots must be strictly increasing")
    tc = np.clip(t, knots[0], knots[-1])
    out = np.zeros((len(tc), len(knots)), dtype=np.float64)
    right = np.searchsorted(knots, tc, side="right")
    right = np.clip(right, 1, len(knots) - 1)
    left = right - 1
    span = knots[right] - knots[left]
    a = (tc - knots[left]) / span
    rows = np.arange(len(tc))
    out[rows, left] = 1.0 - a
    out[rows, right] += a
    out[tc <= knots[0]] = np.eye(1, len(knots), 0)[0]
    out[tc >= knots[-1]] = np.eye(1, len(knots), len(knots) - 1)[0]
    return out


def _column(X, idx, name, default=0.0):
    j = idx.get(name)
    return X[:, j].astype(np.float64) if j is not None else np.full(len(X), default)


def _death_features(dead_blue, dead_red, towers_blue, towers_red,
                    inhib_blue, inhib_red):
    """Nonlinear, side-symmetric teamfight features available in both feeds.

    Historical kill-derived death counts can exceed five when a player dies
    twice inside the approximate respawn window, so enforce the physical
    range before fitting. Base pressure ignores outer towers and treats a
    previously destroyed inhibitor as evidence that the base has been opened.
    """
    dead_blue = np.clip(np.asarray(dead_blue, dtype=np.float64), 0.0, 5.0)
    dead_red = np.clip(np.asarray(dead_red, dtype=np.float64), 0.0, 5.0)
    dead_adv = dead_red - dead_blue
    dead_adv_sq = np.sign(dead_adv) * dead_adv * dead_adv
    dead_count_sq_adv = dead_red * dead_red - dead_blue * dead_blue
    blue_access = np.maximum(np.asarray(towers_blue, dtype=np.float64) - 5.0, 0.0)
    red_access = np.maximum(np.asarray(towers_red, dtype=np.float64) - 5.0, 0.0)
    blue_access += 2.0 * (np.asarray(inhib_blue, dtype=np.float64) > 0.0)
    red_access += 2.0 * (np.asarray(inhib_red, dtype=np.float64) > 0.0)
    dead_base_pressure = dead_red * blue_access - dead_blue * red_access
    return dead_adv, dead_adv_sq, dead_count_sq_adv, dead_base_pressure


def state_values_from_matrix(X, names):
    """Production feature contract from stored WPX rows."""
    X = np.asarray(X)
    idx = {str(n): i for i, n in enumerate(names)}
    gk = _column(X, idx, "gold_k")
    role = np.column_stack([_column(X, idx, "gold_" + r) for r in ("top", "jng", "mid", "bot")])
    alloc = role - gk[:, None] / 5.0
    dead_adv, dead_adv_sq, dead_count_sq_adv, dead_base_pressure = _death_features(
        _column(X, idx, "dead_blue"), _column(X, idx, "dead_red"),
        _column(X, idx, "towers_blue"), _column(X, idx, "towers_red"),
        _column(X, idx, "inhib_blue"), _column(X, idx, "inhib_red"))
    vals = np.column_stack([
        gk, _column(X, idx, "gold_mom"), _column(X, idx, "cs_k"), alloc,
        _column(X, idx, "d_kill"), _column(X, idx, "d_tower"),
        _column(X, idx, "d_dragon"), _column(X, idx, "d_baron"),
        _column(X, idx, "baron_active"),
        _column(X, idx, "d_inhib"), _column(X, idx, "d_elder"),
        _column(X, idx, "elder_buff"),
        _column(X, idx, "soul"), dead_adv, dead_adv_sq, dead_count_sq_adv,
        dead_base_pressure,
        _column(X, idx, "hp_pool"),
        _column(X, idx, "lvl_k"), _column(X, idx, "has_hp"),
        _column(X, idx, "gold_rel"),
        _column(X, idx, "t_since_kill", default=10.0),
    ])
    if vals.shape[1] != len(STATE_FEATURES):
        raise AssertionError("state feature contract is out of sync")
    return vals


def state_values_from_live(state):
    """Same production feature contract from an official-feed state dict."""
    s = dict(state)
    gk = float(s.get("gold_diff_k", 0.0) or 0.0)
    total_gold_k = (float(s.get("gold_blue", 0) or 0)
                    + float(s.get("gold_red", 0) or 0)) / 1000.0
    role = list(s.get("gold_role") or [gk / 5.0] * 5)
    role = (role + [gk / 5.0] * 5)[:5]
    drag_b = int(s.get("drag_blue", 0) or 0)
    drag_r = int(s.get("drag_red", 0) or 0)
    inh_b = float(s.get("inhib_blue", 0) or 0)
    inh_r = float(s.get("inhib_red", 0) or 0)
    dead_adv, dead_adv_sq, dead_count_sq_adv, dead_base_pressure = _death_features(
        float(s.get("dead_blue", 0) or 0), float(s.get("dead_red", 0) or 0),
        float(s.get("towers_blue", 0) or 0), float(s.get("towers_red", 0) or 0),
        inh_b, inh_r)
    vals = [
        gk,
        gk - float(s.get("gold_diff_prev_k", gk) or 0.0),
        float(s.get("cs_diff_k", 0.0) or 0.0),
        *[float(role[i] or 0.0) - gk / 5.0 for i in range(4)],
        float(s.get("kills", 0) or 0),
        float(s.get("towers", 0) or 0),
        float(s.get("dragons", drag_b - drag_r) or 0),
        float(s.get("barons", 0) or 0),
        float(s.get("baron_active", 0) or 0),
        float(s.get("inhibs", inh_b - inh_r) or 0),
        float(s.get("elders", 0) or 0),
        float(s.get("elder_active", 0) or 0),
        float((drag_b >= 4) - (drag_r >= 4)),
        float(dead_adv), float(dead_adv_sq), float(dead_count_sq_adv),
        float(dead_base_pressure),
        float(s.get("hp_pool", 0.0) or 0.0),
        float(s.get("lvl_k", 0.0) or 0.0),
        float(s.get("has_hp", 0.0) or 0.0),
        gk / total_gold_k if total_gold_k > 1.0 else 0.0,
        float(s.get("t_since_kill_min", 10.0)),
    ]
    return np.asarray(vals, dtype=np.float64)


def pregame_values_from_matrix(X, names):
    idx = {str(n): i for i, n in enumerate(names)}
    return np.column_stack([_column(np.asarray(X), idx, n) for n in PREGAME_FEATURES])


def pregame_values_from_live(state):
    s = dict(state)
    # Same fallback as wpx.live_vector: a missed gol.gg lookup borrows the OE
    # Elo (near-identical scale) rather than reading as "even teams".
    if s.get("elo_gg") is None:
        s["elo_gg"] = s.get("elo_oe", 0.0)
    return np.asarray([float(s.get(n, 0.0) or 0.0) for n in PREGAME_FEATURES], dtype=np.float64)


def _champ_matrix(C, n_champs):
    C = np.asarray(C)
    out = np.zeros((len(C), n_champs), dtype=np.float64)
    if not n_champs:
        return out
    rows = np.arange(len(C))
    for k in range(min(10, C.shape[1])):
        ids = C[:, k].astype(int)
        ok = (ids >= 0) & (ids < n_champs)
        out[rows[ok], ids[ok]] += 1.0 if k < 5 else -1.0
    return out


def _scale_fit(raw, q=0.005):
    raw = np.asarray(raw, dtype=np.float64)
    lo = np.quantile(raw, q, axis=0)
    hi = np.quantile(raw, 1.0 - q, axis=0)
    clipped = np.clip(raw, lo, hi)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std, lo, hi


def _scale_apply(raw, mean, std, lo, hi):
    return (np.clip(np.asarray(raw, dtype=np.float64), lo, hi) - mean) / std


def _fit_static_logit(A, y, reg, bounds=None, weights=None, maxiter=250):
    from scipy.optimize import minimize

    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg = np.asarray(reg, dtype=np.float64)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)

    finite_bounds = bounds or [(None, None)] * A.shape[1]
    lower = np.asarray([b[0] if b[0] is not None else -20.0 for b in finite_bounds])
    upper = np.asarray([b[1] if b[1] is not None else 20.0 for b in finite_bounds])

    def objective(beta):
        if not np.all(np.isfinite(beta)):
            return 1e100, np.zeros_like(beta)
        beta = np.clip(beta, lower, upper)
        eta = np.einsum("ij,j->i", A, beta, optimize=True)
        resid = w * (_sigmoid(eta) - y)
        loss = np.sum(w * (np.logaddexp(0.0, eta) - y * eta))
        loss += 0.5 * np.sum(reg * beta * beta)
        grad = np.einsum("ij,i->j", A, resid, optimize=True) + reg * beta
        return float(loss), grad

    result = minimize(objective, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B",
                      bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-6})
    if not result.success:
        log.warning("pregame optimizer: %s", result.message)
    coef = np.clip(result.x, lower, upper)
    if not np.all(np.isfinite(coef)):
        raise FloatingPointError("pregame optimizer returned non-finite coefficients")
    return coef


def fit_pregame(raw, C, y, champ_names, l2=10.0, l2_champ=150.0):
    raw = np.asarray(raw, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean, std, lo, hi = _scale_fit(raw)
    Z = _scale_apply(raw, mean, std, lo, hi)
    S = _champ_matrix(C, len(champ_names))
    A = np.column_stack([np.ones(len(y)), Z, S])
    reg = np.concatenate([[0.0], np.full(Z.shape[1], l2), np.full(S.shape[1], l2_champ)])
    bounds = [(-10.0, 10.0)] + [(0.0, 10.0)] * Z.shape[1] + [(-5.0, 5.0)] * S.shape[1]
    beta = _fit_static_logit(A, y, reg, bounds=bounds)
    return {
        "mean": mean, "std": std, "lo": lo, "hi": hi,
        "intercept": float(beta[0]), "team_beta": beta[1:1 + Z.shape[1]],
        "champ_beta": beta[1 + Z.shape[1]:], "champ_names": np.asarray(champ_names),
    }


def predict_pregame(model, raw, C):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    Z = _scale_apply(raw, model["mean"], model["std"], model["lo"], model["hi"])
    S = _champ_matrix(C, len(model["champ_beta"]))
    team = np.einsum("ij,j->i", Z, model["team_beta"], optimize=True)
    champ = np.einsum("ij,j->i", S, model["champ_beta"], optimize=True)
    eta = model["intercept"] + team + champ
    return eta, team, champ


def _fold_ids(gids, k=5, seed=19):
    gids = np.asarray(gids)
    rng = np.random.default_rng(seed)
    order = np.arange(len(gids))
    rng.shuffle(order)
    folds = np.empty(len(gids), dtype=int)
    folds[order] = np.arange(len(gids)) % max(2, min(k, len(gids)))
    return folds


def oof_pregame(raw, C, y, gids, champ_names, k=5):
    raw = np.asarray(raw)
    folds = _fold_ids(gids, k)
    out_team = np.zeros(len(y), dtype=np.float64)
    out_champ = np.zeros(len(y), dtype=np.float64)
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        model = fit_pregame(raw[tr], C[tr], y[tr], champ_names)
        _, team, champ = predict_pregame(model, raw[te], C[te])
        out_team[te] = model["intercept"] + team
        out_champ[te] = champ
    return out_team, out_champ


def _game_balanced_weights(gids):
    _, inv, counts = np.unique(gids, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w / w.mean()


def fit_champ_state(X, names, C, y, rows, n_champs, l2_champ=CHAMP_STATE_L2,
                    cap_min=CHAMP_STATE_CAP_MIN):
    """Per-champion in-game coefficients, champscale spec.

    Joint ridge-IRLS of the outcome on the full exploration feature set, its
    xtime expansion, and signed champion presence x min(t, cap): the champion
    block absorbs what champions predict *conditional on game state*, which
    the pregame outcome fit cannot see.  Returns only the champion block.
    """
    from .wpx import ALREADY_INTERACTED

    X = np.asarray(X)
    idx = {str(n): i for i, n in enumerate(names)}
    rich = [i for i, n in enumerate(names) if str(n) != "draft"]
    xcols = [c for c in rich if str(names[c]) not in ALREADY_INTERACTED
             and str(names[c]) != "bias"]
    t_col = idx["t"]
    Xm = X[rows]
    B = Xm[:, rich].astype(np.float64)
    t_raw = Xm[:, t_col:t_col + 1].astype(np.float64)
    tt = np.minimum(t_raw, cap_min / 30.0)
    S = _champ_matrix(np.asarray(C)[rows], n_champs) * tt
    A = np.hstack([B, Xm[:, xcols].astype(np.float64) * t_raw, S])
    ytr = np.asarray(y)[rows].astype(np.float64)
    beta = np.zeros(A.shape[1])
    reg = np.full(A.shape[1], 1.0)
    reg[0] = 0.0
    if n_champs:
        reg[-n_champs:] = l2_champ
    for _ in range(30):
        p = _sigmoid(A @ beta)
        W = p * (1 - p) + 1e-9
        step = np.linalg.solve((A * W[:, None]).T @ A + np.diag(reg),
                               A.T @ (p - ytr) + reg * beta)
        beta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    if not np.all(np.isfinite(beta)):
        raise FloatingPointError("champ-state IRLS returned non-finite coefficients")
    return beta[-n_champs:].copy() if n_champs else np.zeros(0)


def champ_state_scores(beta, C):
    """Signed per-game champion score: own champions add, enemy subtract."""
    return _champ_matrix(C, len(beta)) @ np.asarray(beta, dtype=np.float64)


def oof_champ_state(X, names, C, y, gid, rows_pool, train_gids, n_champs,
                    k=5, seed=41):
    """Out-of-fold champion-state score per training game.

    ``rows_pool`` are the candidate fitting rows (fixed-minute training rows);
    each game's score comes from a fit that excluded that game.
    """
    gid = np.asarray(gid)
    train_gids = np.asarray(train_gids)
    folds = _fold_ids(train_gids, k, seed=seed)
    fold_of = {g: f for g, f in zip(train_gids, folds)}
    pool_gid = gid[rows_pool]
    scores = {}
    game_rows = _game_rows(gid)
    row_of_game = {g: i for g, i in zip(gid[game_rows], game_rows)}
    for f in range(folds.max() + 1):
        fit_rows = rows_pool[np.asarray([fold_of.get(g) != f for g in pool_gid])]
        beta = fit_champ_state(X, names, C, y, fit_rows, n_champs)
        for g in train_gids[folds == f]:
            row = row_of_game[g]
            scores[g] = float(champ_state_scores(beta, np.asarray(C)[row:row + 1])[0])
    return scores


def prior_values_from_matrix(model, X, names, C):
    """[prior_team_logit, prior_champ_logit, prior_champ_state] for stored rows."""
    pre = model["pregame"]
    pre_raw = pregame_values_from_matrix(X, names)
    _, team, champ = predict_pregame(pre, pre_raw, C)
    score = champ_state_scores(model["champ_state"]["beta"], np.asarray(C))
    return np.column_stack([pre["intercept"] + team, champ, score])


def fit_state_model(raw, y, gids, t_min, knots=TIME_KNOTS, l2=STATE_L2,
                    smooth=STATE_SMOOTH, maxiter=300,
                    monotone_features=MONOTONE_FEATURES):
    """Fit the bounded smooth coefficient surface.

    ``raw`` columns are PRIOR_INPUTS + STATE_FEATURES.
    """
    from scipy.optimize import minimize

    raw = np.asarray(raw, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    feature_names = PRIOR_INPUTS + STATE_FEATURES
    if raw.shape[1] != len(feature_names):
        raise ValueError("wrong state feature width")
    mean, std, lo, hi = _scale_fit(raw)
    Z = _scale_apply(raw, mean, std, lo, hi)
    B = time_basis(t_min, knots)
    w = _game_balanced_weights(gids)
    nf, nk = Z.shape[1], B.shape[1]

    # rows: time intercept, then one smooth coefficient curve per feature.
    bounds = [(-10.0, 10.0)] * nk
    monotone_features = set(monotone_features)
    for name in feature_names:
        b = (0.0, 10.0) if name in monotone_features else (-10.0, 10.0)
        bounds.extend([b] * nk)

    def objective(flat):
        if not np.all(np.isfinite(flat)):
            return 1e100, np.zeros_like(flat)
        flat = np.clip(flat, -10.0, 10.0)
        theta = flat.reshape(nf + 1, nk)
        coef = np.einsum("ik,fk->if", B, theta[1:], optimize=True)
        eta = np.einsum("ik,k->i", B, theta[0], optimize=True) + np.sum(Z * coef, axis=1)
        resid = w * (_sigmoid(eta) - y)
        loss = np.sum(w * (np.logaddexp(0.0, eta) - y * eta))
        grad = np.empty_like(theta)
        grad[0] = np.einsum("ik,i->k", B, resid, optimize=True)
        grad[1:] = np.einsum("if,ik->fk", Z * resid[:, None], B, optimize=True)

        # Ridge is scale meaningful because every input column is standardized.
        loss += 0.5 * l2 * np.sum(theta[1:] * theta[1:])
        grad[1:] += l2 * theta[1:]
        # Adjacent-knot penalty: coefficient effects vary smoothly with time.
        delta = theta[:, 1:] - theta[:, :-1]
        loss += 0.5 * smooth * np.sum(delta * delta)
        grad[:, :-1] -= smooth * delta
        grad[:, 1:] += smooth * delta
        return float(loss), grad.reshape(-1)

    initial = np.zeros((nf + 1, nk), dtype=np.float64)
    base = np.clip(np.average(y, weights=w), 1e-4, 1 - 1e-4)
    initial[0, :] = np.log(base / (1.0 - base))
    result = minimize(objective, initial.reshape(-1), jac=True, method="L-BFGS-B",
                      bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 2e-5,
                                               "maxcor": 20})
    if not result.success:
        log.warning("state optimizer: %s", result.message)
    coef = np.clip(result.x, -10.0, 10.0)
    if not np.all(np.isfinite(coef)):
        raise FloatingPointError("state optimizer returned non-finite coefficients")
    return {
        "theta": coef.reshape(nf + 1, nk), "feature_names": np.asarray(feature_names),
        "mean": mean, "std": std, "lo": lo, "hi": hi, "knots": np.asarray(knots),
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "objective": float(result.fun), "cal_intercept": 0.0, "cal_slope": 1.0,
    }


def predict_state(model, raw, t_min, components=False):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    Z = _scale_apply(raw, model["mean"], model["std"], model["lo"], model["hi"])
    B = time_basis(t_min, model["knots"])
    effect = np.einsum("ik,fk->if", B, model["theta"], optimize=True)
    parts = np.column_stack([effect[:, 0], Z * effect[:, 1:]])
    slope = float(model.get("cal_slope", 1.0))
    intercept = float(model.get("cal_intercept", 0.0))
    parts *= slope
    parts[:, 0] += intercept
    eta = parts.sum(axis=1)
    if components:
        return _sigmoid(eta), eta, parts
    return _sigmoid(eta)


def _fit_platt_logits(logits, y, gids):
    logits, y, gids = map(np.asarray, (logits, y, gids))
    A = np.column_stack([np.ones(len(y)), logits])
    weights = _game_balanced_weights(gids)
    beta = _fit_static_logit(A, y, np.array([0.0, 1.0]),
                             bounds=[(-5.0, 5.0), (0.05, 5.0)], weights=weights)
    return float(beta[0]), float(beta[1])


def _fit_calibration_intercept(logits, y, gids, slope):
    """Re-center a fixed calibration slope on the final fitted model.

    A temporal calibrator is estimated with an older submodel.  Once the model
    is refit on all available games, carrying that submodel's intercept forward
    would double-count any base-rate drift present in the calibration block.
    """
    logits, y, gids = map(np.asarray, (logits, y, gids))
    weights = _game_balanced_weights(gids)
    intercept = 0.0
    for _ in range(40):
        p = _sigmoid(intercept + slope * logits)
        grad = np.sum(weights * (p - y))
        hess = np.sum(weights * p * (1.0 - p)) + 1e-9
        step = grad / hess
        intercept = float(np.clip(intercept - step, -5.0, 5.0))
        if abs(step) < 1e-9:
            break
    return intercept


def calibrate_state_model(raw, y, gids, t_min, folds=3,
                          l2=STATE_L2, smooth=STATE_SMOOTH):
    """Game-level OOF Platt calibration for the final state surface."""
    raw, y, gids, t_min = map(np.asarray, (raw, y, gids, t_min))
    ug = np.unique(gids)
    gf = _fold_ids(ug, folds, seed=29)
    fold_of = {g: f for g, f in zip(ug, gf)}
    rf = np.asarray([fold_of[g] for g in gids])
    logits = np.zeros(len(y), dtype=np.float64)
    for f in range(gf.max() + 1):
        tr, te = rf != f, rf == f
        sub = fit_state_model(raw[tr], y[tr], gids[tr], t_min[tr],
                              l2=l2, smooth=smooth, maxiter=220)
        logits[te] = predict_state(sub, raw[te], t_min[te], components=True)[1]
    return _fit_platt_logits(logits, y, gids)


def _game_rows(gid):
    gid = np.asarray(gid)
    _, first = np.unique(gid, return_index=True)
    return np.sort(first)


def _temporal_state_calibration(tr_pre, tr_C, tr_y, tr_gid, tr_dates,
                                champ_names, state, y, gid, t_s, seq,
                                cs_by_gid=None,
                                l2=STATE_L2, smooth=STATE_SMOOTH):
    """Calibrate on the newest date block using only earlier games.

    This mirrors deployment more closely than random folds: both the pregame
    stack and the state model that score the calibration games are trained
    strictly on earlier dates.  Complete match dates stay in one block.
    ``cs_by_gid`` carries the main fit's out-of-fold champion-state scores;
    reusing them here (instead of refitting the champion block early-only)
    biases only the calibration estimate, in the conservative direction.
    """
    tr_dates = np.asarray(tr_dates).astype(str)
    if len(tr_dates) < 100 or np.any(tr_dates == ""):
        return None
    n_cal = max(1, int(round(len(tr_dates) * 0.2)))
    cutoff = np.sort(tr_dates)[-n_cal]
    early = tr_dates < cutoff
    if early.sum() < 100 or (~early).sum() < 20:
        return None

    pre = fit_pregame(tr_pre[early], tr_C[early], tr_y[early], champ_names)
    team_oof, champ_oof = oof_pregame(
        tr_pre[early], tr_C[early], tr_y[early], tr_gid[early], champ_names, k=5)
    prior = {g: (pt, pc) for g, pt, pc in
             zip(tr_gid[early], team_oof, champ_oof)}
    _, team, champ = predict_pregame(pre, tr_pre[~early], tr_C[~early])
    prior.update({g: (pre["intercept"] + pt, pc) for g, pt, pc in
                  zip(tr_gid[~early], team, champ)})

    cs_by_gid = cs_by_gid or {}
    training_games = set(tr_gid.tolist())
    rows = (seq < 0) & np.asarray([g in training_games for g in gid])
    row_gid = gid[rows]
    row_prior = np.asarray([prior[g] + (cs_by_gid.get(g, 0.0),)
                            for g in row_gid], dtype=np.float64)
    raw = np.column_stack([row_prior, state[rows]])
    early_games = set(tr_gid[early].tolist())
    fit = np.asarray([g in early_games for g in row_gid])
    model = fit_state_model(raw[fit], y[rows][fit], row_gid[fit],
                            t_s[rows][fit] / 60.0, l2=l2, smooth=smooth)
    logits = predict_state(model, raw[~fit], t_s[rows][~fit] / 60.0,
                           components=True)[1]
    return _fit_platt_logits(logits, y[rows][~fit], row_gid[~fit])


def fit_arrays(X, y, gid, t_s, seq, C, names, champ_names, train_games=None,
               pregame_folds=5, dates=None, state_l2=STATE_L2,
               state_smooth=STATE_SMOOTH, calibration="temporal",
               champ_state_all_rows=CHAMP_STATE_ALL_ROWS,
               champ_state_folds=CHAMP_STATE_FOLDS):
    X, y, gid, t_s, seq, C = map(np.asarray, (X, y, gid, t_s, seq, C))
    game_i = _game_rows(gid)
    game_gid, game_y, game_C = gid[game_i], y[game_i], C[game_i]
    game_dates = np.asarray(dates)[game_i] if dates is not None else None
    game_pre = pregame_values_from_matrix(X[game_i], names)
    if train_games is None:
        train_games = np.ones(len(game_gid), dtype=bool)
    else:
        train_games = np.asarray(train_games, dtype=bool)
    tr_gid = game_gid[train_games]
    tr_pre, tr_C, tr_y = game_pre[train_games], game_C[train_games], game_y[train_games]
    pre_full = fit_pregame(tr_pre, tr_C, tr_y, champ_names)
    pre_team_oof, pre_champ_oof = oof_pregame(tr_pre, tr_C, tr_y, tr_gid, champ_names, k=pregame_folds)

    train_gid_set = set(tr_gid.tolist())
    row_train = np.asarray([g in train_gid_set for g in gid])
    minute_train = (seq < 0) & row_train
    n_champs = len(champ_names)
    cs_rows = np.where(row_train if champ_state_all_rows else minute_train)[0]
    cs_beta = fit_champ_state(X, names, C, y, cs_rows, n_champs)
    cs_by_gid = oof_champ_state(X, names, C, y, gid, cs_rows, tr_gid,
                                n_champs, k=champ_state_folds)

    prior_by_gid = {g: (pt, pc) for g, pt, pc in zip(tr_gid, pre_team_oof, pre_champ_oof)}
    other = ~train_games
    if other.any():
        _, team, champ = predict_pregame(pre_full, game_pre[other], game_C[other])
        prior_by_gid.update({g: (pre_full["intercept"] + pt, pc)
                             for g, pt, pc in zip(game_gid[other], team, champ)})
        other_scores = champ_state_scores(cs_beta, game_C[other])
        cs_by_gid.update({g: float(s) for g, s in zip(game_gid[other], other_scores)})
    row_prior = np.asarray([prior_by_gid[g] + (cs_by_gid[g],) for g in gid],
                           dtype=np.float64)
    state = state_values_from_matrix(X, names)
    state_raw = np.column_stack([row_prior, state])
    raw_train, y_train = state_raw[minute_train], y[minute_train]
    gid_train, time_train = gid[minute_train], t_s[minute_train] / 60.0
    temporal = None
    if calibration == "temporal" and game_dates is not None:
        temporal = _temporal_state_calibration(
            tr_pre, tr_C, tr_y, tr_gid, game_dates[train_games], champ_names,
            state, y, gid, t_s, seq, cs_by_gid=cs_by_gid,
            l2=state_l2, smooth=state_smooth)
    if temporal is None:
        cal_intercept, cal_slope = calibrate_state_model(
            raw_train, y_train, gid_train, time_train,
            l2=state_l2, smooth=state_smooth)
        calibration_used = "game_oof"
    else:
        cal_intercept, cal_slope = temporal
        calibration_used = "temporal_holdout"
    state_model = fit_state_model(raw_train, y_train, gid_train, time_train,
                                  l2=state_l2, smooth=state_smooth)
    if temporal is not None:
        final_logits = predict_state(state_model, raw_train, time_train,
                                     components=True)[1]
        cal_intercept = _fit_calibration_intercept(
            final_logits, y_train, gid_train, cal_slope)
    state_model["cal_intercept"] = cal_intercept
    state_model["cal_slope"] = cal_slope
    state_model["calibration_method"] = calibration_used
    state_model["l2"] = float(state_l2)
    state_model["smooth"] = float(state_smooth)
    return {"pregame": pre_full, "state": state_model,
            "champ_state": {"beta": cs_beta, "cap_min": CHAMP_STATE_CAP_MIN,
                            "l2": CHAMP_STATE_L2},
            "train_games": int(train_games.sum()), "train_states": int(minute_train.sum())}


def save_model(model, path=MODEL_PATH, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pre, state = model["pregame"], model["state"]
    payload = {
        "kind": np.asarray(MODEL_KIND),
        "pre_names": np.asarray(PREGAME_FEATURES),
        "pre_mean": pre["mean"], "pre_std": pre["std"], "pre_lo": pre["lo"], "pre_hi": pre["hi"],
        "pre_intercept": np.asarray(pre["intercept"]), "pre_team_beta": pre["team_beta"],
        "pre_champ_beta": pre["champ_beta"], "champ_names": pre["champ_names"],
        "state_names": state["feature_names"], "state_mean": state["mean"], "state_std": state["std"],
        "state_lo": state["lo"], "state_hi": state["hi"], "theta": state["theta"],
        "time_knots": state["knots"], "cal_intercept": np.asarray(state.get("cal_intercept", 0.0)),
        "cal_slope": np.asarray(state.get("cal_slope", 1.0)),
        "state_l2": np.asarray(state.get("l2", STATE_L2)),
        "state_smooth": np.asarray(state.get("smooth", STATE_SMOOTH)),
        "calibration_method": np.asarray(state.get("calibration_method", "unknown")),
        "champ_state_beta": np.asarray(model["champ_state"]["beta"], dtype=np.float64),
        "champ_state_cap_min": np.asarray(float(model["champ_state"]["cap_min"])),
        "champ_state_l2": np.asarray(float(model["champ_state"]["l2"])),
        "meta": np.asarray(json.dumps(meta or {}, sort_keys=True)),
    }
    np.savez_compressed(path, **payload)
    return path


def load_model(path=MODEL_PATH):
    d = np.load(path, allow_pickle=False)
    kind = str(d["kind"].item())
    if kind != MODEL_KIND:
        raise ValueError("unsupported model kind %s" % kind)
    return {
        "pregame": {
            "mean": d["pre_mean"], "std": d["pre_std"], "lo": d["pre_lo"], "hi": d["pre_hi"],
            "intercept": float(d["pre_intercept"]), "team_beta": d["pre_team_beta"],
            "champ_beta": d["pre_champ_beta"], "champ_names": d["champ_names"],
        },
        "state": {
            "feature_names": d["state_names"], "mean": d["state_mean"], "std": d["state_std"],
            "lo": d["state_lo"], "hi": d["state_hi"], "theta": d["theta"], "knots": d["time_knots"],
            "cal_intercept": float(d["cal_intercept"]) if "cal_intercept" in d.files else 0.0,
            "cal_slope": float(d["cal_slope"]) if "cal_slope" in d.files else 1.0,
            "l2": float(d["state_l2"]) if "state_l2" in d.files else STATE_L2,
            "smooth": float(d["state_smooth"]) if "state_smooth" in d.files else STATE_SMOOTH,
            "calibration_method": (str(d["calibration_method"].item())
                                   if "calibration_method" in d.files else "unknown"),
        },
        "champ_state": {
            "beta": d["champ_state_beta"],
            "cap_min": float(d["champ_state_cap_min"]),
            "l2": float(d["champ_state_l2"]),
        },
        "meta": json.loads(str(d["meta"].item())),
    }


def fit_full(dataset_path=None, model_path=MODEL_PATH):
    dataset_path = dataset_path or os.path.join(OUT_DIR, "states.npz")
    d = np.load(dataset_path, allow_pickle=True)
    t0 = time.time()
    model = fit_arrays(d["X"], d["y"], d["gid"], d["t"], d["seq"], d["C"],
                       list(d["names"]), list(d["champ_names"]),
                       dates=d["date"] if "date" in d.files else None)
    meta = {"fitted_at": int(time.time()), "dataset": os.path.basename(dataset_path),
            "games": model["train_games"], "minute_states": model["train_states"],
            "seconds": round(time.time() - t0, 2), "game_balanced": True,
            "causal_fixed_minutes": True,
            "state_l2": model["state"]["l2"],
            "state_smooth": model["state"]["smooth"],
            "calibration": model["state"]["calibration_method"]}
    path = save_model(model, model_path, meta)
    log.info("wpgam: saved %s (%d games / %d minute states; %.1fs)", path,
             meta["games"], meta["minute_states"], meta["seconds"])
    return path


def _live_champ_row(champ_names, blue_champs, red_champs):
    lut = {_norm_champ(n): i for i, n in enumerate(champ_names)}
    row = np.full((1, 10), -1, dtype=np.int32)
    unknown = []
    for side, champs, off in ((1, blue_champs, 0), (-1, red_champs, 5)):
        for k, champ in enumerate(list(champs)[:5]):
            j = lut.get(_norm_champ(champ))
            if j is None:
                if champ:
                    unknown.append(champ)
            else:
                row[0, off + k] = j
    return row, unknown


def predict_live(state, blue_champs=(), red_champs=(), path=MODEL_PATH):
    model = load_model(path)
    pre = model["pregame"]
    C, unknown = _live_champ_row(pre["champ_names"], blue_champs, red_champs)
    pre_raw = pregame_values_from_live(state)
    _, pre_team, pre_champ = predict_pregame(pre, pre_raw, C)
    pre_team_total = pre["intercept"] + float(pre_team[0])
    champ_state = float(champ_state_scores(model["champ_state"]["beta"], C)[0])
    state_raw = np.concatenate([[pre_team_total, float(pre_champ[0]), champ_state],
                                state_values_from_live(state)])[None, :]
    t_min = float(state.get("t_min", 0.0) or 0.0)
    p, eta, parts = predict_state(model["state"], state_raw, [t_min], components=True)

    # Split the state model's pregame-logit contribution into team and champion
    # pieces without changing their sum.  Centering belongs to the team/prior
    # component; champion contribution combines the pregame champion logit and
    # the in-game champion-state channel.
    lo_prior = float(parts[0, 1])
    lo_champ_pre = float(parts[0, 2])
    lo_champ_state = float(parts[0, 3])
    lo_champ = lo_champ_pre + lo_champ_state
    lo_time = float(parts[0, 0])
    state_parts = dict(zip(STATE_FEATURES, parts[0, 1 + len(PRIOR_INPUTS):]))
    lo_state = float(sum(state_parts.values()))
    lo_deaths = float(sum(state_parts.get(n, 0.0) for n in
                          ("dead_adv", "dead_adv_sq", "dead_count_sq_adv",
                           "dead_base_pressure")))
    lo_baron_active = float(state_parts.get("baron_active", 0.0))
    lo_elder_active = float(state_parts.get("elder_active", 0.0))
    dead_blue = float(state.get("dead_blue", 0) or 0)
    dead_red = float(state.get("dead_red", 0) or 0)
    dead_adv = dead_red - dead_blue
    return {"p_blue": round(float(p[0]), 4), "unknown_champions": unknown,
            "lo_prior": round(lo_prior, 3), "lo_state": round(lo_state, 3),
            "lo_champ": round(lo_champ, 3), "lo_time": round(lo_time, 3),
            "lo_champ_state": round(lo_champ_state, 3),
            "lo_deaths": round(lo_deaths, 3),
            "lo_baron_active": round(lo_baron_active, 3),
            "lo_elder_active": round(lo_elder_active, 3),
            "terminal_state": bool(t_min >= 35.0 and max(dead_blue, dead_red) >= 4.0
                                   and min(dead_blue, dead_red) <= 1.0),
            "model_kind": MODEL_KIND}


def _date_split(d, holdout=0.2):
    gid = d["gid"]
    first = _game_rows(gid)
    game_gid = gid[first]
    if "date" in d.files:
        raw = np.asarray(d["date"])[first].astype(str)
        valid = raw != ""
        if valid.all() and valid.sum() >= 20:
            n_test = max(1, int(round(len(game_gid) * holdout)))
            cutoff = np.sort(raw)[-n_test]
            # Keep an entire match day on one side of the boundary.
            train = raw < cutoff
            if train.any() and (~train).any():
                return first, train, "date"
    order = np.argsort(game_gid, kind="stable")
    method = "game_id_proxy"
    n_test = max(1, int(round(len(game_gid) * holdout)))
    test_idx = order[-n_test:]
    train = np.ones(len(game_gid), dtype=bool)
    train[test_idx] = False
    return first, train, method


def _metric_summary(p, y, gids, bootstrap=1000, seed=23):
    p, y, gids = np.asarray(p), np.asarray(y), np.asarray(gids)
    loss = (p - y) ** 2
    ug = np.unique(gids)
    per_game = np.asarray([loss[gids == g].mean() for g in ug])
    rng = np.random.default_rng(seed)
    draws = rng.choice(per_game, size=(bootstrap, len(per_game)), replace=True).mean(axis=1)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    return {"states": int(len(y)), "games": int(len(ug)), "brier_state": float(loss.mean()),
            "brier_game": float(per_game.mean()),
            "brier_game_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "logloss_state": float(ll)}


def evaluate_walk_forward(dataset_path=None, holdout=0.2, bootstrap=1000):
    dataset_path = dataset_path or os.path.join(OUT_DIR, "states.npz")
    d = np.load(dataset_path, allow_pickle=True)
    first, game_train, method = _date_split(d, holdout)
    model = fit_arrays(d["X"], d["y"], d["gid"], d["t"], d["seq"], d["C"],
                       list(d["names"]), list(d["champ_names"]),
                       train_games=game_train,
                       dates=d["date"] if "date" in d.files else None)
    game_gid = d["gid"][first]
    test_gids = set(game_gid[~game_train].tolist())
    test = (d["seq"] < 0) & np.asarray([g in test_gids for g in d["gid"]])

    pre = model["pregame"]
    def predict_rows(mask):
        prior = prior_values_from_matrix(model, d["X"][mask], list(d["names"]),
                                         d["C"][mask])
        raw = np.column_stack([prior,
                               state_values_from_matrix(d["X"][mask], list(d["names"]))])
        state_p = predict_state(model["state"], raw, d["t"][mask] / 60.0)
        pre_p = _sigmoid(prior[:, 0] + prior[:, 1])
        return state_p, pre_p

    p, p_pre = predict_rows(test)
    y, gids, ts = d["y"][test], d["gid"][test], d["t"][test]
    out = {"split": method, "train_games": model["train_games"],
           "test_games": len(test_gids), "overall": _metric_summary(p, y, gids, bootstrap),
           "pregame_baseline": _metric_summary(p_pre, y, gids, bootstrap)}
    out["by_phase"] = {}
    for name, lo, hi in (("early", 0, 900), ("mid", 900, 1500), ("late", 1500, 10 ** 9)):
        m = (ts >= lo) & (ts < hi)
        if m.any():
            out["by_phase"][name] = _metric_summary(p[m], y[m], gids[m], bootstrap)
    names = {str(n): i for i, n in enumerate(d["names"])}
    test_X = d["X"][test]
    dead_blue = np.clip(test_X[:, names["dead_blue"]], 0, 5)
    dead_red = np.clip(test_X[:, names["dead_red"]], 0, 5)
    late_four_vs_one = ((ts >= 35 * 60) & (np.maximum(dead_blue, dead_red) >= 4)
                        & (np.minimum(dead_blue, dead_red) <= 1))
    out["by_situation"] = {}
    if late_four_vs_one.any():
        out["by_situation"]["late_four_vs_one"] = _metric_summary(
            p[late_four_vs_one], y[late_four_vs_one], gids[late_four_vs_one], bootstrap)
    out["event_aligned"] = {}
    test_game = np.asarray([g in test_gids for g in d["gid"]])
    for platform in ("pm", "ks"):
        aligned = (d["seq"] >= 0) & test_game & np.isfinite(d[platform])
        if aligned.any():
            model_p, _ = predict_rows(aligned)
            event_y, event_gid = d["y"][aligned], d["gid"][aligned]
            out["event_aligned"][platform] = {
                "model": _metric_summary(model_p, event_y, event_gid, bootstrap),
                "market": _metric_summary(d[platform][aligned], event_y, event_gid, bootstrap),
            }
    return out


def report_walk_forward(result):
    print("constrained GAM walk-forward (%s): %d train / %d test games" %
          (result["split"], result["train_games"], result["test_games"]))
    for name, m in [("overall", result["overall"])] + list(result["by_phase"].items()):
        print("  %-8s Brier(state)=%.4f  Brier(game)=%.4f  95%% CI %.4f..%.4f  logloss=%.4f  n=%d/%d games" %
              (name, m["brier_state"], m["brier_game"], m["brier_game_ci95"][0],
               m["brier_game_ci95"][1], m["logloss_state"], m["states"], m["games"]))
    for name, m in result.get("by_situation", {}).items():
        print("  %-20s Brier(state)=%.4f  Brier(game)=%.4f  n=%d/%d games" %
              (name, m["brier_state"], m["brier_game"], m["states"], m["games"]))
    b = result["pregame_baseline"]
    print("  pregame  Brier(state)=%.4f  Brier(game)=%.4f  logloss=%.4f" %
          (b["brier_state"], b["brier_game"], b["logloss_state"]))
    for platform, scores in result.get("event_aligned", {}).items():
        print("  event %-2s model=%.4f market=%.4f (%d points / %d games)" %
              (platform.upper(), scores["model"]["brier_state"],
               scores["market"]["brier_state"], scores["model"]["states"],
               scores["model"]["games"]))
