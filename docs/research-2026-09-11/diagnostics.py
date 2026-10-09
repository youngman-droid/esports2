"""Read-only diagnostics of saved, already-inspected model predictions.

Run from anywhere with the project NumPy environment. This does not fit models,
read the database, score new dates, or modify production/evaluation artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def game_average(values, gids):
    unique, inverse = np.unique(gids, return_inverse=True)
    return unique, np.bincount(inverse, weights=values) / np.bincount(inverse)


def metrics(p, y, gids):
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    _, losses = game_average((p - y) ** 2, gids)
    q = np.clip(p, 1e-6, 1 - 1e-6)
    _, logloss = game_average(-y * np.log(q) - (1 - y) * np.log1p(-q), gids)
    _, residual = game_average(y - p, gids)
    return dict(games=len(losses), states=len(y), brier_game=float(losses.mean()),
                logloss_game=float(logloss.mean()),
                observed_minus_predicted=float(residual.mean()))


def paired(p, ref, y, gids, cluster_map):
    unique, losses = game_average((p - y) ** 2 - (ref - y) ** 2, gids)
    _, inverse = np.unique([cluster_map[int(g)] for g in unique], return_inverse=True)
    totals = np.bincount(inverse, weights=losses)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(20260911)
    draws = rng.integers(len(totals), size=(5000, len(totals)))
    bootstrap = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return dict(delta_brier=float(losses.mean()),
                ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                games=len(unique), clusters=len(totals))


def main(output):
    paths = {
        "dataset": ROOT / "data/wpx/states.npz",
        "plan": ROOT / "data/wpx/methods_v9/plan.json",
        "predictions": ROOT / "data/wpx/methods_v9/rolling_predictions.npz",
        "saved_scores": ROOT / "data/wpx/methods_v9/rolling.json",
        "resources": ROOT / "data/wpx/adapt_cache/408b0e7b37ff020354920067/resources.npz",
        "major_dataset": ROOT / "data/wpx/major_v9/states.npz",
        "model": ROOT / "data/wpx/model_live_gam.npz",
        "manifest": ROOT / "data/wpx/live_stack.json",
        "registry": ROOT / "data/wpx/evaluation_registry.json",
    }
    hashes = {key: sha256(path) for key, path in paths.items()}
    plan = json.loads(paths["plan"].read_text())
    assert hashes["dataset"] == plan["dataset_sha256"]
    assert hashes["resources"] == plan["resources_sha256"]
    with np.load(paths["predictions"]) as archive:
        saved = {key: archive[key] for key in archive.files}
    gids, y = saved.pop("gid"), saved.pop("y")
    with np.load(paths["dataset"]) as archive:
        fixed = archive["seq"] < 0
        use = fixed & np.isin(archive["gid"], np.unique(gids))
        np.testing.assert_array_equal(archive["gid"][use], gids)
        np.testing.assert_array_equal(archive["y"][use], y)
        dates = archive["date"][use]
        names = list(archive["names"])
        X = archive["X"][use]
        league = archive["league"][use]
        all_games = len(np.unique(archive["gid"]))
        fixed_states = int(fixed.sum())
    assert max(dates) <= plan["consumed_through"]
    with np.load(paths["resources"]) as archive:
        clusters = dict(zip(archive["series_gid"].astype(int), archive["series_key"].astype(str)))
    with np.load(paths["major_dataset"]) as archive:
        major = np.isin(gids, np.unique(archive["gid"]))
    computed = {key: metrics(p, y, gids) for key, p in saved.items()}
    originals = json.loads(paths["saved_scores"].read_text())
    for key, m in computed.items():
        for field in ("games", "states", "brier_game", "logloss_game"):
            np.testing.assert_allclose(m[field], originals[key]["metrics"][field], atol=1e-12, rtol=0)
    core, rich = saved["constrained_gam"], saved["rich_gam"]
    hp = X[:, names.index("has_hp")] > .5
    prior_signal = ((np.abs(X[:, names.index("elo_oe")]) > 1e-12)
                    | (np.abs(X[:, names.index("pelo_oe")]) > 1e-12))
    masks = {"all": np.ones(len(gids), dtype=bool), "major": major,
             "nonmajor": ~major, "hp_present": hp, "hp_missing": ~hp,
             "oe_prior_nonzero": prior_signal, "oe_prior_missing_or_even": ~prior_signal}
    months = np.array([date[:7] for date in dates])
    for month in np.unique(months):
        masks["excluding_" + month] = months != month
    comparisons = {}
    for name, mask in masks.items():
        comparisons[name] = dict(core=metrics(core[mask], y[mask], gids[mask]),
                                 rich=metrics(rich[mask], y[mask], gids[mask]),
                                 paired=paired(rich[mask], core[mask], y[mask], gids[mask], clusters))
    _, first = np.unique(gids, return_index=True)
    league_counts = {str(value): int(np.sum(league[first] == value)) for value in np.unique(league)}
    result = dict(
        protocol="Exploratory saved-prediction audit; no new model selection or confirmatory outcome block.",
        cohort=dict(all_dataset_games=all_games, all_dataset_fixed_states=fixed_states,
                    replay_games=len(first), replay_states=len(gids),
                    first_date=str(min(dates)), last_date=str(max(dates))),
        notes=["Slice games are rebalanced within each slice; HP-present and HP-missing games can overlap.",
               "OE prior missing-or-even is a feature proxy, not a verified missingness label.",
               "Leave-month-out checks remove scored rows only; they do not refit monthly models.",
               "Cluster intervals are descriptive and unadjusted for multiple inspected slices."],
        input_sha256=hashes, source_sha256=sha256(Path(__file__).resolve()),
        verified_saved_metrics=computed, rich_versus_core=comparisons,
        replay_league_game_counts=league_counts)
    for key in ("dataset", "model", "manifest", "registry"):
        assert sha256(paths[key]) == hashes[key]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(cohort=result["cohort"], comparisons=comparisons), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
