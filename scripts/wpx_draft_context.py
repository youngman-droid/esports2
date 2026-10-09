"""Isolated composition/Fearless screen on already-consumed saved draft games.

Reconstructs and verifies the existing frozen, player-controlled draft baseline.
The residual uses that fixed offset; training offsets are in-sample predictions
from its original pre-January fit, explicitly not an independent OOF control.
Validation chooses baseline unless the paired upper Brier bound is negative
and log loss does not regress. Later consumed outcomes remain diagnostics.
No database writes, new game collection, production promotion or registration.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpcomposition as composition, wpfearless as fearless
from scripts import draft_comfort_compare as control


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_baseline(games, directory):
    """Verify current reconstruction against saved baseline validation scores."""
    directory = Path(directory)
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        vocabulary = archive["vocabulary"].astype(str).tolist()
        player_scale, nbase, beta = archive["player_scale"], int(archive["nbase"]), archive["draft"]
    if nbase != 5 or beta.shape != (nbase + len(vocabulary),):
        raise ValueError("Expected frozen player-controlled draft baseline")
    lookup = {name: i for i, name in enumerate(vocabulary)}
    ii, jj, vv = [], [], []
    history = control.history_features(games)
    base = []
    for i, game in enumerate(games):
        h = history[game["id"]]
        base.append(np.r_[1., h[0], h[3] / player_scale])
        for name, value in control.signed_draft(game).items():
            if name in lookup:
                ii.append(i)
                jj.append(lookup[name])
                vv.append(value)
    draft = sparse.csr_matrix((vv, (ii, jj)), shape=(len(games), len(vocabulary)))
    baseline = expit(sparse.hstack((sparse.csr_matrix(base), draft), format="csr") @ beta)
    report = json.loads((directory / "report.json").read_text())
    if report.get("train_before") != "2026-01-16" or report.get("diagnostic_before") != "2026-09-03":
        raise ValueError("Baseline consumed-data split contract differs")
    y = np.array([g["y"] for g in games])
    for label, mask in (("validation", np.array(["2026-01-16" <= g["day"] < "2026-05-01" for g in games])),
                        ("diagnostic", np.array(["2026-05-01" <= g["day"] < "2026-09-03" for g in games]))):
        measured = control.metrics(y[mask], baseline[mask])
        reference = report["models"]["draft"][label]
        if (measured["games"] != reference["games"] or not np.isfinite([measured["brier"], measured["logloss"]]).all()
                or any(abs(measured[key] - reference[key]) > 1e-10 for key in ("brier", "logloss"))):
            raise ValueError("Current baseline reconstruction does not reproduce its saved metrics")
    return np.asarray(baseline)


def extract(games, source_root, bundles=None):
    """Only saved inputs enter; absent Fearless certificates stay unavailable."""
    source_root, bundles = Path(source_root), bundles or {}
    catalogs, annotations = {}, {}
    comp_rows, fearless_rows, coverage, reasons = [], [], Counter(), Counter()
    sources = {}
    for i, game in enumerate(games):
        patch = game.get("patch") or ""
        parsed = composition._patch(patch)
        version = ".".join(map(str, parsed + (1,) if len(parsed or ()) == 2 else parsed or ()))
        if version not in catalogs:
            path = source_root / "catalogs" / (version + ".json")
            labels = source_root / "annotations" / (version + ".json")
            catalogs[version] = composition.load_catalog(path) if path.exists() else None
            annotations[version] = json.loads(labels.read_text()) if labels.exists() else None
            for source in (path, labels):
                if source.exists():
                    sources[str(source)] = sha(source)
        # The archive preserves day, not draft seconds; use day opening as a
        # conservative source-availability gate rather than guessing a clock.
        draft_ts = datetime.fromisoformat(game["day"]).replace(tzinfo=timezone.utc).timestamp()
        comp = composition.features(game["blue"]["picks"], game["red"]["picks"], patch,
                                     catalog=catalogs[version], annotations=annotations[version], as_of_ts=draft_ts)
        bundle = bundles.get(str(game["id"]))
        if bundle is not None:
            context = bundle.get("context") or {}
            valid = (context.get("archive_game_id") == game["id"]
                     and context.get("blue_archive_team") == game["blue_team"]
                     and context.get("red_archive_team") == game["red_team"]
                     and context.get("picks") == {s: game[s]["picks"] for s in ("blue", "red")}
                     and context.get("players") == {s: {r: str(p) for r, p in game[s]["players"].items()} for s in ("blue", "red")})
            if not valid:
                raise ValueError("Fearless certificate does not identify the same archived draft and players")
            fear = fearless.features(context, bundle.get("prior_games"), bundle.get("rules"),
                                     bundle.get("player_history"), draft_ts=draft_ts)
        else:
            fear = fearless._empty("missing_archived_series_rule_certificate")
        comp_rows.append(comp["features"])
        fearless_rows.append(fear["features"])
        coverage["composition_any"] += int(comp["available"])
        coverage["composition_complete"] += int(comp["complete"])
        coverage["fearless_any"] += int(fear["available"])
        for name, known in zip(comp["names"], comp["known"]):
            coverage[name] += int(known)
        reasons["composition:" + comp["reason"]] += 1
        reasons["fearless:" + fear["reason"]] += 1
        if (i + 1) % 5000 == 0:
            print("Captured %d already-consumed drafts" % (i + 1), flush=True)
    return (np.asarray(comp_rows), np.asarray(fearless_rows), dict(coverage), dict(reasons), sources)


def run(games, baseline, extras, output):
    y = np.array([g["y"] for g in games])
    gid = np.array([g["id"] for g in games])
    train = np.array([g["day"] < "2026-01-16" for g in games])
    validation = np.array(["2026-01-16" <= g["day"] < "2026-05-01" for g in games])
    later = np.array(["2026-05-01" <= g["day"] < "2026-09-03" for g in games])
    report = dict(protocol="exploratory consumed-history frozen-draft residual; no promotion",
                  training_offset="original baseline in-sample pre-January predictions; not independent OOF predictions",
                  train_before="2026-01-16", validation_before="2026-05-01", diagnostic_before="2026-09-03",
                  selection="validation upper paired Brier bound < 0 and no logloss regression; otherwise baseline",
                  production_changed=False, candidates={})
    report.update(games=len(games), min_outcome_date=min(g["day"] for g in games),
                  max_outcome_date=max(g["day"] for g in games))
    probabilities = dict(baseline=baseline)
    for name, module in (("composition", composition), ("fearless", fearless)):
        extra = extras[name]
        if not np.any(extra[train] != 0):
            report["candidates"][name] = dict(evaluated=False, reason="no_supported_training_features")
            probabilities[name] = baseline.copy()
            continue
        model = module.fit(extra[train], baseline[train], y[train], gid[train], spec=dict(l2=800., smooth=70.))
        module.save(model, output / (name + ".npz"))
        p = module.predict(model, extra, baseline)
        restored = module.load(output / (name + ".npz"))
        if not np.array_equal(p, module.predict(restored, extra, baseline)):
            raise ValueError("Residual artifact roundtrip differs")
        if not np.array_equal(p[~np.any(extra != 0, axis=1)], baseline[~np.any(extra != 0, axis=1)]):
            raise ValueError("Missing-source baseline fallback differs")
        result = dict(evaluated=True, training_supported=int(np.any(extra[train] != 0, axis=1).sum()))
        for label, mask in (("validation", validation), ("diagnostic", later)):
            compare = control.interval([g for g, keep in zip(games, mask) if keep],
                                       ((p - y) ** 2 - (baseline - y) ** 2)[mask])
            result[label] = dict(metrics=control.metrics(y[mask], p[mask]),
                                 baseline=control.metrics(y[mask], baseline[mask]), paired=compare)
        report["candidates"][name] = result
        probabilities[name] = p
    eligible = [name for name, result in report["candidates"].items()
                if result.get("evaluated") and result["validation"]["paired"]["ci95"][1] < 0
                and result["validation"]["metrics"]["logloss"] <= result["validation"]["baseline"]["logloss"]]
    report["selected"] = min(eligible, key=lambda name: report["candidates"][name]["validation"]["metrics"]["brier"]) if eligible else "baseline"
    np.savez_compressed(output / "predictions.npz", gid=gid, y=y, composition=extras["composition"],
                        fearless=extras["fearless"], **{"p_" + k: v for k, v in probabilities.items()})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/wpx/draft_comfort_comparison_20260916/inputs.json")
    parser.add_argument("--baseline-dir", default="data/wpx/draft_comfort_comparison_20260916/player_controlled")
    parser.add_argument("--source-root", default="data/composition")
    parser.add_argument("--fearless-bundles", help="Explicit certified archived series/rule/player snapshots keyed by game ID")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    games = json.loads(Path(args.input).read_text())
    if not games or any(not isinstance(g.get("day"), str) or g["day"] >= "2026-09-03" for g in games):
        raise ValueError("Only already-consumed outcomes through September 2, 2026 may enter")
    if len({g["id"] for g in games}) != len(games):
        raise ValueError("Duplicate archived game IDs")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), Path(composition.__file__), Path(fearless.__file__),
                    Path("lol_ticker/wpresidual.py"), Path("scripts/draft_comfort_compare.py"), Path("lol_ticker/draft.py"),
                    Path(args.input), Path(args.baseline_dir) / "model.npz", Path(args.baseline_dir) / "report.json"]
    if args.fearless_bundles:
        source_paths.append(Path(args.fearless_bundles))
    hashes = {str(path): sha(path) for path in source_paths}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in source_paths:
        if path.suffix == ".py":
            (snapshot / path.name).write_bytes(path.read_bytes())
    baseline = frozen_baseline(games, args.baseline_dir)
    bundles = json.loads(Path(args.fearless_bundles).read_text()) if args.fearless_bundles else None
    comp, fear, coverage, reasons, sources = extract(games, args.source_root, bundles)
    report = run(games, baseline, dict(composition=comp, fearless=fear), output)
    report.update(coverage=coverage, coverage_reasons=reasons, source_catalogs=sources)
    (output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2))
    hashes.update(sources)
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError("An evaluated dependency changed during the run")
    (output / "completion.json").write_text(json.dumps(dict(completed=True, production_changed=False,
        dependencies=hashes, output_hashes={str(path.name): sha(path) for path in output.iterdir() if path.is_file()}), sort_keys=True, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
