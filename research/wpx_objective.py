"""Isolated objective/trend screens on audited, already-consumed corrected rows.

No production dispatch, registration, outcome inventory or database write.
Use --coverage-only to inspect actual historical support without fitting.
Objective events require a supplied completeness manifest or --read-only-events;
the database query reads only input events belonging to corrected archive IDs.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpobjective, wptrend, wpresidual, wpgam, wpbench
from research import wpx_combat as historical
from research import wpx_methods_v9 as methods

log = logging.getLogger("opportunities")
SPEC = dict(l2=800., smooth=70.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_states(rows, gold):
    idx = {n: i for i, n in enumerate(rows["names"])}
    if gold.shape != (len(rows["gid"]), 10) or not np.isfinite(gold).all() or np.any(gold < 0):
        raise ValueError("Complete, matching canonical individual gold is required")
    if not np.allclose((gold[:, :5].sum(axis=1)-gold[:, 5:].sum(axis=1))/1000.,
                       rows["X"][:, idx["gold_k"]], atol=2e-5, rtol=0):
        raise ValueError("Individual gold disagrees with corrected baseline rows")
    for i in np.lexsort((rows["prediction_clock_s"], rows["gid"])):
        # ts here is an offline monotone frame key in the game's clock, not
        # a fabricated historical wall timestamp. It is never market-aligned.
        clock = float(rows["prediction_clock_s"][i])
        state = dict(game_id=int(rows["gid"][i]), clock_s=clock, ts=clock,
                     gold_blue=float(gold[i, :5].sum()), gold_red=float(gold[i, 5:].sum()),
                     kills=float(rows["X"][i, idx["d_kill"]]),
                     towers=float(rows["X"][i, idx["d_tower"]]),
                     dragons=float(rows["X"][i, idx["d_dragon"]]),
                     barons=float(rows["X"][i, idx["d_baron"]]),
                     elders=float(rows["X"][i, idx["d_elder"]]),
                     inhibs=float(rows["X"][i, idx["d_inhib"]]))
        state["counter_differences"] = {kind: float(rows["X"][i, idx["d_"+kind]])
                                       for kind in ("kill", "tower", "dragon", "baron", "elder", "inhib")}
        yield int(i), state


def extract_trend(rows, gold):
    extra = np.zeros((len(rows["gid"]), len(wptrend.FEATURE_NAMES)), dtype=np.float32)
    known = np.zeros_like(extra, dtype=bool)
    tracker = wptrend.new_tracker()
    reset_count = 0
    for i, state in _source_states(rows, gold):
        observation = wptrend.observation_from_state(state)
        observation["gold_roles"] = gold[i].astype(float).tolist()
        block = wptrend.capture(state, tracker, attempt_id=state["game_id"], observation=observation,
                               max_gap_s=90., anchor_slack_s=0.)
        extra[i], known[i] = block["features"], block["known"]
        reset_count += int(block["reset"])
    return dict(extra=extra, known=known, available=known.any(axis=1), resets=reset_count,
                limitations=["minute sources support exact120/300s anchors, not30s",
                    "same-clock HP histories are unavailable in the aged corrected telemetry archive",
                    "historical role-slot identity retains the canonical archive's assumptions",
                    "lead volatility is an inspection summary; not a signed residual input"])


def read_events(gids, *, event_file=None, read_only_events=False):
    if event_file is not None:
        payload = json.loads(Path(event_file).read_text())
        if not payload.get("source") or not isinstance(payload.get("events"), dict):
            raise ValueError("Events file needs source, events and complete_game_ids metadata")
        allowed = set(map(int, gids))
        records = {int(g): v for g, v in payload["events"].items() if int(g) in allowed}
        complete = set(map(int, payload.get("complete_game_ids") or [])) & allowed
        return records, complete, dict(source=payload["source"], file_sha256=sha(event_file))
    if not read_only_events:
        raise ValueError("Objective timing requires --events-json or explicit --read-only-events")
    import psycopg
    from psycopg.rows import dict_row
    from lol_ticker import config
    records = {}
    with psycopg.connect(config.PG_DSN, row_factory=dict_row,
                        options="-c default_transaction_read_only=on -c statement_timeout=60000") as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        for event in conn.execute("""SELECT game_id,seq,time_s,action,side FROM golgg_events
            WHERE game_id=ANY(%s) AND (action='baron' OR action LIKE 'dragon%%')
            ORDER BY game_id,seq""", (list(map(int, gids)),)):
            gid = int(event.pop("game_id"))
            records.setdefault(gid, []).append(event)
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # Event completeness remains a source assumption; the adapter independently
    # checks all corrected objective counts and active buffs before accepting it.
    return records, set(records), dict(source="read_only_golgg_events_for_corrected_game_ids", sha256=digest)


def extract_objective(rows, events, complete):
    extra = np.zeros((len(rows["gid"]), len(wpobjective.FEATURE_NAMES)), dtype=np.float32)
    known = np.zeros_like(extra, dtype=bool)
    idx = {n: i for i, n in enumerate(rows["names"])}
    rejected = []
    ordered = np.argsort(rows["gid"], kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(rows["gid"][ordered]) != 0)+1, len(ordered)]
    for start, end in zip(boundaries, boundaries[1:]):
        indices = ordered[start:end]
        gid = rows["gid"][indices[0]]
        if int(gid) not in complete:
            continue
        source = events.get(int(gid), [])
        blocks, valid = [], True
        for i in indices:
            clock = float(rows["prediction_clock_s"][i])
            counters = {k: [0, 0] for k in ("dragon", "baron", "elder")}
            acquired = {k: [None, None] for k in ("baron", "elder")}
            for e in source:
                t, side, action = e.get("time_s"), e.get("side"), e.get("action") or ""
                if t is None or t < 0 or t > clock or side not in ("blue", "red"):
                    continue
                kind = "elder" if action == "dragon:elder" else "dragon" if action.startswith("dragon") else action
                if kind in counters:
                    slot = 0 if side == "blue" else 1
                    counters[kind][slot] += 1
                    if kind in acquired:
                        old = acquired[kind][slot]
                        acquired[kind][slot] = max(float(t), old) if old is not None else float(t)
            expected = {"d_"+k: v[0]-v[1] for k, v in counters.items()}
            expected.update(drag_blue=counters["dragon"][0], drag_red=counters["dragon"][1])
            block = wpobjective.historical_features(source, clock, events_complete=True)
            for kind, column in (("baron", "baron_active"), ("elder", "elder_buff")):
                # The legacy historical binary uses a closed expiry boundary;
                # a new remaining-seconds feature is zero at that boundary.
                b, r = acquired[kind]
                duration = wpobjective.BUFF_SECONDS[kind]
                expected[column] = int(b is not None and clock-b <= duration)-int(r is not None and clock-r <= duration)
            if any(abs(float(rows["X"][i, idx[k]])-v) > 2e-5 for k, v in expected.items()):
                valid = False; break
            blocks.append(block)
        if not valid:
            rejected.append(int(gid)); continue
        for i, block in zip(indices, blocks):
            extra[i], known[i] = block["features"], block["known"]
    return dict(extra=extra, known=known, available=known.any(axis=1), rejected_game_ids=rejected,
                limitations=["historical event-stream completeness is a source assumption checked against all corrected objective counts/active buffs",
                    "no patch-certified spawn profiles in this initial screen; spawn/soul-window columns remain unavailable",
                    "aged historical HP cannot certify same-clock objective/readiness couplings",
                    "remaining team buff time is an opportunity proxy; individual buff holders unknown"])


def coverage(rows, features, names, mask=None):
    mask = np.ones(len(rows["gid"]), dtype=bool) if mask is None else np.asarray(mask)
    available = mask & features["available"]
    return dict(states=int(mask.sum()), games=int(len(np.unique(rows["gid"][mask]))),
                available_states=int(available.sum()), available_games=int(len(np.unique(rows["gid"][available]))),
                nonzero_states=int((mask & np.any(features["extra"] != 0, axis=1)).sum()),
                known_states={name: int((mask & features["known"][:, j]).sum()) for j, name in enumerate(names)})


def stage(output, label, block, rows, features, series, cache, baseline_plan, module):
    target = output/label; target.mkdir()
    baseline, masks, calibration, hashes, _, _ = historical.baseline_stage(cache, label, block, rows, baseline_plan)
    fit, val = masks["fit"], masks["validation"]
    log.info("%s %s fit: %d training/%d scored states; %d nonzero training rows", module.KIND, label,
             int(fit.sum()), int(val.sum()), int(np.any(features["extra"][fit] != 0, axis=1).sum()))
    model = module.fit(features["extra"][fit], baseline["fit"], rows["y"][fit], rows["gid"][fit], rows["t_min"][fit], spec=SPEC)
    module.save(model, target/"residual.npz")
    raw = module.predict(module.load(target/"residual.npz"), features["extra"][val], baseline["validation"], rows["t_min"][val])
    ref, predicted = wpbench._apply_platt(baseline["validation"], calibration), wpbench._apply_platt(raw, calibration)
    missing = ~features["available"][val]
    if not np.array_equal(predicted[missing], ref[missing]):
        raise ValueError("Unavailable rows changed their frozen-core baseline probability")
    scores = dict(metrics=methods.metrics(predicted, rows["y"][val], rows["gid"][val]),
                  baseline=methods.metrics(ref, rows["y"][val], rows["gid"][val]),
                  paired=methods.paired(predicted, ref, rows["y"][val], rows["gid"][val], series))
    np.savez_compressed(target/"predictions.npz", gid=rows["gid"][val], t=rows["t"][val], baseline=ref, candidate=predicted)
    result = dict(block=block, baseline_hashes=hashes, calibration=calibration, scores=scores,
                  missing_fallback_exact=True, fit_coverage=coverage(rows, features, module.FEATURE_NAMES, fit),
                  validation_coverage=coverage(rows, features, module.FEATURE_NAMES, val))
    methods.dump(target/"report.json", result)
    log.info("%s %s paired delta Brier %.8f", module.KIND, label, scores["paired"]["delta_brier"])
    return result


def main(args):
    module = wpobjective if args.family == "objective" else wptrend
    output, dataset, cache = args.out, args.dataset, args.baseline_cache
    if output.exists():
        raise FileExistsError("Use a new isolated output directory")
    sources = {str(Path(m.__file__)): sha(m.__file__) for m in (module, wpresidual, wpgam, wpbench, historical, methods)}
    sources[str(Path(__file__))] = sha(__file__)
    protected_paths = [Path(wpgam.MODEL_PATH), Path(wpgam.OUT_DIR)/"model_live.npz", Path(wpgam.OUT_DIR)/"live_stack.json",
                       Path(wpgam.OUT_DIR)/"evaluation_registry.json", Path(wpgam.OUT_DIR)/"historical_evaluation_registry.json"]
    protected = {str(p): sha(p) for p in protected_paths}
    rows, manifest = historical.load_rows(dataset)
    baseline_plan = json.loads((cache/"plan.json").read_text())
    if baseline_plan["dataset_sha256"] != sha(dataset):
        raise ValueError("Corrected dataset does not match the saved chronological baseline")
    resources = cache.parent/"resources"/"resources.npz"
    if sha(resources) != baseline_plan["resources_sha256"]:
        raise ValueError("Canonical role-gold/series resource cache changed")
    with np.load(resources, allow_pickle=False) as z:
        gold = z["gold"]
        series = dict(zip(z["series_gid"].astype(int), z["series_key"].astype(str)))
    source = dict(resources_sha256=sha(resources))
    if args.family == "trend":
        features = extract_trend(rows, gold)
    else:
        events, complete, event_source = read_events(np.unique(rows["gid"]), event_file=args.events_json,
                                                    read_only_events=args.read_only_events)
        features = extract_objective(rows, events, complete); source.update(events=event_source)
    output.mkdir(parents=True)
    if args.family == "objective":
        methods.dump(output/"objective_event_inputs.json", dict(source=event_source["source"],
            source_provenance=event_source, events=events, complete_game_ids=sorted(complete)))
    np.savez_compressed(output/"features.npz", gid=rows["gid"], t=rows["t"], extra=features["extra"],
                        known=features["known"], available=features["available"], names=np.array(module.FEATURE_NAMES))
    snapshots = output/"source_snapshot"; snapshots.mkdir()
    for path in sources:
        (snapshots/Path(path).name).write_bytes(Path(path).read_bytes())
    plan = dict(family=args.family, spec=SPEC, input_contract=module.INPUT_CONTRACT, dataset_sha256=sha(dataset),
                manifest_sha256=sha(str(dataset)+".manifest.json"), baseline_plan_sha256=sha(cache/"plan.json"),
                features_sha256=sha(output/"features.npz"), input_sources=source, sources=sources, protected=protected,
                limitations=features["limitations"]+[manifest["identity_limitation"], "consumed historical diagnostic, not promotion evidence"],
                coverage=coverage(rows, features, module.FEATURE_NAMES),
                capture_limits=dict(max_gap_s=90, anchor_slack_s=0) if args.family == "trend" else None)
    methods.dump(output/"plan.json", plan)
    report = dict(completed=True, coverage_only=bool(args.coverage_only), coverage=plan["coverage"],
                  limitations=plan["limitations"], production_changed=False, registration_changed=False,
                  rejected_games=len(features.get("rejected_game_ids", [])))
    if not args.coverage_only:
        development = [stage(output, "development_"+str(i+1), block, rows, features, series, cache, baseline_plan, module)
                       for i, block in enumerate(baseline_plan["development"])]
        latest = development[-1]["scores"]
        selected = ("residual" if latest["paired"]["ci95"][1] < 0 and
                    latest["metrics"]["logloss_game"] <= latest["baseline"]["logloss_game"] else "baseline")
        methods.dump(output/"selection.json", dict(selected=selected, stage="development_3", frozen=True))
        report.update(selected=selected, development=development,
            frozen=stage(output, "frozen", baseline_plan["final"], rows, features, series, cache, baseline_plan, module))
    if any(sha(path) != value for path, value in protected.items()):
        raise ValueError("Protected production artifacts changed during the screen")
    if any(sha(path) != value for path, value in sources.items()):
        raise ValueError("Evaluated source changed during the screen; use a fresh run")
    methods.dump(output/"report.json", report)
    methods.dump(output/"completion.json", dict(completed=True,
        artifacts={str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
        sources=sources, production_changed=False, registration_changed=False))
    log.info("Completed %s %s: %s", args.family, "coverage" if args.coverage_only else "screen", plan["coverage"])


def parse_args(default_family="objective"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("objective", "trend"), default=default_family)
    parser.add_argument("--dataset", type=Path, default=Path(wpgam.OUT_DIR)/"states_inputs_v2_canonical_before_2026-09-03.npz")
    parser.add_argument("--baseline-cache", type=Path, default=Path(wpgam.OUT_DIR)/"action_20260911/resource_study")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--events-json", type=Path)
    parser.add_argument("--read-only-events", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main(parse_args())
