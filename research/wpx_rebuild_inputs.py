"""Rebuild consumed historical inputs into an immutable, versioned artifact.

Example: python3 scripts/wpx_rebuild_inputs.py --before 2026-09-03 \
    --out data/wpx/states_inputs_v2_before_2026-09-03.npz
"""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from lol_ticker import db, wpx, wpx_inputs


def audit(path):
    with np.load(path, allow_pickle=False) as z:
        names = z["names"].tolist()
        used = z["X"][:, names.index("has_hp")] > 0
        clock = z["prediction_clock_s"]
        lo, hi, age = z["hp_clock_lo_s"], z["hp_clock_hi_s"], z["hp_age_upper_s"]
        future = used & (hi > clock)
        assert not future.any(), "future observation joined"
        assert np.all(np.isfinite(lo[used]) & np.isfinite(hi[used]))
        assert np.all((0 <= lo[used]) & (lo[used] <= hi[used]))
        assert np.allclose(age[used], clock[used] - lo[used])
        assert np.all((age[used] >= 0) & (age[used] <= 90))
        # Stored wall timestamps are integral; +1 safely covers their rounding.
        assert np.all(z["hp_observation_ts"][used] + 1 - z["hp_origin_ts"][used] <= clock[used])
        assert np.all(z["X"][:, [names.index("items_done"), names.index("item_gold_k")]] == 0)
        assert str(z["input_contract_sha256"].item()) == wpx_inputs.INPUT_CONTRACT_SHA256
        assert np.all(z["X"][:, [names.index("dead_blue"), names.index("dead_red")]] <= 5)
        cutoff = str(z["cutoff_exclusive"].item())
        if cutoff:
            assert np.all(z["date"] < cutoff)
        fixed = z["seq"] < 0
        assert np.array_equal(clock, np.maximum(0, z["t"] - (z["seq"] >= 0))), "prediction clock mismatch"
        return {"rows": len(clock), "games": len(np.unique(z["gid"])), "hp_rows": int(used.sum()),
                "fixed_rows": int(fixed.sum()), "hp_fixed_rows": int((fixed & used).sum()),
                "future_timestamp_joins": int(future.sum()), "max_hp_age_upper_s": float(age[used].max()) if used.any() else None,
                "input_contract_sha256": wpx_inputs.INPUT_CONTRACT_SHA256, "cutoff_exclusive": cutoff}


def audit_source_joins(conn, path):
    """Independently validate every accepted HP vector against its raw source.

    This uses vector inequalities at the stored interval endpoints, rather than
    calling the builder's bracket-selection implementation again.
    """
    from collections import defaultdict
    with np.load(path, allow_pickle=False) as z:
        names = z["names"].tolist(); X = z["X"]
        used = X[:, names.index("has_hp")] > 0
        gids = z["gid"][used]; ts = z["hp_observation_ts"][used]
        lows, highs = z["hp_clock_lo_s"][used], z["hp_clock_hi_s"][used]
        needed = sorted(set(int(g) for g in gids))
        raw = {}
        for r in conn.execute("""SELECT fg.golgg_game_id gid,fm.ts,fm.data
            FROM feed_games fg JOIN feed_minutes fm USING(esports_game_id)
            WHERE fg.status='scraped' AND fg.golgg_game_id=ANY(%s)""", (needed,)):
            key = (r["gid"], r["ts"])
            assert key not in raw or raw[key] == r["data"], "ambiguous raw HP observation"
            raw[key] = r["data"]
        timelines = defaultdict(dict)
        for r in conn.execute("SELECT game_id,slot,minute,gold FROM golgg_timeline WHERE game_id=ANY(%s)", (needed,)):
            timelines[r["game_id"]].setdefault(r["minute"] * 60, [np.nan] * 10)[r["slot"]] = r["gold"]
        timeline_arrays = {gid: (np.array(sorted(values)), np.array([values[t] for t in sorted(values)]))
                           for gid, values in timelines.items()}
        unique = np.unique(np.column_stack((gids, ts, lows, highs)), axis=0)
        for gid, observation_ts, lo, hi in unique:
            row = raw[(int(gid), int(observation_ts))]
            observed = np.asarray(row["gdb"] + row["gdr"])
            times, values = timeline_arrays[int(gid)]
            lower = values[times == lo]
            assert len(lower) == 1 and np.all(lower[0] <= observed) and np.any(lower[0] < observed)
            prefix = values[times <= hi]
            assert np.all(np.isfinite(prefix)) and np.all(np.diff(prefix, axis=0) >= 0)
            assert np.any(np.all(prefix >= observed, axis=1) & np.any(prefix > observed, axis=1))
        expected = []
        for gid, observation_ts in zip(gids, ts):
            row = raw[(int(gid), int(observation_ts))]
            blue, red = row["hpb"], row["hpr"]
            lb, lr = row.get("lvb") or [], row.get("lvr") or []
            levels = ((sum(lb) - sum(lr)) / 5 if len(lb) == len(lr) == 5
                      and all(isinstance(x, (int, float)) for x in lb + lr) else 0)
            expected.append([sum(blue) - sum(red), sum(x < .3 for x in blue), sum(x < .3 for x in red), levels, 1])
        actual = X[used][:, [names.index(n) for n in ("hp_pool", "hp_low_b", "hp_low_r", "lvl_k", "has_hp")]]
        np.testing.assert_allclose(actual, np.asarray(expected), atol=1e-6)
        return {"source_verified_hp_rows": len(gids), "unique_observation_intervals": len(unique),
                "source_verified_games": len(needed), "raw_feature_mismatches": 0,
                "gold_bound_violations": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True, help="Exclusive consumed-history cutoff")
    ap.add_argument("--out", required=True)
    ap.add_argument("--origin-cache-dir", default="data/wpx/feed_origins_v1")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--verify-source", action="store_true", help="Recheck all accepted HP rows against raw DB inputs")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not a.audit_only:
        with db.connect() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            wpx.build(conn, output_path=a.out, before=a.before, origin_cache_dir=a.origin_cache_dir)
    report = audit(a.out)
    if a.verify_source:
        with db.connect() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            report.update(audit_source_joins(conn, a.out))
    report["sha256"] = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
