"""Shared, append-only outcome exposure and frozen evaluation contracts.

Legacy ledgers stay untouched. Their union is imported conservatively: a date
watermark quarantines even games absent from today's dataset, while explicit
canonical IDs prevent a later date correction from making an outcome fresh.
Reserve outcomes before prediction/scoring; failed runs remain exposed.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import tempfile

import numpy as np

from . import config

PATH = os.path.join(config.REPO_ROOT, "data", "wpx", "outcome_exposure.json")
LEGACY_NAMES = ("evaluation_registry.json", "historical_evaluation_registry.json")
KIND = "wpx_outcome_exposure_v1"
PLAN_KIND = "wpx_frozen_experiment_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_game_id(gid):
    """Historical datasets use gol.gg IDs, never dataset row numbers."""
    if isinstance(gid, str) and gid.startswith("golgg:"):
        gid = gid.split(":", 1)[1]
    value = int(gid)
    if value <= 0:
        raise ValueError("game ID must be a positive gol.gg ID")
    return "golgg:%d" % value


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".exposure-", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(value, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def _locked(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield
        fcntl.flock(fh, fcntl.LOCK_UN)


def load(path=PATH):
    if not os.path.exists(path):
        return {"kind": KIND, "consumed_through": "", "games": {},
                "history": [], "plans": {}}
    with open(path) as fh:
        value = json.load(fh)
    if value.get("kind") != KIND:
        raise ValueError("unsupported canonical exposure inventory")
    return value


def migrate(gids, dates, path=PATH, legacy_paths=None, write=True):
    """Union all known legacy histories without editing or erasing them.

    With no prior ledger, conservatively quarantine all supplied history.
    ``write=False`` supports a read-only preview; scoring callers persist the
    migration before doing any model work.
    """
    gids, dates = np.asarray(gids), np.asarray(dates).astype(str)
    if len(gids) != len(dates) or not len(gids) or np.any(dates == ""):
        raise ValueError("complete game IDs and dates required for migration")
    legacy_paths = legacy_paths if legacy_paths is not None else [
        os.path.join(os.path.dirname(path), name) for name in LEGACY_NAMES]

    def merge():
        inventory = load(path)
        known = {h.get("source_sha256") for h in inventory["history"]}
        available = False
        for source in legacy_paths:
            if not os.path.exists(source):
                continue
            available = True
            with open(source) as fh:
                legacy = json.load(fh)
            if legacy.get("kind") != "wpx_evaluation_registry_v1":
                raise ValueError("unsupported legacy exposure registry: %s" % source)
            watermark = str(legacy["consumed_through"])
            inventory["consumed_through"] = max(inventory["consumed_through"], watermark)
            source_hash = sha256(source)
            if source_hash not in known:
                inventory["history"].append({"action": "legacy_union_import",
                    "recorded_at": _now(), "source_path": os.path.abspath(source),
                    "source_sha256": source_hash, "legacy_registry": legacy})
                known.add(source_hash)
        if not available and not inventory["history"]:
            inventory["consumed_through"] = max(dates)
            inventory["history"].append({"action": "migration_quarantine",
                "recorded_at": _now(), "reason": "all outcomes present at adoption",
                "consumed_through": max(dates)})
        for gid, date in zip(gids, dates):
            key = canonical_game_id(gid)
            if date <= inventory["consumed_through"] and key not in inventory["games"]:
                inventory["games"][key] = {"date": str(date), "reason": "legacy_union"}
        if write:
            _write(path, inventory)
        return inventory

    if not write:
        return merge()
    with _locked(path):
        return merge()


def fresh_mask(inventory, gids, dates):
    return np.asarray([str(date) > inventory["consumed_through"] and
        canonical_game_id(gid) not in inventory["games"]
        for gid, date in zip(gids, dates)], dtype=bool)


def freeze_plan(experiment, provenance, population, rule, start_after,
                path=PATH):
    """Register an immutable, prospective fixed-endpoint experiment.

    The rule explicitly chooses a series/date cluster, endpoint, precision and
    sample floor. Population must be declared before the first eligible date.
    """
    required = {"candidate_sha256", "incumbent_sha256", "input_contract_sha256",
                "source_revision", "training_cutoff"}
    if required - provenance.keys() or any(not provenance[k] for k in required):
        raise ValueError("incomplete experiment provenance")
    if (rule.get("metric") != "game_balanced_brier" or
            rule.get("bootstrap_unit") not in {"series_date", "date"} or
            rule.get("endpoint") != "fixed_date" or not rule.get("end_date") or
            int(rule.get("minimum_games", 0)) < 2 or
            float(rule.get("precision_target", 0)) <= 0 or
            "min_improvement" not in rule):
        raise ValueError("explicit frozen metric, clustering, endpoint and precision required")
    if not population or str(start_after) < datetime.now(timezone.utc).date().isoformat():
        raise ValueError("population must start after registration, never on historical outcomes")
    if str(rule["end_date"]) <= str(start_after):
        raise ValueError("endpoint must follow experiment start")
    body = {"kind": PLAN_KIND, "experiment": str(experiment),
            "provenance": provenance, "population": population, "rule": rule,
            "start_after": str(start_after), "frozen_at": _now()}
    with _locked(path):
        inventory = load(path)
        existing = [p for p in inventory["plans"].values()
                    if p["experiment"] == str(experiment)]
        if existing:
            candidate = dict(body, frozen_at=existing[0]["frozen_at"])
            if digest(candidate) != existing[0]["plan_sha256"]:
                raise ValueError("experiment already frozen with a different plan")
            return existing[0]
        plan = dict(body, plan_sha256=digest(body))
        inventory["plans"][plan["plan_sha256"]] = plan
        inventory["history"].append({"action": "freeze_plan", "recorded_at": _now(),
            "plan_sha256": plan["plan_sha256"]})
        _write(path, inventory)
        return plan


def reserve(plan_sha256, gids, dates, path=PATH):
    """Atomically burn an entire frozen block before outcomes are scored."""
    with _locked(path):
        inventory = load(path)
        plan = inventory["plans"].get(plan_sha256)
        if not plan:
            raise ValueError("unknown frozen experiment")
        if digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan_sha256:
            raise ValueError("frozen experiment content hash mismatch")
        if any(h.get("action") == "reserve_outcomes" and
               h.get("plan_sha256") == plan_sha256 for h in inventory["history"]):
            raise ValueError("frozen endpoint already reserved; reuse is diagnostic only")
        dates = np.asarray(dates).astype(str)
        if (not len(dates) or not fresh_mask(inventory, gids, dates).all() or
                np.any(dates <= plan["start_after"]) or
                np.any(dates > plan["rule"]["end_date"])):
            raise ValueError("block contains exposed or out-of-plan outcomes")
        if datetime.now(timezone.utc).date().isoformat() <= plan["rule"]["end_date"]:
            raise ValueError("fixed endpoint has not closed")
        if len(set(map(canonical_game_id, gids))) < plan["rule"]["minimum_games"]:
            raise ValueError("frozen endpoint below minimum game count")
        ids = [canonical_game_id(g) for g in gids]
        for key, date in zip(ids, dates):
            inventory["games"][key] = {"date": str(date), "reason": "reserved_outcome",
                "plan_sha256": plan_sha256}
        inventory["history"].append({"action": "reserve_outcomes", "recorded_at": _now(),
            "plan_sha256": plan_sha256, "game_ids": ids})
        _write(path, inventory)
        return plan


def expose(gids, dates, experiment, plan_sha256, path=PATH, feed_game_ids=None):
    """Record inspected forward outcomes before publishing scores.

    ``gids`` contains gol.gg IDs or None for currently unlinked feed games.
    Missing links are retained by feed ID and conservatively quarantine dates
    through that outcome's date; a later historical backfill cannot relabel it
    fresh. Known mappings only burn their explicit canonical game identity.
    """
    gids, dates = list(gids), list(map(str, dates))
    if len(gids) != len(dates) or any(not date for date in dates):
        raise ValueError("exposure IDs and dates must align")
    feed_game_ids = [None] * len(gids) if feed_game_ids is None else list(feed_game_ids)
    if len(feed_game_ids) != len(gids):
        raise ValueError("feed IDs must align with canonical IDs")
    with _locked(path):
        inventory = load(path)
        pending = inventory.setdefault("unlinked_feed_games", {})
        added, missing = [], []
        for gid, date, feed_id in zip(gids, dates, feed_game_ids):
            record = {"date": date, "experiment": str(experiment),
                      "plan_sha256": str(plan_sha256), "reason": "inspected_forward_outcome"}
            if gid is None:
                if feed_id is None:
                    raise ValueError("unlinked outcome requires its feed game ID")
                key = str(feed_id)
                if key not in pending:
                    pending[key] = record
                    missing.append(key)
                inventory["consumed_through"] = max(inventory["consumed_through"], date)
            else:
                key = canonical_game_id(gid)
                if key not in inventory["games"]:
                    inventory["games"][key] = record
                    added.append(key)
        if added or missing:
            inventory["history"].append({"action": "inspected_forward_outcomes",
                "recorded_at": _now(), "experiment": str(experiment),
                "plan_sha256": str(plan_sha256), "game_ids": added,
                "unlinked_feed_game_ids": missing,
                "unlinked_policy": "quarantine all dates through unlinked outcome date"})
            _write(path, inventory)
        return {"canonical_games_added": len(added), "unlinked_games_added": len(missing),
                "consumed_through": inventory["consumed_through"]}


def cluster_labels(gids, dates, match_ids=None, unit="series_date"):
    """Use date+match when present; a missing match falls back to whole date."""
    dates = np.asarray(dates).astype(str)
    if len(gids) != len(dates) or np.any(dates == ""):
        raise ValueError("dated game rows required for clustered inference")
    if unit == "date" or match_ids is None:
        return dates
    if unit != "series_date" or len(match_ids) != len(gids):
        raise ValueError("unsupported clustering contract")
    # Missing a match anywhere on a date forces whole-date clustering, so a
    # partially identified series cannot be broken into independent units.
    missing_dates = {date for date, mid in zip(dates, match_ids)
                     if mid is None or str(mid) in {"", "0", "None", "nan"}}
    return np.asarray([date if date in missing_dates else "%s:%s" % (date, mid)
                       for date, mid in zip(dates, match_ids)])


def paired_interval(candidate, incumbent, y, gids, clusters=None,
                    bootstrap=5000, seed=83):
    """Resample whole clusters; retain equal total weight for every game."""
    candidate, incumbent, y, gids = map(np.asarray, (candidate, incumbent, y, gids))
    if not (len(candidate) == len(incumbent) == len(y) == len(gids)) or not len(y):
        raise ValueError("paired predictions must have identical nonempty rows")
    if (not np.isfinite(candidate).all() or not np.isfinite(incumbent).all() or
            not np.isfinite(y).all() or np.any((candidate < 0) | (candidate > 1)) or
            np.any((incumbent < 0) | (incumbent > 1))):
        raise ValueError("finite probabilities required")
    clusters = np.asarray(gids if clusters is None else clusters).astype(str)
    if len(clusters) != len(gids):
        raise ValueError("cluster labels must align with prediction rows")
    ug = np.unique(gids)
    delta, game_clusters = [], []
    row_delta = (candidate - y) ** 2 - (incumbent - y) ** 2
    for gid in ug:
        mask = gids == gid
        labels = np.unique(clusters[mask])
        if len(labels) != 1:
            raise ValueError("one game crosses bootstrap clusters")
        delta.append(row_delta[mask].mean())
        game_clusters.append(labels[0])
    delta, game_clusters = np.asarray(delta), np.asarray(game_clusters)
    units = np.unique(game_clusters)
    sums = np.asarray([delta[game_clusters == unit].sum() for unit in units])
    counts = np.asarray([(game_clusters == unit).sum() for unit in units])
    ci = None
    if len(units) >= 2 and bootstrap > 0:
        rng = np.random.default_rng(seed)
        draws = np.empty(bootstrap)
        # Bounded batches avoid allocating bootstrap x corpus-sized matrices.
        for start in range(0, bootstrap, 128):
            ix = rng.integers(len(units), size=(min(128, bootstrap - start), len(units)))
            draws[start:start + len(ix)] = sums[ix].sum(axis=1) / counts[ix].sum(axis=1)
        ci = np.quantile(draws, [.025, .975]).tolist()
    return {"candidate_minus_incumbent": float(delta.mean()), "ci95": ci,
            "games": int(len(ug)), "clusters": int(len(units)), "game_balanced": True}
