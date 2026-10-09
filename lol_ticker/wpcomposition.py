"""Patch-bound, sourced composition features; no automatic ability guesses.

Data Dragon supplies literal Tank tags, attack range, and defensive stat growth.
Those are named proxies, not measured teamfight power. Engage, disengage,
waveclear, damage profile and late scaling require explicit reviewed capability
annotations with patch and primary-source evidence. A family needs all ten
champions; incomplete families contribute zero and retain a coverage reason.
This module captures inputs only and has no production scoring dispatch.
"""
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from .sqpairs import key as champion_key


KIND = "patch_composition_v1"
ROLES = ("top", "jng", "mid", "bot", "sup")
FEATURE_NAMES = ["frontline_tank_tag_count", "mean_attack_range", "hp_growth_sum",
                 "armor_growth_sum", "engage_count", "disengage_count",
                 "waveclear_count", "physical_primary_count", "magic_primary_count",
                 "mixed_primary_count", "damage_type_balance", "late_scaling_count"]
STATIC_NAMES = FEATURE_NAMES[:4]
ANNOTATED_FLAGS = {"engage_count": "engage", "disengage_count": "disengage",
                   "waveclear_count": "waveclear", "late_scaling_count": "late_scaling"}
ROLE_ALIASES = dict(top="top", jungle="jng", jng="jng", mid="mid",
                    bottom="bot", bot="bot", support="sup", sup="sup")
_PRIMARY_HOSTS = {"developer.riotgames.com", "ddragon.leagueoflegends.com",
                  "www.leagueoflegends.com", "leagueoflegends.com",
                  "raw.communitydragon.org", "communitydragon.org"}
INPUT_CONTRACT = dict(kind=KIND, names=FEATURE_NAMES, role_order=list(ROLES),
                      sources="exact-version static catalog; timestamped reviewed primary-source capability annotations",
                      completeness="each family requires ten known champions; missing families zero",
                      orientation="blue minus red; proxies have explicit names; no availability intercept")


def _finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _patch(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", value):
        return None
    return tuple(int(x) for x in value.split("."))


def _same_patch(game_patch, source_patch):
    game, source = _patch(game_patch), _patch(source_patch)
    return bool(game and source and len(source) == 3
                and (game == source if len(game) == 3 else game == source[:2]))


def _primary(url):
    return isinstance(url, str) and urlparse(url).scheme == "https" and urlparse(url).hostname in _PRIMARY_HOSTS


def _empty(reason, patch):
    return dict(kind=KIND, available=False, complete=False, reason=reason, patch=patch,
                names=list(FEATURE_NAMES), features=[0.] * len(FEATURE_NAMES),
                known=[False] * len(FEATURE_NAMES), coverage={name: 0 for name in FEATURE_NAMES},
                reasons={name: reason for name in FEATURE_NAMES},
                blue={}, red={}, provenance={}, limitations=[
                    "Tank tag and defensive stat growth are literal static proxies, not measured frontline or scaling strength.",
                    "Attack range describes basic attacks, not effective spell reach.",
                    "Reviewed capability labels describe a champion, not a calibrated win-probability effect.",
                    "Damage profile weights each champion equally and does not estimate dealt-damage shares."])


def catalog_from_payload(payload, *, source_url, available_from_ts, raw_sha256=None):
    """Wrap a saved immutable Data Dragon response with explicit release evidence.

    ``available_from_ts`` is when the source version was available, not today's
    retrieval timestamp. Callers must retain the release-date evidence beside
    the snapshot. Historical scoring never substitutes the newest release.
    """
    if (not isinstance(payload, Mapping) or _patch(payload.get("version")) is None
            or len(_patch(payload.get("version"))) != 3
            or not isinstance(payload.get("data"), Mapping)
            or not _primary(source_url) or not _finite(available_from_ts)):
        raise ValueError("A versioned primary-source catalog and source availability time are required")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = raw_sha256 or hashlib.sha256(raw).hexdigest()
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Catalog source hash must be SHA-256")
    return dict(version=payload["version"], data=dict(payload["data"]), source_url=source_url,
                source_sha256=digest, available_from_ts=float(available_from_ts),
                catalog_sha256=hashlib.sha256(json.dumps(
                    {"version": payload["version"], "data": payload["data"]},
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest())


def load_catalog(path):
    """Load an explicit saved wrapper; serving never downloads a latest patch."""
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Invalid composition catalog")
    return value


def _catalog_ok(catalog, patch, as_of_ts):
    try:
        return (isinstance(catalog, Mapping) and _same_patch(patch, catalog.get("version"))
            and isinstance(catalog.get("data"), Mapping) and _primary(catalog.get("source_url"))
            and isinstance(catalog.get("source_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", catalog["source_sha256"])
            and _finite(catalog.get("available_from_ts"))
            and catalog["available_from_ts"] <= as_of_ts
            and catalog.get("catalog_sha256") == hashlib.sha256(json.dumps(
                {"version": catalog.get("version"), "data": catalog.get("data")},
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
    except (TypeError, ValueError, OverflowError):
        return False


def _annotation(record, trait, patch, as_of_ts):
    if not isinstance(record, Mapping) or not _same_patch(patch, record.get("version")):
        return None
    traits = record.get("traits")
    annotation = traits.get(trait) if isinstance(traits, Mapping) else None
    if not isinstance(annotation, Mapping):
        return None
    evidence = annotation.get("evidence_urls")
    if (annotation.get("review_status") != "reviewed" or not isinstance(annotation.get("reviewer"), str)
            or not annotation["reviewer"]
            or not isinstance(evidence, list) or not evidence or not all(_primary(url) for url in evidence)
            or not _finite(annotation.get("source_available_from_ts"))
            or annotation["source_available_from_ts"] > as_of_ts):
        return None
    value = annotation.get("value")
    if trait == "primary_damage":
        return value if isinstance(value, str) and value in {"physical", "magic", "mixed"} else None
    return int(value) if isinstance(value, bool) or (type(value) is int and value in (0, 1)) else None


def features(blue, red, patch, *, catalog=None, annotations=None, as_of_ts=None):
    """Capture completed-draft traits in verified role order, blue minus red.

    ``blue`` and ``red`` are role->champion maps, not pick-order lists.
    Availability is family-specific; all unknown features are exactly zero.
    Annotation records are keyed by canonical champion name and exact static
    version, and require reviewed, timestamped primary-source evidence.
    """
    out = _empty("missing_draft_or_source", patch)
    if not _finite(as_of_ts) or not _patch(patch):
        out["reason"] = "missing_patch_or_prediction_time"
        return out
    if (not isinstance(blue, Mapping) or not isinstance(red, Mapping)
            or set(blue) != set(ROLES) or set(red) != set(ROLES)
            or any(not isinstance(c, str) or not champion_key(c) for c in list(blue.values()) + list(red.values()))):
        out["reason"] = "incomplete_role_draft"
        return out
    champs = [champion_key(side[r]) for side in (blue, red) for r in ROLES]
    if len(set(champs)) != 10:
        out["reason"] = "duplicate_draft_champion"
        return out
    source_ok = _catalog_ok(catalog, patch, as_of_ts)
    by_champ = {}
    if source_ok:
        for name, record in catalog["data"].items():
            # Patch 16.15.1's summary also contains Jade_* League Classic
            # records sharing standard champion display names. Keep the base
            # roster only; a display-name join must never overwrite SR stats.
            if isinstance(name, str) and "_" not in name and isinstance(record, Mapping):
                by_champ[champion_key(record.get("name") or name)] = record
                by_champ[champion_key(name)] = record
        out["provenance"]["static"] = {key: catalog[key] for key in
                                             ("version", "source_url", "source_sha256", "available_from_ts", "catalog_sha256")}
        out["provenance"]["version_resolution"] = "exact_release" if len(_patch(patch)) == 3 else "major_minor_version_only"
    records = [by_champ.get(champ) for champ in champs]
    values = {}
    for name in STATIC_NAMES:
        observed = []
        for record in records:
            if not isinstance(record, Mapping):
                observed.append(None)
                continue
            if name == "frontline_tank_tag_count":
                tags = record.get("tags")
                observed.append(float("Tank" in tags) if isinstance(tags, list) and all(isinstance(t, str) for t in tags) else None)
            else:
                stat = dict(mean_attack_range="attackrange", hp_growth_sum="hpperlevel",
                            armor_growth_sum="armorperlevel")[name]
                stats = record.get("stats")
                raw = stats.get(stat) if isinstance(stats, Mapping) else None
                observed.append(float(raw) if _finite(raw) and raw >= 0 else None)
        out["coverage"][name] = sum(v is not None for v in observed)
        if all(v is not None for v in observed):
            b, r = sum(observed[:5]), sum(observed[5:])
            if name == "mean_attack_range":
                b, r = b / 5., r / 5.
            values[name] = (b, r)
        out["reasons"][name] = "complete_static_source" if name in values else "missing_or_mismatched_static_source"
    annotations = annotations if isinstance(annotations, Mapping) else {}
    annotation_records = []
    for champ in champs:
        record = annotations.get(champ)
        try:
            json.dumps(record, allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            record = None
        annotation_records.append(record)
    for name, trait in ANNOTATED_FLAGS.items():
        observed = [_annotation(record, trait, patch, as_of_ts) for record in annotation_records]
        out["coverage"][name] = sum(v is not None for v in observed)
        if all(v is not None for v in observed):
            values[name] = (float(sum(observed[:5])), float(sum(observed[5:])))
        out["reasons"][name] = "complete_reviewed_capabilities" if name in values else "missing_reviewed_capabilities"
    damage = [_annotation(record, "primary_damage", patch, as_of_ts) for record in annotation_records]
    for name in FEATURE_NAMES[7:11]:
        out["coverage"][name] = sum(v is not None for v in damage)
        out["reasons"][name] = "complete_reviewed_damage_profiles" if all(v is not None for v in damage) else "missing_reviewed_damage_profiles"
    if all(v is not None for v in damage):
        by_side = []
        for side in (damage[:5], damage[5:]):
            physical, magic, mixed = (side.count(kind) for kind in ("physical", "magic", "mixed"))
            # Equal champion weights are an explicit categorical mix proxy.
            balance = 2. * min(physical + .5 * mixed, magic + .5 * mixed) / 5.
            by_side.append([float(physical), float(magic), float(mixed), balance])
        for j, name in enumerate(FEATURE_NAMES[7:11]):
            values[name] = (by_side[0][j], by_side[1][j])
    out["blue"] = {name: pair[0] for name, pair in values.items()}
    out["red"] = {name: pair[1] for name, pair in values.items()}
    out["features"] = [values[name][0] - values[name][1] if name in values else 0. for name in FEATURE_NAMES]
    out["known"] = [name in values for name in FEATURE_NAMES]
    out["available"] = any(out["known"])
    out["complete"] = all(out["known"])
    out["reason"] = "complete_composition" if out["complete"] else "partial_composition" if out["available"] else "no_sourced_traits"
    out["provenance"]["annotation_records"] = {champ: record for champ, record in zip(champs, annotation_records) if record is not None}
    return out


def from_metadata(metadata, patch, *, catalog=None, annotations=None, as_of_ts=None):
    """Verify official role/participant metadata and use completed champions."""
    sides = []
    for side, allowed in (("blue", set(range(1, 6))), ("red", set(range(6, 11)))):
        team = metadata.get(side + "TeamMetadata") if isinstance(metadata, Mapping) else None
        players = team.get("participantMetadata") if isinstance(team, Mapping) else None
        if not isinstance(players, list) or len(players) != 5:
            return _empty("incomplete_participant_metadata", patch)
        roles, ids = {}, set()
        for player in players:
            if not isinstance(player, Mapping):
                return _empty("invalid_participant_metadata", patch)
            pid, role, champion = player.get("participantId"), player.get("role"), player.get("championId")
            if (type(pid) is not int or pid not in allowed or pid in ids
                    or not isinstance(role, str) or role not in ROLE_ALIASES or ROLE_ALIASES[role] in roles
                    or not isinstance(champion, str) or not champion_key(champion)):
                return _empty("unverified_participant_roles", patch)
            ids.add(pid)
            roles[ROLE_ALIASES[role]] = champion
        if ids != allowed or set(roles) != set(ROLES):
            return _empty("unverified_participant_roles", patch)
        sides.append(roles)
    return features(*sides, patch, catalog=catalog, annotations=annotations, as_of_ts=as_of_ts)


def capture(metadata, patch, *, as_of_ts, source_root=None):
    """Capture using local per-release catalogs and optional reviewed labels.

    The default is ``data/composition/{catalogs,annotations}/VERSION.json``.
    Missing files yield an explicit unavailable block; no network or mutable
    latest-version lookup occurs in the recording path.
    """
    from . import config
    root = Path(source_root) if source_root is not None else Path(config.REPO_ROOT) / "data" / "composition"
    parsed = _patch(patch)
    if parsed is None:
        return _empty("missing_patch", patch)
    version = ".".join(map(str, parsed if len(parsed) == 3 else parsed + (1,)))
    catalog_path, annotation_path = root / "catalogs" / (version + ".json"), root / "annotations" / (version + ".json")
    try:
        catalog = load_catalog(catalog_path) if catalog_path.exists() else None
        annotations = json.loads(annotation_path.read_text()) if annotation_path.exists() else None
    except (OSError, ValueError, TypeError):
        return _empty("invalid_local_composition_sources", patch)
    return from_metadata(metadata, patch, catalog=catalog, annotations=annotations, as_of_ts=as_of_ts)


def fit(extra, baseline_p, y, gids, *, spec=None):
    """Isolated draft-time signed residual; the caller freezes the core offset."""
    import numpy as np
    from . import wpresidual
    return wpresidual.fit(extra, baseline_p, y, gids, np.zeros(len(y)), feature_names=FEATURE_NAMES,
                          candidate_kind=KIND, input_contract=INPUT_CONTRACT,
                          spec=spec, coefficient_bounds=(-10., 10.), time_knots=[0., 1.])


def predict(model, extra, baseline_p):
    import numpy as np
    from . import wpresidual
    wpresidual.validate(model, candidate_kind=KIND, feature_names=FEATURE_NAMES, input_contract=INPUT_CONTRACT)
    return wpresidual.predict(model, extra, baseline_p, np.zeros(len(baseline_p)))


def save(model, path):
    from . import wpresidual
    wpresidual.validate(model, candidate_kind=KIND, feature_names=FEATURE_NAMES, input_contract=INPUT_CONTRACT)
    wpresidual.save(model, path)


def load(path):
    from . import wpresidual
    model = wpresidual.load(path)
    wpresidual.validate(model, candidate_kind=KIND, feature_names=FEATURE_NAMES, input_contract=INPUT_CONTRACT)
    return model
