"""Sourced composition proxies aligned to consumed corrected-core fixed rows.

Only the four literal Data Dragon proxies enter this adapter. Reviewed ability
annotations are absent from the archived inputs, so no engage, waveclear,
damage, or scaling labels are inferred. Saved static releases must have been
available at the opening of the archived UTC day. This module reads inputs
only; it does not inspect outcomes or write serving artifacts.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpcomposition as composition


FEATURE_NAMES = list(composition.STATIC_NAMES)
INPUT_CONTRACT = dict(
    kind="corrected_core_static_composition_v1", names=FEATURE_NAMES,
    role_order=list(composition.ROLES), orientation="blue minus red",
    source="saved exact Data Dragon release; two-component patches resolve to .1",
    availability="source available no later than UTC opening of archived game day",
    missing="unknown family is exactly zero; no availability intercept",
    limitations=["literal tags and stat growth are proxies, not measured ability strength",
                 "canonical archive role slots are inherited from the audited dataset",
                 "reviewed ability annotations are not present and are not inferred"],
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _day_open(day):
    """Require a day, not an ambiguous local timestamp or a guessed draft time."""
    try:
        parsed = datetime.strptime(day, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError("Archived composition inputs require ISO UTC days") from exc
    if parsed.strftime("%Y-%m-%d") != day or day > "2026-09-02":
        raise ValueError("Only consumed ISO days through September 2, 2026 are allowed")
    return parsed.replace(tzinfo=timezone.utc).timestamp()


def collect(dataset, rows, source_root=Path("data/composition")):
    """Return four static features and provenance in the exact fixed-row order.

    ``rows`` is the output of ``wpx_combat.load_rows``. Its gid, t, and date
    arrays must match the dataset's ``seq < 0`` rows exactly. A game must have
    one constant completed draft, patch, and day across every fixed row; a
    first-row shortcut must never mask contradictory archived inputs.
    """
    dataset, source_root = Path(dataset), Path(source_root)
    with np.load(dataset, allow_pickle=False) as archive:
        required = {"gid", "t", "seq", "date", "patch", "C", "champ_names"}
        if not required.issubset(archive.files):
            raise ValueError("Composition requires fixed-row identity and completed draft inputs")
        seq = archive["seq"]
        if seq.ndim != 1:
            raise ValueError("Invalid archive sequence shape")
        fixed = seq < 0
        values = {}
        for key in ("gid", "t", "date", "patch", "C"):
            raw = archive[key]
            if len(raw) != len(seq):
                raise ValueError("Archive composition arrays do not share a row count")
            values[key] = raw[fixed]
        champion_names = archive["champ_names"]
    if (values["C"].shape != (len(values["gid"]), 10)
            or not np.issubdtype(values["C"].dtype, np.integer)
            or champion_names.ndim != 1
            or any(values[key].ndim != 1 for key in ("gid", "t", "date", "patch"))):
        raise ValueError("Expected ten integer canonical champion slots per fixed row")
    for key in ("gid", "t", "date"):
        if key not in rows or not np.array_equal(np.asarray(rows[key]), values[key]):
            raise ValueError("Composition fixed-row %s alignment differs from corrected-core rows" % key)
    if not len(values["gid"]):
        raise ValueError("A nonempty fixed-row archive is required")
    gids, first, inverse = np.unique(values["gid"], return_index=True, return_inverse=True)
    for key in ("C", "patch", "date"):
        if not np.array_equal(values[key], values[key][first][inverse]):
            raise ValueError("Archived %s is inconsistent within a game" % key)
    day_ts = {day: _day_open(day) for day in np.unique(values["date"].astype(str))}
    catalogs, sources, catalog_provenance = {}, {}, {}
    per_game_extra = np.zeros((len(gids), len(FEATURE_NAMES)), dtype=np.float32)
    per_game_known = np.zeros_like(per_game_extra, dtype=bool)
    reasons, versions, resolutions = Counter(), Counter(), Counter()
    for i, index in enumerate(first):
        patch, day = str(values["patch"][index]), str(values["date"][index])
        parsed = composition._patch(patch)
        version = ".".join(map(str, parsed + (1,) if len(parsed or ()) == 2 else parsed or ()))
        if version not in catalogs:
            path = source_root / "catalogs" / (version + ".json")
            catalogs[version] = None
            if version and path.is_file():
                sources[str(path)] = sha(path)
                try:
                    catalogs[version] = composition.load_catalog(path)
                except (ValueError, TypeError, UnicodeError):
                    pass  # An invalid optional catalog remains unavailable.
        slots = values["C"][index]
        picks = [str(champion_names[int(slot)]) if 0 <= slot < len(champion_names) else "" for slot in slots]
        blue, red = (dict(zip(composition.ROLES, picks[:5])),
                     dict(zip(composition.ROLES, picks[5:])))
        capture = composition.features(blue, red, patch, catalog=catalogs[version], as_of_ts=day_ts[day])
        per_game_extra[i] = capture["features"][:len(FEATURE_NAMES)]
        per_game_known[i] = capture["known"][:len(FEATURE_NAMES)]
        reasons[capture["reason"]] += 1
        if "static" in capture["provenance"]:
            catalog_provenance[version] = capture["provenance"]["static"]
            versions[version] += 1
            resolutions[capture["provenance"]["version_resolution"]] += 1
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError("A composition source changed during extraction")
    extra, known = per_game_extra[inverse], per_game_known[inverse]
    available = np.any(known, axis=1)
    if not np.isfinite(extra).all() or np.any(extra[~known] != 0):
        raise ValueError("Unknown or invalid composition values cannot enter a candidate")
    return dict(gid=values["gid"].copy(), t=values["t"].copy(), extra=extra,
        known=known, available=available, names=np.asarray(FEATURE_NAMES),
        provenance=dict(input_contract=INPUT_CONTRACT, dataset_sha256=sha(dataset),
            source_sha256=sha(Path(__file__)), composition_source_sha256=sha(Path(composition.__file__)),
            sources=sources, catalogs=catalog_provenance, utc_day_gate=True,
            coverage=dict(games=len(gids), covered_games=int(np.any(per_game_known, axis=1).sum()),
                states=len(extra), covered_states=int(available.sum()),
                known_games={name: int(per_game_known[:, j].sum()) for j, name in enumerate(FEATURE_NAMES)},
                known_states={name: int(known[:, j].sum()) for j, name in enumerate(FEATURE_NAMES)},
                reasons=dict(reasons), resolved_versions=dict(versions), version_resolution=dict(resolutions))))
