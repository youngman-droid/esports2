"""Read-only serving of the separately built player rating artifact.

Requests never fit a model, open the match database, or alter deployed priors.
Atomic artifact replacement permits refreshes while the dashboard is open.
"""
import copy
import json
import math
import threading
from pathlib import Path

from . import config

ARTIFACT_PATH = Path(config.REPO_ROOT) / "data" / "player_ratings" / "ratings.json"
PAGE_PATH = Path(__file__).with_name("player_rankings.html")
_cache_lock = threading.Lock()
_cache_key = None
_cache_payload = None


class RankingsUnavailable(Exception):
    """The ranking artifact must be built or repaired before it can be served."""


def _reject_nonfinite(value):
    raise ValueError("non-finite rating value: " + value)


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        _reject_nonfinite(value)
    return result


def ratings():
    """Return the current artifact, refreshing on replacement without DB access."""
    global _cache_key, _cache_payload
    with _cache_lock:
        try:
            stat = ARTIFACT_PATH.stat()
            key = (str(ARTIFACT_PATH), stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if key != _cache_key:
                with ARTIFACT_PATH.open(encoding="utf-8") as stream:
                    payload = json.load(stream, parse_constant=_reject_nonfinite,
                                        parse_float=_finite_float)
                if not isinstance(payload, dict) or not isinstance(payload.get("players"), list):
                    raise ValueError("invalid ranking artifact")
                if not isinstance(payload.get("meta"), dict):
                    raise ValueError("missing ranking provenance")
                _cache_payload = payload
                _cache_key = key
            return copy.deepcopy(_cache_payload)
        except (OSError, ValueError) as error:
            raise RankingsUnavailable(
                "Player ratings are unavailable. Build them with: "
                "python3 -m lol_ticker.player_ratings build "
                "--output data/player_ratings/ratings.json"
            ) from error


def archive():
    """Public early-era results; records without complete player lineups do not enter individual fits."""
    path = ARTIFACT_PATH.with_name("early_archive.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    except (OSError, ValueError) as error:
        raise RankingsUnavailable("The early competitive archive is unavailable.") from error
