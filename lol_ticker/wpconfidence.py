"""Isolated confidence-aware rating candidate; never modifies production priors.

Confidence uses only explicitly available, past-dated source metadata. Unknown
confidence preserves the incumbent value rather than inventing missing ratings.
Per-side ratings shrink toward the same neutral 1500 anchor before differencing.
"""
import copy
import datetime as dt
import math
from collections.abc import Mapping
from numbers import Real

KIND = "confidence_prior_candidate_v1"
DEFAULT_SPEC = dict(age_half_life_days=60., support_games=20., strength=.5,
                    neutral_elo=1500.)
INPUT_CONTRACT = dict(name=KIND, metadata="explicit past source date; no future counts",
    missing="preserve baseline; missing confidence is unavailable, not confidence zero",
    identity="side rating shrinks around common neutral anchor; preserve source roster",
    operation="isolated candidate only; no production dispatch")
CHANNELS = {"elo_oe": ("oe", "elo_blue", "elo_red"),
            "pelo_oe": ("oe", "pelo_blue", "pelo_red"),
            "elo_gg": ("golgg", "gg_elo_blue", "gg_elo_red"),
            "elo_gg_fast": ("golgg", "gg_elo_fast_blue", "gg_elo_fast_red")}


def _number(x):
    return isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x)


def _timestamp(value):
    if _number(value):
        return float(value)
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return None  # Never silently reinterpret a local timestamp as UTC.
        return value.timestamp()
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time(), dt.timezone.utc).timestamp()
    if isinstance(value, str):
        try:
            if len(value) == 10:
                return _timestamp(dt.date.fromisoformat(value))
            return _timestamp(dt.datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _spec(spec):
    spec = dict(DEFAULT_SPEC, **(spec or {}))
    if set(spec) != set(DEFAULT_SPEC) or any(not _number(v) for v in spec.values()):
        raise ValueError("Invalid confidence specification")
    if (spec["age_half_life_days"] <= 0 or spec["support_games"] <= 0
            or not 0 <= spec["strength"] <= 1 or spec["neutral_elo"] <= 0):
        raise ValueError("Invalid confidence bounds")
    return spec


def _age(source, as_of):
    stamp = _timestamp(source.get("date", source.get("last_ts")))
    if stamp is None:
        return None, "missing_source_date"
    if stamp > as_of:
        return None, "future_source_date"
    return (as_of-stamp)/86400., "past_source_date"


def side_confidence(source, *, as_of_ts, roster=None, player_adjustment=None,
                    player=False, spec=None):
    """Read confidence without treating absent sample support as zero games.

    Team games may be supplied as ``games``/``sample_games`` only when counted
    before the source timestamp. Adjusted-player confidence uses the actual
    resolved player identities and their own ``last_ts``/``games`` metadata.
    """
    spec, as_of = _spec(spec), _timestamp(as_of_ts)
    if as_of is None:
        raise ValueError("An explicit timezone-aware or epoch as-of time is required")
    out = dict(available=False, confidence=None, reason="missing_source", factors={},
               age_days=None, source_game_id=None, source_roster=[], resolved_players=[])
    actual = (player and isinstance(player_adjustment, Mapping)
              and player_adjustment.get("applied") is True)
    if actual and not player_adjustment.get("resolved_players"):
        out["reason"] = "missing_adjusted_player_identities"
        return out
    if not isinstance(source, Mapping):
        source = {}
    if source.get("available") is not True and not actual:
        return out
    age, reason = (None, "actual_resolved_lineup") if actual else _age(source, as_of)
    out.update(age_days=age, reason=reason, source_game_id=source.get("game_id"),
               source=source.get("source"), source_roster=list(source.get("players") or []),
               source_date_precision=("UTC_day_proxy" if isinstance(source.get("date"), str)
                   and len(source["date"]) == 10 else "timestamp"))
    if age is None and not actual:
        return out
    factors = {} if actual else {"age": 2. ** (-age/spec["age_half_life_days"])}
    n = source.get("sample_games", source.get("games"))
    if n is not None and not actual:
        if not _number(n) or n < 0 or n != int(n):
            out["reason"] = "invalid_sample_support"
            return out
        # A recent count must be certified no later than the rating source,
        # not recomputed from a later snapshot and attached to an old row.
        count_at = _timestamp(source.get("sample_count_as_of_ts", source.get("date")))
        source_at = _timestamp(source.get("date"))
        if count_at is None or count_at > as_of or (source_at is not None and count_at > source_at):
            out["reason"] = "uncertified_or_future_sample_support"
            return out
        window = source.get("sample_window_days")
        if window is not None and (not _number(window) or window <= 0):
            out["reason"] = "invalid_sample_window"
            return out
        factors["support"] = n/(n+spec["support_games"])
    if isinstance(roster, Mapping) and roster.get("available") is True:
        # Continuity must describe this exact rating source, not a newer roster.
        same_source = (roster.get("reference_game_id") == source.get("game_id")
                       and roster.get("reference_source") == source.get("source"))
        matched = roster.get("matched")
        if same_source and _number(matched) and matched == int(matched) and 0 <= matched <= 5:
            factors["continuity"] = matched/5.
    if player and isinstance(player_adjustment, Mapping):
        resolved = player_adjustment.get("resolved_players")
        if isinstance(resolved, list) and resolved:
            keys = [r.get("matched_name") for r in resolved if isinstance(r, Mapping)]
            if len(keys) != len(resolved) or None in keys or len(set(keys)) != len(keys) or len(keys) > 5:
                out["reason"] = "ambiguous_player_identity"
                return out
            reliabilities = []
            for row in resolved:
                a, r = _age(row, as_of)
                games = row.get("games")
                if a is None or not _number(games) or games < 0 or games != int(games):
                    out["reason"] = r if a is None else "invalid_player_support"
                    return out
                reliabilities.append(2.**(-a/spec["age_half_life_days"])*games/(games+spec["support_games"]))
            factors["resolved_support"] = sum(reliabilities)/5.
            out["resolved_players"] = [dict(matched_name=r["matched_name"], live_name=r.get("live_name"),
                games=int(r["games"]), last_ts=r.get("last_ts")) for r in resolved]
            if player_adjustment.get("applied") is True:
                # This prior was recomputed for the actual lineup. The previous
                # team's lineup continuity and date no longer describe it.
                factors.pop("continuity", None)
                factors.pop("age", None)
    confidence = math.prod(factors.values())
    out.update(available=True, confidence=confidence, reason="certified_metadata", factors=factors,
               support_available=("support" in factors or "resolved_support" in factors),
               continuity_available="continuity" in factors,
               support_window_days=source.get("sample_window_days"))
    return out


def capture(priors, *, as_of_ts, teams=None, spec=None):
    """Return metadata and an isolated adjusted-prior candidate, JSON-safe.

    Unknown confidence, absent original coverage, or absent per-side ratings
    leaves that original channel exactly unchanged. Pass this block through
    capture; use ``adjusted_priors`` only in an explicitly evaluated adapter.
    """
    spec = _spec(spec)
    if not isinstance(priors, Mapping):
        priors = {}
    provenance = priors.get("prior_provenance") or {}
    availability = priors.get("prior_availability") or {}
    roster, adjustments = priors.get("roster") or {}, priors.get("pelo_adjustment") or {}
    adjusted, channels = copy.deepcopy(dict(priors)), {}
    for channel, (family, blue_key, red_key) in CHANNELS.items():
        sides = {}
        for side in ("blue", "red"):
            sides[side] = side_confidence((provenance.get(family) or {}).get(side),
                as_of_ts=as_of_ts, roster=roster.get(side), player_adjustment=adjustments.get(side),
                player=channel == "pelo_oe", spec=spec)
        usable = (availability.get(channel) is True and all(s["available"] for s in sides.values())
                  and _number(priors.get(channel)) and _number(priors.get(blue_key))
                  and _number(priors.get(red_key)))
        applied, reason = False, "missing_coverage_or_side_ratings"
        if usable:
            # Refuse an identity/value mismatch rather than shrink unrelated
            # per-side ratings into the recorded differential.
            original = (priors[blue_key]-priors[red_key])/400.
            if abs(original-priors[channel]) <= 1e-9:
                values = []
                for side, key in (("blue", blue_key), ("red", red_key)):
                    retain = 1.-spec["strength"]*(1.-sides[side]["confidence"])
                    value = spec["neutral_elo"]+retain*(priors[key]-spec["neutral_elo"])
                    adjusted[key] = value
                    values.append(value)
                adjusted[channel] = (values[0]-values[1])/400.
                applied, reason = True, "confidence_shrinkage_candidate"
            else:
                reason = "side_rating_differential_mismatch"
        channels[channel] = dict(available=usable, applied=applied, reason=reason, sides=sides,
                                 original=priors.get(channel), candidate=adjusted.get(channel))
    return dict(kind=KIND, available=any(c["applied"] for c in channels.values()),
                as_of_ts=_timestamp(as_of_ts), teams=list(teams or []), spec=spec,
                input_contract=dict(INPUT_CONTRACT), channels=channels, adjusted_priors=adjusted,
                production_applied=False)
