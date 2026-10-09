"""Optional research capture shared by the live recorder and offline callers.

Capture never applies candidate predictions. Catalogs and certified rule
profiles are local inputs, and every unavailable family carries its reason.
Stateful trackers belong to one game attempt and use the pause-aware clock.
"""
from collections.abc import Mapping
import logging

log = logging.getLogger(__name__)


def _guard(module, name, callback):
    try:
        return callback()
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, OSError, RuntimeError) as error:
        log.warning("%s research capture unavailable: %s", name, error)
        names = list(getattr(module, "FEATURE_NAMES", []))
        return dict(available=False, reason="invalid_" + name + "_inputs", names=names,
                    features=[0.] * len(names), known=[False] * len(names),
                    production_applied=False)


def capture_draft(metadata, *, draft_ts, context=None):
    """Capture role-verified draft sources once for each game attempt.

    Explicit context supports ``composition_catalog``, ``composition_annotations``,
    ``composition_root``, an explicit ``composition_patch`` mapping,
    ``sq_table_path`` and a ``fearless`` adapter payload.
    Missing context keeps the per-release local catalog and shipped SQ defaults.
    Fearless needs its own certified rule/history payload to become available.
    """
    from . import wpcomposition, wpfearless, wpsqearly
    context = dict(context) if isinstance(context, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    from . import wpcombined_prod
    context = wpcombined_prod.capture_context(metadata, context)
    patch = metadata.get("patchVersion")
    composition_patch = context.get("composition_patch", patch)
    if context.get("composition_catalog") is not None or context.get("composition_annotations") is not None:
        composition = lambda: wpcomposition.from_metadata(metadata, composition_patch,
            catalog=context.get("composition_catalog"), annotations=context.get("composition_annotations"),
            as_of_ts=draft_ts)
    else:
        composition = lambda: wpcomposition.capture(metadata, composition_patch, as_of_ts=draft_ts,
                                                    source_root=context.get("composition_root"))
    fearless = context.get("fearless") or ({"series_id": context["series_id"],
                                           "game_num": context.get("game_num")}
                                          if context.get("series_id") else None)
    def capture_fearless():
        if isinstance(fearless, Mapping) and isinstance(fearless.get("context"), Mapping):
            certified = fearless["context"]
            for key in ("series_id", "game_num"):
                if context.get(key) is not None and str(context[key]) != str(certified.get(key)):
                    raise ValueError("certified Fearless bundle differs from current " + key)
        return wpfearless.capture(metadata, fearless, as_of_ts=draft_ts,
                                 source_root=context.get("fearless_root"))
    snapshot = dict(source="official_game_metadata", patch=patch, game_start_ts=draft_ts,
                    series_id=context.get("series_id"), game_num=context.get("game_num"), teams={})
    for side in ("blue", "red"):
        team = metadata.get(side + "TeamMetadata") or {}
        team = team if isinstance(team, Mapping) else {}
        participants = team.get("participantMetadata") or []
        participants = participants if isinstance(participants, (list, tuple)) else []
        snapshot["teams"][side] = dict(team_id=team.get("esportsTeamId"), participants=[
            dict(participant_id=p.get("participantId"), esports_player_id=p.get("esportsPlayerId"),
                 summoner_name=p.get("summonerName"), role=p.get("role"), champion_id=p.get("championId"))
            for p in participants if isinstance(p, Mapping)])
    composition_block = _guard(wpcomposition, "composition", composition)
    composition_block.update(feed_patch=patch, resolved_patch=composition_patch,
                             patch_mapping="explicit_context" if composition_patch != patch else "feed_version")
    return {
        "draft_snapshot": snapshot,
        "composition": composition_block,
        "sq_early": _guard(wpsqearly, "sq_early", lambda: wpsqearly.live_capture(
            metadata, table_path=context.get("sq_table_path"))),
        "fearless": _guard(wpfearless, "fearless", capture_fearless),
    }


def capture_priors(priors, *, as_of_ts, teams=None):
    from . import wpconfidence
    return _guard(wpconfidence, "prior_confidence", lambda: wpconfidence.capture(
        priors, as_of_ts=as_of_ts, teams=teams))


def capture_dynamic(state, trackers, *, attempt_id, metadata=None, frame=None, context=None):
    """Attach objective and trajectory blocks; mutate caller-owned history only."""
    from . import wpobjective, wptrend
    context = dict(context) if isinstance(context, Mapping) else {}
    objective_tracker = trackers.setdefault("objective", wpobjective.new_tracker())
    trend_tracker = trackers.setdefault("trajectory", wptrend.new_tracker())
    state["objective_opportunities"] = _guard(wpobjective, "objective_opportunities",
        lambda: wpobjective.capture(state, objective_tracker, attempt_id=attempt_id,
                                    timing_rules=context.get("objective_timing_rules")))
    state["trajectory"] = _guard(wptrend, "trajectory", lambda: wptrend.capture(
        state, trend_tracker, attempt_id=attempt_id,
        observation=wptrend.live_observation(metadata, frame, state)
        if metadata is not None and frame is not None else None,
        max_gap_s=context.get("trajectory_max_gap_s", 20.),
        anchor_slack_s=context.get("trajectory_anchor_slack_s", 15.)))
    state["trajectory"]["sampling_protocol"] = "recorder_sparse_windows_v1"
    exact_tracker = trackers.setdefault("trajectory_exact", wptrend.new_tracker())
    state["trajectory_exact"] = _guard(wptrend, "trajectory_exact", lambda: wptrend.capture(
        state, exact_tracker, attempt_id=attempt_id,
        observation=wptrend.live_observation(metadata, frame, state)
        if metadata is not None and frame is not None else None,
        max_gap_s=90., anchor_slack_s=0.))
    state["trajectory_exact"]["sampling_protocol"] = "exact_120_300s_v1"
    return state
