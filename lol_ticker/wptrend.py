"""Bounded causal game-clock trajectories and an isolated frozen-core residual.

No future interpolation or wall-clock motion through pauses occurs. Full
window coverage, a causal anchor and complete endpoint channels are required.
Historical minute samples cannot supply the thirty-second window. Side-neutral
lead volatility is captured for analysis but is not a blue-advantage regressor.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

import numpy as np


KIND = "causal_trajectory_residual_v1"
ROLES = ("top", "jng", "mid", "bot", "sup")
WINDOWS = (30, 120, 300)
OBJECTIVES = ("tower", "baron", "dragon", "elder", "inhib")
FEATURE_NAMES = [name for window in WINDOWS for name in
    (["lead_change_%ds_k" % window]
     + ["gold_change_%s_%ds_k" % (role, window) for role in ROLES]
     + ["hp_change_%s_%ds" % (role, window) for role in ROLES]
     + ["alive_change_%s_%ds" % (role, window) for role in ROLES]
     + ["kill_change_%ds" % window]
     + ["%s_change_%ds" % (kind, window) for kind in OBJECTIVES])]
INPUT_CONTRACT = {
    "name": KIND, "orientation": "current minus causal anchor, blue minus red",
    "windows_s": list(WINDOWS), "clock": "pause-aware game seconds; no future interpolation",
    "coverage": "anchor precedes target within explicit slack; bounded gaps; independently gated endpoints",
    "roles": "metadata role and participant-ID join live; certified corrected archive role order historically",
    "health": "same-clock endpoint health only; carried archive HP cannot become current",
    "counts": "actual side counts or explicit signed counter differences; signed decreases do not imply remake",
    "missing": "zero features and exact baseline fallback; no missingness intercept",
    "volatility": "side-neutral lead standard deviation/reversals reported only; excluded from signed residual",
}
_ALIASES = dict(top="top", jng="jng", jungle="jng", mid="mid", bot="bot", bottom="bot", sup="sup", support="sup")
_COUNT_KEYS = {"kill": ("kills_blue", "kills_red"), "tower": ("towers_blue", "towers_red"),
               "baron": ("barons_blue", "barons_red"), "dragon": ("drag_blue", "drag_red"),
               "elder": ("elders_blue", "elders_red"), "inhib": ("inhib_blue", "inhib_red")}
_DIFF_KEYS = dict(kill="kills", tower="towers", baron="barons", dragon="dragons", elder="elders", inhib="inhibs")


def _number(v):
    return isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) and math.isfinite(v)


def _vector(value, length, *, high=None):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length:
        return None
    if any(not _number(v) or v < 0 or high is not None and v > high for v in value):
        return None
    return list(map(float, value))


def new_tracker():
    return dict(attempt_id=None, last_ts=None, last_clock_s=None, records=[], last_block=None)


def _empty(reason):
    return dict(available=False, reason=reason, names=list(FEATURE_NAMES),
                features=[0.] * len(FEATURE_NAMES), known=[False] * len(FEATURE_NAMES),
                windows={}, source="causal_frame_history", excluded_inputs=["items", "positions", "side_neutral_volatility"])


def observation_from_state(state):
    """Read explicit current canonical combat observations and team counters."""
    if not isinstance(state, Mapping):
        return {}
    combat = state.get("combat") or {}
    raw = combat.get("observation") or {}
    source_ts = state.get("ts", combat.get("observation_ts"))
    fresh = (combat.get("age_upper_s") == 0 and
             (combat.get("observation_ts") is None or source_ts == combat["observation_ts"]))
    canonical = raw.get("role_order") == list(ROLES)
    gold = _vector(list(raw.get("gdb") or []) + list(raw.get("gdr") or []), 10) if canonical and fresh else None
    hp = _vector(list(raw.get("hpb") or []) + list(raw.get("hpr") or []), 10, high=1.) if canonical and fresh else None
    counts = {kind: _vector([state.get(k) for k in keys], 2) for kind, keys in _COUNT_KEYS.items()}
    supplied_diffs = state.get("counter_differences") or {}
    if not isinstance(supplied_diffs, Mapping):
        supplied_diffs = {}
    diffs = {kind: float(pair[0]-pair[1]) if pair is not None else
             float(supplied_diffs[kind]) if _number(supplied_diffs.get(kind)) else
             float(state[_DIFF_KEYS[kind]]) if _number(state.get(_DIFF_KEYS[kind])) else None
             for kind, pair in counts.items()}
    return dict(ts=source_ts, clock_s=state.get("clock_s"), game_state=state.get("game_state"),
                gold_roles=gold, hp=hp, gold_totals=_vector([state.get("gold_blue"), state.get("gold_red")], 2),
                counts=counts, count_diffs=diffs, source="same_frame_canonical_state")


def live_observation(metadata, frame, state):
    """Preserve role gold even during opening frames with missing health.

    Participant IDs and roles must be complete and unique. Team-level channels
    remain usable if canonical role identity or health is unavailable.
    """
    out = observation_from_state(state)
    out.update(ts=state.get("ts", (state.get("combat") or {}).get("observation_ts")), source="same_window_frame")
    if not isinstance(metadata, Mapping) or not isinstance(frame, Mapping):
        return out
    ordered = []
    for side, ids in (("blue", set(range(1, 6))), ("red", set(range(6, 11)))):
        team_metadata, team_frame = metadata.get(side + "TeamMetadata"), frame.get(side + "Team")
        if not isinstance(team_metadata, Mapping) or not isinstance(team_frame, Mapping):
            return out
        entries = team_metadata.get("participantMetadata")
        players = team_frame.get("participants")
        if not isinstance(entries, list) or len(entries) != 5 or not isinstance(players, list) or len(players) != 5:
            return out
        roles, observed = {}, {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                return out
            pid, original_role = entry.get("participantId"), entry.get("role")
            role = _ALIASES.get(original_role) if isinstance(original_role, str) else None
            if not _number(pid) or pid != int(pid) or pid not in ids or role is None or role in roles or pid in roles.values():
                return out
            roles[role] = int(pid)
        for player in players:
            if not isinstance(player, Mapping):
                return out
            pid = player.get("participantId")
            if not _number(pid) or pid != int(pid) or pid not in ids or pid in observed:
                return out
            observed[int(pid)] = player
        if set(roles) != set(ROLES) or set(observed) != ids:
            return out
        ordered.extend(observed[roles[role]] for role in ROLES)
    out["gold_roles"] = _vector([p.get("totalGold") for p in ordered], 10)
    hp = []
    for player in ordered:
        current, maximum = player.get("currentHealth"), player.get("maxHealth")
        if not _number(current) or not _number(maximum) or maximum <= 0 or not 0 <= current <= maximum:
            hp = None; break
        hp.append(float(current / maximum))
    out["hp"] = hp
    return out


def _reset_detected(previous, current):
    if current["clock_s"] < previous["clock_s"]:
        return True
    for field in ("gold_totals", "gold_roles"):
        if previous.get(field) is not None and current.get(field) is not None:
            if any(a < b for a, b in zip(current[field], previous[field])):
                return True
    for kind in ("kill", "baron", "dragon", "elder"):
        before, after = previous["counts"].get(kind), current["counts"].get(kind)
        if before is not None and after is not None and any(a < b for a, b in zip(after, before)):
            return True
    return False


def capture(state, tracker, *, attempt_id=None, observation=None, max_gap_s=10.,
            anchor_slack_s=1., max_frames=1024):
    """Capture bounded 30/120/300-second histories without replaying retries.

    The caller supplies pause-aware clock_s and attempt identity. Same-frame
    retries return the original block, older frames do not alter history, and
    clock/cumulative-counter regression clears history. Wall time through a
    pause never fills a missing game-time window.
    """
    if not isinstance(tracker, dict):
        raise TypeError("tracker must be a caller-owned dict")
    if (not _number(max_gap_s) or max_gap_s <= 0 or not _number(anchor_slack_s) or anchor_slack_s < 0
            or not isinstance(max_frames, int) or max_frames < 2):
        raise ValueError("Invalid history coverage limits")
    record = deepcopy(observation if observation is not None else observation_from_state(state))
    if (not isinstance(record, dict) or not _number(record.get("ts")) or not _number(record.get("clock_s"))
            or record["clock_s"] < 0):
        return _empty("invalid_frame_clock")
    # Never let an explicitly supplied observation belong to another frame.
    expected_ts = state.get("ts", (state.get("combat") or {}).get("observation_ts"))
    if record["clock_s"] != state.get("clock_s") or expected_ts is not None and record["ts"] != expected_ts:
        return _empty("observation_frame_mismatch")
    record["gold_totals"] = _vector(record.get("gold_totals"), 2)
    record["gold_roles"] = _vector(record.get("gold_roles"), 10)
    record["hp"] = _vector(record.get("hp"), 10, high=1.)
    record["counts"] = {k: _vector((record.get("counts") or {}).get(k), 2) for k in _COUNT_KEYS}
    supplied_diffs = record.get("counter_differences") or record.get("count_diffs") or {}
    if not isinstance(supplied_diffs, Mapping):
        supplied_diffs = {}
    record["count_diffs"] = {k: float(pair[0]-pair[1]) if pair is not None else
                            float(supplied_diffs[k]) if _number(supplied_diffs.get(k)) else None
                            for k, pair in record["counts"].items()}
    attempt = attempt_id if attempt_id is not None else state.get("game_id")
    same = tracker.get("attempt_id") == attempt
    if same and tracker.get("last_ts") is not None and record["ts"] <= tracker["last_ts"]:
        return deepcopy(tracker["last_block"]) if record["ts"] == tracker["last_ts"] else _empty("out_of_order_frame")
    records = tracker.get("records") or []
    reset = not same or bool(records and _reset_detected(records[-1], record))
    if reset:
        records = []
    # Repeated wall-time frames in a pause occupy one game-clock instant.
    # Otherwise a long pause would bias volatility and exhaust the frame cap.
    if records and records[-1]["clock_s"] == record["clock_s"]:
        records[-1] = record
    else:
        records.append(record)
    floor = record["clock_s"] - max(WINDOWS) - anchor_slack_s - max_gap_s
    records = [r for r in records if r["clock_s"] >= floor][-max_frames:]
    block = _empty("insufficient_causal_history")
    feature_index = {n: i for i, n in enumerate(FEATURE_NAMES)}
    def put(name, value):
        j = feature_index[name]; block["features"][j] = float(value); block["known"][j] = True
    for window in WINDOWS:
        target = record["clock_s"] - window
        candidates = [i for i, r in enumerate(records) if r["clock_s"] <= target]
        info = dict(available=False, reason="missing_causal_anchor", anchor_clock_s=None,
                    actual_window_s=None, lead_volatility_k=None, lead_reversals=None)
        block["windows"][str(window)] = info
        if not candidates:
            continue
        start = candidates[-1]
        anchor, prefix = records[start], records[start:]
        if target - anchor["clock_s"] > anchor_slack_s:
            info["reason"] = "anchor_too_old"; continue
        if any(b["clock_s"]-a["clock_s"] > max_gap_s for a, b in zip(prefix, prefix[1:])):
            info["reason"] = "history_gap"; continue
        info.update(available=True, reason="causal_window", anchor_clock_s=float(anchor["clock_s"]),
                    actual_window_s=float(record["clock_s"]-anchor["clock_s"]), samples=len(prefix))
        for field, stem, divisor in (("gold_roles", "gold_change", 1000.), ("hp", "hp_change", 1.)):
            if anchor.get(field) is not None and record.get(field) is not None:
                old, new = anchor[field], record[field]
                for j, role in enumerate(ROLES):
                    suffix = "_k" if field == "gold_roles" else ""
                    put("%s_%s_%ds%s" % (stem, role, window, suffix),
                        ((new[j]-new[j+5])-(old[j]-old[j+5])) / divisor)
        if anchor.get("hp") is not None and record.get("hp") is not None:
            old, new = anchor["hp"], record["hp"]
            for j, role in enumerate(ROLES):
                put("alive_change_%s_%ds" % (role, window),
                    (int(new[j] > 0)-int(new[j+5] > 0))-(int(old[j] > 0)-int(old[j+5] > 0)))
        if anchor.get("gold_totals") is not None and record.get("gold_totals") is not None:
            old, new = anchor["gold_totals"], record["gold_totals"]
            put("lead_change_%ds_k" % window, ((new[0]-new[1])-(old[0]-old[1])) / 1000.)
        for kind in _COUNT_KEYS:
            old, new = anchor["count_diffs"].get(kind), record["count_diffs"].get(kind)
            if old is not None and new is not None:
                put("%s_change_%ds" % (kind, window), new-old)
        if all(r.get("gold_totals") is not None for r in prefix):
            lead = [(r["gold_totals"][0]-r["gold_totals"][1]) / 1000. for r in prefix]
            signs = [int(np.sign(x)) for x in lead if x != 0]
            info.update(lead_volatility_k=float(np.std(lead)),
                        lead_reversals=sum(a != b for a, b in zip(signs, signs[1:])))
    block.update(available=any(block["known"]), reason="partial_causal_history" if any(block["known"]) else block["reason"],
                 observation_ts=float(record["ts"]), clock_s=float(record["clock_s"]), attempt_id=attempt,
                 reset=bool(reset), retained_frames=len(records), max_gap_s=float(max_gap_s),
                 anchor_slack_s=float(anchor_slack_s))
    tracker.update(attempt_id=attempt, last_ts=record["ts"], last_clock_s=record["clock_s"],
                   records=records, last_block=deepcopy(block))
    return block


def historical_features(states, *, gold=None, role_order=None, max_gap_s=90.):
    """Adapt ordered, corrected fixed-minute rows without future interpolation.

    Optional individual gold must already match these rows in certified role
    order. Carried HP is excluded through observation_from_state's age gate.
    Exact minute anchors support 120/300 seconds; thirty seconds stays unknown.
    """
    if gold is not None:
        gold = np.asarray(gold)
        if (role_order != list(ROLES) or gold.shape != (len(states), 10)
                or not np.isfinite(gold).all() or np.any(gold < 0)):
            raise ValueError("Historical individual gold requires certified canonical role order and complete matching rows")
    tracker, output = new_tracker(), []
    for i, state in enumerate(states):
        observation = observation_from_state(state)
        if gold is not None:
            observation["gold_roles"] = gold[i].astype(float).tolist()
        output.append(capture(state, tracker, attempt_id=state.get("game_id"), observation=observation,
                              max_gap_s=max_gap_s, anchor_slack_s=0.))
    return output


def fit(extra, baseline_p, y, gids, t, spec=None):
    from . import wpresidual
    return wpresidual.fit(extra, baseline_p, y, gids, t, feature_names=FEATURE_NAMES,
        candidate_kind=KIND, input_contract=INPUT_CONTRACT, spec=spec, coefficient_bounds=(0., 10.))


def predict(model, extra, baseline_p, t):
    from . import wpresidual
    wpresidual.validate(model, candidate_kind=KIND, feature_names=FEATURE_NAMES, input_contract=INPUT_CONTRACT)
    return wpresidual.predict(model, extra, baseline_p, t)


def save(model, path):
    from . import wpresidual
    wpresidual.validate(model, candidate_kind=KIND, feature_names=FEATURE_NAMES, input_contract=INPUT_CONTRACT)
    return wpresidual.save(model, path)


def load(path):
    from . import wpresidual
    model = wpresidual.load(path)
    wpresidual.validate(model, candidate_kind=KIND, feature_names=FEATURE_NAMES, input_contract=INPUT_CONTRACT)
    return model
