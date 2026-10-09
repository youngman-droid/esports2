"""Causal objective opportunities; capture and isolated frozen-core experiments.

Remaining team-buff timers are observed tracker inputs, not proof that every
living player still holds the buff. Spawn inference needs a caller-supplied,
patch-matched timing profile. Position, individual buff holders and respawn
seconds are deliberately unavailable. This module has no production dispatch.
"""
from copy import deepcopy
import math
from collections.abc import Mapping
from numbers import Real

import numpy as np


KIND = "objective_opportunities_residual_v1"
FEATURE_NAMES = ["baron_time_adv", "elder_time_adv",
    "baron_alive_opportunity", "elder_alive_opportunity",
    "baron_living_gold_opportunity_k", "elder_living_gold_opportunity_k",
    "dragon_alive_window_adv", "dragon_living_gold_window_adv_k",
    "soul_threat_window_adv", "baron_alive_window_adv", "elder_alive_window_adv"]
BUFF_SECONDS = {"baron": 180., "elder": 150.}
INPUT_CONTRACT = {
    "name": KIND, "orientation": "blue minus red; shared side functions",
    "timers": "provided live team timers or causal historical event times; buff-holder identity unknown",
    "spawn": "explicit patch-matched timing rules; conservative upper remaining time in 60-second opportunity window",
    "combat": "same-clock verified ten-player combat observation only; aged health does not become current readiness",
    "missing": "independent known masks and zero correction; no missingness intercept",
    "unsupported": ["position", "individual_buff_holders", "respawn_seconds", "summoner_cooldowns"],
}
_COUNTERS = {"baron": ("barons_blue", "barons_red"),
             "elder": ("elders_blue", "elders_red"),
             "dragon": ("drag_blue", "drag_red")}


def _number(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _counts(state, kind):
    values = [state.get(k) for k in _COUNTERS[kind]]
    return list(map(int, values)) if all(_number(v) and v >= 0 and v == int(v) for v in values) else None


def _rules(rules, patch):
    if not isinstance(rules, Mapping) or not rules.get("source") or not patch or rules.get("patch") != patch:
        return None
    keys = ("dragon_first_s", "dragon_respawn_s", "baron_first_s", "baron_respawn_s",
            "elder_first_after_soul_s", "elder_respawn_s")
    if any(k in rules and (not _number(rules[k]) or rules[k] <= 0) for k in keys):
        return None
    return dict(rules)


def new_tracker():
    return {"attempt_id": None, "clock_s": None, "ts": None, "counts": {},
            "last_events": {}, "zero_history": {}, "last_block": None}


def _empty(reason):
    return dict(available=False, reason=reason, names=list(FEATURE_NAMES),
                features=[0.] * len(FEATURE_NAMES), known=[False] * len(FEATURE_NAMES),
                source="causal_objective_tracker", raw={}, excluded_inputs=INPUT_CONTRACT["unsupported"][:])


def _spawn_intervals(clock, counts, last_events, zero_history, rules):
    """Intervals in remaining game seconds; no wall-clock countdown during pauses."""
    if rules is None:
        return {}
    result = {}
    soul = counts.get("dragon") is not None and max(counts["dragon"]) >= 4
    for kind in ("baron", "dragon", "elder"):
        if kind == "dragon" and soul:
            continue
        interval, delay = last_events.get(kind), rules.get(kind + "_respawn_s")
        if kind == "elder" and not interval:
            if not soul:
                continue
            interval, delay = last_events.get("dragon"), rules.get("elder_first_after_soul_s")
        if interval is not None and delay is not None:
            lo, hi = interval
            result[kind] = [max(0., lo + delay - clock), max(0., hi + delay - clock)]
        elif kind != "elder" and zero_history.get(kind) and rules.get(kind + "_first_s") is not None:
            remaining = max(0., rules[kind + "_first_s"] - clock)
            result[kind] = [remaining, remaining]
    return result


def observation_features(state, *, spawn_intervals=None):
    """Extract independently gated, signed features from available state only."""
    block = _empty("missing_objective_inputs")
    if not isinstance(state, Mapping):
        return block
    values, known = block["features"], block["known"]
    names = {n: i for i, n in enumerate(FEATURE_NAMES)}
    def put(name, value):
        values[names[name]], known[names[name]] = float(value), True
    timers = {}
    for kind, maximum in BUFF_SECONDS.items():
        pair = [state.get(kind + "_timer_" + side + "_s") for side in ("blue", "red")]
        if all(_number(x) and 0 <= x <= maximum for x in pair):
            timers[kind] = list(map(float, pair))
            put(kind + "_time_adv", (pair[0] - pair[1]) / maximum)
    combat = state.get("combat") or {}
    raw, channels = combat.get("raw") or {}, combat.get("channel_available") or {}
    # A carried historical health sample cannot certify readiness at a later
    # objective opportunity. Its source age is retained, but the coupling is off.
    frame_ts = state.get("ts", combat.get("observation_ts"))
    fresh = (combat.get("available") is True and combat.get("age_upper_s") == 0
             and _number(combat.get("observation_ts")) and _number(frame_ts)
             and combat["observation_ts"] == frame_ts)
    readiness = {}
    for channel in ("alive", "living_gold_k"):
        vector = raw.get(channel)
        if fresh and channels.get(channel) is True and isinstance(vector, (list, tuple)) and len(vector) == 10:
            if all(_number(v) and v >= 0 and (channel != "alive" or v <= 1) for v in vector):
                readiness[channel] = [sum(vector[:5]), sum(vector[5:])]
    for kind, pair in timers.items():
        weight = [v / BUFF_SECONDS[kind] for v in pair]
        for channel, suffix, divisor in (("alive", "alive_opportunity", 5.),
                                          ("living_gold_k", "living_gold_opportunity_k", 1.)):
            if channel in readiness:
                b, r = readiness[channel]
                put(kind + "_" + suffix, (weight[0] * b - weight[1] * r) / divisor)
    spawns = {}
    for kind, interval in (spawn_intervals or {}).items():
        if (kind not in ("dragon", "baron", "elder") or not isinstance(interval, (list, tuple))
                or len(interval) != 2 or not all(_number(v) and v >= 0 for v in interval) or interval[0] > interval[1]):
            continue
        # Use the upper bound: an uncertain earlier spawn never creates a
        # stronger opportunity than the latest admissible acquisition allows.
        proximity = max(0., 1. - float(interval[1]) / 60.)
        spawns[kind] = dict(remaining_lo_s=float(interval[0]), remaining_hi_s=float(interval[1]), proximity=proximity)
        if "alive" in readiness:
            b, r = readiness["alive"]
            put(kind + "_alive_window_adv", proximity * (b-r) / 5.)
        if kind == "dragon":
            if "living_gold_k" in readiness:
                b, r = readiness["living_gold_k"]
                put("dragon_living_gold_window_adv_k", proximity * (b-r))
            dragons = _counts(state, "dragon")
            if dragons is not None:
                put("soul_threat_window_adv", proximity * ((dragons[0] == 3) - (dragons[1] == 3)))
    block.update(available=any(known), reason="partial_objective_inputs" if any(known) else block["reason"],
                 raw=dict(buff_remaining_s=timers, spawn=spawns, readiness=readiness,
                          combat_age_upper_s=combat.get("age_upper_s")),
                 observation_ts=state.get("ts"), clock_s=state.get("clock_s"))
    return block


def capture(state, tracker, *, attempt_id=None, timing_rules=None, max_event_interval_s=90.):
    """Idempotent live capture; mutate only a bounded caller-owned tracker.

    Same timestamp retries return their original block. Unknown older frames
    cannot replay transitions. A new attempt, later clock regression or a
    decreasing cumulative objective counter resets event history. Transitions
    supply acquisition intervals rather than invented exact kill seconds.
    """
    if not isinstance(tracker, dict):
        raise TypeError("tracker must be a caller-owned dict")
    if not _number(max_event_interval_s) or max_event_interval_s < 0:
        raise ValueError("max_event_interval_s must be nonnegative")
    if not isinstance(state, Mapping) or not _number(state.get("clock_s")) or state["clock_s"] < 0:
        return _empty("invalid_game_clock")
    ts = state.get("ts", (state.get("combat") or {}).get("observation_ts"))
    if not _number(ts):
        return _empty("missing_frame_timestamp")
    clock = float(state["clock_s"])
    attempt = attempt_id if attempt_id is not None else state.get("game_id")
    same_attempt = tracker.get("attempt_id") == attempt
    if same_attempt and tracker.get("ts") is not None and ts <= tracker["ts"]:
        return deepcopy(tracker["last_block"]) if ts == tracker["ts"] else _empty("out_of_order_frame")
    counts = {k: _counts(state, k) for k in _COUNTERS}
    reset = (not same_attempt or tracker.get("clock_s") is not None and clock < tracker["clock_s"]
             or any(counts[k] is not None and tracker.get("counts", {}).get(k) is not None
                    and any(a < b for a, b in zip(counts[k], tracker["counts"][k])) for k in counts))
    if reset or tracker.get("ts") is None:
        tracker.clear(); tracker.update(new_tracker()); tracker["attempt_id"] = attempt
        tracker["zero_history"] = {k: counts[k] == [0, 0] for k in counts}
    previous_clock = tracker.get("clock_s")
    for kind in counts:
        before, after = tracker.get("counts", {}).get(kind), counts[kind]
        if after is None:
            tracker["last_events"].pop(kind, None)
            tracker["zero_history"][kind] = False
        if before is not None and after is not None and sum(after) > sum(before):
            if previous_clock is not None and clock - previous_clock <= max_event_interval_s:
                tracker["last_events"][kind] = [previous_clock, clock]
            else:
                tracker["last_events"].pop(kind, None)
            tracker["zero_history"][kind] = False
    profile = _rules(timing_rules, state.get("patch"))
    spawn = _spawn_intervals(clock, counts, tracker["last_events"], tracker["zero_history"], profile)
    block = observation_features(state, spawn_intervals=spawn)
    block.update(attempt_id=attempt, reset=bool(reset), rules_available=profile is not None,
                 timing_rules=profile, acquisition_intervals=deepcopy(tracker["last_events"]))
    tracker.update(ts=float(ts), clock_s=clock, counts=counts, last_block=deepcopy(block))
    return block


def historical_features(events, clock_s, *, events_complete=False, combat=None,
                        timing_rules=None, patch=None):
    """Adapt a complete historical event stream; ignore every future event.

    Only explicit earlier Baron/Elder/dragon kills enter. Caller must certify
    event completeness and use the corrected prediction clock. Optional aged
    HP remains unavailable for objective/readiness interactions.
    """
    if not events_complete or not _number(clock_s) or clock_s < 0:
        return _empty("unverified_historical_events")
    counts = {k: [0, 0] for k in _COUNTERS}
    last, side_last = {}, {k: [None, None] for k in BUFF_SECONDS}
    for event in events:
        time_s, side, action = event.get("time_s"), event.get("side"), event.get("action") or ""
        if not _number(time_s) or time_s < 0 or time_s > clock_s or side not in ("blue", "red"):
            continue
        kind = "elder" if action == "dragon:elder" else "dragon" if action.startswith("dragon") else action
        if kind not in counts:
            continue
        slot = 0 if side == "blue" else 1
        counts[kind][slot] += 1
        if kind not in last or time_s > last[kind][0]:
            last[kind] = [float(time_s), float(time_s)]
        if kind in side_last:
            previous = side_last[kind][slot]
            side_last[kind][slot] = max(float(time_s), previous) if previous is not None else float(time_s)
    state = dict(clock_s=float(clock_s), combat=combat or {}, patch=patch)
    for kind, keys in _COUNTERS.items():
        state.update(zip(keys, counts[kind]))
    for kind, duration in BUFF_SECONDS.items():
        for slot, side in enumerate(("blue", "red")):
            acquisition = side_last[kind][slot]
            state[kind + "_timer_" + side + "_s"] = max(0., duration - (clock_s-acquisition)) if acquisition is not None else 0.
    profile = _rules(timing_rules, patch)
    spawn = _spawn_intervals(clock_s, counts, last, {k: sum(v) == 0 for k, v in counts.items()}, profile)
    block = observation_features(state, spawn_intervals=spawn)
    block.update(source="complete_historical_event_prefix", rules_available=profile is not None,
                 timing_rules=profile, acquisition_intervals=last)
    return block


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
