"""Versioned historical input contract and conservative telemetry primitives.

The archive stores sparse wall-clock windows, not a verified pause/attempt clock.
Recovering the original timestamp repairs provenance but cannot manufacture a
game clock. Such observations remain unavailable until a clock is certified.
"""
import collections
import hashlib
import json
import math


INPUT_CONTRACT = {
    "name": "wpx_inputs_v2",
    "hp": "strict all-player cumulative-gold clock bracket plus recovered pregame wall-origin upper bound; ambiguous clocks unavailable",
    "hp_max_age_s": 90,
    "hp_pre_observation": "zero features and has_hp=0",
    "death": "official HP if trusted; otherwise unique victim timer estimate, uncertainty recorded",
    "items": "ablated_v1: items_done and item_gold_k zero in training and candidate serving",
    "gold_cs": "latest observed complete minute <= prediction clock; no interpolation",
    "events": "event game clock <= prediction clock, pre-event rows evaluated one second earlier",
}
INPUT_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(INPUT_CONTRACT, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def archived_observation(row, origin=None):
    """Wrap a stored row without relabeling a nominal minute as observation time.

    ``verified_game_clock_s`` is accepted only together with explicit clock
    verification and an attempt identifier. The current importer intentionally
    does not produce those fields: sparse frames cannot establish them.
    """
    origin = origin or {}
    data = row.get("data") or {}
    provenance = data.get("_observation") or {}
    clock = provenance.get("verified_game_clock_s")
    trusted = (provenance.get("clock_status") == "verified_pause_attempt_clock"
               and bool(provenance.get("attempt_id"))
               and isinstance(clock, (int, float)) and math.isfinite(clock) and clock >= 0)
    ts = row.get("ts")
    original_ts = origin.get("origin_ts")
    return {
        "data": data, "_clock_s": float(clock) if trusted else None,
        "_clock_trusted": trusted, "_observation_ts": ts,
        "_origin_ts": original_ts,
        "_wall_elapsed_s": ts - original_ts if ts is not None and original_ts is not None else None,
        "_clock_source": "verified_pause_attempt_clock" if trusted else "untrusted_sparse_wall_clock",
        "_attempt_id": provenance.get("attempt_id") if trusted else None,
        "_origin_source": origin.get("source", "missing_origin"),
    }


def bracket_archived_observations(rows, origin, gold, champions):
    """Conservative clock intervals, never an alleged exact pause-adjusted clock.

    Assumptions: per-player earned totalGold is monotone within a game attempt;
    the two feeds identify the same players/champions in the same role slots.
    Exact identities, a pregame opening, strict lower/upper gold brackets and
    monotone prefixes are required. The original elapsed wall time is a second
    upper bound: pauses/init delays make this gate later, never earlier.
    Missing windows therefore reduce coverage instead of being guessed pauses.
    """
    import re
    def token(value):
        cleaned = re.sub(r"[^a-z0-9]", "", str(value).lower())
        # Canonical Riot internal IDs used by the established live adapter.
        return {"monkeyking": "wukong", "renata": "renataglasc", "nunu": "nunuwillump"}.get(cleaned, cleaned)
    def rejected(reason):
        return {r["minute"]: dict(archived_observation(r, origin), _clock_source=reason,
                                 _clock_s=None, _clock_trusted=False)
                for r in rows}
    if not origin or origin.get("origin_ts") is None:
        return rejected("missing_original_origin")
    metadata = origin.get("game_metadata") or {}
    source_champs = []
    for side, expected_ids in (("blue", list(range(1, 6))), ("red", list(range(6, 11)))):
        players = (metadata.get(side + "TeamMetadata") or {}).get("participantMetadata") or []
        if [p.get("participantId") for p in players] != expected_ids:
            return rejected("unverified_participant_order")
        roles = [p.get("role") for p in players]
        if roles != ["top", "jungle", "mid", "bottom", "support"]:
            return rejected("unverified_role_order")
        source_champs.extend(p.get("championId") for p in players)
    if len(champions) != 10 or list(map(token, champions)) != list(map(token, source_champs)):
        return rejected("opening_final_champion_mismatch")
    opening = origin.get("first_frame") or {}
    if any((opening.get(side + "Team") or {}).get("totalGold") != 0 for side in ("blue", "red")):
        return rejected("ambiguous_opening")
    timeline = [(m * 60, list(v[0]) + list(v[1])) for m, v in sorted(gold.items())]
    if not timeline or any(len(v) != 10 or any(x is None for x in v) for _, v in timeline):
        return rejected("incomplete_gold_timeline")
    out, previous, reset = {}, None, False
    for row in sorted(rows, key=lambda r: (r.get("ts") or 0, r["minute"])):
        wrapped = archived_observation(row, origin)
        data = row.get("data") or {}
        observation_champs = (data.get("_observation") or {}).get("champions")
        if observation_champs is not None and list(map(token, observation_champs)) != list(map(token, champions)):
            wrapped.update(_clock_trusted=False, _clock_s=None, _clock_source="observation_champion_mismatch")
            out[row["minute"]] = wrapped
            continue
        if (("pidb" in data and data["pidb"] != list(range(1, 6)))
                or ("pidr" in data and data["pidr"] != list(range(6, 11)))):
            wrapped.update(_clock_trusted=False, _clock_s=None, _clock_source="observation_participant_order_mismatch")
            out[row["minute"]] = wrapped
            continue
        observed = list(data.get("gdb") or []) + list(data.get("gdr") or [])
        reason = "unbracketed_gold_clock"
        if len(observed) == 10 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in observed):
            if previous is not None and any(a < b for a, b in zip(observed, previous)):
                reset = True
            previous = observed
            lo = hi = None
            last = None
            for clock, values in timeline:
                if last is not None and any(a < b for a, b in zip(values, last)):
                    reason = "gold_timeline_reset"; break
                last = values
                if all(a <= b for a, b in zip(values, observed)) and any(a < b for a, b in zip(values, observed)):
                    lo = clock
                if all(a >= b for a, b in zip(values, observed)) and any(a > b for a, b in zip(values, observed)):
                    hi = clock; break
            if reset:
                reason = "feed_counter_reset"
            elif lo is not None and hi is not None and 0 < hi - lo <= 90 and row.get("ts") is not None:
                # Existing ts BIGINT truncates the actual observation second.
                # The +1 is an upper bound, never backdate fractional timestamps.
                wall_hi = row["ts"] + 1.0 - origin["origin_ts"]
                upper = max(float(hi), wall_hi)
                if upper - lo <= 90:
                    wrapped.update(_clock_s=upper, _clock_lo_s=float(lo), _clock_trusted=True,
                                   _clock_source="gold_bracket_wall_upper_v1", _clock_gold_hi_s=float(hi))
                else:
                    reason = "clock_interval_too_wide"
        if not wrapped["_clock_trusted"]:
            wrapped["_clock_source"] = reason
        out[row["minute"]] = wrapped
    return out


def estimated_deaths(events, t_s):
    """Deduplicate a victim's latest observed death; timer remains an estimate.

    A kill by a previously dead player establishes that player was alive at
    that event. Missing victim identities do not become anonymous extra deaths.
    """
    dead_until = {}
    unresolved = 0
    for e in sorted(events, key=lambda e: (e.get("time_s") or 0, e.get("seq") or 0)):
        et = e.get("time_s")
        if e.get("action") != "kill" or et is None or et > t_s:
            continue
        killer = e.get("player")
        if killer:
            dead_until.pop((e.get("side"), killer), None)
        side, victim = e.get("victim_side"), e.get("target")
        if side not in ("blue", "red") or not victim:
            unresolved += 1
            continue
        # Match the existing rough timer scale, anchored to the death time.
        # This is never represented as a measured alive/dead state.
        dead_until[(side, victim)] = et + min(60.0, 8.0 + 1.6 * et / 60.0)
    counts = [sum(side == target and until > t_s for (side, _), until in dead_until.items())
              for target in ("blue", "red")]
    return {"counts": [min(5, n) for n in counts], "source": "victim_timer_estimate",
            "uncertain": True, "unresolved_victims": unresolved}


def replay_inventory(events, t_s, prices, *, price_patch, game_patch):
    """Replay supported inventory transitions against an explicit patch table.

    The stored gol.gg undo event has item_id=-1 and loses before/after identity.
    Inferring which transaction it undoes is unsafe. That player's inventory is
    therefore unavailable from that point forward. Destruction removes upgrade
    components/consumables; sales remove completed items as well as their value.
    This helper is for input diagnostics; v2 modeling ablates both item channels.
    """
    if not game_patch or price_patch != game_patch:
        return {"available": False, "reason": "patch_price_mismatch", "items_done": 0, "item_gold": 0}
    inv = collections.defaultdict(collections.Counter)
    sides, invalid = {}, set()
    for e in sorted(events, key=lambda e: (e.get("t") or 0, e.get("seq") or 0)):
        if e.get("t") is None or e["t"] > t_s:
            continue
        player, item = e.get("player_id"), e.get("item_id")
        sides[player] = e.get("side")
        kind = e.get("event")
        if kind == "ITEM_UNDO":
            invalid.add(player)
        elif kind == "ITEM_PURCHASED":
            inv[player][item] += 1
        elif kind in ("ITEM_SOLD", "ITEM_DESTROYED"):
            if inv[player][item] <= 0:
                invalid.add(player)
            else:
                inv[player][item] -= 1
        else:
            invalid.add(player)
    if invalid:
        return {"available": False, "reason": "ambiguous_or_incomplete_transitions", "items_done": 0, "item_gold": 0}
    total = done = 0
    for player, inventory in inv.items():
        side = sides.get(player)
        if side not in ("blue", "red"):
            return {"available": False, "reason": "unknown_side", "items_done": 0, "item_gold": 0}
        sign = 1 if side == "blue" else -1
        for item, n in inventory.items():
            if n <= 0:
                continue
            if item not in prices:
                return {"available": False, "reason": "missing_patch_price", "items_done": 0, "item_gold": 0}
            total += sign * n * prices[item]
            done += sign * n * (prices[item] >= 2200)
    return {"available": True, "reason": "patch_inventory", "items_done": done, "item_gold": total}


def load_patch_item_prices(patch, cache_dir, *, allow_download=False):
    """Load an immutable Data Dragon .1 patch table, with explicit provenance.

    Never falls back to today's global item prices. A game recorded only as a
    major/minor patch resolves to that patch's .1 static-data release, rather
    than claiming unavailable subpatch precision.
    """
    import pathlib
    import re
    import urllib.request
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", patch or ""):
        raise ValueError("a numeric Data Dragon patch is required")
    version = patch if patch.count(".") == 2 else patch + ".1"
    path = pathlib.Path(cache_dir) / (version + ".json")
    if path.exists():
        result = json.loads(path.read_text())
    else:
        if not allow_download:
            raise FileNotFoundError(path)
        url = "https://ddragon.leagueoflegends.com/cdn/%s/data/en_US/item.json" % version
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
        payload = json.loads(raw)
        if payload.get("version") != version:
            raise ValueError("Data Dragon returned a different version")
        result = {"version": version, "requested_patch": patch, "source": url,
                  "response_sha256": hashlib.sha256(raw).hexdigest(),
                  "resolution": "exact_release" if patch.count(".") == 2 else "patch_first_release",
                  "prices": {k: int(v["gold"]["total"]) for k, v in payload["data"].items()}}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            json.dump(result, handle, sort_keys=True)
    if result.get("version") != version:
        raise ValueError("cached patch version mismatch")
    return {int(k): int(v) for k, v in result["prices"].items()}, result


def inventory_from_db(conn, game_id, t_s, cache_dir, *, allow_download=False):
    """Diagnostic current inventory from every stored transition and game patch."""
    game = conn.execute("SELECT patch FROM golgg_games WHERE game_id=%s", (game_id,)).fetchone()
    if not game or not game.get("patch"):
        return {"available": False, "reason": "missing_game_patch", "items_done": 0, "item_gold": 0}
    events = [dict(row) for row in conn.execute("""SELECT b.player_id,b.seq,b.build_time t,b.event,b.item_id,p.side
        FROM golgg_builds b JOIN golgg_players p USING(game_id,player_id)
        WHERE b.game_id=%s ORDER BY b.build_time,b.player_id,b.seq""", (game_id,))]
    if len({e["player_id"] for e in events}) != 10:
        return {"available": False, "reason": "incomplete_player_builds", "items_done": 0, "item_gold": 0}
    try:
        prices, provenance = load_patch_item_prices(game["patch"], cache_dir, allow_download=allow_download)
    except (FileNotFoundError, ValueError):
        return {"available": False, "reason": "missing_patch_prices", "items_done": 0, "item_gold": 0}
    result = replay_inventory(events, t_s, prices, price_patch=game["patch"], game_patch=game["patch"])
    result.update(game_id=game_id, game_patch=game["patch"], price_provenance=provenance,
                  transition_count=sum(e["t"] is not None and e["t"] <= t_s for e in events))
    # The price map itself is already immutable in the cache; avoid duplicating it.
    result["price_provenance"] = {k: v for k, v in provenance.items() if k != "prices"}
    return result
