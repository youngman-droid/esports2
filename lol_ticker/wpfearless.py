"""Causal Fearless availability and observed player pools under explicit rules.

No league-name default implies a format. A rule profile must identify an exact
tournament and phase, its effective window, and primary-source evidence. Series
records need a certified complete prefix and stable team IDs across side swaps.
Player pools count earlier observed role/champion appearances, never outcomes;
they describe supported experience, not mastery or guaranteed legal choices.
"""
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from .sqpairs import key as champion_key
from .wpcomposition import ROLES, ROLE_ALIASES


KIND = "certified_fearless_pool_v1"
FEATURE_NAMES = (["remaining_pool_" + role for role in ROLES]
                 + ["pool_retained_" + role for role in ROLES]
                 + ["pick_support_log_" + role for role in ROLES]
                 + ["unseen_pick_adv_" + role for role in ROLES])
INPUT_CONTRACT = dict(kind=KIND, names=FEATURE_NAMES, role_order=list(ROLES),
                      rules="explicit primary-source profile scoped to tournament, phase and effective window",
                      series="certified complete earlier-game prefix, stable team IDs, current game identity",
                      pools="timestamped prior role/champion appearances; no outcomes; minimum three champion and ten player games",
                      completeness="each role pair requires both player histories; missing role pairs zero",
                      orientation="blue minus red; no availability intercept")
FIRST_STAND_2025_RULE_TEMPLATE = {
    "rule_id": "riot_first_stand_2025_full_fearless",
    "tournament_id": "First Stand 2025",
    "phases": ["knockout"],
    "effective_from_ts": 1741564800,  # March 10, 2025 UTC
    "effective_until_ts": 1742169600,  # March 17 exclusive; event ends March 16
    "source_published_ts": 1736294400,  # January 8, 2025 UTC
    "source_url": "https://lolesports.com/en-US/news/lol-esports-in-2025",
    "mode": "both_teams", "reset_before_games": [], "max_games": 5,
    "verified": True,
    "scope_note": "Only First Stand 2025; exact phase/series identity must be separately certified.",
}
FIRST_STAND_2025_ROUND_ROBIN_RULE_TEMPLATE = dict(
    FIRST_STAND_2025_RULE_TEMPLATE,
    rule_id="riot_first_stand_2025_full_fearless_round_robin",
    phases=["round_robin"], max_games=3)


def _finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _empty(reason):
    return dict(kind=KIND, available=False, series_available=False, reason=reason, names=list(FEATURE_NAMES),
                features=[0.] * len(FEATURE_NAMES), known=[False] * len(FEATURE_NAMES),
                coverage={"players": 0, "prior_games": 0}, blocked={"blue": [], "red": []},
                player_pools={}, provenance={}, limitations=[
                    "Observed pools measure earlier appearances, not champion proficiency or win strength.",
                    "Remaining pools apply the certified Fearless restriction only; current bans and disabled champions are separate.",
                    "Player identities and series boundaries must be certified by the source adapter."])


def _rule_reason(context, rules, draft_ts):
    if not isinstance(rules, Mapping) or rules.get("verified") is not True:
        return "uncertified_rules"
    url = rules.get("source_url")
    if not isinstance(url, str):
        return "missing_primary_rule_source"
    source = urlparse(url)
    if source.scheme != "https" or source.hostname not in {
            "lolesports.com", "competitiveops.riotgames.com", "www.leagueoflegends.com",
            "leagueoflegends.com", "static.riotgames.com", "assets.contentstack.io"}:
        return "missing_primary_rule_source"
    if (not isinstance(rules.get("rule_id"), str) or not rules["rule_id"]
            or not isinstance(rules.get("mode"), str) or rules["mode"] not in {"both_teams", "own_team", "none"}
            or type(rules.get("max_games")) is not int or not 1 <= rules["max_games"] <= 7
            or not isinstance(rules.get("reset_before_games"), list)
            or any(type(x) is not int or not 2 <= x <= rules["max_games"] for x in rules["reset_before_games"])
            or len(set(rules["reset_before_games"])) != len(rules["reset_before_games"])):
        return "invalid_rule_profile"
    if (rules.get("tournament_id") != context.get("tournament_id")
            or not isinstance(rules.get("phases"), list) or context.get("phase") not in rules["phases"]):
        return "rule_tournament_or_phase_mismatch"
    if (not all(_finite(rules.get(k)) for k in ("source_published_ts", "effective_from_ts", "effective_until_ts"))
            or not rules["source_published_ts"] <= draft_ts
            or not rules["effective_from_ts"] <= draft_ts < rules["effective_until_ts"]):
        return "rule_time_mismatch"
    return None


def build_player_pools(records, *, as_of_ts, lookback_days=365, min_champion_games=3,
                       identity_namespace="caller_certified"):
    """Count timestamped prior appearances, deduplicating game/player identity.

    Records are ``game_id, player_id, role, champion, completed_ts``. The current
    observation and future games are excluded strictly. A snapshot carries its
    upper cutoff and source coverage so it cannot masquerade as a full roster.
    """
    if (not _finite(as_of_ts) or type(lookback_days) is not int or lookback_days <= 0
            or type(min_champion_games) is not int or min_champion_games < 1
            or not isinstance(identity_namespace, str) or not identity_namespace):
        raise ValueError("Finite cutoff and positive pool support settings are required")
    pools, seen, accepted, rejected, excluded = defaultdict(lambda: defaultdict(Counter)), {}, 0, 0, 0
    latest = None
    for record in records:
        if not isinstance(record, Mapping):
            rejected += 1
            continue
        ts, role = record.get("completed_ts"), record.get("role")
        player, game, champion = record.get("player_id"), record.get("game_id"), record.get("champion")
        if (not _finite(ts) or role not in ROLES or player is None or not str(player)
                or game is None or not isinstance(champion, str) or not champion_key(champion)):
            rejected += 1
            continue
        if not as_of_ts - lookback_days * 86400 <= ts < as_of_ts:
            excluded += 1
            continue
        identity, value = (str(game), str(player)), (role, champion_key(champion), float(ts))
        if identity in seen:
            if seen[identity] != value:
                raise ValueError("Conflicting historical game/player appearance")
            continue
        seen[identity] = value
        pools[str(player)][role][value[1]] += 1
        accepted += 1
        latest = ts if latest is None else max(latest, ts)
    return dict(kind="causal_player_pools_v1", available=bool(accepted), as_of_ts=float(as_of_ts),
                identity_namespace=identity_namespace,
                latest_observation_ts=float(latest) if latest is not None else None,
                lookback_days=lookback_days, min_champion_games=min_champion_games,
                records=accepted, rejected_records=rejected, excluded_records=excluded,
                pools={p: {r: dict(c) for r, c in role.items()} for p, role in pools.items()})


def _role_map(value):
    return (isinstance(value, Mapping) and set(value) == set(ROLES)
            and all(isinstance(c, str) and champion_key(c) for c in value.values()))


def features(context, prior_games, rules, player_history, *, draft_ts,
             min_champion_games=3, min_player_games=10):
    """Extract remaining observed pools from a certified completed draft.

    Context contains ``series_id,tournament_id,phase,game_num,blue_team_id,
    red_team_id,history_complete,picks,players``. Picks and players each map
    side->role->stable ID/name. Prior records carry the same IDs and picks plus
    ``completed_ts`` and form exactly games 1..game_num-1. No outcomes are read.
    """
    if not isinstance(context, Mapping) or not _finite(draft_ts):
        return _empty("missing_series_context")
    if context.get("history_complete") is not True or not context.get("series_id"):
        return _empty("uncertified_series_prefix")
    reason = _rule_reason(context, rules, draft_ts)
    if reason:
        return _empty(reason)
    game_num = context.get("game_num")
    if type(game_num) is not int or not 1 <= game_num <= rules["max_games"]:
        return _empty("invalid_series_game_number")
    teams = [context.get(side + "_team_id") for side in ("blue", "red")]
    if any(t is None or not str(t) for t in teams) or str(teams[0]) == str(teams[1]):
        return _empty("invalid_series_teams")
    teams = [str(t) for t in teams]
    picks, players = context.get("picks"), context.get("players")
    if (not isinstance(picks, Mapping) or not isinstance(players, Mapping)
            or any(not _role_map(picks.get(s)) for s in ("blue", "red"))
            or any(not isinstance(players.get(s), Mapping) or set(players[s]) != set(ROLES)
                   or any(p is None or not str(p) for p in players[s].values()) for s in ("blue", "red"))):
        return _empty("incomplete_current_draft_or_players")
    current_champs = [champion_key(picks[s][r]) for s in ("blue", "red") for r in ROLES]
    current_players = [str(players[s][r]) for s in ("blue", "red") for r in ROLES]
    if len(set(current_champs)) != 10 or len(set(current_players)) != 10:
        return _empty("duplicate_current_champion_or_player")
    if not isinstance(prior_games, (list, tuple)) or len(prior_games) != game_num - 1:
        return _empty("incomplete_series_history")
    if not all(isinstance(g, Mapping) for g in prior_games):
        return _empty("invalid_series_history")
    if (any(type(g.get("game_num")) is not int for g in prior_games)
            or sorted(g["game_num"] for g in prior_games) != list(range(1, game_num))):
        return _empty("noncontiguous_series_history")
    used = {team: set() for team in teams}
    for game in sorted(prior_games, key=lambda g: g["game_num"]):
        if (str(game.get("series_id")) != str(context["series_id"])
                or not _finite(game.get("completed_ts")) or game["completed_ts"] >= draft_ts
                or game.get("tournament_id") != context["tournament_id"]):
            return _empty("series_identity_or_time_mismatch")
        prior_teams = [str(game.get(s + "_team_id")) for s in ("blue", "red")]
        gpicks = game.get("picks")
        if set(prior_teams) != set(teams) or not isinstance(gpicks, Mapping) or any(not _role_map(gpicks.get(s)) for s in ("blue", "red")):
            return _empty("invalid_prior_series_draft")
        champs = [champion_key(gpicks[s][r]) for s in ("blue", "red") for r in ROLES]
        if len(set(champs)) != 10:
            return _empty("duplicate_prior_champion")
        if game["game_num"] in rules["reset_before_games"]:
            used = {team: set() for team in teams}
        for side, team in zip(("blue", "red"), prior_teams):
            blocked = set.union(*used.values()) if rules["mode"] == "both_teams" else used[team] if rules["mode"] == "own_team" else set()
            selected = {champion_key(gpicks[side][r]) for r in ROLES}
            if selected & blocked:
                return _empty("prior_draft_contradicts_rules")
        for side, team in zip(("blue", "red"), prior_teams):
            used[team].update(champion_key(gpicks[side][r]) for r in ROLES)
    if game_num in rules["reset_before_games"]:
        used = {team: set() for team in teams}
    blocked = {side: (set.union(*used.values()) if rules["mode"] == "both_teams"
                      else used[team] if rules["mode"] == "own_team" else set())
               for side, team in zip(("blue", "red"), teams)}
    if any({champion_key(picks[s][r]) for r in ROLES} & blocked[s] for s in ("blue", "red")):
        return _empty("current_draft_contradicts_rules")
    # Certified series removals remain useful capture even while player-ID
    # history is unavailable. They still produce no fitted pool correction.
    out = _empty("missing_or_future_player_history")
    out.update(series_available=True, blocked={side: sorted(value) for side, value in blocked.items()})
    out["coverage"]["prior_games"] = len(prior_games)
    rule_provenance = {key: rules[key] for key in ("rule_id", "tournament_id", "phases", "effective_from_ts",
        "effective_until_ts", "source_published_ts", "source_url", "mode", "reset_before_games", "max_games", "verified")}
    out["provenance"] = dict(rules=rule_provenance, series_id=str(context["series_id"]), game_num=game_num,
                             draft_ts=float(draft_ts))
    if (type(min_champion_games) is not int or min_champion_games < 1
            or type(min_player_games) is not int or min_player_games < 1):
        raise ValueError("Positive observed-pool support thresholds are required")
    if (not isinstance(player_history, Mapping) or player_history.get("kind") != "causal_player_pools_v1"
            or player_history.get("available") is not True or not _finite(player_history.get("as_of_ts"))
            or player_history["as_of_ts"] > draft_ts
            or not _finite(player_history.get("latest_observation_ts"))
            or player_history["latest_observation_ts"] >= draft_ts
            or not isinstance(player_history.get("pools"), Mapping)):
        return out
    out["reason"] = "incomplete_player_pool_support"
    raw = {side: [] for side in ("blue", "red")}
    known_roles = []
    for role in ROLES:
        supported = []
        for side in ("blue", "red"):
            player = str(players[side][role])
            role_history = player_history["pools"].get(player)
            player_role = role_history.get(role) if isinstance(role_history, Mapping) else None
            valid = isinstance(player_role, Mapping) and all(
                isinstance(champ, str) and type(n) is int and n > 0 for champ, n in player_role.items())
            total = sum(player_role.values()) if valid else 0
            ready = valid and total >= min_player_games
            supported.append(ready)
            if ready:
                pool = {champion_key(champ) for champ, n in player_role.items() if n >= min_champion_games}
                remaining = pool - blocked[side]
                selected_n = player_role.get(champion_key(picks[side][role]), 0)
                raw[side].append([float(len(remaining)), float(len(remaining) / len(pool)) if pool else 0.,
                                  math.log1p(selected_n), -float(selected_n == 0)])
                out["player_pools"][player + ":" + role] = dict(total_games=total, supported_pool=sorted(pool),
                    remaining_pool=sorted(remaining), selected_champion_games=int(selected_n))
                out["coverage"]["players"] += 1
            else:
                raw[side].append([0.] * 4)
        known_roles.append(all(supported))
    # Incomplete role pairs are individually zero. No intercept is permitted;
    # missing complete player support never shifts the baseline on its own.
    out["known"] = [known_roles[r] for _ in range(4) for r in range(5)]
    out["features"] = [raw["blue"][r][family] - raw["red"][r][family] if known_roles[r] else 0.
                       for family in range(4) for r in range(5)]
    out["available"] = any(known_roles)
    out["complete"] = all(known_roles)
    out["reason"] = "complete_fearless_pools" if all(known_roles) else "partial_fearless_pools" if any(known_roles) else "incomplete_player_pool_support"
    out["provenance"] = dict(rules=rule_provenance, series_id=str(context["series_id"]), game_num=game_num,
                             draft_ts=float(draft_ts), pool_as_of_ts=float(player_history["as_of_ts"]),
                             min_champion_games=min_champion_games, min_player_games=min_player_games)
    return out


def capture(metadata, context=None, *, as_of_ts, source_root=None):
    """Use a supplied certified bundle or a local series snapshot.

    Bundle keys are ``context,prior_games,rules,player_history``. Without a
    bundle, ``context={'series_id': ID}`` loads ``data/fearless/series/ID.json``.
    A default player snapshot can live at ``data/fearless/player_pools.json``.
    Missing certification yields zero features rather than guessing a format.
    """
    from . import config
    root = Path(source_root) if source_root is not None else Path(config.REPO_ROOT) / "data" / "fearless"
    bundle = context if isinstance(context, Mapping) and "context" in context else None
    local_bundle = bundle is None
    if bundle is None and isinstance(context, Mapping) and context.get("series_id"):
        series_id = str(context["series_id"])
        if re.fullmatch(r"[A-Za-z0-9_.-]+", series_id) and series_id not in {".", ".."}:
            path = root / "series" / (series_id + ".json")
            try:
                bundle = json.loads(path.read_text()) if path.exists() else None
            except (OSError, ValueError):
                return _empty("invalid_local_series_context")
    if not isinstance(bundle, Mapping) or not isinstance(bundle.get("context"), Mapping):
        return _empty("missing_certified_series_context")
    current = dict(bundle["context"])
    if local_bundle and (type(context.get("game_num")) is not int
                         or context["game_num"] != current.get("game_num")
                         or str(context["series_id"]) != str(current.get("series_id"))):
        return _empty("local_series_game_identity_mismatch")
    picks, players = {}, {}
    for side, ids in (("blue", set(range(1, 6))), ("red", set(range(6, 11)))):
        team = metadata.get(side + "TeamMetadata") if isinstance(metadata, Mapping) else None
        participants = team.get("participantMetadata") if isinstance(team, Mapping) else None
        if not isinstance(participants, list) or len(participants) != 5:
            return _empty("incomplete_live_participant_metadata")
        picks[side], players[side], seen = {}, {}, set()
        for participant in participants:
            if not isinstance(participant, Mapping):
                return _empty("invalid_live_participant_metadata")
            pid, role = participant.get("participantId"), participant.get("role")
            player_id, champion = participant.get("esportsPlayerId"), participant.get("championId")
            if (type(pid) is not int or pid not in ids or pid in seen or not isinstance(role, str) or role not in ROLE_ALIASES
                    or ROLE_ALIASES[role] in picks[side] or player_id is None or not str(player_id)
                    or not isinstance(champion, str) or not champion_key(champion)):
                return _empty("unverified_live_player_roles")
            seen.add(pid)
            picks[side][ROLE_ALIASES[role]] = champion
            players[side][ROLE_ALIASES[role]] = str(player_id)
        team_id = team.get("esportsTeamId")
        if team_id is None or str(team_id) != str(current.get(side + "_team_id")):
            return _empty("live_series_team_identity_mismatch")
    # Snapshot picks/rosters cannot overwrite the actual current draft.
    current.update(picks=picks, players=players)
    history = bundle.get("player_history")
    if history is None:
        path = root / "player_pools.json"
        try:
            history = json.loads(path.read_text()) if path.exists() else None
        except (OSError, ValueError):
            return _empty("invalid_local_player_pools")
    if isinstance(history, Mapping) and history.get("identity_namespace") != "riot_esports_player_id":
        out = features(current, bundle.get("prior_games"), bundle.get("rules"), None, draft_ts=as_of_ts)
        if out["series_available"]:
            out["reason"] = "unresolved_live_player_identity_namespace"
        return out
    return features(current, bundle.get("prior_games"), bundle.get("rules"), history, draft_ts=as_of_ts)


def fit(extra, baseline_p, y, gids, *, spec=None):
    """Fit an isolated positive pool-advantage residual on frozen draft odds."""
    import numpy as np
    from . import wpresidual
    return wpresidual.fit(extra, baseline_p, y, gids, np.zeros(len(y)), feature_names=FEATURE_NAMES,
                          candidate_kind=KIND, input_contract=INPUT_CONTRACT,
                          spec=spec, coefficient_bounds=(0., 10.), time_knots=[0., 1.])


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
