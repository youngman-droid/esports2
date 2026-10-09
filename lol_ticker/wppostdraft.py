"""Build completed-draft inputs without requiring or reading live-game data.

This is an isolated historical input export, not a model fit. Outcomes are
labels only. Stored sequential ratings describe the state before each map;
downstream model fitting and calibration still need their own date cutoffs.
"""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import draft, wpgam, wpx


CONTRACT = {
    "name": "wpx_postdraft_v1",
    "state": "one completed-draft row at t=0 per map; all gameplay features neutral",
    "pregame_features": list(wpgam.PREGAME_FEATURES),
    "champions": "ten validated side/team/role slots from golgg_players",
    "ratings": "stored sequential pregame OE and gol.gg ratings, OE side orientation validated",
    "series": "only earlier-numbered maps in the same match, no later date or known later start",
    "labels": "current-map winner_side is used only for y",
    "missing_priors": "neutral fallback, individually recorded in input provenance",
    "timing_limitation": "historical completed picks do not certify when each draft lock was observed live",
}


def _champion_key(value):
    key = wpgam._norm_champ(value).replace("&", "")
    return {"monkeyking": "wukong", "renata": "renataglasc", "nunu": "nunuwillump"}.get(key, key)


def _date(value):
    return datetime.date.fromisoformat(str(value))


def _complete_champions(game, players):
    if len(players) != 10 or {p.get("slot") for p in players} != set(range(10)):
        raise ValueError("incomplete_role_slots")
    players = sorted(players, key=lambda p: p["slot"])
    roles = ({"top"}, {"jungle", "jng"}, {"mid", "middle"},
             {"adc", "bot", "bottom"}, {"support", "sup"})
    for p in players:
        side = "blue" if p["slot"] < 5 else "red"
        if (p.get("side") != side or
                draft.norm_team(p.get("team") or "") != draft.norm_team(game[side + "_team"]) or
                str(p.get("role") or "").lower() not in roles[p["slot"] % 5] or
                not str(p.get("champion") or "").strip()):
            raise ValueError("inconsistent_champion_identity")
    champions = [p["champion"] for p in players]
    keys = list(map(_champion_key, champions))
    if len(set(keys)) != 10:
        raise ValueError("duplicate_champions")
    for side, start in (("blue", 0), ("red", 5)):
        picks = game.get(side + "_picks")
        if (not isinstance(picks, list) or len(picks) != 5 or
                sorted(map(_champion_key, picks)) != sorted(keys[start:start + 5])):
            raise ValueError("incomplete_or_conflicting_draft")
    return champions


def _pregame_row(game, previous):
    """Construct features from a pregame allowlist; never inspect its label."""
    missing = []
    orientation = "missing"
    sign = 1.0
    if game.get("oe_blue_team") is not None or game.get("oe_red_team") is not None:
        gg = tuple(draft.norm_team(game[s + "_team"]) for s in ("blue", "red"))
        oe = tuple(draft.norm_team(game.get("oe_" + s + "_team") or "") for s in ("blue", "red"))
        if oe == gg:
            orientation = "same"
        elif oe == gg[::-1]:
            orientation, sign = "reversed", -1.0
        else:
            raise ValueError("oe_team_mismatch")
        if game.get("game_start") is not None:
            day = datetime.datetime.fromtimestamp(game["game_start"], datetime.timezone.utc).date()
            if day != _date(game["date"]):
                raise ValueError("oe_date_mismatch")

    def difference(blue, red, name, scale=1.0, oe=False):
        values = (game.get(blue), game.get(red))
        if (oe and orientation == "missing") or any(
                v is None or not math.isfinite(float(v)) for v in values):
            missing.append(name)
            return 0.0
        return (float(values[0]) - float(values[1])) / scale * (sign if oe else 1.0)

    features = {
        "elo_oe": difference("oe_elo_b", "oe_elo_r", "elo_oe", 400., oe=True),
        "pelo_oe": difference("oe_pelo_b", "oe_pelo_r", "pelo_oe", 400., oe=True),
        "form_diff": difference("form_blue", "form_red", "form_diff", oe=True),
        "elo_gg": difference("elo_blue_pre", "elo_red_pre", "elo_gg", 400.),
        "elo_gg_fast": difference("elo_blue_pre_fast", "elo_red_pre_fast", "elo_gg_fast", 400.),
        "series_diff": 0.0,
    }
    prior_ids = []
    teams = {draft.norm_team(game[s + "_team"]) for s in ("blue", "red")}
    for prior in previous:
        if (game.get("match_id") is None or prior.get("match_id") != game["match_id"] or
                not game.get("game_num") or not prior.get("game_num") or
                prior["game_num"] >= game["game_num"] or prior["game_id"] == game["game_id"] or
                _date(prior["date"]) > _date(game["date"]) or
                (game.get("game_start") is not None and prior.get("game_start") is not None and
                 prior["game_start"] >= game["game_start"]) or
                prior.get("winner_side") not in ("blue", "red") or
                {draft.norm_team(prior[s + "_team"]) for s in ("blue", "red")} != teams):
            continue
        winner = draft.norm_team(prior[prior["winner_side"] + "_team"])
        features["series_diff"] += 1. if winner == draft.norm_team(game["blue_team"]) else -1.
        prior_ids.append(prior["game_id"])
    if game.get("game_num", 1) and len(prior_ids) != game["game_num"] - 1:
        missing.append("series_history_incomplete")
    row = np.zeros(len(wpx.FEATURE_NAMES), dtype=np.float32)
    names = {name: i for i, name in enumerate(wpx.FEATURE_NAMES)}
    row[names["bias"]] = 1.
    row[names["t_since_kill"]] = 10.
    for name in wpgam.PREGAME_FEATURES:
        row[names[name]] = features[name]
    return row, {"oe_orientation": orientation, "neutral_fallbacks": missing,
                 "series_prior_gids": sorted(prior_ids)}


def build(conn, output_path, before, after=None, league_prefix=None, champion_names=None):
    """Export maps with ``after <= date < before`` using a read-only connection.

    A supplied champion vocabulary is retained exactly; unseen names get -1.
    Without one, the export creates a deterministic vocabulary of accepted rows.
    Existing artifacts and the legacy states.npz path cannot be overwritten.
    """
    path = Path(output_path).resolve()
    if path == (Path(wpx.OUT_DIR) / "states.npz").resolve():
        raise ValueError("postdraft export cannot overwrite legacy states.npz")
    manifest_path = Path(str(path) + ".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError(path)
    cutoff = _date(before)
    start = _date(after) if after is not None else None
    if start is not None and start >= cutoff:
        raise ValueError("after must precede exclusive before date")
    readonly = conn.execute("SELECT current_setting('transaction_read_only') AS read_only").fetchone()
    if readonly["read_only"] != "on":
        raise ValueError("postdraft export requires a read-only database transaction")
    prefix = None if league_prefix is None else str(league_prefix).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    games = conn.execute("""SELECT g.game_id,g.match_id,g.game_num,g.trname,g.date,g.patch,
            g.blue_team,g.red_team,g.blue_picks,g.red_picks,g.winner_side,g.oe_game_id,
            g.elo_blue_pre,g.elo_red_pre,g.elo_blue_pre_fast,g.elo_red_pre_fast,
            o.blue_team AS oe_blue_team,o.red_team AS oe_red_team,o.date_utc AS game_start,
            r.elo_blue AS oe_elo_b,r.elo_red AS oe_elo_r,
            r.pelo_blue AS oe_pelo_b,r.pelo_red AS oe_pelo_r,r.form_blue,r.form_red
        FROM golgg_games g LEFT JOIN oe_games o ON o.game_id=g.oe_game_id
        LEFT JOIN oe_ratings r ON r.game_id=g.oe_game_id
        WHERE g.date < %s::date AND (%s::date IS NULL OR g.date >= %s::date)
          AND (%s::text IS NULL OR g.trname LIKE %s)
        ORDER BY g.date,g.game_id""", (before, after, after, prefix, prefix)).fetchall()
    gids = [g["game_id"] for g in games]
    players = collections.defaultdict(list)
    if gids:
        for p in conn.execute("""SELECT game_id,slot,side,role,team,champion
                FROM golgg_players WHERE game_id=ANY(%s) ORDER BY game_id,slot""", (gids,)):
            players[p["game_id"]].append(p)
    matches = sorted({g["match_id"] for g in games if g.get("match_id") is not None})
    previous = collections.defaultdict(list)
    if matches:
        for p in conn.execute("""SELECT g.game_id,g.match_id,g.game_num,g.date,
                g.blue_team,g.red_team,g.winner_side,o.date_utc AS game_start
            FROM golgg_games g LEFT JOIN oe_games o ON o.game_id=g.oe_game_id
            WHERE g.match_id=ANY(%s) AND g.date < %s::date
            ORDER BY g.match_id,g.game_num,g.game_id""", (matches, before)):
            previous[p["match_id"]].append(p)
    accepted, rejected = [], []
    for game in games:
        try:
            day = _date(game["date"])
            if day >= cutoff or (start is not None and day < start):
                raise ValueError("outside_date_window")
            if (game.get("winner_side") not in ("blue", "red") or
                    not game.get("blue_team") or not game.get("red_team") or
                    draft.norm_team(game["blue_team"]) == draft.norm_team(game["red_team"])):
                raise ValueError("invalid_game_identity_or_label")
            champions = _complete_champions(game, players[game["game_id"]])
            row, provenance = _pregame_row(game, previous[game.get("match_id")])
        except (ValueError, TypeError) as exc:
            rejected.append({"gid": game["game_id"], "reason": str(exc)})
            continue
        accepted.append((game, champions, row, provenance))
    vocabulary = list(champion_names) if champion_names is not None else sorted(
        {champion for _, champions, _, _ in accepted for champion in champions})
    lookup = {_champion_key(name): i for i, name in enumerate(vocabulary)}
    if len(lookup) != len(vocabulary):
        raise ValueError("champion vocabulary has duplicate canonical names")
    encoded, provenance_rows = [], []
    for game, champions, _, provenance in accepted:
        encoded.append([lookup.get(_champion_key(c), -1) for c in champions])
        provenance_rows.append(dict(provenance, gid=game["game_id"], oe_game_id=game.get("oe_game_id"),
                                    unknown_champions=[c for c in champions if _champion_key(c) not in lookup]))
    n = len(accepted)
    field = lambda key, default="": np.asarray([g.get(key) if g.get(key) is not None else default for g, _, _, _ in accepted])
    payload = dict(X=np.asarray([x for _, _, x, _ in accepted], dtype=np.float32).reshape(n, len(wpx.FEATURE_NAMES)),
                   y=np.asarray([float(g["winner_side"] == "blue") for g, _, _, _ in accepted], dtype=np.float32),
                   gid=field("game_id").astype(np.int64), t=np.zeros(n, dtype=np.int32), seq=np.full(n, -1, dtype=np.int32),
                   C=np.asarray(encoded, dtype=np.int32).reshape(n, 10), date=field("date").astype(str),
                   league=field("trname").astype(str), patch=field("patch").astype(str),
                   oe_game_id=field("oe_game_id").astype(str), game_start=field("game_start", -1).astype(np.int64),
                   names=np.asarray(wpx.FEATURE_NAMES), champ_names=np.asarray(vocabulary, dtype=str),
                   pm=np.full(n, np.nan, dtype=np.float32), ks=np.full(n, np.nan, dtype=np.float32),
                   cutoff_exclusive=np.asarray(before), cutoff_inclusive=np.asarray(after or ""),
                   input_contract=np.asarray(json.dumps(CONTRACT, sort_keys=True)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    manifest = dict(dataset=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), input_contract=CONTRACT,
                    rows=n, games=n, candidate_games=len(games), cutoff_exclusive=before, cutoff_inclusive=after,
                    league_prefix=league_prefix, max_date=max(payload["date"], default=None),
                    rejected=rejected, input_provenance=provenance_rows,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return str(path)
