"""Exploratory player rankings from Oracle's Elixir CSVs.

The CLI defaults to competition-adjusted logistic win RAPM, displayed as wins
added per 100 otherwise evenly matched games. A legacy gold-margin model remains
available for reproducibility. Historical fits are past-only; fixed parameters
and chronological validation are reported in the artifact. Nothing here writes
to the ticker database or production betting models.
"""
from __future__ import annotations

import argparse
import calendar
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import logging
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import expit
from scipy import sparse
from scipy.sparse.linalg import splu

ROLES = ("top", "jng", "mid", "bot", "sup")
DEFAULT_SOURCES = tuple(Path("data/oe") / f"oe_{year}.csv" for year in range(2014, 2027))
INTERNATIONAL_LEAGUES = frozenset(("WLDs", "MSI", "EWC", "FST", "IEM", "IWCI", "Rift Rivals", "RR", "ASCI", "EUM", "EM"))
CONTEXT_EVENTS = INTERNATIONAL_LEAGUES | {"KeSPA", "KeSPA Cup", "DCup"}
COMPETITION_ALIASES = {"OGN": "LCK", "NA LCS": "LCS", "EU LCS": "LEC"}
VERSION = 1
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Appearance:
    player_id: str
    name: str
    role: str
    side: str
    team: str
    team_id: str
    champion: str
    total_gold: float
    gold_at15: float | None
    provisional: bool = False


@dataclass(frozen=True)
class Game:
    game_id: str
    played_at: datetime
    league: str
    patch: str
    players: tuple[Appearance, ...]
    gold_margin: float
    blue_win: bool
    duration_seconds: float = 1800


@dataclass
class Fit:
    players: list[str]
    index: dict[str, int]
    coefficients: np.ndarray
    lane_prior: np.ndarray
    lane_games: Counter
    lane_effective_games: Counter
    controls: dict[str, int]
    precision: sparse.csc_matrix
    residual_variance: float
    weights: np.ndarray
    raw_coefficients: np.ndarray | None = None
    player_regions: dict[str, str] | None = None
    regional_strength: dict[str, float] | None = None
    win_log_odds: np.ndarray | None = None


def _datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def _cutoff(value: str | datetime | None, latest: datetime) -> datetime:
    if value is None:
        return latest
    if isinstance(value, str) and len(value.strip()) == 10:
        return _datetime(value) + timedelta(days=1) - timedelta(microseconds=1)
    return _datetime(value)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_mapping(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
        raise ValueError(f"Expected a JSON object mapping player IDs to strings: {path}")
    return payload


def read_games(source_paths: Iterable[str | Path], aliases: Mapping[str, str] | None = None, *, require_gold: bool = True) -> tuple[list[Game], dict]:
    """Read a minimal, validated view. IDs are never merged solely by IGN."""
    aliases = aliases or {}
    records: dict[str, dict] = {}
    rejected = Counter()
    source_rows = 0
    duplicate_rows = 0
    for source_path in source_paths:
        with Path(source_path).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("position") not in ROLES:
                    continue
                source_rows += 1
                game_id = row.get("gameid", "").strip()
                try:
                    played_at = _datetime(row.get("date", ""))
                except (ValueError, TypeError):
                    rejected["invalid_date_rows"] += 1
                    continue
                if not game_id:
                    rejected["missing_game_id_rows"] += 1
                    continue
                length = _number(row.get("gamelength"))
                result = row.get("result", "")
                record = records.setdefault(game_id, {"date": played_at, "league": row.get("league", ""),
                    "patch": row.get("patch", ""), "length": _number(row.get("gamelength")),
                    "players": {}, "win": row.get("result") == "1", "invalid": False,
                    "results": {}, "teams": {}})
                if (record["date"] != played_at or record["league"] != row.get("league", "")
                        or record["patch"] != row.get("patch", "") or length != record["length"]
                        or length is None or length <= 0 or result not in ("0", "1")):
                    record["invalid"] = True
                side = row.get("side", "").lower()
                role = row["position"]
                if side not in ("blue", "red"):
                    record["invalid"] = True
                    continue
                team_id = row.get("teamid") or row.get("teamname", "")
                team_key = (team_id, row.get("teamname", ""))
                if (not team_id or side in record["teams"] and record["teams"][side] != team_key
                        or side in record["results"] and record["results"][side] != result):
                    record["invalid"] = True
                record["teams"][side] = team_key
                record["results"][side] = result
                name = row.get("playername", "").strip()
                player_id = row.get("playerid", "").strip()
                provisional = not player_id
                if not player_id:
                    scope = "|".join((row.get("league", ""), row.get("teamid") or row.get("teamname", ""), name.casefold()))
                    player_id = "provisional:" + hashlib.sha256(scope.encode()).hexdigest()[:20]
                # Explicit aliases are allowed, but cycles are rejected.
                seen = set()
                while player_id in aliases:
                    if player_id in seen:
                        raise ValueError("Player identity aliases contain a cycle")
                    seen.add(player_id)
                    player_id = aliases[player_id]
                gold = _number(row.get("totalgold"))
                if not name or (require_gold and (gold is None or gold < 0)):
                    record["invalid"] = True
                    continue
                gold_at15 = _number(row.get("goldat15"))
                if gold_at15 is not None and gold_at15 < 0:
                    rejected["negative_gold_at15_rows_skipped"] += 1
                    gold_at15 = None
                appearance = Appearance(player_id, name, role, side, row.get("teamname", ""),
                    team_id, row.get("champion", ""), gold if gold is not None and gold >= 0 else 0, gold_at15, provisional)
                key = (side, role)
                if key in record["players"]:
                    if record["players"][key] == appearance:
                        duplicate_rows += 1
                    else:
                        record["invalid"] = True
                else:
                    record["players"][key] = appearance
                if side == "blue":
                    record["win"] = row.get("result") == "1"
    games = []
    expected = {(side, role) for side in ("blue", "red") for role in ROLES}
    for game_id, record in records.items():
        if record["invalid"] or set(record["players"]) != expected:
            rejected["invalid_or_incomplete_games"] += 1
            continue
        if (record["results"].get("blue") == record["results"].get("red")
                or record["teams"]["blue"][0] == record["teams"]["red"][0]):
            rejected["invalid_team_or_result_games"] += 1
            continue
        seconds = record["length"]
        if seconds is None or seconds <= 600:
            rejected["invalid_duration_games"] += 1
            continue
        players = tuple(record["players"][(side, role)] for side in ("blue", "red") for role in ROLES)
        if len({p.player_id for p in players}) != 10:
            rejected["duplicate_player_games"] += 1
            continue
        gold_margin = sum((1 if p.side == "blue" else -1) * p.total_gold for p in players) / (seconds / 60)
        games.append(Game(game_id, record["date"], record["league"], record["patch"], players, gold_margin, record["win"], seconds))
    games.sort(key=lambda game: (game.played_at, game.game_id))
    return games, {"player_rows_read": source_rows, "duplicate_player_rows_ignored": duplicate_rows,
        "valid_games_read": len(games), "rejected": dict(rejected)}


def _weights(games: Sequence[Game], cutoff: datetime, half_life_days: float) -> np.ndarray:
    if half_life_days <= 0 or not math.isfinite(half_life_days):
        raise ValueError("half_life_days must be finite and positive")
    return np.array([2 ** (-(cutoff - game.played_at).total_seconds() / 86400 / half_life_days) for game in games])


def _patch(patch: str) -> str:
    return ".".join(patch.split(".")[:2]) or "unknown"


def _regional_context(games: Sequence[Game], initial: Mapping[str, str] | None = None):
    """Domestic affiliations known at game time; never read future appearances."""
    latest = dict(initial or {})
    team_home = {}
    contexts = {}
    for game in games:
        context = {}
        for side in ("blue", "red"):
            roster = [p for p in game.players if p.side == side]
            team = roster[0].team_id
            if game.league not in CONTEXT_EVENTS:
                home = COMPETITION_ALIASES.get(game.league, game.league)
                team_home[team] = home
            else:
                known = Counter(latest[p.player_id] for p in roster if p.player_id in latest)
                home = team_home.get(team, known.most_common(1)[0][0] if known else "unlinked")
            for player in roster:
                if game.league not in CONTEXT_EVENTS:
                    latest[player.player_id] = home
                context[player.player_id] = home
        contexts[game.game_id] = context
    return contexts, latest


def _design(games: Sequence[Game], player_index: dict[str, int], *, lane: bool = False,
            controls: dict[str, int] | None = None, regions: Mapping[str, Mapping[str, str]] | None = None) -> tuple[sparse.csr_matrix, np.ndarray, list[int], dict[str, int], Counter]:
    """Build signed lineup (or two-player lane) design; unseen test features are zero."""
    training = controls is None
    controls = {} if controls is None else dict(controls)
    width = len(player_index)
    rows, columns, values, targets, game_rows = [], [], [], [], []
    lane_counts = Counter()

    def add_control(row: int, key: str, value: float = 1):
        if training and key not in controls:
            controls[key] = width + len(controls)
        if key in controls:
            rows.append(row); columns.append(controls[key]); values.append(value)

    for game_number, game in enumerate(games):
        if lane and game.duration_seconds < 900:
            continue
        pairs = {role: [next(p for p in game.players if p.side == side and p.role == role)
                         for side in ("blue", "red")] for role in ROLES} if lane else {"all": game.players}
        for role, players in pairs.items():
            if lane and any(p.gold_at15 is None for p in players):
                continue
            row = len(targets)
            target = (players[0].gold_at15 - players[1].gold_at15) / 15 if lane else game.gold_margin
            targets.append(target); game_rows.append(game_number)
            for player in players:
                sign = 1 if player.side == "blue" else -1
                if player.player_id in player_index:
                    rows.append(row); columns.append(player_index[player.player_id]); values.append(sign)
                if regions is not None:
                    region = regions.get(game.game_id, {}).get(player.player_id, "unlinked")
                    if region != "unlinked":
                        add_control(row, f"region:{region}", sign)
                if player.champion:
                    add_control(row, f"champion:{player.role}:{player.champion}", sign)
                if lane:
                    lane_counts[player.player_id] += 1
            if lane:
                add_control(row, f"side:role:{role}")
                add_control(row, f"side:patch:{role}:{_patch(game.patch)}")
            else:
                add_control(row, "side:global")
                add_control(row, f"side:league:{game.league}")
                add_control(row, f"side:patch:{_patch(game.patch)}")
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=(len(targets), width + len(controls))).tocsr()
    matrix.eliminate_zeros()
    return matrix, np.array(targets), game_rows, controls, lane_counts


def _solve(matrix: sparse.csr_matrix, targets: np.ndarray, weights: np.ndarray,
           player_count: int, controls: Mapping[str, int], alpha: float,
           prior: np.ndarray | None = None) -> tuple[np.ndarray, sparse.csc_matrix, float]:
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("ridge alpha must be finite and positive")
    penalties = np.full(matrix.shape[1], alpha, dtype=float)
    # Side intercepts are nuisance adjustments, with mild rather than player-level shrinkage.
    for name, index in controls.items():
        if name.startswith("region:"):
            penalties[index] = 2.0
        if name == "side:global" or name.startswith("side:role:"):
            penalties[index] = 0.01
    weighted = matrix.multiply(np.sqrt(weights)[:, None]).tocsr()
    precision = (weighted.T @ weighted + sparse.diags(penalties)).tocsc()
    precision.eliminate_zeros()
    rhs = matrix.T @ (weights * targets)
    if prior is not None:
        rhs[:player_count] += penalties[:player_count] * prior
    coefficients = splu(precision).solve(np.asarray(rhs))
    residual = targets - matrix @ coefficients
    variance = max(float(np.dot(weights, residual ** 2) / max(weights.sum(), 1)), 1)
    return coefficients, precision, variance


def wins_added(coefficient):
    """Expected additional wins per 100 otherwise evenly matched games."""
    return 100 * (expit(coefficient) - .5)


def _solve_logistic(matrix, targets, weights, player_count, controls, alpha):
    """Penalized Bernoulli likelihood; Newton steps with backtracking."""
    penalties = np.full(matrix.shape[1], alpha, dtype=float)
    for name, index in controls.items():
        if name.startswith("region:"):
            penalties[index] = 2.0
        if name == "side:global":
            penalties[index] = .01
    coefficients = np.zeros(matrix.shape[1])
    def objective(value):
        eta = matrix @ value
        return float(np.dot(weights, np.logaddexp(0, eta) - targets * eta) + .5 * np.dot(penalties, value ** 2))
    for iteration in range(40):
        probability = expit(matrix @ coefficients)
        curvature = weights * np.maximum(probability * (1 - probability), 1e-8)
        precision = (matrix.T @ matrix.multiply(curvature[:, None]) + sparse.diags(penalties)).tocsc()
        gradient = np.asarray(matrix.T @ (weights * (probability - targets))) + penalties * coefficients
        step = splu(precision, permc_spec="MMD_AT_PLUS_A").solve(gradient)
        if np.max(np.abs(step)) < 1e-7:
            return coefficients, precision, 1.0
        scale = 1.0
        before = objective(coefficients)
        while objective(coefficients - scale * step) > before and scale > 2 ** -20:
            scale *= .5
        coefficients -= scale * step
    raise RuntimeError("Win model failed to converge after 40 Newton iterations")


def fit_win_ratings(games, cutoff, *, half_life_days=150, ridge_alpha=100, **unused):
    if not games or any(game.played_at > cutoff for game in games):
        raise ValueError("Win fits require nonempty past-only games")
    if ridge_alpha <= 0 or not math.isfinite(ridge_alpha):
        raise ValueError("ridge_alpha must be finite and positive")
    players = sorted({p.player_id for game in games for p in game.players})
    index = {player: i for i, player in enumerate(players)}
    weights = _weights(games, cutoff, half_life_days)
    included = np.flatnonzero(weights >= 2 ** -8)
    fitting = [games[i] for i in included]
    regions, latest = _regional_context(games)
    matrix, _, _, controls, _ = _design(fitting, index, regions=regions)
    coefficients, precision, variance = _solve_logistic(matrix, np.array([g.blue_win for g in fitting], dtype=float),
        weights[included], len(players), controls, ridge_alpha)
    raw = coefficients.copy()
    strengths = {key.removeprefix("region:"): float(coefficients[i]) for key, i in controls.items() if key.startswith("region:")}
    for player, i in index.items():
        coefficients[i] += strengths.get(latest.get(player), 0)
    # Display on a standard 100-game, neutral-matchup scale. Prediction retains log odds.
    fit = Fit(players, index, wins_added(coefficients), np.zeros(len(players)), Counter(), Counter(),
        controls, precision, variance, weights, raw, latest, strengths)
    fit.win_log_odds = coefficients[:len(players)].copy()
    return fit


def win_validation(games, cutoff, *, half_life_days=150, ridge_alpha=100, holdout_days=90, **unused):
    boundary = cutoff - timedelta(days=holdout_days)
    train = [g for g in games if g.played_at < boundary]
    test = [g for g in games if boundary <= g.played_at <= cutoff]
    if len(train) < 30 or len(test) < 10:
        return {"available": False, "reason": "Need at least 30 training and 10 holdout games"}
    training_cutoff = boundary - timedelta(microseconds=1)
    fit = fit_win_ratings(train, training_cutoff, half_life_days=half_life_days, ridge_alpha=ridge_alpha)
    teams = sorted({p.team_id for g in train for p in g.players})
    team_index = {team: i for i, team in enumerate(teams)}
    def team_matrix(games):
        rows, columns, values = [], [], []
        for row, game in enumerate(games):
            for side, sign in (("blue", 1), ("red", -1)):
                team = next(p.team_id for p in game.players if p.side == side)
                if team in team_index:
                    rows.append(row); columns.append(team_index[team]); values.append(sign)
            rows.append(row); columns.append(len(teams)); values.append(1)
        return sparse.coo_matrix((values, (rows, columns)), shape=(len(games), len(teams)+1)).tocsr()
    targets = np.array([g.blue_win for g in train], dtype=float)
    coefficients, _, _ = _solve_logistic(team_matrix(train), targets, fit.weights, len(teams), {"side:global": len(teams)}, ridge_alpha)
    predictions = {"win_rapm": expit(_predict(fit, test)), "team_logistic": expit(team_matrix(test) @ coefficients),
        "side_only": np.full(len(test), np.average(targets, weights=fit.weights)), "coin_flip": np.full(len(test), .5)}
    actual = np.array([g.blue_win for g in test], dtype=float)
    metrics = {}
    for name, predicted in predictions.items():
        clipped = np.clip(predicted, 1e-9, 1-1e-9)
        metrics[name] = {"brier_score": round(float(np.mean((predicted-actual)**2)), 6),
            "log_loss": round(float(-np.mean(actual*np.log(clipped)+(1-actual)*np.log1p(-clipped))), 6),
            "winner_accuracy": round(float(np.mean((predicted >= .5)==actual)), 4)}
    return {"available": True, "train_games": len(train), "test_games": len(test),
        "training_end": train[-1].played_at.isoformat(), "holdout_start": boundary.isoformat(), "holdout_end": cutoff.isoformat(),
        "metrics": metrics, "limitations": "Frozen 90-day holdout; hyperparameters fixed, not tuned. Predictive accuracy does not establish causal player value."}


def fit_ratings(games: Sequence[Game], cutoff: datetime, *, half_life_days: float = 150,
                ridge_alpha: float = 100, lane_alpha: float = 40,
                prior_weight: float = 1) -> Fit:
    """Fit only supplied games, all of which must be at or before cutoff."""
    if not games:
        raise ValueError("No valid games at or before the rating cutoff")
    if any(game.played_at > cutoff for game in games):
        raise ValueError("Future games cannot enter a historical rating fit")
    if prior_weight < 0 or not math.isfinite(prior_weight):
        raise ValueError("prior_weight must be finite and nonnegative")
    if ridge_alpha <= 0 or lane_alpha <= 0 or not math.isfinite(ridge_alpha) or not math.isfinite(lane_alpha):
        raise ValueError("ridge_alpha and lane_alpha must be finite and positive")
    players = sorted({p.player_id for game in games for p in game.players})
    player_index = {player: index for index, player in enumerate(players)}
    game_weights = _weights(games, cutoff, half_life_days)
    # Below 1/256 weight, history remains in coverage and earlier snapshots but
    # is dropped from this solve to keep daily full-archive rebuilds practical.
    included = np.flatnonzero(game_weights >= 2 ** -8)
    fitting_games = [games[index] for index in included]
    fitting_weights = game_weights[included]
    lane_matrix, lane_targets, lane_game_rows, lane_controls, lane_counts = _design(fitting_games, player_index, lane=True)
    lane_prior = np.zeros(len(players))
    lane_effective = Counter()
    if len(lane_targets):
        lane_weights = fitting_weights[np.asarray(lane_game_rows)]
        lane_coefficients, _, _ = _solve(lane_matrix, lane_targets, lane_weights,
            len(players), lane_controls, lane_alpha)
        lane_prior = lane_coefficients[:len(players)]
        # Coverage uses the same paired observations and cutoff as the prior fit.
        for game, weight in zip(games, game_weights):
            if game.duration_seconds < 900:
                continue
            for role in ROLES:
                pair = [p for p in game.players if p.role == role]
                if all(p.gold_at15 is not None for p in pair):
                    for player in pair:
                        lane_effective[player.player_id] += float(weight)
    regions, latest_regions = _regional_context(games)
    matrix, targets, _, controls, _ = _design(fitting_games, player_index, regions=regions)
    coefficients, precision, variance = _solve(matrix, targets, fitting_weights,
        len(players), controls, ridge_alpha, prior=prior_weight * lane_prior)
    raw_coefficients = coefficients.copy()
    regional_strength = {name.removeprefix("region:"): float(coefficients[index])
                         for name, index in controls.items() if name.startswith("region:")}
    for player, index in player_index.items():
        coefficients[index] += regional_strength.get(latest_regions.get(player), 0)
    return Fit(players, player_index, coefficients, lane_prior, lane_counts,
        lane_effective, controls, precision, variance, game_weights,
        raw_coefficients, latest_regions, regional_strength)


def _posterior_standard_errors(fit: Fit, *, probes: int = 192) -> tuple[np.ndarray, str]:
    """Inverse precision diagonal, exact on small fits; deterministic probing otherwise.

    Probing preserves teammate collinearity (unlike reciprocating the diagonal).
    This is a conditional Gaussian posterior approximation, not a calibrated CI.
    """
    factor = splu(fit.precision)
    dimension, count = fit.precision.shape[0], len(fit.players)
    # Total impact = individual deviation + shared competition coefficient.
    rows, cols, values = list(range(count)), list(range(count)), [1.] * count
    for player, index in fit.index.items():
        region_index = fit.controls.get("region:" + (fit.player_regions or {}).get(player, "unlinked"))
        if region_index is not None:
            rows.append(index); cols.append(region_index); values.append(1.)
    projection = sparse.csr_matrix((values, (rows, cols)), shape=(count, dimension))
    if dimension <= 700:
        inverse = factor.solve(projection.T.toarray())
        diagonal = np.asarray(projection.multiply(inverse.T).sum(axis=1)).ravel()
        method = "exact inverse precision diagonal with regional covariance"
    else:
        # Probe only the individual diagonal, then add shared-region variance and
        # individual/region covariance exactly. Probing the combined projection
        # directly has excessive variance because thousands share one region.
        rng = np.random.default_rng(20261004)
        accumulator = np.zeros(count)
        for _ in range(probes):
            z = rng.choice([-1., 1.], size=dimension)
            accumulator += z[:count] * factor.solve(z)[:count]
        diagonal = np.maximum(accumulator / probes, 1 / fit.precision.diagonal()[:count])
        for name, region_index in fit.controls.items():
            if not name.startswith("region:"):
                continue
            unit = np.zeros(dimension); unit[region_index] = 1
            column = factor.solve(unit)
            for player, index in fit.index.items():
                if (fit.player_regions or {}).get(player) == name.removeprefix("region:"):
                    diagonal[index] += column[region_index] + 2 * column[index]
        method = f"{probes} individual diagonal probes plus exact regional variance/covariance"
    return np.sqrt(fit.residual_variance * np.maximum(diagonal, 0)), method


def _predict(fit: Fit, games: Sequence[Game]) -> np.ndarray:
    regions, _ = _regional_context(games, fit.player_regions)
    matrix, _, _, _, _ = _design(games, fit.index, controls=fit.controls, regions=regions)
    return np.asarray(matrix @ (fit.raw_coefficients if fit.raw_coefficients is not None else fit.coefficients))


def _team_baseline(train: Sequence[Game], test: Sequence[Game], cutoff: datetime,
                   half_life_days: float, alpha: float) -> np.ndarray:
    team_ids = sorted({p.team_id for game in train for p in game.players})
    index = {team: position for position, team in enumerate(team_ids)}

    def matrix(games):
        rows, columns, values = [], [], []
        for row, game in enumerate(games):
            for side, sign in (("blue", 1), ("red", -1)):
                team_id = next(p.team_id for p in game.players if p.side == side)
                if team_id in index:
                    rows.append(row); columns.append(index[team_id]); values.append(sign)
            rows.append(row); columns.append(len(index)); values.append(1)
        return sparse.coo_matrix((values, (rows, columns)), shape=(len(games), len(index) + 1)).tocsr()
    coefficients, _, _ = _solve(matrix(train), np.array([g.gold_margin for g in train]),
        _weights(train, cutoff, half_life_days), len(index), {"side:global": len(index)}, alpha)
    return np.asarray(matrix(test) @ coefficients)


def chronological_validation(games: Sequence[Game], cutoff: datetime, *, half_life_days: float = 150,
                             ridge_alpha: float = 100, lane_alpha: float = 40,
                             prior_weight: float = 1, holdout_days: int = 90) -> dict:
    """Frozen pre-holdout fits. Neither holdout outcomes nor lane stats enter fitting."""
    boundary = cutoff - timedelta(days=holdout_days)
    train = [game for game in games if game.played_at < boundary]
    test = [game for game in games if boundary <= game.played_at <= cutoff]
    if len(train) < 30 or len(test) < 10:
        return {"available": False, "reason": "Need at least 30 training and 10 holdout games",
            "train_games": len(train), "test_games": len(test)}
    training_cutoff = boundary - timedelta(microseconds=1)
    informed = fit_ratings(train, training_cutoff, half_life_days=half_life_days,
        ridge_alpha=ridge_alpha, lane_alpha=lane_alpha, prior_weight=prior_weight)
    pure = fit_ratings(train, training_cutoff, half_life_days=half_life_days,
        ridge_alpha=ridge_alpha, lane_alpha=lane_alpha, prior_weight=0)
    targets = np.array([game.gold_margin for game in test])
    outcomes = np.array([game.blue_win for game in test])
    neutral = float(np.average([g.gold_margin for g in train], weights=informed.weights))
    predictions = {"lane_informed_rapm": _predict(informed, test), "pure_rapm": _predict(pure, test),
        "team_ridge": _team_baseline(train, test, training_cutoff, half_life_days, ridge_alpha),
        "side_only": np.full(len(test), neutral), "zero_margin": np.zeros(len(test))}
    metrics = {name: {"rmse_gold_per_minute": round(float(np.sqrt(np.mean((prediction - targets) ** 2))), 3),
        "mae_gold_per_minute": round(float(np.mean(np.abs(prediction - targets))), 3),
        "winner_accuracy": round(float(np.mean((prediction > 0) == outcomes)), 4)}
        for name, prediction in predictions.items()}
    unseen = sum(p.player_id not in informed.index for game in test for p in game.players)
    model_rmse = metrics["lane_informed_rapm"]["rmse_gold_per_minute"]
    baseline_rmse = metrics["team_ridge"]["rmse_gold_per_minute"]
    return {"available": True, "design": "Single frozen chronological holdout; fixed exploratory hyperparameters; no tuning",
        "train_games": len(train), "test_games": len(test), "train_end": train[-1].played_at.isoformat(),
        "holdout_start": boundary.isoformat(), "holdout_end": test[-1].played_at.isoformat(),
        "unseen_player_appearances": unseen, "unseen_player_fraction": round(unseen / (10 * len(test)), 5),
        "metrics": metrics, "beats_team_baseline": model_rmse < baseline_rmse,
        "limitations": "One holdout is a diagnostic, not evidence of calibrated player talent or future win probabilities."}


def _metadata(games: Sequence[Game], fit: Fit, cutoff: datetime, active_days: int) -> tuple[dict, dict]:
    stats: dict[str, dict] = {}
    parent = {player: player for player in fit.players}

    def find(player):
        while parent[player] != player:
            parent[player] = parent[parent[player]]
            player = parent[player]
        return player

    for game, weight in zip(games, fit.weights):
        first = find(game.players[0].player_id)
        for player in game.players:
            parent[find(player.player_id)] = first
            stat = stats.setdefault(player.player_id, {"games": 0, "effective_games": 0., "recent_games": 0,
                "roles": Counter(), "leagues": Counter(), "rosters": Counter(), "international_games": 0,
                "first_seen": game.played_at, "identity_status": "provisional" if player.provisional else "source_id"})
            stat.update(name=player.name, team=player.team, latest_league=game.league,
                last_seen=game.played_at, latest_role=player.role)
            stat["games"] += 1; stat["effective_games"] += float(weight)
            stat["recent_games"] += int((cutoff - game.played_at).total_seconds() <= active_days * 86400)
            stat["roles"][player.role] += float(weight)
            if game.league not in CONTEXT_EVENTS:
                stat["leagues"][game.league] += float(weight)
                stat["latest_home_league"] = game.league
            roster = tuple(sorted(p.player_id for p in game.players if p.side == player.side))
            stat["rosters"][roster] += float(weight)
            stat["international_games"] += int(game.league in INTERNATIONAL_LEAGUES)
    roots = sorted({find(player) for player in fit.players})
    components = {root: number + 1 for number, root in enumerate(roots)}
    component_players = Counter(find(player) for player in fit.players)
    component_games = Counter(find(game.players[0].player_id) for game in games)
    component_leagues = defaultdict(set)
    for game in games:
        component_leagues[find(game.players[0].player_id)].add(game.league)
    for player_id, stat in stats.items():
        root = find(player_id)
        stat["comparison_component"] = components[root]
        stat["component_players"] = component_players[root]
        stat["component_games"] = component_games[root]
        stat["role"] = stat["roles"].most_common(1)[0][0]
        # Recent home competition is more useful than filtering everyone under "Worlds".
        stat["league"] = stat.get("latest_home_league", stat["latest_league"])
    league_games = Counter(game.league for game in games)
    connectivity = {"component_count": len(roots), "components": [
        {"id": components[root], "players": component_players[root], "games": component_games[root],
         "leagues": sorted(component_leagues[root])} for root in roots],
        "international_event_games": sum(league_games[league] for league in INTERNATIONAL_LEAGUES),
        "warning": "Shared competition strength uses international and transfer bridges; a connected historical graph does not guarantee precise current global comparisons."}
    return stats, connectivity


def _role_scores(stats: dict, fit: Fit) -> tuple[dict[str, float], dict]:
    scores, normalization = {}, {}
    for role in ROLES:
        reference = [player for player, stat in stats.items() if stat["role"] == role
            and stat["recent_games"] > 0 and stat["effective_games"] >= 10]
        if len(reference) < 2:
            reference = [player for player, stat in stats.items() if stat["role"] == role]
        values = np.array([fit.coefficients[fit.index[player]] for player in reference])
        weights = np.array([min(stats[player]["effective_games"], 100) for player in reference])
        if len(values) and weights.sum() > 0:
            mean = float(np.average(values, weights=weights))
            std = max(float(np.sqrt(np.average((values - mean) ** 2, weights=weights))), 1e-6 if fit.win_log_odds is not None else 1)
        else:
            mean, std = 0, 1
        normalization[role] = {"mean_impact": round(mean, 5), "standard_deviation": round(std, 5), "reference_players": len(reference)}
        for player, stat in stats.items():
            if stat["role"] == role:
                scores[player] = 50 + 10 * (fit.coefficients[fit.index[player]] - mean) / std
    return scores, normalization


def _month_cutoffs(cutoff: datetime, months: int) -> list[datetime]:
    result = []
    year, month = cutoff.year, cutoff.month
    # Completed months only; current fit supplies the newest trajectory point.
    for _ in range(months):
        month -= 1
        if month == 0:
            year -= 1; month = 12
        last_day = calendar.monthrange(year, month)[1]
        result.append(datetime(year, month, last_day, 23, 59, 59, 999999, tzinfo=timezone.utc))
    return sorted(result)


def _age(birthday: str | None, cutoff: datetime) -> float | None:
    if birthday is None:
        return None
    try:
        born = date.fromisoformat(birthday)
    except ValueError:
        return None
    today = cutoff.date()
    years = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    # Feb 29 anniversaries use March 1 in non-leap years, consistently with
    # the month/day integer-age test and the browser's age-at-peak calculation.
    def anniversary(year):
        if born.month == 2 and born.day == 29 and not calendar.isleap(year):
            return date(year, 3, 1)
        return date(year, born.month, born.day)
    previous = anniversary(born.year + years)
    following = anniversary(born.year + years + 1)
    age = years + (today - previous).days / (following - previous).days
    return round(age, 4) if 0 <= age <= 100 else None


def _source_manifest(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def build_ratings(source_paths: Iterable[str | Path] = DEFAULT_SOURCES, *, as_of: str | datetime | None = None,
                  birthdays_path: str | Path | None = None, aliases_path: str | Path | None = None,
                  half_life_days: float = 150, ridge_alpha: float = 100, lane_alpha: float = 40,
                  prior_weight: float = 1, active_days: int = 180, history_months: int = 24,
                  validate: bool = True, uncertainty_probes: int = 192, outcome: str = "wins") -> dict:
    """Build a JSON-compatible artifact without modifying sources or production state."""
    if active_days < 1 or history_months < 0 or uncertainty_probes < 1:
        raise ValueError("active_days and uncertainty_probes must be positive; history_months must be nonnegative")
    if outcome not in ("gold", "wins"):
        raise ValueError("outcome must be gold or wins")
    fitter = fit_win_ratings if outcome == "wins" else fit_ratings
    sources = [Path(path) for path in source_paths]
    games, ingestion = read_games(sources, _json_mapping(aliases_path), require_gold=outcome == "gold")
    if not games:
        raise ValueError("No complete, valid ten-player games in the supplied CSVs")
    cutoff = _cutoff(as_of, games[-1].played_at)
    excluded_future = sum(game.played_at > cutoff for game in games)
    games = [game for game in games if game.played_at <= cutoff]
    LOGGER.info("Fitting current ratings: %s games", len(games))
    fit = fitter(games, cutoff, half_life_days=half_life_days, ridge_alpha=ridge_alpha,
        lane_alpha=lane_alpha, prior_weight=prior_weight)
    LOGGER.info("Estimating conditional uncertainty")
    standard_errors, uncertainty_method = _posterior_standard_errors(fit, probes=uncertainty_probes)
    stats, connectivity = _metadata(games, fit, cutoff, active_days)
    scores, normalization = _role_scores(stats, fit)
    histories = defaultdict(list)
    snapshots = _month_cutoffs(cutoff, history_months)
    # Annual archive snapshots extend the trajectory to the first fitted season.
    if history_months:
        snapshots += [datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
                      for year in range(games[0].played_at.year, cutoff.year)
                      if datetime(year, 12, 31, tzinfo=timezone.utc) < min(snapshots or [cutoff])]
    for snapshot in sorted(set(snapshots)):
        past = [game for game in games if game.played_at <= snapshot]
        if len(past) < 10:
            continue
        LOGGER.info("Historical snapshot %s: %s games", snapshot.date(), len(past))
        historical_fit = fitter(past, snapshot, half_life_days=half_life_days,
            ridge_alpha=ridge_alpha, lane_alpha=lane_alpha, prior_weight=prior_weight)
        historical_stats, _ = _metadata(past, historical_fit, snapshot, active_days)
        historical_scores, _ = _role_scores(historical_stats, historical_fit)
        for player in historical_fit.players:
            if historical_stats[player]["recent_games"]:
                histories[player].append({"date": snapshot.date().isoformat(),
                    "rating": round(historical_scores[player], 2),
                    "impact": round(float(historical_fit.coefficients[historical_fit.index[player]]), 3),
                    "team": historical_stats[player]["team"], "league": historical_stats[player]["league"], "games": historical_stats[player]["games"], "effective_games": round(historical_stats[player]["effective_games"], 2), "role": historical_stats[player]["role"]})
    birthdays = _json_mapping(birthdays_path)
    names = defaultdict(set)
    for player, stat in stats.items():
        names[stat["name"].casefold()].add(player)
    players = []
    for player_id in fit.players:
        stat = stats[player_id]; index = fit.index[player_id]
        impact, sd = float(fit.coefficients[index]), float(standard_errors[index])
        ci_low, ci_high = impact - 1.96 * sd, impact + 1.96 * sd
        if outcome == "wins":
            log_odds = fit.win_log_odds[index]
            ci_low, ci_high = wins_added(log_odds - 1.96 * sd), wins_added(log_odds + 1.96 * sd)
            sd *= 100 * expit(log_odds) * (1 - expit(log_odds))
        effective = stat["effective_games"]
        lane_effective = fit.lane_effective_games[player_id]
        current = {"date": cutoff.date().isoformat(), "rating": round(scores[player_id], 2),
            "impact": round(impact, 3), "team": stat["team"], "league": stat["league"], "games": stat["games"], "effective_games": round(effective, 2), "role": stat["role"]}
        trajectory = histories[player_id] + [current]
        peak_candidates = [point for point in trajectory if point.get("effective_games", 0) >= 20]
        peak = max(peak_candidates, key=lambda point: point["impact"], default=None)
        players.append({"player_id": player_id, "name": stat["name"], "team": stat["team"],
            "league": stat["league"], "latest_league": stat["latest_league"], "role": stat["role"],
            "rating": current["rating"], "impact": current["impact"], "standard_error": round(sd, 3),
            "ci_low": round(float(ci_low), 3), "ci_high": round(float(ci_high), 3),
            "games": stat["games"], "recent_games": stat["recent_games"], "effective_games": round(effective, 2),
            "lane_prior": round(float(fit.lane_prior[index]), 3), "lane_games": fit.lane_games[player_id],
            "lane_effective_games": round(lane_effective, 2), "lane_coverage": round(lane_effective / effective, 4) if effective else 0,
            "first_seen": stat["first_seen"].date().isoformat(), "last_seen": stat["last_seen"].date().isoformat(),
            "active": bool(stat["recent_games"]), "identity_status": stat["identity_status"],
            "duplicate_name": len(names[stat["name"].casefold()]) > 1,
            "roster_diversity": len(stat["rosters"]),
            "dominant_roster_share": round(max(stat["rosters"].values()) / effective, 4) if effective else 0,
            "international_games": stat["international_games"], "comparison_component": stat["comparison_component"],
            "competition_strength": round((fit.regional_strength or {}).get((fit.player_regions or {}).get(player_id), 0), 3),
            "individual_deviation": round(impact - (fit.regional_strength or {}).get((fit.player_regions or {}).get(player_id), 0), 3),
            "component_players": stat["component_players"], "birthday": birthdays.get(player_id), "age": _age(birthdays.get(player_id), cutoff),
            "history": trajectory, "career_peak": peak,
            "peak_impact": peak["impact"] if peak else None,
            "peak_rating": peak["rating"] if peak else None})
    players.sort(key=lambda player: (-player["impact"], player["player_id"]))
    role_rank = Counter()
    for rank, player in enumerate(players, 1):
        role_rank[player["role"]] += 1
        player["rank"] = rank; player["role_rank"] = role_rank[player["role"]]
    LOGGER.info("Chronological validation")
    validation = (win_validation if outcome == "wins" else chronological_validation)(games, cutoff, half_life_days=half_life_days,
        ridge_alpha=ridge_alpha, lane_alpha=lane_alpha, prior_weight=prior_weight) if validate else {"available": False, "reason": "Skipped by request"}
    early_path = Path("data/player_ratings/early_archive.json")
    early_archive = json.loads(early_path.read_text()).get("meta") if early_path.exists() else None
    payload = {"meta": {"version": VERSION, "model": "Hierarchical competition-adjusted, lane-informed margin RAPM (exploratory)",
        "created_at": datetime.now(timezone.utc).isoformat(), "as_of": cutoff.isoformat(),
        "early_archive": early_archive, "data_start": games[0].played_at.isoformat(), "data_end": games[-1].played_at.isoformat(),
        "games": len(games), "player_count": len(players), "half_life_days": half_life_days,
        "fit_window_days": half_life_days * 8, "minimum_fit_weight": 2 ** -8, "ridge_alpha": ridge_alpha, "lane_alpha": lane_alpha, "prior_weight": prior_weight,
        "active_days": active_days, "unit": "gold/min", "ranking_field": "impact",
        "score_formula": "50 + 10 * (impact - role mean) / role standard deviation; role-relative index",
        "regional_strength": fit.regional_strength, "regional_ridge_alpha": 2.0, "role_normalization": normalization, "uncertainty_method": uncertainty_method,
        "interval_definition": "Approximate 95% conditional Gaussian model interval; estimated lane prior and hyperparameters held fixed; not calibrated talent confidence",
        "source_files": [str(source.resolve()) for source in sources],
        "source_manifest": [_source_manifest(source) for source in sources],
        "birthdays_source": str(Path(birthdays_path).resolve()) if birthdays_path else None,
        "birthdays_manifest": _source_manifest(Path(birthdays_path)) if birthdays_path else None,
        "identity_aliases_source": str(Path(aliases_path).resolve()) if aliases_path else None,
        "identity_aliases_manifest": _source_manifest(Path(aliases_path)) if aliases_path else None,
        "limitations": [
            "Covers players observed in these CSVs, not every player worldwide.",
            "Each fit uses the most recent eight half-lives (1200 days by default); earlier games remain in archive coverage and historical snapshots, not the current solve.",
            "Terminal team gold margin divided by game minutes is the response; impact is an adjusted association, not causal individual value or predicted wins.",
            "Lane prior comes from same-role gold differences at 15 minutes, adjusted for opponent and champion; it favors early resource accumulation and may undervalue supports or low-resource play.",
            "Using lane impact as a terminal-margin shrinkage target is an exploratory modeling assumption; all hyperparameters are fixed rather than tuned.",
            "Players who usually play together cannot be cleanly separated by lineup results; lane priors supply differentiation but do not solve attribution.",
            "Shared competition coefficients are learned jointly from international and transfer bridges, with shrinkage toward zero. Sparse bridges and qualified-team selection still limit global comparisons.",
            "Uncertainty is conditional on the chosen model and estimated lane prior; it excludes identity errors, unmeasured context and prior/hyperparameter uncertainty.",
            "Source player IDs can change. Same-name identities remain distinct unless explicitly aliased; missing IDs use team/league-scoped provisional identities.",
            "Birthdays are published Leaguepedia dates matched to stable source IDs; unresolved identities and missing dates remain unknown. Annual archive and recent monthly history refits use only observations available by each snapshot. Career peak is the largest sampled impact with at least 20 decayed games; unequal career lengths and sampling affect comparisons.",
            "This model does not reproduce DARKO's predictive, multimetric daily projection system."]},
        "players": players, "diagnostics": {"ingestion": {**ingestion, "future_games_excluded": excluded_future},
            "validation": validation, "connectivity": connectivity,
            "roster_locked_players": sum(p["dominant_roster_share"] >= .9 for p in players),
            "provisional_players": sum(p["identity_status"] == "provisional" for p in players),
            "age_coverage": sum(p["age"] is not None for p in players)},
        "options": {"roles": list(ROLES), "leagues": sorted({player["league"] for player in players})}}
    if outcome == "wins":
        payload["meta"].update({"outcome": "wins", "model": "Competition-adjusted logistic win RAPM (exploratory)",
            "unit": "wins added / 100 games", "impact_formula": "100 * (sigmoid(player log odds + competition log odds) - 0.5)",
            "reference": "Zero-effect player in an otherwise evenly matched game; not replacement level",
            "lane_alpha": None, "prior_weight": 0,
            "regional_strength": {key: round(float(wins_added(value)), 5) for key, value in fit.regional_strength.items()},
            "interval_definition": "Approximate 95% conditional Laplace interval in log odds, transformed to wins added per 100 games",
            "limitations": [
                "Fits game wins directly; no gold outcomes or lane-gold priors enter the win fit.",
                "Wins added is a standardized rate: 100 * (sigmoid(total player log odds) - 0.5), against a zero-effect reference in an otherwise evenly matched game. It is not observed or cumulative career wins.",
                "Lineup effects add in log odds; displayed wins-added rates do not add together. No causal or replacement-level claim is made.",
                "Teammates who always play together cannot be separated from results alone; coefficients share credit and shrink toward zero.",
                "Competition strength depends on international and transfer bridges. Sparse or disconnected comparisons remain uncertain.",
                "Champion-role and side-by-league/patch controls, recency half-life and penalties are fixed exploratory choices; conditional intervals omit model misspecification and hyperparameter uncertainty.",
                "Each fit uses eight half-lives of match evidence. Earlier games remain in historical snapshots and archive coverage.",
                "Coverage is limited to complete ten-player source games. Early archive results without reliable player lineups cannot enter individual fits.",
                "Birthdays and reviewed identity aliases come from Leaguepedia. Missing ages stay unknown; historical fits use past data only.",
                "This is not a reproduction of DARKO."]})
        for player in players:
            i = fit.index[player["player_id"]]
            region = fit.regional_strength.get(fit.player_regions.get(player["player_id"]), 0)
            player["win_log_odds"] = round(float(fit.win_log_odds[i]), 6)
            player["competition_strength"] = round(float(wins_added(region)), 3)
            player["individual_deviation"] = round(float(wins_added(fit.win_log_odds[i] - region)), 3)
            for key in ("lane_prior", "lane_games", "lane_effective_games", "lane_coverage"):
                player.pop(key, None)
    return payload



def load_ratings(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("players"), list) or "meta" not in payload:
        raise ValueError("Not a player-ratings artifact")
    if payload["meta"].get("version") != VERSION:
        raise ValueError("Unsupported player-ratings artifact version")
    return payload


def filter_players(payload: dict, *, search: str = "", roles: Iterable[str] | None = None,
                   leagues: Iterable[str] | None = None, active_only: bool = True,
                   min_games: int = 0, min_effective_games: float = 0) -> list[dict]:
    role_set, league_set = set(roles or ()), set(leagues or ())
    query = search.casefold().strip()
    return [player for player in payload["players"]
        if (not active_only or player["active"])
        and (not role_set or player["role"] in role_set)
        and (not league_set or player["league"] in league_set)
        and player["games"] >= min_games and player["effective_games"] >= min_effective_games
        and (not query or query in " ".join(str(player.get(field, "")) for field in
            ("name", "team", "league", "player_id")).casefold())]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build a read-only ranking artifact from OE CSVs")
    build.add_argument("--outcome", choices=("wins", "gold"), default="wins", help="Fit game wins (default) or legacy gold margin")
    build.add_argument("--sources", nargs="+", type=Path, default=DEFAULT_SOURCES)
    build.add_argument("--output", type=Path, default=Path("data/player_ratings/ratings.json"))
    build.add_argument("--as-of", help="Inclusive UTC cutoff (YYYY-MM-DD or ISO timestamp); defaults to latest source game")
    build.add_argument("--birthdays", type=Path, default=Path("data/player_ratings/birthdays.json") if Path("data/player_ratings/birthdays.json").exists() else None, help="Optional JSON mapping source player ID to YYYY-MM-DD birthday")
    build.add_argument("--aliases", type=Path, default=Path("data/player_ratings/identity_aliases.json") if Path("data/player_ratings/identity_aliases.json").exists() else None, help="Optional reviewed JSON mapping previous source player IDs to canonical IDs")
    build.add_argument("--half-life-days", type=float, default=150)
    build.add_argument("--ridge-alpha", type=float, default=100)
    build.add_argument("--lane-alpha", type=float, default=40)
    build.add_argument("--prior-weight", type=float, default=1)
    build.add_argument("--active-days", type=int, default=180)
    build.add_argument("--history-months", type=int, default=24)
    build.add_argument("--skip-validation", action="store_true")
    build.add_argument("--uncertainty-probes", type=int, default=192)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    input_paths = list(args.sources) + [path for path in (args.birthdays, args.aliases) if path is not None]
    if args.output.resolve() in {source.resolve() for source in input_paths}:
        parser.error("Output must not overwrite an input file")
    try:
        payload = build_ratings(args.sources, as_of=args.as_of, birthdays_path=args.birthdays,
            aliases_path=args.aliases, half_life_days=args.half_life_days, ridge_alpha=args.ridge_alpha,
            lane_alpha=args.lane_alpha, prior_weight=args.prior_weight, active_days=args.active_days,
            history_months=args.history_months, validate=not args.skip_validation, outcome=args.outcome,
            uncertainty_probes=args.uncertainty_probes)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False, suffix=".json.tmp") as handle:
        json.dump(payload, handle, allow_nan=False, separators=(",", ":"))
        temporary = Path(handle.name)
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "games": payload["meta"]["games"],
        "players": payload["meta"]["player_count"], "as_of": payload["meta"]["as_of"],
        "validation": payload["diagnostics"]["validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
