import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lol_ticker import player_ratings as ratings


START = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)


def game(number, *, played_at=None, league="LCK", team_a="A", team_b="B", lane=True):
    blue_a = number % 2 == 0
    lane_differences = (1500, 600, 150, -450, -900)
    players = []
    for side in ("blue", "red"):
        team = team_a if (side == "blue") == blue_a else team_b
        is_a = team == team_a
        for role, difference in zip(ratings.ROLES, lane_differences):
            players.append(ratings.Appearance(
                player_id=f"{team}:{role}", name=f"{team} {role}", role=role, side=side,
                team=team, team_id=team, champion=f"{role} champion",
                total_gold=12000 + (500 if is_a else -500),
                gold_at15=5000 + (difference / 2 if is_a else -difference / 2) if lane else None))
    return ratings.Game(str(number), played_at or START + timedelta(days=number), league,
        "15.1", tuple(players), (5000 if blue_a else -5000) / 30, blue_a, 1800)


def rows(games):
    result = []
    for match in games:
        for player in match.players:
            result.append({"gameid": match.game_id, "date": match.played_at.isoformat(),
                "league": match.league, "patch": match.patch, "gamelength": str(match.duration_seconds),
                "position": player.role, "side": player.side.title(), "playerid": player.player_id,
                "playername": player.name, "teamid": player.team_id, "teamname": player.team,
                "champion": player.champion, "totalgold": str(player.total_gold),
                "goldat15": "" if player.gold_at15 is None else str(player.gold_at15),
                "result": str(int(match.blue_win if player.side == "blue" else not match.blue_win))})
    return result


def write_csv(path, values):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


class PlayerRatingsTests(unittest.TestCase):
    def test_signed_lineup_and_paired_lane_design(self):
        matches = [game(0)]
        index = {player.player_id: number for number, player in enumerate(matches[0].players)}
        matrix, targets, _, _, _ = ratings._design(matches, index)
        self.assertEqual(matrix.shape[0], 1)
        np.testing.assert_array_equal(matrix.toarray()[0, :10], [1] * 5 + [-1] * 5)
        self.assertAlmostEqual(targets[0], 5000 / 30)
        lane, targets, _, _, counts = ratings._design(matches, index, lane=True)
        self.assertEqual(lane.shape[0], 5)
        self.assertEqual((lane[:, :10] != 0).sum(axis=1).tolist(), [[2]] * 5)
        self.assertAlmostEqual(targets[0], 100)
        self.assertEqual(counts["A:top"], 1)

    def test_lane_prior_differentiates_locked_teammates_without_claiming_identification(self):
        matches = [game(number) for number in range(60)]
        cutoff = matches[-1].played_at
        pure = ratings.fit_ratings(matches, cutoff, prior_weight=0)
        informed = ratings.fit_ratings(matches, cutoff)
        a_values = [pure.coefficients[pure.index[f"A:{role}"]] for role in ratings.ROLES]
        self.assertLess(max(a_values) - min(a_values), 1e-9)
        self.assertGreater(informed.coefficients[informed.index["A:top"]],
                           informed.coefficients[informed.index["A:sup"]] + 10)
        stats, _ = ratings._metadata(matches, informed, cutoff, 180)
        self.assertEqual(len(stats["A:top"]["rosters"]), 1)
        self.assertEqual(informed.lane_games["A:top"], 60)

    def test_missing_lane_stats_shrink_to_neutral_and_short_games_skip_lane(self):
        matches = [game(number, lane=False) for number in range(20)]
        fit = ratings.fit_ratings(matches, matches[-1].played_at)
        np.testing.assert_array_equal(fit.lane_prior, np.zeros(10))
        self.assertEqual(sum(fit.lane_games.values()), 0)
        self.assertEqual(sum(fit.lane_effective_games.values()), 0)
        short = [replace(game(number), duration_seconds=800) for number in range(20)]
        fit = ratings.fit_ratings(short, short[-1].played_at)
        self.assertEqual(sum(fit.lane_games.values()), 0)

    def test_posterior_accounts_for_locked_roster_collinearity(self):
        matches = [game(number) for number in range(100)]
        fit = ratings.fit_ratings(matches, matches[-1].played_at)
        errors, method = ratings._posterior_standard_errors(fit)
        naive = np.sqrt(fit.residual_variance / fit.precision.diagonal()[:10])
        self.assertTrue(np.all(errors > naive * 1.05))
        self.assertIn("exact", method)
        self.assertTrue(np.all(np.isfinite(errors)))

    def test_recency_decay_and_future_fit_guard(self):
        matches = [game(0), game(1, played_at=START + timedelta(days=150))]
        weights = ratings._weights(matches, matches[-1].played_at, 150)
        np.testing.assert_allclose(weights, [.5, 1])
        with self.assertRaisesRegex(ValueError, "Future"):
            ratings.fit_ratings(matches, START)

    def test_nonfinite_hyperparameters_fail_before_building_a_broken_artifact(self):
        matches = [game(0), game(1)]
        for parameter in ("half_life_days", "ridge_alpha", "lane_alpha", "prior_weight"):
            for value in (float("nan"), float("inf")):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaises(ValueError):
                        ratings.fit_ratings(matches, matches[-1].played_at, **{parameter: value})

    def test_parser_rejects_conflicting_metadata_and_remakes(self):
        mutations = ((0, "gamelength", "1700"), (1, "patch", "15.2"),
                     (2, "result", "maybe"), (3, "result", "0"),
                     (4, "teamname", "Unexpected"), (5, "date", "2026-01-01"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.csv"
            for row_number, field, value in mutations:
                malformed = rows([game(0)])
                malformed[row_number][field] = value
                write_csv(path, malformed)
                accepted, _ = ratings.read_games([path])
                self.assertEqual(accepted, [], (field, value))
            for duration in (0, 599, 600):
                write_csv(path, rows([replace(game(0), duration_seconds=duration)]))
                self.assertEqual(ratings.read_games([path])[0], [])
            values = rows([game(0)])
            values[0]["goldat15"] = "-1"
            write_csv(path, values)
            accepted, diagnostics = ratings.read_games([path])
            self.assertEqual(len(accepted), 1)
            self.assertIsNone(accepted[0].players[0].gold_at15)
            self.assertEqual(diagnostics["rejected"]["negative_gold_at15_rows_skipped"], 1)

    def test_ids_never_merge_by_name_and_missing_ids_are_scoped(self):
        matches = [game(0), game(1, league="LEC", team_a="C", team_b="D")]
        values = rows(matches)
        for value in values:
            if value["position"] == "top":
                value["playername"] = "Same Name"
                value["playerid"] = ""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.csv"
            write_csv(path, values)
            accepted, _ = ratings.read_games([path])
            top_ids = {p.player_id for match in accepted for p in match.players if p.role == "top"}
            self.assertEqual(len(top_ids), 4)
            self.assertTrue(all(player.startswith("provisional:") for player in top_ids))
            values = rows(matches)
            for value in values:
                value["playername"] = "Same Name"
            write_csv(path, values)
            accepted, _ = ratings.read_games([path])
            self.assertEqual(len({p.player_id for match in accepted for p in match.players}), 20)

    def test_explicit_aliases_and_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.csv"
            write_csv(path, rows([game(0)]))
            accepted, _ = ratings.read_games([path], {"A:top": "canonical:top"})
            self.assertIn("canonical:top", {p.player_id for p in accepted[0].players})
            with self.assertRaisesRegex(ValueError, "cycle"):
                ratings.read_games([path], {"A:top": "B:top", "B:top": "A:top"})

    def test_cutoff_and_history_are_invariant_to_appended_future_outliers(self):
        past = [game(number) for number in range(70)]
        cutoff = past[-1].played_at
        future = game(999, played_at=cutoff + timedelta(days=1), league="FUTURE", team_a="Future", team_b="Unknown")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.csv"
            write_csv(path, rows(past))
            before = ratings.build_ratings([path], as_of=cutoff, history_months=2, validate=False)
            write_csv(path, rows(past + [future]))
            after = ratings.build_ratings([path], as_of=cutoff, history_months=2, validate=False)
            self.assertEqual(before["players"], after["players"])
            self.assertEqual(after["meta"]["games"], 70)
            self.assertEqual(after["diagnostics"]["ingestion"]["future_games_excluded"], 1)
            for player in after["players"]:
                self.assertTrue(all(point["date"] <= cutoff.date().isoformat() for point in player["history"]))
            january = ratings.build_ratings([path], as_of="2025-01-31", history_months=0, validate=False)
            by_id = {player["player_id"]: player for player in january["players"]}
            for player in after["players"]:
                point = next(point for point in player["history"] if point["date"] == "2025-01-31")
                self.assertEqual(point["impact"], by_id[player["player_id"]]["impact"])
                self.assertEqual(point["rating"], by_id[player["player_id"]]["rating"])
            json.dumps(after, allow_nan=False)

    def test_chronological_holdout_is_frozen_and_reports_baselines(self):
        matches = [game(number) for number in range(150)]
        cutoff = matches[-1].played_at
        result = ratings.chronological_validation(matches, cutoff, holdout_days=30)
        self.assertTrue(result["available"])
        self.assertLess(result["train_end"], result["holdout_start"])
        self.assertEqual(result["train_games"] + result["test_games"], 150)
        self.assertEqual(set(result["metrics"]), {"lane_informed_rapm", "pure_rapm", "team_ridge", "side_only", "zero_margin"})
        self.assertEqual(result["unseen_player_appearances"], 0)
        self.assertIn("no tuning", result["design"])
        json.dumps(result, allow_nan=False)

    def test_payload_filter_age_components_and_source_fingerprints(self):
        matches = [game(number) for number in range(20)]
        matches += [game(100, league="LEC", team_a="C", team_b="D")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.csv"
            birthdays = Path(directory) / "birthdays.json"
            birthdays.write_text(json.dumps({"A:top": "2000-01-01"}))
            write_csv(path, rows(matches))
            payload = ratings.build_ratings([path], history_months=0, validate=False, birthdays_path=birthdays)
            self.assertEqual(payload["diagnostics"]["connectivity"]["component_count"], 2)
            self.assertEqual(payload["diagnostics"]["age_coverage"], 1)
            self.assertEqual(len(payload["meta"]["source_manifest"][0]["sha256"]), 64)
            self.assertEqual(payload["meta"]["source_manifest"][0]["size_bytes"], path.stat().st_size)
            selected = ratings.filter_players(payload, roles=["top"], leagues=["LCK"], search="A", min_games=20)
            self.assertEqual([player["player_id"] for player in selected], ["A:top"])
            self.assertGreater(selected[0]["age"], 25)
            self.assertTrue(all(player["age"] is None for player in payload["players"] if player["player_id"] != "A:top"))
            output = Path(directory) / "ratings.json"
            output.write_text(json.dumps(payload, allow_nan=False))
            self.assertEqual(ratings.load_ratings(output), payload)


if __name__ == "__main__":
    unittest.main()
