import copy
import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from lol_ticker import wpgam, wppostdraft, wpx


def fixture():
    champions = ["champ%d" % i for i in range(10)]
    game = dict(game_id=10, match_id=8, game_num=2, trname="LPL 2026 Split 2",
                date=datetime.date(2026, 5, 1), patch="16.8", blue_team="Alpha", red_team="Beta",
                winner_side="blue", blue_picks=champions[:5], red_picks=champions[5:],
                oe_game_id="oe10", oe_blue_team="Alpha", oe_red_team="Beta", game_start=1777636800,
                oe_elo_b=1600., oe_elo_r=1400., oe_pelo_b=1700., oe_pelo_r=1500.,
                form_blue=.8, form_red=.3, elo_blue_pre=1800., elo_red_pre=1600.,
                elo_blue_pre_fast=1900., elo_red_pre_fast=1500.)
    players = [dict(game_id=10, slot=i, role=("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")[i % 5],
                    side="blue" if i < 5 else "red", team="Alpha" if i < 5 else "Beta", champion=c)
               for i, c in enumerate(champions)]
    prior = dict(game_id=8, match_id=8, game_num=1, date=game["date"],
                 blue_team="Beta", red_team="Alpha", winner_side="red", game_start=game["game_start"] - 3600)
    return game, players, prior


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0]

    def __iter__(self):
        return iter(self.rows)


class ReadOnlyFixture:
    def __init__(self, games, players, previous):
        self.games, self.players, self.previous = games, players, previous
        self.queries = []

    def execute(self, query, params=None):
        normalized = " ".join(query.lower().split())
        self.queries.append((normalized, params))
        assert normalized.startswith("select "), normalized
        for forbidden in ("golgg_timeline", "golgg_events", "golgg_builds", "golgg_items", "feed_", "duration_s", "blue_gold", "red_gold", "rapm_", "draft_outcome"):
            assert forbidden not in normalized, normalized
        if "current_setting" in normalized:
            return Rows([{"read_only": "on"}])
        if "from golgg_players" in normalized:
            return Rows(self.players)
        if "g.match_id=any" in normalized:
            return Rows(self.previous)
        return Rows(self.games)


class PostdraftTests(unittest.TestCase):
    def test_export_needs_no_timeline_duration_or_gameplay_and_keeps_frozen_vocabulary(self):
        game, players, prior = fixture()
        conn = ReadOnlyFixture([game], players, [prior])
        with tempfile.TemporaryDirectory() as folder:
            output = wppostdraft.build(conn, Path(folder) / "draft.npz", "2026-09-03",
                                      after="2026-05-01", league_prefix="LPL ", champion_names=["champ0"])
            with np.load(output) as data:
                self.assertEqual(data["C"].tolist(), [[0] + [-1] * 9])
                self.assertEqual(data["y"].tolist(), [1.])
                self.assertEqual(data["t"].tolist(), [0])
                self.assertEqual(data["seq"].tolist(), [-1])
                values = dict(zip(data["names"], data["X"][0]))
                allowed = set(wpgam.PREGAME_FEATURES) | {"bias", "t_since_kill"}
                self.assertTrue(all(v == 0 for k, v in values.items() if k not in allowed))
                self.assertEqual(values["series_diff"], 1.)
                self.assertEqual(values["t_since_kill"], 10.)
                self.assertTrue(np.isnan(data["pm"]).all() and np.isnan(data["ks"]).all())
            manifest = json.loads(Path(output + ".manifest.json").read_text())
            self.assertEqual(manifest["input_provenance"][0]["series_prior_gids"], [8])
            self.assertEqual(len(manifest["input_provenance"][0]["unknown_champions"]), 9)

    def test_oe_reversed_orientation_flips_only_oe_priors(self):
        game, _, prior = fixture()
        original, _ = wppostdraft._pregame_row(game, [prior])
        game.update(oe_blue_team="Beta", oe_red_team="Alpha")
        reversed_row, provenance = wppostdraft._pregame_row(game, [prior])
        for name in ("elo_oe", "pelo_oe", "form_diff"):
            index = wpx.FEATURE_NAMES.index(name)
            self.assertEqual(original[index], -reversed_row[index])
        for name in ("elo_gg", "elo_gg_fast", "series_diff"):
            index = wpx.FEATURE_NAMES.index(name)
            self.assertEqual(original[index], reversed_row[index])
        self.assertEqual(provenance["oe_orientation"], "reversed")
        game["oe_red_team"] = "Wrong opponent"
        with self.assertRaisesRegex(ValueError, "oe_team_mismatch"):
            wppostdraft._pregame_row(game, [prior])

    def test_current_result_and_future_maps_never_change_features(self):
        game, _, prior = fixture()
        baseline, provenance = wppostdraft._pregame_row(game, [prior])
        future_map = dict(prior, game_id=11, game_num=3, winner_side="blue")
        future_date = dict(prior, game_id=12, date=datetime.date(2026, 5, 2))
        future_start = dict(prior, game_id=13, game_start=game["game_start"] + 1)
        changed = dict(game, winner_side="red", blue_gold=999999, blue_kills=200, duration_s=9999)
        actual, actual_provenance = wppostdraft._pregame_row(changed, [prior, future_map, future_date, future_start, changed])
        np.testing.assert_array_equal(actual, baseline)
        self.assertEqual(actual_provenance, provenance)

    def test_incomplete_and_conflicting_drafts_fail_closed(self):
        game, players, _ = fixture()
        for broken in (players[:-1], [dict(p, slot=0) for p in players],
                       [dict(p, side="red") if p["slot"] == 0 else p for p in players],
                       [dict(p, team="Other") if p["slot"] == 0 else p for p in players],
                       [dict(p, role="ADC") if p["slot"] == 0 else p for p in players]):
            with self.assertRaises(ValueError):
                wppostdraft._complete_champions(game, broken)
        conflict = copy.deepcopy(game)
        conflict["blue_picks"][0] = "Wrong champion"
        with self.assertRaisesRegex(ValueError, "conflicting_draft"):
            wppostdraft._complete_champions(conflict, players)

    def test_missing_oe_priors_are_neutral_and_flagged(self):
        game, _, _ = fixture()
        game.update(oe_blue_team=None, oe_red_team=None, oe_game_id=None, game_start=None)
        row, provenance = wppostdraft._pregame_row(game, [])
        for name in ("elo_oe", "pelo_oe", "form_diff"):
            self.assertEqual(row[wpx.FEATURE_NAMES.index(name)], 0.)
            self.assertIn(name, provenance["neutral_fallbacks"])
        self.assertIn("series_history_incomplete", provenance["neutral_fallbacks"])

    def test_date_cutoffs_rechecked_before_export(self):
        game, players, _ = fixture()
        outside = dict(game, game_id=11, date=datetime.date(2026, 9, 3))
        before = dict(game, game_id=12, date=datetime.date(2026, 4, 30))
        conn = ReadOnlyFixture([game, outside, before], players, [])
        with tempfile.TemporaryDirectory() as folder:
            output = wppostdraft.build(conn, Path(folder) / "draft.npz", "2026-09-03", after="2026-05-01")
            result = json.loads(Path(output + ".manifest.json").read_text())
            self.assertEqual(result["games"], 1)
            self.assertEqual([r["gid"] for r in result["rejected"]], [11, 12])

    def test_existing_legacy_and_writable_connection_rejected(self):
        conn = mock.Mock()
        with self.assertRaises(ValueError):
            wppostdraft.build(conn, Path(wpx.OUT_DIR) / "states.npz", "2026-09-03")
        with tempfile.NamedTemporaryFile() as existing, self.assertRaises(FileExistsError):
            wppostdraft.build(conn, existing.name, "2026-09-03")
        conn.execute.assert_not_called()
        conn.execute.return_value.fetchone.return_value = {"read_only": "off"}
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "read-only"):
            wppostdraft.build(conn, Path(folder) / "draft.npz", "2026-09-03")


if __name__ == "__main__":
    unittest.main()
