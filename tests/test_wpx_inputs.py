import copy
import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from lol_ticker import wpx, wpx_inputs


def fixture():
    champs = ["champ%d" % i for i in range(10)]
    origin = {"origin_ts": 1000.25, "source": "archive", "first_frame": {
        "blueTeam": {"totalGold": 0}, "redTeam": {"totalGold": 0}}, "game_metadata": {}}
    for side, offset in (("blue", 0), ("red", 5)):
        origin["game_metadata"][side + "TeamMetadata"] = {"participantMetadata": [
            {"participantId": offset + i + 1, "championId": champs[offset + i], "role": role}
            for i, role in enumerate(("top", "jungle", "mid", "bottom", "support"))]}
    gold = {m: [[500 + 500 * m] * 5, [500 + 500 * m] * 5] for m in range(5)}
    row = {"minute": 1, "ts": 1070, "data": {"gdb": [1100] * 5, "gdr": [1100] * 5,
        "hpb": [0, .5, 1, 1, 1], "hpr": [1] * 5, "lvb": [3] * 5, "lvr": [3] * 5}}
    return champs, origin, gold, row


class ClockTests(unittest.TestCase):
    def test_gold_bracket_cannot_be_joined_before_upper_bound(self):
        champs, origin, gold, row = fixture()
        hp = wpx_inputs.bracket_archived_observations([row], origin, gold, champs)
        self.assertEqual(hp[1]["_clock_lo_s"], 60)
        self.assertEqual(hp[1]["_clock_s"], 120)
        self.assertEqual(wpx._hp_observation(hp, 119)["features"][-1], 0)
        got = wpx._hp_observation(hp, 120)
        self.assertEqual(got["features"][-1], 1)
        self.assertEqual(got["dead_blue"], 1)
        self.assertEqual(got["age_upper_s"], 60)
        self.assertEqual(wpx._hp_observation(hp, 151)["features"][-1], 0)

    def test_future_hp_mutation_and_insertion_do_not_change_earlier_features(self):
        champs, origin, gold, row = fixture()
        baseline = wpx._hp_observation(wpx_inputs.bracket_archived_observations([row], origin, gold, champs), 120)
        future = copy.deepcopy(row); future.update(minute=3, ts=1190)
        # Even a future reset that resembles early gold cannot become an early HP observation.
        future["data"].update(gdb=[700] * 5, gdr=[700] * 5, hpb=[0] * 5)
        hp = wpx_inputs.bracket_archived_observations([row, future], origin, gold, champs)
        self.assertEqual(wpx._hp_observation(hp, 120), baseline)
        self.assertFalse(hp[3]["_clock_trusted"])

    def test_unknown_pause_widens_interval_and_fails_closed(self):
        champs, origin, gold, row = fixture()
        row["ts"] += 300
        hp = wpx_inputs.bracket_archived_observations([row], origin, gold, champs)
        self.assertFalse(hp[1]["_clock_trusted"])
        self.assertEqual(hp[1]["_clock_source"], "clock_interval_too_wide")

    def test_missing_opening_champion_mismatch_and_counter_resets_fail_closed(self):
        champs, origin, gold, row = fixture()
        for altered_origin in (None, dict(origin, first_frame={}), dict(origin, game_metadata={})):
            hp = wpx_inputs.bracket_archived_observations([row], altered_origin, gold, champs)
            self.assertFalse(hp[1]["_clock_trusted"])
        hp = wpx_inputs.bracket_archived_observations([row], origin, gold, champs[::-1])
        self.assertFalse(hp[1]["_clock_trusted"])
        gold[1] = [[400] * 5, [400] * 5]
        hp = wpx_inputs.bracket_archived_observations([row], origin, gold, champs)
        self.assertFalse(hp[1]["_clock_trusted"])

    def test_plateau_is_not_an_exact_timestamp(self):
        champs, origin, gold, row = fixture()
        row["data"].update(gdb=[1000] * 5, gdr=[1000] * 5)
        hp = wpx_inputs.bracket_archived_observations([row], origin, gold, champs)
        self.assertFalse(hp[1]["_clock_trusted"])

    def test_known_riot_aliases_preserve_exact_identity(self):
        champs, origin, gold, row = fixture()
        champs[0] = "Wukong"; champs[1] = "Renata Glasc"
        players = origin["game_metadata"]["blueTeamMetadata"]["participantMetadata"]
        players[0]["championId"] = "MonkeyKing"; players[1]["championId"] = "Renata"
        hp = wpx_inputs.bracket_archived_observations([row], origin, gold, champs)
        self.assertTrue(hp[1]["_clock_trusted"])

    def test_new_observation_identity_mismatch_is_excluded(self):
        champs, origin, gold, row = fixture()
        row["data"]["pidb"] = [2, 1, 3, 4, 5]
        hp = wpx_inputs.bracket_archived_observations([row], origin, gold, champs)
        self.assertFalse(hp[1]["_clock_trusted"])
        row["data"].pop("pidb")
        row["data"]["_observation"] = {"champions": champs[::-1]}
        hp = wpx_inputs.bracket_archived_observations([row], origin, gold, champs)
        self.assertFalse(hp[1]["_clock_trusted"])

    def test_nominal_minute_is_never_a_clock_fallback(self):
        _, _, _, row = fixture()
        self.assertEqual(wpx._hp_observation({0: row["data"]}, 60)["features"][-1], 0)
        self.assertEqual(wpx._hp_observation({0: {"_clock_s": 120, "data": row["data"]}}, 0)["features"][-1], 0)

    def test_rebuild_rejects_legacy_and_existing_output_before_querying(self):
        conn = mock.Mock()
        with self.assertRaises(ValueError):
            wpx.build(conn, output_path=pathlib.Path(wpx.OUT_DIR) / "states.npz")
        with tempfile.NamedTemporaryFile() as handle, self.assertRaises(FileExistsError):
            wpx.build(conn, output_path=handle.name)
        conn.execute.assert_not_called()


class DeathTests(unittest.TestCase):
    def test_pre_event_prediction_does_not_include_other_events_in_same_second(self):
        game = {"oe_elo_b": None, "oe_elo_r": None, "oe_pelo_b": None, "oe_pelo_r": None}
        future = dict(seq=1, time_s=601, action="kill", side="blue", player="b", target="r", victim_side="red")
        baseline = wpx._state(game, {}, {}, [], [], 0, 600)
        # idx can contain an earlier sequence at the displayed event timestamp;
        # the pre-event state itself is evaluated one second earlier.
        observed = wpx._state(game, {}, {}, [future], [], 1, 600)
        np.testing.assert_array_equal(observed, baseline)

    def test_repeated_victim_is_one_death_and_future_events_do_not_count(self):
        events = [dict(seq=i, time_s=600+i, action="kill", side="blue", player="b", target="r", victim_side="red")
                  for i in range(7)]
        events.append(dict(seq=9, time_s=999, action="kill", side="red", player="r2", target="b2", victim_side="blue"))
        result = wpx_inputs.estimated_deaths(events, 610)
        self.assertEqual(result["counts"], [0, 1])
        self.assertTrue(result["uncertain"])
        self.assertEqual(wpx_inputs.estimated_deaths(events, 700)["counts"], [0, 0])

    def test_kill_by_victim_establishes_return_to_life(self):
        events = [dict(seq=1, time_s=600, action="kill", side="blue", player="b", target="r", victim_side="red"),
                  dict(seq=2, time_s=610, action="kill", side="red", player="r", target="b", victim_side="blue")]
        self.assertEqual(wpx_inputs.estimated_deaths(events, 610)["counts"], [1, 0])


class InventoryTests(unittest.TestCase):
    def run_inventory(self, events, t=100):
        return wpx_inputs.replay_inventory(events, t, {1: 1000, 2: 3000}, price_patch="15.16", game_patch="15.16")

    def event(self, seq, event, item, t=10):
        return dict(seq=seq, t=t, event=event, item_id=item, player_id=1, side="blue")

    def test_upgrade_destruction_and_sale_match_current_inventory(self):
        es = [self.event(1, "ITEM_PURCHASED", 1), self.event(2, "ITEM_DESTROYED", 1),
              self.event(3, "ITEM_PURCHASED", 2)]
        self.assertEqual(self.run_inventory(es)["item_gold"], 3000)
        es.append(self.event(4, "ITEM_SOLD", 2))
        got = self.run_inventory(es)
        self.assertTrue(got["available"])
        self.assertEqual((got["item_gold"], got["items_done"]), (0, 0))

    def test_ambiguous_undo_is_unknown_and_does_not_modify_earlier_inventory(self):
        es = [self.event(1, "ITEM_PURCHASED", 2), self.event(2, "ITEM_UNDO", -1, t=20)]
        self.assertEqual(self.run_inventory(es, t=19)["item_gold"], 3000)
        self.assertFalse(self.run_inventory(es, t=20)["available"])

    def test_patch_mismatch_and_unowned_destruction_fail_closed(self):
        self.assertFalse(self.run_inventory([self.event(1, "ITEM_DESTROYED", 1)])["available"])
        got = wpx_inputs.replay_inventory([], 100, {}, price_patch="16.16", game_patch="15.16")
        self.assertEqual(got["reason"], "patch_price_mismatch")

    def test_patch_cache_never_falls_back_to_global_table(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(FileNotFoundError):
            wpx_inputs.load_patch_item_prices("15.16", folder)


class ImporterTests(unittest.TestCase):
    def test_import_has_no_database_or_scrape_side_effects(self):
        path = pathlib.Path(__file__).resolve().parents[1] / "scripts/feed_backfill.py"
        spec = importlib.util.spec_from_file_location("feed_backfill", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch("lol_ticker.db.connect") as connect:
            spec.loader.exec_module(module)
        connect.assert_not_called()

    def test_origin_is_persisted_before_missing_minute_requests(self):
        path = pathlib.Path(__file__).resolve().parents[1] / "scripts/feed_backfill.py"
        spec = importlib.util.spec_from_file_location("feed_backfill_test", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        first = {"rfc460Timestamp": "2025-08-20T23:47:29.605Z"}
        sink = mock.Mock()
        opening = {"frames": [first], "gameMetadata": {}}
        def request(*args, **kwargs):
            if len(args) == 1:
                return opening
            sink.assert_called_once()
            return None
        with mock.patch.object(module, "_get", side_effect=request), mock.patch.object(module.time, "sleep"):
            status, rows, provenance = module.scrape_game("test", origin_sink=sink)
        self.assertEqual(status, "unavailable")
        self.assertEqual(rows, [])
        self.assertEqual(provenance["origin_timestamp"], first["rfc460Timestamp"])
        self.assertAlmostEqual(provenance["origin_ts"], 1755733649.605, places=3)
        self.assertEqual(provenance["clock_status"], "wall_origin_only_not_game_clock")


if __name__ == "__main__":
    unittest.main()
