"""Role-aware research capture remains independent of the deployed forecast."""
import copy
import json
import unittest
from unittest import mock

import numpy as np

from lol_ticker import live, wpgam, wpcombat


class LiveCombatTests(unittest.TestCase):
    def setUp(self):
        self.t0 = live._ts("2026-08-29T00:00:00Z")
        roles = ["top", "jungle", "mid", "bottom", "support"]
        gold = [9000, 9000, 9000, 14000, 4000]
        levels = [14, 14, 14, 16, 11]
        self.metadata = {"patchVersion": "16.16.1"}
        self.frame = {"rfc460Timestamp": "2026-08-29T00:20:00Z",
                      "gameState": "in_game"}
        for side, offset in (("blue", 0), ("red", 5)):
            self.metadata[side + "TeamMetadata"] = {
                "esportsTeamId": side,
                "participantMetadata": [
                    {"participantId": offset + i + 1, "role": role,
                     "championId": "Ahri", "summonerName": side + role}
                    for i, role in enumerate(roles)]}
            self.frame[side + "Team"] = {
                "totalGold": sum(gold), "totalKills": 10, "towers": 5,
                "inhibitors": 0, "barons": 0, "dragons": [],
                "participants": [
                    {"participantId": offset + i + 1, "totalGold": gold[i],
                     "currentHealth": 1000, "maxHealth": 1000,
                     "level": levels[i], "creepScore": 200}
                    for i in range(5)]}

    def state(self, frame=None, metadata=None, previous=None, details=None):
        return live._frame_state(
            self.metadata if metadata is None else metadata,
            self.frame if frame is None else frame,
            previous, details, {}, self.t0)

    @staticmethod
    def features(block):
        return dict(zip(block["names"], block["features"]))

    def test_adc_and_support_deaths_differ_despite_identical_aggregate_state(self):
        adc_dead = copy.deepcopy(self.frame)
        support_dead = copy.deepcopy(self.frame)
        adc_dead["blueTeam"]["participants"][3]["currentHealth"] = 0
        support_dead["blueTeam"]["participants"][4]["currentHealth"] = 0
        adc_state, support_state = self.state(adc_dead), self.state(support_dead)
        for key in ("dead_blue", "dead_red", "hp_pool", "lvl_k", "gold_diff_k"):
            self.assertEqual(adc_state[key], support_state[key])
        adc, support = self.features(adc_state["combat"]), self.features(support_state["combat"])
        self.assertEqual(adc["alive_bot"], -1)
        self.assertEqual(adc["alive_sup"], 0)
        self.assertEqual(support["alive_bot"], 0)
        self.assertEqual(support["alive_sup"], -1)
        self.assertEqual(adc["living_gold_bot_k"], -14)
        self.assertEqual(support["living_gold_sup_k"], -4)

    def test_metadata_and_window_permutations_preserve_role_snapshot(self):
        metadata, frame = copy.deepcopy(self.metadata), copy.deepcopy(self.frame)
        for side in ("blue", "red"):
            entries = metadata[side + "TeamMetadata"]["participantMetadata"]
            metadata[side + "TeamMetadata"]["participantMetadata"] = [entries[i] for i in (3, 0, 4, 1, 2)]
            players = frame[side + "Team"]["participants"]
            frame[side + "Team"]["participants"] = [players[i] for i in (4, 2, 0, 3, 1)]
        self.assertEqual(self.state()["combat"], self.state(frame, metadata)["combat"])
        self.assertEqual(self.state()["combat"]["observation"]["pidb"], [1, 2, 3, 4, 5])

    def test_missing_roles_or_metadata_are_unavailable_without_breaking_state(self):
        metadata = copy.deepcopy(self.metadata)
        del metadata["blueTeamMetadata"]["participantMetadata"][3]["role"]
        for missing in ({}, metadata):
            with self.subTest(metadata=missing):
                state = self.state(metadata=missing)
                block = state["combat"]
                self.assertFalse(block["available"])
                self.assertFalse(any(block["known"]))
                self.assertEqual(block["features"], [0.0] * len(wpcombat.FEATURE_NAMES))
                self.assertEqual(state["has_hp"], 1)
                self.assertEqual(state["gold_diff_k"], 0)

    def test_combat_health_uses_only_the_current_window_frame(self):
        previous = copy.deepcopy(self.frame)
        for side in ("blue", "red"):
            for player in previous[side + "Team"]["participants"]:
                player["currentHealth"] = 0
        details = {"participants": [
            {"participantId": pid, "currentHealth": 0, "items": []}
            for pid in range(1, 11)]}
        block = self.state(previous=previous, details=details)["combat"]
        self.assertEqual(block, self.state()["combat"])
        self.assertTrue(block["available"])
        self.assertEqual(block["raw"]["alive"], [1.0] * 10)
        self.assertEqual(block["source"], "official_window")
        self.assertEqual(block["observation_timestamp"], self.frame["rfc460Timestamp"])
        self.assertEqual(block["observation_ts"], self.t0 + 1200)
        self.assertEqual(block["age_upper_s"], 0)
        json.dumps(block, allow_nan=False)

    def test_direct_fetch_and_frame_series_capture_the_same_block(self):
        opening = copy.deepcopy(self.frame)
        opening["rfc460Timestamp"] = live._iso(self.t0)
        details = {"frames": [{"participants": [
            {"participantId": pid, "items": [], "currentHealth": 0}
            for pid in range(1, 11)]}]}
        responses = [
            {"gameMetadata": self.metadata, "frames": [opening]},
            {"frames": [self.frame]}, details,
            {"frames": [opening]}, details]
        with mock.patch.object(live, "_get", side_effect=responses):
            fetched = live._fetch_state("synthetic-game")
        self.assertEqual(fetched["combat"], self.state()["combat"])

    def test_capture_does_not_change_either_production_feature_contract(self):
        state = self.state()
        legacy = dict(state)
        del legacy["combat"]
        for names in (wpgam.LEGACY_STATE_FEATURES, wpgam.STATE_FEATURES):
            np.testing.assert_array_equal(
                wpgam.state_values_from_live(state, names),
                wpgam.state_values_from_live(legacy, names))

    def test_optional_combat_validation_failure_does_not_abort_live_state(self):
        with mock.patch.object(wpcombat, "live_observation", side_effect=ValueError("bad metadata")):
            state = self.state()
        self.assertFalse(state["combat"]["available"])
        self.assertEqual(state["combat"]["observation"]["reason"], "invalid_live_combat")
        self.assertEqual(state["dead_blue"], 0)


if __name__ == "__main__":
    unittest.main()
