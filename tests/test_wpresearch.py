import copy
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

from lol_ticker import live, wpgam, wpresearch, wpobjective, wpcomposition, wpfearless


class ResearchCaptureIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Research source fixtures must not inherit the machine's active
        # production table pin or automatic gameplay-patch catalog mapping.
        pointer = mock.patch("lol_ticker.wpcombined_prod.load_active", return_value=None)
        pointer.start()
        self.addCleanup(pointer.stop)
        self.t0 = live._ts("2026-08-29T00:00:00Z")
        self.metadata = {"patchVersion": "16.16.1"}
        champions = ["Aatrox", "Amumu", "Ahri", "Ashe", "Alistar",
                     "Camille", "Corki", "Diana", "Draven", "Elise"]
        for side, offset in (("blue", 0), ("red", 5)):
            self.metadata[side + "TeamMetadata"] = dict(esportsTeamId=side, participantMetadata=[
                dict(participantId=offset + i + 1, role=role, championId=champions[offset + i],
                     summonerName=side + str(i), esportsPlayerId=side + str(i))
                for i, role in enumerate(["top", "jungle", "mid", "bottom", "support"])])

    def frame(self, second):
        out = dict(rfc460Timestamp=live._iso(self.t0 + second), gameState="in_game")
        for side, offset in (("blue", 0), ("red", 5)):
            gold = [500 + second * (2 if side == "blue" else 1)] * 5
            out[side + "Team"] = dict(totalGold=sum(gold), totalKills=0, towers=0, inhibitors=0,
                barons=0, dragons=[], participants=[dict(participantId=offset + i + 1,
                    totalGold=gold[i], currentHealth=1000, maxHealth=1000, level=1, creepScore=10)
                    for i in range(5)])
        return out

    def state(self, second, trackers, *, clock=None, attempt="g"):
        frame = self.frame(second)
        state = live._frame_state(self.metadata, frame, None, None, {}, self.t0,
                                  game_clock_s=second if clock is None else clock)
        state.update(game_state=frame["gameState"], baron_timer_blue_s=0., baron_timer_red_s=0.,
                     elder_timer_blue_s=0., elder_timer_red_s=0.)
        return wpresearch.capture_dynamic(state, trackers, attempt_id=attempt,
                                          metadata=self.metadata, frame=frame)

    def test_missing_local_draft_sources_produce_complete_unavailable_blocks(self):
        with tempfile.TemporaryDirectory() as root:
            blocks = wpresearch.capture_draft(self.metadata, draft_ts=self.t0,
                context=dict(composition_root=root, fearless_root=root, sq_table_path=root + "/missing.npz"))
        self.assertEqual(set(blocks), {"composition", "fearless", "sq_early", "draft_snapshot"})
        self.assertTrue(all(blocks[k]["available"] is False for k in ("composition", "fearless", "sq_early")))
        self.assertEqual(blocks["draft_snapshot"]["teams"]["blue"]["participants"][0]["esports_player_id"], "blue0")
        json.dumps(blocks, allow_nan=False)

    def test_recorder_window_cadence_supports_bounded_causal_lookbacks(self):
        trackers = {}
        # 15s laps fetching one10s window alternate1s/11s gaps.
        for start in range(0, 321, 10):
            if start % 30 == 10:
                continue
            for second in range(start, start + 10):
                state = self.state(second, trackers)
        windows = state["trajectory"]["windows"]
        for window in (30, 120, 300):
            self.assertTrue(windows[str(window)]["available"])
            self.assertGreaterEqual(windows[str(window)]["actual_window_s"], window)
            self.assertLessEqual(windows[str(window)]["actual_window_s"], window + 15)
        self.assertEqual(state["trajectory"]["sampling_protocol"], "recorder_sparse_windows_v1")
        self.assertEqual(state["trajectory"]["max_gap_s"], 20.)
        json.dumps(state, allow_nan=False)

    def test_pause_wall_time_and_same_frame_retry_cannot_add_game_history(self):
        trackers = {}
        original = self.state(0, trackers)
        paused = self.state(120, trackers, clock=0)
        self.assertFalse(paused["trajectory"]["available"])
        self.assertEqual(paused["trajectory"]["retained_frames"], 1)
        tracker_snapshot = copy.deepcopy(trackers)
        repeated = self.state(120, trackers, clock=0)
        self.assertEqual(repeated["trajectory"], paused["trajectory"])
        self.assertEqual(trackers, tracker_snapshot)
        self.assertEqual(original["objective_opportunities"]["features"],
                         repeated["objective_opportunities"]["features"])

    def test_new_attempt_clears_both_tracker_histories(self):
        trackers = {}
        for second in range(31):
            state = self.state(second, trackers)
        self.assertTrue(state["trajectory"]["available"])
        reset = self.state(40, trackers, clock=0, attempt="remake")
        self.assertTrue(reset["trajectory"]["reset"])
        self.assertFalse(reset["trajectory"]["available"])
        self.assertTrue(reset["objective_opportunities"]["reset"])

    def test_capture_keeps_both_production_feature_vectors_unchanged(self):
        before = live._frame_state(self.metadata, self.frame(20), None, None, {}, self.t0)
        after = copy.deepcopy(before)
        wpresearch.capture_dynamic(after, {}, attempt_id="g", metadata=self.metadata, frame=self.frame(20))
        after["prior_confidence"] = wpresearch.capture_priors({}, as_of_ts=self.t0)
        after.update(wpresearch.capture_draft(self.metadata, draft_ts=self.t0))
        for names in (wpgam.LEGACY_STATE_FEATURES, wpgam.STATE_FEATURES):
            np.testing.assert_array_equal(wpgam.state_values_from_live(after, names),
                                          wpgam.state_values_from_live(before, names))

    def test_optional_capture_validation_failure_keeps_other_families(self):
        with mock.patch.object(wpobjective, "capture", side_effect=ValueError("bad profile")):
            state = self.state(30, {})
        self.assertEqual(state["objective_opportunities"]["reason"], "invalid_objective_opportunities_inputs")
        self.assertEqual(state["trajectory"]["retained_frames"], 1)

    def test_recent_sample_counts_bind_to_each_actual_team_rating_source(self):
        old_cache = getattr(live.team_priors, "_cache", None)
        if hasattr(live.team_priors, "_cache"):
            del live.team_priors._cache
        def restore():
            if old_cache is None and hasattr(live.team_priors, "_cache"):
                del live.team_priors._cache
            elif old_cache is not None:
                live.team_priors._cache = old_cache
        self.addCleanup(restore)
        row = dict(game_id="latest", blue_team="Alpha", red_team="Beta", winner="Alpha",
            date_utc=self.t0 - 86400, elo_blue=1500., elo_red=1500., pelo_blue=1500., pelo_red=1500.,
            players_blue=["a"] * 5, players_red=["b"] * 5)
        earlier = dict(row, game_id="earlier", red_team="Gamma", date_utc=self.t0 - 172800)
        conn = mock.MagicMock()
        conn.execute.return_value.fetchall.side_effect = [[row, earlier], []]
        with mock.patch.object(live.time, "time", return_value=self.t0):
            priors = live.team_priors(conn, ["Alpha", "Beta"])
        sources = priors["prior_provenance"]["oe"]
        self.assertEqual(sources["blue"]["sample_games"], 2)
        self.assertEqual(sources["red"]["sample_games"], 1)
        self.assertEqual(sources["blue"]["sample_count_as_of_ts"], row["date_utc"])
        self.assertEqual(sources["blue"]["sample_window_days"], 120)
        confidence = wpresearch.capture_priors(priors, as_of_ts=self.t0, teams=["Alpha", "Beta"])
        self.assertTrue(confidence["channels"]["elo_oe"]["applied"])
        self.assertAlmostEqual(confidence["channels"]["elo_oe"]["sides"]["blue"]["factors"]["support"], 2/22)
        self.assertAlmostEqual(confidence["channels"]["elo_oe"]["sides"]["red"]["factors"]["support"], 1/21)

    def test_malformed_optional_catalog_cannot_abort_research_capture(self):
        catalog = wpcomposition.catalog_from_payload(
            dict(version="16.16.1", data=dict(Ahri=dict(id="Ahri", name="Ahri", tags=["Mage"], stats=["bad"]))),
            source_url="https://ddragon.leagueoflegends.com/cdn/16.16.1/data/en_US/champion.json",
            available_from_ts=self.t0 - 100)
        blocks = wpresearch.capture_draft(self.metadata, draft_ts=self.t0,
                                           context={"composition_catalog": catalog})
        numerical_range = blocks["composition"]["names"].index("mean_attack_range")
        self.assertFalse(blocks["composition"]["known"][numerical_range])
        self.assertIn("draft_snapshot", blocks)
        json.dumps(blocks, allow_nan=False)

    def test_explicit_fearless_bundle_must_match_known_current_game(self):
        for outer in (dict(series_id="other", game_num=2), dict(series_id="series", game_num=4)):
            context = dict(outer, fearless={"context": {"series_id": "series", "game_num": 2}})
            with mock.patch.object(wpfearless, "capture", return_value={"available": True}) as capture:
                blocks = wpresearch.capture_draft(self.metadata, draft_ts=self.t0, context=context)
            self.assertFalse(blocks["fearless"]["available"])
            capture.assert_not_called()

    def test_full_server_build_requires_explicit_composition_patch_mapping(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["patchVersion"] = "16.16.707.2100"
        data = {p["championId"]: dict(id=p["championId"], name=p["championId"], tags=["Mage"],
                                      stats=dict(attackrange=550, hpperlevel=100, armorperlevel=4))
                for side in ("blue", "red") for p in metadata[side + "TeamMetadata"]["participantMetadata"]}
        catalog = wpcomposition.catalog_from_payload(
            dict(version="16.16.1", data=data),
            source_url="https://ddragon.leagueoflegends.com/cdn/16.16.1/data/en_US/champion.json",
            available_from_ts=self.t0 - 100)
        unknown = wpresearch.capture_draft(metadata, draft_ts=self.t0,
                                            context={"composition_catalog": catalog})
        self.assertFalse(unknown["composition"]["available"])
        mapped = wpresearch.capture_draft(metadata, draft_ts=self.t0,
            context={"composition_catalog": catalog, "composition_patch": "16.16"})
        self.assertTrue(mapped["composition"]["available"])
        self.assertEqual(mapped["composition"]["feed_patch"], metadata["patchVersion"])
        self.assertEqual(mapped["composition"]["patch_mapping"], "explicit_context")
        self.assertEqual(mapped["draft_snapshot"]["patch"], metadata["patchVersion"])

    def test_first_remake_frame_uses_new_lineup_for_priors_and_confidence(self):
        from lol_ticker import wpx
        old_cache = live._cache
        live._cache = {}
        self.addCleanup(setattr, live, "_cache", old_cache)
        conn = mock.MagicMock(); conn.execute.return_value = []
        live._cache["g"] = live._new_cache(conn, self.metadata, self.t0)
        remake_metadata = copy.deepcopy(self.metadata)
        remake_metadata["blueTeamMetadata"]["participantMetadata"][0].update(
            championId="Lux", summonerName="new-top", esportsPlayerId="new-top-id")
        window = dict(gameMetadata=remake_metadata, frames=[self.frame(1200)])
        forecast = dict(p_blue=.6, lo_prior=0., lo_state=0., lo_champ=0., lo_time=0.)
        def adjust(_conn, _teams, priors, blue, red):
            result = dict(priors)
            result["pelo_adjustment"] = dict(blue=dict(applied=True,
                resolved_players=[dict(matched_name=blue[0], games=20, last_ts=self.t0)]))
            return result
        with mock.patch.object(live, "_get", side_effect=[window, {"frames": []}, {"frames": []}]), \
                mock.patch.object(live, "_restart_t0", return_value=self.t0 + 1200), \
                mock.patch.object(live, "lineup_adjusted_priors", side_effect=adjust) as adjusted, \
                mock.patch.object(wpx, "predict_live", return_value=forecast):
            result = live.estimate_series(conn, "g", priors={"pelo_oe": .1}, teams=["Blue", "Red"])
        self.assertEqual(adjusted.call_args.args[3][0], "new-top")
        confidence = result["frames"][0]["prior_confidence"]
        self.assertEqual(confidence["adjusted_priors"]["pelo_adjustment"]["blue"]["resolved_players"][0]["matched_name"],
                         "new-top")
        self.assertEqual(result["frames"][0]["p_blue"], .6)


if __name__ == "__main__":
    unittest.main()
