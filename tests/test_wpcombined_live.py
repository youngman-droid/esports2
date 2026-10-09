import copy
import unittest

import numpy as np

from lol_ticker import wpcombined, wpcombined_live, wpcomposition, wpobjective, wpsqearly, wptrend


class CombinedLiveInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = dict(objective=wpobjective.FEATURE_NAMES, trend=wptrend.FEATURE_NAMES,
                     composition=wpcomposition.FEATURE_NAMES, sq=["prior_patch_pair_score"])
        specs = {k: dict(feature_names=list(reversed(v)), bounds=(-10., 10.) if k == "composition" else (0., 10.))
                 for k, v in names.items()}
        specs["sq"].update(bounds=(0., 2.5), time_knots=[0., 5., 10., 15., 20.],
                           zero_after_min=20., monotone_decay=True, scale=.25)
        cls.model = wpcombined.fit({k: np.zeros((2, len(v))) for k, v in names.items()},
                                   [.4, .6], [0, 1], [1, 2], [0., 5.], specs=specs)
        # Positive coefficients even on unsupported columns prove that the
        # adapter's historical support policy is independent of zero weights.
        for family in cls.model["blocks"]:
            family["coefficients"][:] = .1
            if family["name"] == "sq":
                family["coefficients"][-1] = 0.
        wpcombined.validate(cls.model)

    def state(self, clock=480.):
        state = dict(game_id="game", attempt_id="game", clock_s=clock, ts=10000.+clock,
                     patch="16.16.1", baron_timer_blue_s=90., baron_timer_red_s=0.,
                     elder_timer_blue_s=0., elder_timer_red_s=75.)
        state["objective_opportunities"] = wpobjective.observation_features(state)
        history = []
        for c in np.arange(0., clock+1, 60.):
            record = dict(ts=10000.+c, clock_s=c, gold_totals=[2500.+20*c, 2500.+10*c],
                          gold_roles=[500.+4*c]*5+[500.+2*c]*5, hp=[1.]*10,
                          counts={k: [int(c//120), 0] for k in wptrend._COUNT_KEYS},
                          source="same_window_frame")
            history.append(record)
        state["trajectory_history"] = dict(source="same_window_frame", role_order=list(wptrend.ROLES),
                                            attempt_id="game", records=history)
        state["trajectory_exact"] = wpcombined_live.exact_trajectory(state, state["trajectory_history"])
        blue = dict(zip(wpcomposition.ROLES, ["A", "B", "C", "D", "E"]))
        red = dict(zip(wpcomposition.ROLES, ["F", "G", "H", "I", "J"]))
        payload = dict(version="16.16.1", data={c: dict(name=c, tags=["Tank"] if c < "F" else [],
                                  stats=dict(attackrange=200. if c < "F" else 100.,
                                             hpperlevel=100., armorperlevel=4.)) for c in "ABCDEFGHIJ"})
        catalog = wpcomposition.catalog_from_payload(payload,
            source_url="https://ddragon.leagueoflegends.com/cdn/16.16.1/data/en_US/champion.json",
            available_from_ts=9000.)
        state["composition"] = wpcomposition.features(blue, red, "16.16.1", catalog=catalog, as_of_ts=state["ts"])
        state["composition"]["feed_patch"] = state["patch"]
        state["sq_early"] = dict(kind=wpsqearly.KIND, available=True, pair_score=.15, coverage=.9,
                                  patch="16.16", source_patches=["16.15"], table_sha256="a"*64)
        return state

    def by_name(self, features, family, name):
        block = next(f for f in self.model["blocks"] if f["name"] == family)
        return features[family][0, block["feature_names"].index(name)]

    def test_artifact_ordering_and_only_historically_supported_channels(self):
        state = self.state()
        state["objective_opportunities"]["features"] = [.5, -.5]+[12.]*9
        state["objective_opportunities"]["known"] = [True]*11
        features, detail = wpcombined_live.blocks(state, self.model)
        self.assertEqual(set(detail["available_families"]), {"objective", "trend", "composition", "sq"})
        self.assertEqual(self.by_name(features, "objective", "baron_time_adv"), .5)
        self.assertEqual(self.by_name(features, "objective", "elder_time_adv"), -.5)
        self.assertEqual(self.by_name(features, "trend", "lead_change_120s_k"), 1.2)
        self.assertEqual(self.by_name(features, "trend", "gold_change_top_300s_k"), .6)
        self.assertEqual(self.by_name(features, "composition", "frontline_tank_tag_count"), 5.)
        self.assertEqual(self.by_name(features, "sq", "prior_patch_pair_score"), .15)
        for family in self.model["blocks"]:
            for name in family["feature_names"]:
                if name not in wpcombined_live.SUPPORTED[family["name"]]:
                    self.assertEqual(self.by_name(features, family["name"], name), 0.)

    def test_sparse_approximate_anchor_is_withheld_per_window(self):
        state = self.state()
        state.pop("trajectory_history")
        state["trajectory_exact"]["windows"]["120"].update(actual_window_s=135., anchor_clock_s=345.)
        features, detail = wpcombined_live.blocks(state, self.model)
        self.assertEqual(self.by_name(features, "trend", "lead_change_120s_k"), 0.)
        self.assertEqual(self.by_name(features, "trend", "lead_change_300s_k"), 3.)
        self.assertIn("nonexact_or_incomplete_trajectory_window", detail["families"]["trend"]["reasons"])

    def test_current_frame_identity_and_bounded_complete_history_are_required(self):
        state = self.state()
        state.pop("trajectory_history")
        state["objective_opportunities"]["observation_ts"] += 1
        state["trajectory_exact"]["max_gap_s"] = 91.
        features, _ = wpcombined_live.blocks(state, self.model)
        np.testing.assert_array_equal(features["objective"], 0.)
        np.testing.assert_array_equal(features["trend"], 0.)
        self.assertTrue(np.any(features["composition"]))

    def test_trusted_history_rebuild_requires_exact_current_record_and_attempt(self):
        state = self.state()
        history = state["trajectory_history"]
        block = wpcombined_live.exact_trajectory(state, history)
        self.assertEqual(block["windows"]["120"]["actual_window_s"], 120.)
        for mutation in (lambda h: h.update(attempt_id="other"),
                         lambda h: h["records"][-1].update(clock_s=state["clock_s"]+1),
                         lambda h: h["records"][-1].update(ts=state["ts"]+1),
                         lambda h: h["records"].pop(),
                         lambda h: h.update(role_order=list(reversed(wptrend.ROLES)))):
            changed = copy.deepcopy(history)
            mutation(changed)
            self.assertIsNone(wpcombined_live.exact_trajectory(state, changed))

    def test_malformed_vector_and_invalid_individual_fields_fail_closed(self):
        state = self.state()
        state["composition"]["names"].append("duplicate_length")
        objective = state["objective_opportunities"]
        objective["features"][:2] = [float("nan"), True]
        features, _ = wpcombined_live.blocks(state, self.model)
        np.testing.assert_array_equal(features["composition"], 0.)
        np.testing.assert_array_equal(features["objective"], 0.)
        state["objective_opportunities"]["features"][:2] = [.5, -.5]
        state["objective_opportunities"]["known"][:2] = [1, True]
        features, _ = wpcombined_live.blocks(state, self.model)
        self.assertEqual(self.by_name(features, "objective", "baron_time_adv"), 0.)
        self.assertEqual(self.by_name(features, "objective", "elder_time_adv"), -.5)
        state["objective_opportunities"]["features"] = np.array(.5)
        features, _ = wpcombined_live.blocks(state, self.model)
        np.testing.assert_array_equal(features["objective"], 0.)

    def test_composition_source_time_patch_and_complete_champion_coverage(self):
        for mutation in (lambda b: b["provenance"]["static"].update(available_from_ts=99999.),
                         lambda b: b["provenance"]["static"].update(version="16.15.1"),
                         lambda b: b.update(feed_patch="16.15.1")):
            state = self.state()
            mutation(state["composition"])
            features, _ = wpcombined_live.blocks(state, self.model)
            np.testing.assert_array_equal(features["composition"], 0.)
        state = self.state()
        state["composition"]["coverage"]["frontline_tank_tag_count"] = 9
        features, _ = wpcombined_live.blocks(state, self.model)
        self.assertEqual(self.by_name(features, "composition", "frontline_tank_tag_count"), 0.)
        self.assertEqual(self.by_name(features, "composition", "mean_attack_range"), 100.)

    def test_sq_prior_patch_coverage_table_binding_and_exact_twenty_minute_cutoff(self):
        for mutation in (lambda b: b.update(source_patches=["16.16"]),
                         lambda b: b.update(coverage=.79), lambda b: b.update(pair_score=True),
                         lambda b: b.update(patch="16.15")):
            state = self.state()
            mutation(state["sq_early"])
            features, _ = wpcombined_live.blocks(state, self.model)
            np.testing.assert_array_equal(features["sq"], 0.)
        model = copy.deepcopy(self.model)
        model["source_checks"] = {"sq_table_sha256": "b"*64}
        features, _ = wpcombined_live.blocks(self.state(), model)
        np.testing.assert_array_equal(features["sq"], 0.)
        state = self.state(1200.)
        features, _ = wpcombined_live.blocks(state, self.model)
        np.testing.assert_array_equal(features["sq"], 0.)
        self.assertTrue(np.any(features["trend"]))

    def test_explicit_training_support_and_zero_weights_narrow_available_features(self):
        model = copy.deepcopy(self.model)
        model["trained_support"] = {k: list(v) for k, v in wpcombined_live.SUPPORTED.items()}
        model["trained_support"]["composition"] = []
        next(f for f in model["blocks"] if f["name"] == "sq")["coefficients"][:] = 0.
        features, detail = wpcombined_live.blocks(self.state(), model)
        np.testing.assert_array_equal(features["composition"], 0.)
        np.testing.assert_array_equal(features["sq"], 0.)
        self.assertEqual(set(detail["available_families"]), {"objective", "trend"})

    def test_missing_captures_preserve_exact_baseline_including_endpoints(self):
        features, detail = wpcombined_live.blocks({}, self.model)
        self.assertEqual(detail["available_families"], [])
        for baseline in (0., .1234567890123, .5, 1.):
            self.assertEqual(wpcombined.predict(self.model, features, [baseline], [5.])[0], baseline)


if __name__ == "__main__":
    unittest.main()
