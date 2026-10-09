import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lol_ticker import wpcombat, wpobjective, wptrend


def state(clock=1000, ts=None):
    observation = dict(available=True, hpb=[1., 1., 1., 1., 1.], hpr=[1., 1., 1., 1., 0.],
                       gdb=[10000.] * 5, gdr=[10000.] * 5, lvb=[14.] * 5, lvr=[14.] * 5)
    combat = wpcombat.observation_features(observation)
    combat["observation_ts"] = 100000 + clock if ts is None else ts
    return dict(ts=combat["observation_ts"], clock_s=clock, patch="16.16.1", game_id="g", combat=combat,
                baron_timer_blue_s=180., baron_timer_red_s=0., elder_timer_blue_s=0., elder_timer_red_s=0.,
                barons_blue=1, barons_red=0, elders_blue=0, elders_red=0, drag_blue=3, drag_red=2)


RULES = dict(source="test_fixture_explicit_rules", patch="16.16.1", dragon_first_s=300., dragon_respawn_s=300.,
             baron_first_s=1200., baron_respawn_s=360., elder_first_after_soul_s=360., elder_respawn_s=360.)


class ObjectiveTests(unittest.TestCase):
    def features(self, block):
        return dict(zip(block["names"], block["features"]))

    def known(self, block):
        return dict(zip(block["names"], block["known"]))

    def test_remaining_time_changes_opportunity_without_inventing_positions(self):
        start = state()
        earlier = wpobjective.observation_features(start)
        later = copy.deepcopy(start); later["baron_timer_blue_s"] = 30
        self.assertEqual(self.features(earlier)["baron_time_adv"], 1)
        self.assertEqual(self.features(earlier)["baron_living_gold_opportunity_k"], 50)
        self.assertAlmostEqual(self.features(wpobjective.observation_features(later))["baron_living_gold_opportunity_k"], 50/6)
        self.assertIn("position", earlier["excluded_inputs"])
        self.assertIn("respawn_seconds", earlier["excluded_inputs"])
        json.dumps(earlier, allow_nan=False)

    def test_aged_health_does_not_become_current_readiness(self):
        value = state(); value["combat"]["age_upper_s"] = 30
        result = wpobjective.observation_features(value)
        self.assertTrue(self.known(result)["baron_time_adv"])
        self.assertFalse(self.known(result)["baron_alive_opportunity"])
        self.assertEqual(self.features(result)["baron_living_gold_opportunity_k"], 0)

    def test_missing_or_different_combat_timestamp_is_not_current_readiness(self):
        for timestamp in (None, 999):
            value = state(); value["combat"]["observation_ts"] = timestamp
            block = wpobjective.observation_features(value)
            self.assertFalse(self.known(block)["baron_alive_opportunity"])
            self.assertTrue(self.known(block)["baron_time_adv"])

    def test_spawn_requires_patch_matched_explicit_rules(self):
        value = state(290); value.update(barons_blue=0, drag_blue=0, drag_red=0)
        for rules in (None, dict(RULES, patch="16.15.1")):
            block = wpobjective.capture(value, wpobjective.new_tracker(), timing_rules=rules)
            self.assertFalse(block["rules_available"])
            self.assertFalse(self.known(block)["dragon_alive_window_adv"])
        known = wpobjective.capture(value, wpobjective.new_tracker(), timing_rules=RULES)
        self.assertTrue(self.known(known)["dragon_alive_window_adv"])
        self.assertAlmostEqual(known["raw"]["spawn"]["dragon"]["proximity"], 5/6)

    def test_counter_transitions_are_intervals_and_retries_pauses_do_not_replay(self):
        tracker = wpobjective.new_tracker()
        first = state(1000); first.update(barons_blue=0, drag_blue=2)
        wpobjective.capture(first, tracker, timing_rules=RULES)
        second = state(1001)
        block = wpobjective.capture(second, tracker, timing_rules=RULES)
        original = copy.deepcopy(tracker)
        self.assertEqual(block["acquisition_intervals"]["dragon"], [1000., 1001.])
        self.assertEqual(wpobjective.capture(second, tracker, timing_rules=RULES), block)
        self.assertEqual(tracker, original)
        paused = state(1001, ts=102000)
        again = wpobjective.capture(paused, tracker, timing_rules=RULES)
        self.assertEqual(again["raw"]["spawn"], block["raw"]["spawn"])
        self.assertEqual(wpobjective.capture(first, tracker)["reason"], "out_of_order_frame")

    def test_remake_or_missing_counters_cannot_reuse_acquisition_history(self):
        tracker = wpobjective.new_tracker()
        first = state(1000); first.update(barons_blue=0, drag_blue=2)
        wpobjective.capture(first, tracker, timing_rules=RULES)
        wpobjective.capture(state(1001), tracker, timing_rules=RULES)
        remake = state(10, ts=102000); remake.update(barons_blue=0, drag_blue=0, drag_red=0)
        block = wpobjective.capture(remake, tracker, timing_rules=RULES)
        self.assertTrue(block["reset"])
        self.assertEqual(block["acquisition_intervals"], {})
        missing = state(11, ts=102001); del missing["drag_blue"]
        block = wpobjective.capture(missing, tracker, timing_rules=RULES)
        self.assertNotIn("dragon", block["raw"]["spawn"])

    def test_historical_prefix_ignores_future_kills_and_needs_completeness(self):
        events = [dict(time_s=900, side="blue", action="baron"),
                  dict(time_s=1001, side="red", action="dragon:elder")]
        block = wpobjective.historical_features(events, 1000, events_complete=True)
        self.assertEqual(block["raw"]["buff_remaining_s"]["baron"], [80., 0.])
        self.assertEqual(block["raw"]["buff_remaining_s"]["elder"], [0., 0.])
        self.assertEqual(block, wpobjective.historical_features(events[:1], 1000, events_complete=True))
        self.assertFalse(wpobjective.historical_features(events, 1000)["available"])

    def test_soul_threat_stays_separate_from_elder(self):
        events = [dict(time_s=100+i, side="blue", action="dragon:cloud") for i in range(3)]
        events.append(dict(time_s=400, side="red", action="dragon:elder"))
        block = wpobjective.historical_features(events, 402, events_complete=True, patch="16.16.1", timing_rules=RULES)
        self.assertEqual(self.features(block)["soul_threat_window_adv"], 1)
        self.assertGreater(self.features(block)["elder_time_adv"] * -1, 0)

    def test_residual_roundtrip_and_missing_fallback(self):
        n = 80
        extra = np.zeros((n, len(wpobjective.FEATURE_NAMES))); extra[:, 0] = np.tile([-1., 1.], n//2)
        baseline = np.full(n, .5); labels = np.tile([0, 1], n//2)
        model = wpobjective.fit(extra, baseline, labels, np.arange(n), np.full(n, 25.))
        predicted = wpobjective.predict(model, extra, baseline, np.full(n, 25.))
        self.assertGreater(predicted[1], predicted[0])
        np.testing.assert_array_equal(wpobjective.predict(model, np.zeros_like(extra), baseline, np.full(n, 25.)), baseline)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.npz"; wpobjective.save(model, path)
            np.testing.assert_array_equal(wpobjective.predict(wpobjective.load(path), extra, baseline, np.full(n, 25.)), predicted)
        with self.assertRaises(ValueError):
            wptrend.predict(model, np.zeros((n, len(wptrend.FEATURE_NAMES))), baseline, np.full(n, 25.))


if __name__ == "__main__":
    unittest.main()
