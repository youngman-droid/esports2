import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lol_ticker import wptrend


def state(clock, *, ts=None, game="g", increment=0):
    observation = dict(role_order=list(wptrend.ROLES), gdb=[1000.+clock+increment] * 5,
                       gdr=[1000.+clock] * 5, hpb=[1.] * 5, hpr=[1.] * 5)
    ts = clock+100000 if ts is None else ts
    return dict(game_id=game, ts=ts, clock_s=clock,
                combat=dict(observation=observation, observation_ts=ts, age_upper_s=0),
                gold_blue=5000+5*clock+5*increment, gold_red=5000+5*clock,
                kills_blue=clock//30, kills_red=0, towers_blue=clock//120, towers_red=0,
                barons_blue=0, barons_red=0, drag_blue=0, drag_red=0,
                elders_blue=0, elders_red=0, inhib_blue=0, inhib_red=0)


class TrendTests(unittest.TestCase):
    def pairs(self, block, key="features"):
        return dict(zip(block["names"], block[key]))

    def test_three_windows_and_causal_anchor_gold_changes(self):
        tracker = wptrend.new_tracker()
        for clock in range(0, 311, 10):
            block = wptrend.capture(state(clock, increment=clock), tracker)
        self.assertTrue(block["available"])
        values = self.pairs(block)
        self.assertAlmostEqual(values["gold_change_bot_30s_k"], .03)
        self.assertAlmostEqual(values["gold_change_bot_120s_k"], .12)
        self.assertAlmostEqual(values["gold_change_bot_300s_k"], .3)
        self.assertEqual(block["windows"]["120"]["anchor_clock_s"], 190)
        self.assertGreater(block["windows"]["120"]["lead_volatility_k"], 0)
        self.assertNotIn("lead_volatility_k", wptrend.FEATURE_NAMES)
        json.dumps(block, allow_nan=False)

    def test_retry_future_and_out_of_order_cannot_change_earlier_features(self):
        tracker = wptrend.new_tracker()
        for clock in range(0, 41, 10):
            block = wptrend.capture(state(clock, increment=clock), tracker)
        original = copy.deepcopy(tracker)
        changed = state(40, increment=100000)
        self.assertEqual(wptrend.capture(changed, tracker), block)
        self.assertEqual(tracker, original)
        old = wptrend.capture(state(35), tracker)
        self.assertEqual(old["reason"], "out_of_order_frame")
        self.assertEqual(tracker, original)
        frozen = copy.deepcopy(block)
        wptrend.capture(state(50, increment=1000), tracker)
        self.assertEqual(block, frozen)

    def test_pause_wall_time_never_fills_a_window_or_reweights_volatility(self):
        tracker = wptrend.new_tracker()
        for clock in range(0, 31, 10):
            before = wptrend.capture(state(clock, increment=clock), tracker)
        for second in range(1, 301):
            after = wptrend.capture(state(30, increment=30, ts=100030+second), tracker)
        self.assertEqual(before["features"], after["features"])
        self.assertEqual(before["windows"], after["windows"])
        self.assertEqual(after["retained_frames"], 4)

    def test_remake_attempt_counter_reset_and_history_bound(self):
        tracker = wptrend.new_tracker()
        for clock in range(0, 361, 10):
            block = wptrend.capture(state(clock), tracker)
        self.assertLessEqual(block["retained_frames"], 33)
        remake = wptrend.capture(state(0, ts=101000), tracker)
        self.assertTrue(remake["reset"])
        self.assertFalse(remake["available"])
        new_attempt = wptrend.capture(state(30, ts=101030, game="new"), tracker)
        self.assertTrue(new_attempt["reset"])
        self.assertFalse(new_attempt["available"])

    def test_missing_endpoints_and_history_gaps_are_explicit(self):
        tracker = wptrend.new_tracker()
        wptrend.capture(state(0), tracker)
        gap = wptrend.capture(state(30), tracker)
        self.assertEqual(gap["windows"]["30"]["reason"], "history_gap")
        tracker = wptrend.new_tracker()
        for clock in range(0, 31, 10):
            value = state(clock)
            value["combat"]["observation"]["hpb"] = []
            block = wptrend.capture(value, tracker)
        self.assertTrue(self.pairs(block, "known")["gold_change_bot_30s_k"])
        self.assertFalse(self.pairs(block, "known")["hp_change_bot_30s"])

    def test_historical_minutes_support_120_300_but_not_30_or_carried_hp(self):
        values = [state(clock, increment=clock) for clock in range(0, 361, 60)]
        for value in values:
            value["combat"]["age_upper_s"] = 60
        gold = np.array([[1000+s["clock_s"]*2]*5+[1000+s["clock_s"]]*5 for s in values])
        blocks = wptrend.historical_features(values, gold=gold, role_order=list(wptrend.ROLES))
        known = self.pairs(blocks[-1], "known")
        self.assertTrue(known["gold_change_bot_120s_k"])
        self.assertTrue(known["gold_change_bot_300s_k"])
        self.assertFalse(known["gold_change_bot_30s_k"])
        self.assertFalse(known["hp_change_bot_120s"])
        self.assertEqual(blocks[-1]["windows"]["30"]["reason"], "anchor_too_old")
        with self.assertRaises(ValueError):
            wptrend.historical_features(values, gold=gold)

    def test_historical_signed_event_changes_work_with_fixed_gold_and_no_side_counts(self):
        states = [dict(game_id="g", ts=t, clock_s=t, gold_blue=5000, gold_red=5000,
                       counter_differences=dict(kill=kills, tower=towers))
                  for t, kills, towers in ((0, 3, 0), (60, 2, 0), (120, 1, 1))]
        block = wptrend.historical_features(states)[-1]
        values, known = self.pairs(block), self.pairs(block, "known")
        self.assertEqual(values["lead_change_120s_k"], 0)
        self.assertEqual(values["kill_change_120s"], -2)
        self.assertEqual(values["tower_change_120s"], 1)
        self.assertTrue(known["kill_change_120s"])
        self.assertTrue(known["tower_change_120s"])
        self.assertFalse(block["reset"])

    def test_role_gold_live_join_is_independent_of_health_and_list_order(self):
        metadata, frame = {}, {}
        for side, offset in (("blue", 0), ("red", 5)):
            entries = [dict(participantId=offset+i+1, role=role) for i, role in enumerate(wptrend.ROLES)]
            players = [dict(participantId=offset+i+1, totalGold=1000+i*100) for i in range(5)]
            metadata[side+"TeamMetadata"] = dict(participantMetadata=entries[::-1])
            frame[side+"Team"] = dict(participants=[players[i] for i in (2, 4, 1, 3, 0)])
        observed = wptrend.live_observation(metadata, frame, state(0))
        self.assertEqual(observed["gold_roles"], [1000., 1100., 1200., 1300., 1400.] * 2)
        self.assertIsNone(observed["hp"])

    def test_residual_fit_roundtrip_and_unknown_baseline_fallback(self):
        n = 40; extra = np.zeros((n, len(wptrend.FEATURE_NAMES)))
        extra[:, 0] = np.tile([-1., 1.], n//2)
        p, y, t = np.full(n, .5), np.tile([0, 1], n//2), np.full(n, 20.)
        model = wptrend.fit(extra, p, y, np.arange(n), t)
        result = wptrend.predict(model, extra, p, t)
        self.assertGreater(result[1], result[0])
        np.testing.assert_array_equal(wptrend.predict(model, np.zeros_like(extra), p, t), p)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trend.npz"; wptrend.save(model, path)
            np.testing.assert_array_equal(wptrend.predict(wptrend.load(path), extra, p, t), result)


if __name__ == "__main__":
    unittest.main()
