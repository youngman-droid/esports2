"""Causality and source-provenance checks for the combat experiment runner."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research import wpx_combat as runner


def fixture():
    names = ["hp_pool", "hp_low_b", "hp_low_r", "dead_blue", "dead_red", "has_hp"]
    X = np.zeros((3, len(names)))
    X[2] = [-1, 1, 0, 1, 0, 1]
    rows = dict(gid=np.array([7, 7, 7]), y=np.ones(3), date=np.array(["2026-01-01"]*3),
                t=np.array([0, 60, 120]), prediction_clock_s=np.array([0, 60, 120]),
                hp_clock_lo_s=np.array([np.nan, np.nan, 60]), hp_clock_hi_s=np.array([np.nan, np.nan, 120]),
                hp_age_upper_s=np.array([np.nan, np.nan, 60]), hp_observation_ts=np.array([np.nan, np.nan, 100]),
                hp_origin_ts=np.array([np.nan, np.nan, 0]), X=X, names=names, used=np.array([False, False, True]))
    data = dict(hpb=[1, 1, 1, 0, 1], hpr=[1]*5, gdb=[800]*5, gdr=[800]*5,
                lvb=[5]*5, lvr=[5]*5, pidb=list(range(1, 6)), pidr=list(range(6, 11)))
    return rows, {(7, 100): data}, np.array([[0]*10, [600]*10, [1200]*10], dtype=float)


class CombatSourceTests(unittest.TestCase):
    def test_joins_the_exact_audited_observation_and_zeroes_missing_rows(self):
        rows, records, gold = fixture()
        got = runner.extract(rows, records, gold)
        np.testing.assert_array_equal(got["extra"][:2], 0)
        self.assertEqual(got["available"].tolist(), [False, False, True])
        self.assertEqual(got["death"].tolist(), [False, False, True])
        self.assertEqual(got["extra"][2, 3], -1)
        self.assertEqual(got["extra"][2, 13], -.8)

    def test_no_fallback_to_another_minute_or_future_source(self):
        rows, records, gold = fixture()
        with self.assertRaisesRegex(ValueError, "missing"):
            runner.extract(rows, {(7, 101): records[(7, 100)]}, gold)

    def test_changed_gold_cannot_reuse_the_original_clock_audit(self):
        rows, records, gold = fixture()
        records[(7, 100)]["gdb"][3] = 1300
        with self.assertRaisesRegex(ValueError, "clock bracket"):
            runner.extract(rows, records, gold)

    def test_changed_health_or_participant_order_is_rejected(self):
        for key, value in (("hpb", [1]*5), ("pidb", [2, 1, 3, 4, 5])):
            rows, records, gold = fixture()
            records[(7, 100)][key] = value
            with self.assertRaises(ValueError):
                runner.extract(rows, records, gold)

    def test_later_resources_cannot_change_an_earlier_observation(self):
        rows, records, gold = fixture()
        base = runner.extract(rows, records, gold)
        records[(7, 999)] = dict(records[(7, 100)], gdb=[9000]*5, hpb=[0]*5)
        altered = runner.extract(rows, records, gold)
        np.testing.assert_array_equal(base["extra"], altered["extra"])

    def test_future_join_rejected_before_reading_database(self):
        rows, _, _ = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"rows.npz"
            payload = {k: v for k, v in rows.items() if k not in {"used"}}
            payload.update(seq=np.full(3, -1), input_contract_sha256=np.array("fixture"))
            payload["hp_clock_hi_s"] = np.array([np.nan, np.nan, 121])
            np.savez_compressed(path, **payload)
            Path(str(path)+".manifest.json").write_text(json.dumps(dict(sha256=runner.sha(path),
                zero_future_timestamp_joins=True, input_contract_sha256="fixture")))
            with self.assertRaisesRegex(ValueError, "causal clock"):
                runner.load_rows(path)

    def test_health_ablation_does_not_mutate_full_features(self):
        rows, records, gold = fixture()
        extra = runner.extract(rows, records, gold)["extra"]
        before = extra.copy()
        health = runner.variant(extra, "role_health")
        np.testing.assert_array_equal(extra, before)
        np.testing.assert_array_equal(health[:, 10:], 0)


if __name__ == "__main__":
    unittest.main()
