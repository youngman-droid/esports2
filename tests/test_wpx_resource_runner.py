import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research import wpx_resource as runner


class ResourceStudyProtocolTests(unittest.TestCase):
    def test_probability_audit_rejects_nonfinite_and_out_of_range(self):
        part = dict(y=np.zeros(2))
        for value in (np.nan, np.inf, -0.01, 1.01):
            with self.subTest(value=value), patch.object(runner, "predict", return_value=np.full(2, value)), \
                    patch.object(runner.wpbench, "_apply_platt", side_effect=lambda p, _: p):
                with self.assertRaisesRegex(ValueError, "Invalid baseline"):
                    runner.gold_audit({}, {}, part, None)

    def test_selection_rejects_gain_concentrated_in_one_block(self):
        selected, checks = self.selection_for([.40, .51, .51])
        self.assertEqual(selected["family"], "core")
        self.assertFalse(checks["pooled_smooth"]["eligible"])
        self.assertGreater(checks["pooled_smooth"]["leave_one_block_delta"][0], 0)

    def test_selection_accepts_consistent_paired_development_gain(self):
        selected, checks = self.selection_for([.49, .49, .49])
        self.assertEqual(selected["family"], "pooled_smooth")
        self.assertTrue(checks["pooled_smooth"]["eligible"])
        self.assertLess(checks["pooled_smooth"]["ci95"][1], 0)

    @staticmethod
    def selection_for(probabilities):
        families = ("core", "pooled_role", "pooled_constant", "pooled_smooth")
        choices = {f: dict(family=f, index=0, calibration="none",
                          mean_brier=.24 if f == "pooled_smooth" else .25) for f in families}
        blocks, records = [], []
        base = dict(row_game=np.arange(6), gid=np.arange(1, 7), y=np.zeros(6))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for i, probability in enumerate(probabilities):
                blocks.append(dict(validation=np.repeat(np.arange(3) == i, 2)))
                stage = output / ("development_" + str(i + 1))
                stage.mkdir()
                record = {}
                for family in families:
                    key = family + "_0"
                    p = probability if family == "pooled_smooth" else .5
                    np.savez(stage / (key + "_raw.npz"), validation=np.full(2, p))
                    record[key] = dict(methods=dict(none=dict(calibration=dict(intercept=0., slope=1.))))
                records.append(record)
            return runner.robust_selection(choices, records, blocks, base,
                                           {i: str(i) for i in range(1, 7)}, output)


if __name__ == "__main__":
    unittest.main()
