import json
import os
import tempfile
import unittest

import numpy as np

from lol_ticker import wpdeploy, wpx


class WeightSelectionTests(unittest.TestCase):
    def test_validation_selection_prefers_better_component(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        gids = np.arange(4)
        gam = np.array([0.1, 0.2, 0.8, 0.9])
        legacy = 1.0 - gam
        weight, curve = wpdeploy.select_weight(
            gam, legacy, y, gids, grid=[0.0, 0.5, 1.0])
        self.assertEqual(weight, 1.0)
        self.assertEqual(len(curve), 3)

    def test_paired_gate_requires_interval_below_zero(self):
        y = np.array([0.0, 0.0, 1.0, 1.0] * 20)
        gids = np.arange(len(y))
        gam = np.where(y, 0.7, 0.3)
        better = np.where(y, 0.9, 0.1)
        got = wpdeploy.paired_gate(gam, better, y, gids, bootstrap=500)
        self.assertTrue(got["passed"])
        self.assertLess(got["ci95"][1], 0.0)

    def test_shape_audit_rejects_nonmonotone_live_blend(self):
        def gam(state):
            return float(1.0 / (1.0 + np.exp(
                -float(state.get("gold_diff_k", 0.0)))))

        def legacy(state):
            return float(1.0 / (1.0 + np.exp(
                2.0 * float(state.get("gold_diff_k", 0.0)))))

        got = wpdeploy.audit_live_shape(
            w_gam=0.5, predict_gam=gam, predict_legacy=legacy)
        self.assertFalse(got["passed"])
        self.assertTrue(any(row["case"] == "gold"
                            for row in got["failures"]))

    def test_shape_audit_accepts_monotone_gam_without_legacy(self):
        got = wpdeploy.audit_live_shape(
            w_gam=1.0,
            predict_gam=lambda state: 0.5 + 0.01 * float(
                state.get("gold_diff_k", 0.0)),
            predict_legacy=lambda _state: self.fail(
                "standalone GAM audit must not call legacy"))
        self.assertTrue(got["passed"])
        self.assertTrue(got["audited"])


class ManifestTests(unittest.TestCase):
    def test_hash_mismatch_falls_back_to_gam(self):
        with tempfile.TemporaryDirectory() as tmp:
            gam = os.path.join(tmp, "gam.bin")
            legacy = os.path.join(tmp, "legacy.npz")
            stack = os.path.join(tmp, "stack.json")
            with open(gam, "wb") as fh:
                fh.write(b"gam")
            np.savez_compressed(
                legacy, contract=np.asarray(wpx.LEGACY_LIVE_CONTRACT),
                game_balanced=np.asarray(True), converged=np.asarray(True))
            benchmark = {"deployed": True, "deployed_w_gam": 0.4}
            manifest = wpdeploy.build_manifest(benchmark, gam, legacy)
            with open(stack, "w") as fh:
                json.dump(manifest, fh)
            loaded = wpx.load_live_stack(stack, gam, legacy)
            self.assertTrue(loaded["deployed"])
            with open(gam, "ab") as fh:
                fh.write(b"changed")
            loaded = wpx.load_live_stack(stack, gam, legacy)
            self.assertFalse(loaded["deployed"])
            self.assertIn("hash", loaded["reason"])

    def test_legacy_design_excludes_historical_only_features(self):
        names = list(wpx.FEATURE_NAMES)
        rich, _ = wpx._legacy_columns(names)
        selected = {names[i] for i in rich}
        self.assertFalse(selected & wpx.LEGACY_LIVE_EXCLUDED)
        for required in ("gold_k", "dead_blue", "baron_active", "hp_pool"):
            self.assertIn(required, selected)

    def test_registry_quarantines_all_preexisting_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            dates = np.array(["2026-08-01", "2026-08-05", "2026-08-03"])
            registry, initialized = wpdeploy._load_registry(path, dates)
            self.assertTrue(initialized)
            self.assertEqual(registry["consumed_through"], "2026-08-05")
            wpdeploy._write_registry(path, registry)
            loaded, initialized = wpdeploy._load_registry(path, dates)
            self.assertFalse(initialized)
            self.assertEqual(loaded, registry)

    def test_registry_rejects_a_different_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            dates = np.array(["2026-08-01", "2026-08-05"])
            registry, _ = wpdeploy._load_registry(
                path, dates, experiment="live_stack")
            wpdeploy._write_registry(path, registry)
            with self.assertRaises(ValueError):
                wpdeploy._load_registry(
                    path, dates, experiment="historical_odds")


if __name__ == "__main__":
    unittest.main()
