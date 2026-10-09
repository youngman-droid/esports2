import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from lol_ticker import wpdeploy, wpgam, wpx


class TrainingExposureGuardTests(unittest.TestCase):
    def test_fresh_training_outcome_is_rejected_before_model_loading_or_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = os.path.join(folder, "states.npz")
            inventory = os.path.join(folder, "outcome_exposure.json")
            np.savez(dataset, gid=np.array([1, 1, 2]), date=np.array(["2026-09-02", "2026-09-02", "2026-09-03"]),
                     y=np.array([{"never_read": True}], dtype=object))
            ledger = {"kind": wpdeploy.wpexposure.KIND, "consumed_through": "2026-09-02",
                      "games": {}, "history": [], "plans": {}}
            with open(inventory, "w") as handle:
                json.dump(ledger, handle)
            legacy = os.path.join(folder, "evaluation_registry.json")
            with open(legacy, "wb") as handle:
                handle.write(b"legacy must remain untouched")
            before = {p: wpdeploy._sha256(p) for p in (inventory, legacy, dataset)}
            with mock.patch.object(wpgam, "fit_full") as fit, mock.patch.object(wpgam, "load_model") as load:
                with self.assertRaisesRegex(ValueError, "exposed development outcomes only"):
                    wpdeploy.refresh(dataset_path=dataset, gam_path=os.path.join(folder, "gam.npz"), exposure_path=inventory)
            fit.assert_not_called(); load.assert_not_called()
            self.assertEqual(before, {p: wpdeploy._sha256(p) for p in before})

    def test_missing_inventory_does_not_quarantine_supplied_outcomes_implicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = os.path.join(folder, "states.npz"); inventory = os.path.join(folder, "missing.json")
            np.savez(dataset, gid=np.array([1]), date=np.array(["2026-01-01"]))
            with self.assertRaisesRegex(ValueError, "exposed development outcomes only"):
                wpdeploy._require_consumed_training_inputs(dataset, inventory)
            self.assertFalse(os.path.exists(inventory))

    def test_complete_existing_exposure_allows_training_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = os.path.join(folder, "states.npz"); inventory = os.path.join(folder, "inventory.json")
            np.savez(dataset, gid=np.array([1, 1, 2]), date=np.array(["2026-01-01", "2026-01-01", "2026-01-02"]))
            with open(inventory, "w") as handle:
                json.dump({"kind": wpdeploy.wpexposure.KIND, "consumed_through": "2026-01-01",
                           "games": {"golgg:2": {"reason": "already exposed"}}, "history": [], "plans": {}}, handle)
            result = wpdeploy._require_consumed_training_inputs(dataset, inventory)
            self.assertEqual(result["games"], 2)
            self.assertEqual(result["fresh_games"], 0)

    def test_missing_or_inconsistent_dates_fail_before_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            for i, data in enumerate((dict(gid=np.array([1])),
                                      dict(gid=np.array([1]), date=np.array([""])),
                                      dict(gid=np.array([1, 1]), date=np.array(["2026-01-01", "2026-01-02"])))):
                path = os.path.join(folder, "invalid%d.npz" % i); np.savez(path, **data)
                with mock.patch.object(wpgam, "fit_full") as fit, self.assertRaises(ValueError):
                    wpdeploy.refresh(dataset_path=path, exposure_path=os.path.join(folder, "unused.json"))
                fit.assert_not_called()


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
    def test_architecture_change_stages_without_replacing_or_consuming_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            gam = os.path.join(tmp, "gam.npz")
            legacy = os.path.join(tmp, "legacy.npz")
            for path in (gam, legacy):
                with open(path, "wb") as fh: fh.write(b"incumbent")
            def fit_gam(**kwargs):
                with open(kwargs["model_path"], "wb") as fh: fh.write(b"candidate")
            staged = {"kind": wpgam.MODEL_KIND, "state": {"optimizer_success": True}}
            with mock.patch.object(wpgam, "load_model", side_effect=[
                    {"kind": wpgam.LEGACY_MODEL_KIND}, staged]), \
                    mock.patch.object(wpdeploy, "_require_consumed_training_inputs", return_value={}), \
                    mock.patch.object(wpgam, "fit_full", side_effect=fit_gam), \
                    mock.patch.object(wpx, "fit_full_legacy") as fit_legacy, \
                    mock.patch.object(wpx, "legacy_artifact_usable") as check_legacy, \
                    mock.patch.object(wpgam, "predict_live", return_value={"p_blue": .5}), \
                    mock.patch.object(wpx, "_predict_live_legacy") as predict_legacy, \
                    mock.patch.object(wpdeploy.wpaudit, "artifact_shape", return_value={"passed": True}), \
                    mock.patch.object(wpdeploy, "audit_live_shape", return_value={"passed": True}), \
                    mock.patch.object(wpdeploy, "run") as gate:
                result = wpdeploy.refresh(dataset_path="unused", gam_path=gam,
                    legacy_path=legacy, stack_path=os.path.join(tmp,"stack.json"),
                    result_path=os.path.join(tmp,"result.json"))
            self.assertFalse(result["deployed"])
            gate.assert_not_called()
            fit_legacy.assert_not_called()
            check_legacy.assert_not_called()
            predict_legacy.assert_not_called()
            for path in (gam, legacy):
                with open(path,"rb") as fh: self.assertEqual(fh.read(), b"incumbent")
            with open(result["candidate_path"], "rb") as fh:
                self.assertEqual(fh.read(), b"candidate")

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

class DirectEvidenceTests(unittest.TestCase):
    def test_unsupported_adapter_cannot_promote_even_with_valid_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            (candidate, incumbent, dataset, legacy), report = self._fixture(td)
            stack = os.path.join(td, "stack.json")
            with open(stack, "w") as fh:
                json.dump({"stack_sha256": "stack-1", "components": {
                    "gam_sha256": wpdeploy._sha256(incumbent)}}, fh)
            before = {p: wpdeploy._sha256(p) for p in (candidate, incumbent, dataset, legacy, stack)}
            for meta in ({"staged_only": True}, {"production_adapter_required": "unimplemented_v1"}):
                with self.subTest(meta=meta), \
                        mock.patch.object(wpdeploy, "_direct_evidence_valid", return_value=(True, "valid")), \
                        mock.patch.object(wpgam, "load_model", return_value={"meta": meta}), \
                        mock.patch.object(wpdeploy, "audit_live_shape") as audit:
                    with self.assertRaisesRegex(ValueError, "unsupported production adapter"):
                        wpdeploy.promote_candidate(candidate, report["direct_evidence"], dataset,
                            gam_path=incumbent, legacy_path=legacy, stack_path=stack)
                    audit.assert_not_called()
                self.assertEqual(before, {p: wpdeploy._sha256(p) for p in before})

    def _fixture(self, td):
        paths = [os.path.join(td, name) for name in ["candidate", "incumbent", "dataset", "legacy"]]
        for path in paths:
            with open(path, "wb") as fh:
                fh.write(os.path.basename(path).encode())
        candidate, incumbent, dataset, legacy = paths
        provenance = {"candidate_sha256": wpdeploy._sha256(candidate),
            "incumbent_sha256": wpdeploy._sha256(incumbent),
            "dataset_sha256": wpdeploy._sha256(dataset), "source_revision": "source-1",
            "input_contract_sha256": "contract-1", "incumbent_stack_sha256": "stack-1"}
        plan = {"provenance": provenance, "rule": {"metric": "game_balanced_brier",
            "bootstrap_unit": "series_date", "endpoint": "fixed_date",
            "end_date": "2020-01-01", "minimum_games": 100, "min_improvement": .001, "precision_target": .005,
            "max_logloss_regression": 0.0}}
        evidence = dict(provenance, kind="wpx_incumbent_comparison_v1", frozen_plan=plan,
            plan_sha256=wpdeploy.wpexposure.digest(plan), bootstrap_unit="series_date",
            endpoint_complete=True, game_balanced=True, games=120, clusters=50,
            ci95=[-.009, -.002], candidate_minus_incumbent_logloss=-.001,
            candidate_minus_incumbent_calibration_error=-.001)
        return paths, {"direct_evidence": evidence}

    def test_direct_gate_binds_artifacts_dataset_source_and_frozen_rule(self):
        with tempfile.TemporaryDirectory() as td:
            (candidate, incumbent, dataset, _), report = self._fixture(td)
            with mock.patch.object(wpdeploy, "_source_revision", return_value="source-1"):
                self.assertTrue(wpdeploy._direct_evidence_valid(
                    report, candidate, incumbent, dataset)[0])
                report["direct_evidence"]["frozen_plan"]["rule"]["minimum_games"] = 1
                self.assertFalse(wpdeploy._direct_evidence_valid(
                    report, candidate, incumbent, dataset)[0])
            (_, _, _, _), report = self._fixture(td)
            with mock.patch.object(wpdeploy, "_source_revision", return_value="source-2"):
                self.assertIn("source_revision", wpdeploy._direct_evidence_valid(
                    report, candidate, incumbent, dataset)[1])
            with open(candidate, "ab") as fh:
                fh.write(b"new calibration")
            with mock.patch.object(wpdeploy, "_source_revision", return_value="source-1"):
                self.assertIn("candidate_sha256", wpdeploy._direct_evidence_valid(
                    report, candidate, incumbent, dataset)[1])

    def test_precomputed_benchmark_must_match_every_component(self):
        with tempfile.TemporaryDirectory() as td:
            (candidate, _, dataset, legacy), _ = self._fixture(td)
            benchmark = {"kind": "wpx_live_stack_benchmark_v1", "provenance": {
                "dataset_sha256": wpdeploy._sha256(dataset),
                "gam_sha256": wpdeploy._sha256(candidate),
                "legacy_sha256": wpdeploy._sha256(legacy), "source_revision": "source-1"}}
            with mock.patch.object(wpdeploy, "_source_revision", return_value="source-1"):
                wpdeploy._validate_benchmark_provenance(benchmark, dataset, candidate, legacy)
                benchmark["provenance"]["dataset_sha256"] = "different rows"
                with self.assertRaisesRegex(ValueError, "dataset_sha256"):
                    wpdeploy._validate_benchmark_provenance(benchmark, dataset, candidate, legacy)

    def test_same_kind_refresh_stages_and_preserves_incumbent(self):
        with tempfile.TemporaryDirectory() as td:
            gam = os.path.join(td, "gam.npz")
            with open(gam, "wb") as fh:
                fh.write(b"incumbent")
            model = {"kind": wpgam.MODEL_KIND, "state": {"optimizer_success": True}}
            def fit(**kwargs):
                with open(kwargs["model_path"], "wb") as fh:
                    fh.write(b"changed calibration, same kind")
            with mock.patch.object(wpgam, "load_model", return_value=model), \
                    mock.patch.object(wpdeploy, "_require_consumed_training_inputs", return_value={}), \
                    mock.patch.object(wpgam, "fit_full", side_effect=fit), \
                    mock.patch.object(wpgam, "predict_live", return_value={"p_blue": .5}), \
                    mock.patch.object(wpdeploy.wpaudit, "artifact_shape", return_value={"passed": True}), \
                    mock.patch.object(wpdeploy, "audit_live_shape", return_value={"passed": True}), \
                    mock.patch.object(wpdeploy, "run") as run:
                result = wpdeploy.refresh(dataset_path="unused", gam_path=gam,
                    benchmark_result={"kind": "wpx_live_stack_benchmark_v1", "gate": {"passed": True}})
            run.assert_not_called()
            self.assertFalse(result["deployed"])
            with open(gam, "rb") as fh:
                self.assertEqual(fh.read(), b"incumbent")

    def test_exact_candidate_promotion_does_not_refit(self):
        with tempfile.TemporaryDirectory() as td:
            (candidate, incumbent, dataset, legacy), benchmark = self._fixture(td)
            stack, result_path = os.path.join(td, "stack.json"), os.path.join(td, "result.json")
            old_manifest = wpdeploy.build_manifest({"deployed": False}, incumbent, legacy)
            with open(stack, "w") as fh:
                json.dump(old_manifest, fh)
            evidence = benchmark["direct_evidence"]
            evidence["incumbent_stack_sha256"] = old_manifest["stack_sha256"]
            evidence["frozen_plan"]["provenance"]["incumbent_stack_sha256"] = old_manifest["stack_sha256"]
            evidence["plan_sha256"] = wpdeploy.wpexposure.digest(evidence["frozen_plan"])
            candidate_hash, legacy_hash = wpdeploy._sha256(candidate), wpdeploy._sha256(legacy)
            with mock.patch.object(wpdeploy, "_source_revision", return_value="source-1"), \
                    mock.patch.object(wpgam, "load_model", return_value={}), \
                    mock.patch.object(wpgam, "fit_full") as fit, \
                    mock.patch.object(wpdeploy.wpaudit, "artifact_shape", return_value={"passed": True}), \
                    mock.patch.object(wpdeploy, "audit_live_shape", return_value={"passed": True}):
                result = wpdeploy.promote_candidate(candidate, evidence, dataset,
                    gam_path=incumbent, legacy_path=legacy, stack_path=stack, result_path=result_path)
            fit.assert_not_called()
            self.assertTrue(result["benchmark"]["gam_promoted"])
            self.assertEqual(wpdeploy._sha256(incumbent), candidate_hash)
            self.assertEqual(wpdeploy._sha256(legacy), legacy_hash)
            with open(incumbent + ".previous", "rb") as fh:
                self.assertEqual(fh.read(), b"incumbent")
