import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lol_ticker import wpcandidate as candidate, wpgam, wpresource, wpbench
from scripts import wpx_fit_candidate as final_fit
from tests import test_wpgam as gam_tests, test_wpresource as resource_tests


class CandidateExportTests(unittest.TestCase):
    def model(self):
        model = gam_tests.ArtifactTests()._model()
        model["kind"] = wpgam.MODEL_KIND
        model["state"].update(cal_intercept=0., cal_slope=1.)
        model["meta"] = {"input_contract": {"name": "causal"}, "training_cutoff": "2026-09-03T00:00:00+00:00",
                         "dataset_sha256": "frozen-data", "live_calibration": {"intercept": .1, "slope": 1.5},
                         "staged_only": True, "production_adapter_required": "candidate_calibration_v1"}
        return model

    def test_core_adapter_preserves_exact_study_calibration_in_saturated_tail(self):
        model = self.model()
        model["state"]["theta"][:] = 0
        model["state"]["theta"][0] = 20
        state = {"t_min": 20, "gold_blue": 20000, "gold_red": 20000, "gold_diff_k": 0}
        raw = wpgam.predict_live_model(model, state, rounded=False)["p_blue"]
        expected = wpbench._apply_platt([raw], model["meta"]["live_calibration"])[0]
        self.assertEqual(candidate.predict_candidate_model(model, state)["p_blue"], expected)
        self.assertNotEqual(expected, raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"core.npz"
            wpgam.save_model(model, str(path), model["meta"])
            loaded = candidate.load_candidate_model(str(path))
            self.assertEqual(candidate.predict_candidate_model(loaded, state)["p_blue"], expected)
            self.assertEqual(candidate.input_contract(loaded)["calibration_probability_clip"], [1e-5, 1-1e-5])

    def test_native_calibration_matches_study_keeps_component_sum_and_roundtrips(self):
        model = self.model()
        model["meta"].pop("live_calibration")
        state = model["state"]
        state.update(cal_intercept=.1, cal_slope=1.5, calibration_probability_clip=[1e-5, 1-1e-5])
        raw = np.zeros((3, len(state["feature_names"])))
        raw[:, 0] = [-10, 0, 10]
        state["theta"][1] = 4
        uncalibrated = dict(state, cal_intercept=0., cal_slope=1., calibration_probability_clip=None)
        before = wpgam.predict_state(uncalibrated, raw, [10, 20, 30])
        expected = wpbench._apply_platt(before, {"intercept": .1, "slope": 1.5})
        p, eta, parts = wpgam.predict_state(state, raw, [10, 20, 30], components=True)
        np.testing.assert_allclose(p, expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(parts.sum(axis=1), eta)
        np.testing.assert_array_equal(wpgam._sigmoid(eta), p)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"native.npz")
            wpgam.save_model(model, path, model["meta"])
            loaded = wpgam.load_model(path)
            self.assertEqual(loaded["state"]["calibration_probability_clip"], [1e-5, 1-1e-5])
            np.testing.assert_array_equal(wpgam.predict_state(loaded["state"], raw, [10, 20, 30]), p)
            live = {"t_min": 20, "gold_blue": 20000, "gold_red": 10000, "gold_diff_k": 10}
            self.assertEqual(candidate.predict_candidate_model(loaded, live)["p_blue"],
                             wpgam.predict_live_model(loaded, live, rounded=False)["p_blue"])

    def test_older_unclipped_artifacts_retain_bit_exact_predictions(self):
        state = self.model()["state"]
        state.update(cal_intercept=.3, cal_slope=.8)
        raw = np.random.default_rng(4).normal(size=(35, len(state["feature_names"])))
        t = np.linspace(0, 80, len(raw))
        z = wpgam._scale_apply(raw, state["mean"], state["std"], state["lo"], state["hi"])
        effect = np.einsum("ik,fk->if", wpgam.time_basis(t, state["knots"]), state["theta"], optimize=True)
        old_parts = np.column_stack([effect[:, 0], z*effect[:, 1:]])
        old_parts *= state["cal_slope"]; old_parts[:, 0] += state["cal_intercept"]
        expected = wpgam._sigmoid(old_parts.sum(axis=1))
        np.testing.assert_array_equal(wpgam.predict_state(state, raw, t), expected)
        np.testing.assert_array_equal(wpgam.predict_state(dict(state, calibration_probability_clip=None), raw, t), expected)

    def test_native_calibration_rejects_invalid_clip_bounds(self):
        state = self.model()["state"]
        raw = np.zeros((1, len(state["feature_names"])))
        for bounds in ([0, 1], [.9, .1], [np.nan, .9], [1e-5], []):
            with self.assertRaisesRegex(ValueError, "clipping bounds"):
                wpgam.predict_state(dict(state, calibration_probability_clip=bounds), raw, [10])

    def test_pooled_bundle_roundtrip_inference_and_missing_gold_fallback(self):
        core = self.model()
        resource, *_ = resource_tests.ResourceTests().fixture("constant")
        resource.update(input_names=list(wpresource.INPUT_NAMES), time_knots=wpgam.TIME_KNOTS.tolist())
        meta = dict(core["meta"], live_calibration={"intercept": -.1, "slope": .8})
        model = dict(kind=candidate.BUNDLE_KIND, core=core, resource=resource, meta=meta)
        state = {"t_min": 20, "gold_blue": 20000, "gold_red": 10000, "gold_diff_k": 10,
                 "gold_players": [4000]*5+[2000]*5, "gold_role": [2]*5}
        before = candidate.predict_candidate_model(model, state, ["Ahri"]*5, ["Ahri"]*5)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"pooled.npz")
            candidate.save_bundle(core, resource, meta, path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["core_npz"].dtype, np.uint8)
                self.assertEqual(archive["resource_npz"].dtype, np.uint8)
            loaded = candidate.load_candidate_model(path)
            after = candidate.predict_candidate_model(loaded, state, ["Ahri"]*5, ["Ahri"]*5)
            self.assertEqual(before, after)
            self.assertFalse(after["fallback"])
            missing = dict(state, gold_players=None)
            fallback = candidate.predict_candidate_model(loaded, missing, ["Ahri"]*5, ["Ahri"]*5)
            reference = candidate.predict_candidate_model(loaded["core"], missing, ["Ahri"]*5, ["Ahri"]*5)
            self.assertTrue(fallback["fallback"])
            self.assertEqual(fallback["p_blue"], reference["p_blue"])
            self.assertEqual(candidate.input_contract(loaded)["fallback_calibration"], core["meta"]["live_calibration"])

    def test_core_fit_and_calibration_are_disjoint_and_no_refit_occurs_after_calibration(self):
        raw = np.arange(30).reshape(5, 6)
        base = {"y": np.array([0, 1, 0, 1, 0]), "gid": np.arange(5), "t_min": np.arange(5)}
        fit, calibration = np.array([1, 1, 1, 0, 0], dtype=bool), np.array([0, 0, 0, 1, 1], dtype=bool)
        state = {"theta": np.zeros((1, 1))}
        with mock.patch.object(wpgam, "fit_state_model", return_value=state) as fitted, \
                mock.patch.object(wpgam, "predict_state", return_value=np.array([.6, .4])) as predict, \
                mock.patch.object(final_fit.wpadapt, "calibrate", return_value={"intercept": .1, "slope": .9}) as calibrated:
            model, params = final_fit.core_model({"pregame": {}, "champ_state": {}}, raw, base, fit, calibration, "platt")
        fitted.assert_called_once()
        np.testing.assert_array_equal(fitted.call_args.args[0], raw[fit])
        np.testing.assert_array_equal(predict.call_args.args[1], raw[calibration])
        np.testing.assert_array_equal(calibrated.call_args.args[1], base["y"][calibration])
        self.assertEqual(params, {"intercept": .1, "slope": .9})
        self.assertEqual(model["state"]["cal_slope"], .9)
        self.assertEqual(model["state"]["calibration_probability_clip"], [1e-5, 1-1e-5])

    def test_selection_resolves_only_frozen_plan_specs_and_eligible_families(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            selection = {"selected": {"family": "pooled_constant", "index": 1, "calibration": "platt"},
                         "choices": {"core": {"family": "core", "index": 0, "calibration": "temperature"}}}
            (path/"selection.json").write_text(json.dumps(selection))
            plan = {"specs": {"pooled_constant": [{"shrink": 800}, {"shrink": 2000}], "core": [{}]}}
            _, spec = final_fit.selected_settings(path, plan)
            self.assertEqual(spec, {"shrink": 2000})
            selection["selected"]["family"] = "rich_control"
            (path/"selection.json").write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, "not eligible"):
                final_fit.selected_settings(path, plan)

    def test_upstream_attestation_allows_only_state_calibration_and_serialization_changes(self):
        original = "import numpy as np\nCONSTANT=3\ndef fit_pregame(x): return x+CONSTANT\ndef predict_state(x): return x\ndef save_model(x): return x\ndef load_model(x): return x\n"
        native = original.replace("def predict_state(x): return x", "def predict_state(x): return np.clip(x,0,1)")
        self.assertEqual(final_fit.upstream_source_fingerprint(original), final_fit.upstream_source_fingerprint(native))
        changed = native.replace("CONSTANT=3", "CONSTANT=4")
        self.assertNotEqual(final_fit.upstream_source_fingerprint(original), final_fit.upstream_source_fingerprint(changed))
        changed = native.replace("return x+CONSTANT", "return x-CONSTANT")
        self.assertNotEqual(final_fit.upstream_source_fingerprint(original), final_fit.upstream_source_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
