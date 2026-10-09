import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lol_ticker import wpcombined, wpgam, wpresidual, wpsqearly


class JointFrozenCoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(18)
        self.extra = rng.normal(0., 2., (180, 3))
        self.t = np.tile(np.arange(30), 6).astype(float)
        self.gids = np.repeat(np.arange(30), 6)
        self.baseline = np.linspace(.2, .8, 180)
        self.y = rng.binomial(1, wpgam._sigmoid(self.extra[:, 0]*.3))
        self.spec = dict(feature_names=["alive", "gold", "level"], bounds=(0., 10.),
                         l2=800., smooth=70., time_knots=wpgam.TIME_KNOTS.tolist())
        self.sqspec = dict(feature_names=["pair_score"], bounds=(0., 2.5), l2=300., smooth=70.,
                           time_knots=[0., 5., 10., 15., 20.], monotone_decay=True,
                           zero_after_min=20., scale=.25)

    def fit(self):
        return wpcombined.fit({"combat": self.extra}, self.baseline, self.y, self.gids,
                              self.t, specs={"combat": self.spec})

    def test_one_family_matches_existing_frozen_residual(self):
        joint = self.fit()
        reference = wpresidual.fit(self.extra, self.baseline, self.y, self.gids, self.t,
                                   feature_names=self.spec["feature_names"], candidate_kind="test",
                                   input_contract={}, coefficient_bounds=self.spec["bounds"])
        np.testing.assert_allclose(joint["blocks"][0]["scale"], reference["scale"], rtol=1e-14)
        np.testing.assert_allclose(wpcombined.predict(joint, {"combat": self.extra}, self.baseline, self.t),
                                   wpresidual.predict(reference, self.extra, self.baseline, self.t), atol=2e-8)

    def test_sq_single_family_matches_own_monotone_curve(self):
        score = self.extra[:, 0]*.125
        available = np.ones(len(score), bool)
        joint = wpcombined.fit({"sq": score[:, None]}, self.baseline, self.y, self.gids,
                              self.t, specs={"sq": self.sqspec})
        reference = wpsqearly.fit(score, available, self.baseline, self.y, self.gids, self.t)
        np.testing.assert_allclose(wpcombined.predict(joint, {"sq": score[:, None]}, self.baseline, self.t),
                                   wpsqearly.predict(reference, score, available, self.baseline, self.t), atol=2e-6)

    def test_zero_families_preserve_baseline_exactly_including_endpoints(self):
        model = self.fit()
        baseline = np.array([0., .123456789012345, .5, 1.])
        np.testing.assert_array_equal(wpcombined.predict(model, {"combat": np.zeros((4, 3))},
                                                        baseline, [0., 4., 20., 80.]), baseline)

    def test_combined_effect_is_side_antisymmetric_without_intercept(self):
        blocks = {"combat": self.extra, "sq": self.extra[:, :1]*.125}
        specs = {"combat": self.spec, "sq": self.sqspec}
        model = wpcombined.fit(blocks, self.baseline, self.y, self.gids, self.t, specs=specs)
        forward = wpcombined.correction(model, blocks, self.t)
        reverse = wpcombined.correction(model, {k: -v for k, v in blocks.items()}, self.t)
        np.testing.assert_allclose(forward, -reverse, atol=1e-15)
        predictions = wpcombined.predict(model, blocks, self.baseline, self.t)
        swapped = wpcombined.predict(model, {k: -v for k, v in blocks.items()}, 1-self.baseline, self.t)
        np.testing.assert_allclose(predictions, 1-swapped, atol=2e-16)

    def test_sq_late_and_missing_rows_are_zero_and_cap_remains_strict(self):
        spec = dict(self.sqspec, l2=.01, smooth=.01)
        model = wpcombined.fit({"sq": np.full((120, 1), .25)}, np.full(120, .5),
                              np.ones(120), np.arange(120), np.zeros(120), specs={"sq": spec})
        slopes = model["blocks"][0]["coefficients"]
        self.assertEqual(model["optimizer"], "L-BFGS-B then SLSQP")
        self.assertGreater(slopes[0], 2.49999)
        self.assertLessEqual(slopes[0], 2.5)
        self.assertTrue(np.all(np.diff(slopes) <= 0))
        self.assertEqual(slopes[-1], 0.)
        clocks = np.array([0., 4., 10., 19., 20., 25., 999.])
        delta = wpcombined.correction(model, {"sq": np.full((7, 1), .25)}, clocks)
        self.assertTrue(np.all(np.diff(delta) <= 0))
        np.testing.assert_array_equal(delta[clocks >= 20], 0.)
        baseline = np.linspace(.1, .9, 7)
        np.testing.assert_array_equal(wpcombined.predict(model, {"sq": np.zeros((7, 1))}, baseline, clocks), baseline)

    def test_game_balanced_training_scales_and_signed_family(self):
        extra = np.array([[3., 0.], [3., 0.], [3., 0.], [1., 0.]])
        model = wpcombined.fit({"composition": extra}, [.5]*4, [0, 0, 0, 1], [1, 1, 1, 2], [5.]*4,
                              specs={"composition": dict(feature_names=["range", "unknown"], bounds=(-10., 10.))})
        family = model["blocks"][0]
        np.testing.assert_allclose(family["scale"], [np.sqrt(5.), 1.])
        self.assertTrue(np.all(family["coefficients"].reshape(2, -1)[1] == 0))
        self.assertLess(wpcombined.correction(model, {"composition": [[1., 0.]]}, [5.])[0], 0.)

    def test_no_training_support_keeps_zero_coefficients(self):
        model = wpcombined.fit({"combat": np.zeros((2, 3)), "sq": np.zeros((2, 1))},
                              [.2, .8], [0, 1], [1, 2], [0., 5.],
                              specs={"combat": self.spec, "sq": self.sqspec})
        self.assertEqual(model["active_rows"], 0)
        self.assertEqual(model["iterations"], 0)
        for block in model["blocks"]:
            np.testing.assert_array_equal(block["coefficients"], 0.)

    def test_artifact_roundtrip_and_explicit_contract_validation(self):
        model = self.fit()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"joint.npz"
            wpcombined.save(model, path)
            loaded = wpcombined.load(path)
        wpcombined.validate(loaded, specs={"combat": self.spec})
        np.testing.assert_array_equal(wpcombined.predict(model, {"combat": self.extra}, self.baseline, self.t),
                                      wpcombined.predict(loaded, {"combat": self.extra}, self.baseline, self.t))
        wrong = dict(self.spec, smooth=71.)
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            wpcombined.validate(loaded, specs={"combat": wrong})
        corrupt = copy.deepcopy(model)
        corrupt["blocks"][0]["scale"][0] = .9
        with self.assertRaises(ValueError):
            wpcombined.validate(corrupt)

    def test_invalid_inputs_specs_and_family_mismatches_are_rejected(self):
        args = ({"combat": self.extra}, self.baseline, self.y, self.gids, self.t)
        for spec in (dict(self.spec, l2=0.), dict(self.spec, time_knots=[0., 0., 2.]),
                     dict(self.spec, bounds=(1., 2.)), dict(self.spec, unknown=1),
                     dict(self.spec, feature_names=["a", "a", "b"])):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                wpcombined.fit(*args, specs={"combat": spec})
        model = self.fit()
        for blocks, p, t in (({}, [.5], [1.]), ({"combat": [[1., 2.]]}, [.5], [1.]),
                             ({"combat": [[1., np.nan, 0.]]}, [.5], [1.]),
                             ({"combat": [[1., 0., 0.]]}, [1.1], [1.]),
                             ({"combat": [[1., 0., 0.]]}, [.5], [-1.])):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                wpcombined.predict(model, blocks, p, t)

    def test_late_sq_does_not_suppress_other_family(self):
        model = wpcombined.fit({"combat": self.extra, "sq": self.extra[:, :1]*.125},
                              self.baseline, self.y, self.gids, self.t,
                              specs={"combat": self.spec, "sq": self.sqspec})
        without_sq = {"combat": np.ones((2, 3)), "sq": np.zeros((2, 1))}
        with_sq = {"combat": np.ones((2, 3)), "sq": np.full((2, 1), 100.)}
        np.testing.assert_array_equal(wpcombined.correction(model, without_sq, [20., 70.]),
                                      wpcombined.correction(model, with_sq, [20., 70.]))
        self.assertGreater(wpcombined.correction(model, with_sq, [20., 70.])[0], 0.)


if __name__ == "__main__":
    unittest.main()
