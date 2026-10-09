import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lol_ticker import wpresidual


class FrozenCoreResidualTests(unittest.TestCase):
    def fit(self, *, terminal=None, bounds=(0., 10.)):
        extra = np.array([[1., -.2], [-1., .2], [2., -.5], [-2., .5]])
        return wpresidual.fit(extra, [.5] * 4, [1, 0, 1, 0], [1, 2, 3, 4], [1., 2., 5., 9.],
                              feature_names=["benefit", "uncertain"], candidate_kind="test_v1",
                              input_contract={"missing": "zero"}, coefficient_bounds=bounds,
                              time_knots=[0., 10., 20., 30.], zero_after_min=terminal)

    def test_frozen_baseline_exact_fallback_including_endpoints(self):
        model = self.fit()
        baseline = np.array([0., .123456789012345, .5, 1.])
        np.testing.assert_array_equal(wpresidual.predict(model, np.zeros((4, 2)), baseline,
                                                         [0., 4., 20., 80.]), baseline)

    def test_arbitrary_width_positive_physical_effect_and_side_symmetry(self):
        model = self.fit()
        probability = wpresidual.predict(model, [[1., 0.], [2., 0.], [-1., 0.]], [.5] * 3, [5.] * 3)
        self.assertGreater(probability[0], .5)
        self.assertGreaterEqual(probability[1], probability[0])
        self.assertAlmostEqual(probability[0], 1. - probability[2])

    def test_terminal_coefficients_and_predictions_are_exactly_zero(self):
        model = self.fit(terminal=20.)
        np.testing.assert_array_equal(model["coefficients"].reshape(2, 4)[:, 2:], 0.)
        np.testing.assert_array_equal(wpresidual.predict(model, [[30., 7.], [-12., 8.]], [.3, .7],
                                                         [20., 70.]), [.3, .7])
        altered = copy.deepcopy(model)
        altered["coefficients"][-1] = .000001
        with self.assertRaisesRegex(ValueError, "Late coefficients"):
            wpresidual.validate(altered)

    def test_signed_effects_and_per_feature_bounds_are_preserved(self):
        model = self.fit(bounds=[[-10., 10.], [-10., 0.]])
        self.assertTrue(np.all(model["coefficients"].reshape(2, 4)[1] <= 0))
        with self.assertRaises(ValueError):
            self.fit(bounds=[[-10., 10.], [1., 10.]])

    def test_artifact_roundtrip_and_contract_mismatch_rejected(self):
        model = self.fit()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "residual.npz"
            wpresidual.save(model, path)
            loaded = wpresidual.load(path)
        wpresidual.validate(loaded, candidate_kind="test_v1", feature_names=["benefit", "uncertain"],
                            input_contract={"missing": "zero"})
        np.testing.assert_array_equal(loaded["coefficients"], model["coefficients"])
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            wpresidual.validate(loaded, candidate_kind="wrong_v1")

    def test_no_support_remains_baseline_with_unavailable_rows_in_population(self):
        model = wpresidual.fit(np.zeros((2, 3)), [.2, .8], [0, 1], [1, 2], [0., 5.],
                               feature_names=["a", "b", "c"], candidate_kind="empty_v1", input_contract={})
        self.assertEqual(model["training_rows"], 2)
        self.assertEqual(model["active_rows"], 0)
        np.testing.assert_array_equal(model["scale"], np.ones(3))
        np.testing.assert_array_equal(model["coefficients"], 0.)

    def test_invalid_input_and_bad_decay_contract_rejected(self):
        with self.assertRaises(ValueError):
            self.fit(terminal=15.)
        model = self.fit()
        for extra, p, t in (([[float("nan"), 0.]], [.5], [5.]),
                            ([[1.]], [.5], [5.]), ([[1., 0.]], [1.1], [5.]),
                            ([[1., 0.]], [.5], [-1.])):
            with self.subTest(extra=extra, p=p, t=t), self.assertRaises(ValueError):
                wpresidual.predict(model, extra, p, t)


if __name__ == "__main__":
    unittest.main()
