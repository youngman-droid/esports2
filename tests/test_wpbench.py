import unittest

import numpy as np

from lol_ticker import wpbench


class DateBlockTests(unittest.TestCase):
    def test_complete_dates_stay_in_one_block(self):
        dates = np.array(["2026-01-01"] * 3 + ["2026-01-02"] * 3 +
                         ["2026-01-03"] * 3 + ["2026-01-04"] * 3)
        outer = np.ones(len(dates), dtype=bool)
        train, validation, cutoff = wpbench._date_blocks(
            dates, outer, validation_fraction=0.3)
        self.assertEqual(cutoff, "2026-01-03")
        self.assertFalse(np.any(train & validation))
        self.assertTrue(np.all(train | validation))
        for date in np.unique(dates):
            idx = dates == date
            self.assertTrue(train[idx].all() or validation[idx].all())


class CalibrationTests(unittest.TestCase):
    def test_identity_platt_transform(self):
        p = np.array([0.1, 0.4, 0.8])
        got = wpbench._apply_platt(p, {"intercept": 0.0, "slope": 1.0})
        np.testing.assert_allclose(got, p)

    def test_convex_blend_prefers_better_predictions(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        gids = np.arange(4)
        predictions = {
            "good": np.array([0.1, 0.2, 0.8, 0.9]),
            "bad": np.array([0.8, 0.7, 0.3, 0.2]),
        }
        weights = wpbench._fit_blend(predictions, y, gids, ["good", "bad"])
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["good"], 0.99)


class RidgeMethodTests(unittest.TestCase):
    def test_ridge_method_returns_finite_probabilities(self):
        rng = np.random.default_rng(4)
        n = 80
        raw = rng.normal(size=(n, 20))
        y = (raw[:, 2] > 0).astype(float)
        gids = np.repeat(np.arange(20), 4)
        t_min = np.tile([5.0, 10.0, 20.0, 30.0], 20)
        model = wpbench._fit_method(
            "ridge_logit", {"C": 0.1}, raw, y, gids, t_min)
        p = wpbench._predict_method(model, raw, t_min)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.all((p > 0.0) & (p < 1.0)))
        self.assertLess(np.mean((p - y) ** 2), 0.25)


if __name__ == "__main__":
    unittest.main()
