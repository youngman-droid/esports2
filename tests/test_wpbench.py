import unittest
from unittest import mock
import os
import tempfile

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


class NestedCalibrationTests(unittest.TestCase):
    def test_entire_stack_is_fit_strictly_before_calibration(self):
        dates = np.array(["2026-01-%02d" % n for n in range(1, 21)])
        base = {"game_dates": dates, "row_game": np.repeat(np.arange(20), 2)}
        def stack(_base, games):
            return {"train": games[base["row_game"]]}
        with mock.patch.object(wpbench, "_stacked_arrays", side_effect=stack) as fit:
            result, calibration, provenance = wpbench._nested_calibration_stack(
                base, np.ones(20, dtype=bool))
        trained_games = fit.call_args.args[1]
        calibrated_games = np.unique(base["row_game"][calibration])
        self.assertLess(max(dates[trained_games]), min(dates[calibrated_games]))
        self.assertFalse(np.any(result["train"] & calibration))
        self.assertLess(provenance["fit_end"], provenance["calibration_start"])

    def test_benchmark_never_recalibrates_on_fit_or_test_outcomes(self):
        dates = np.array(["2026-01-%02d" % n for n in range(1, 21)])
        game_y = np.arange(20) % 2
        row_game = np.repeat(np.arange(20), 10)
        outer = np.arange(20) < 16
        inner, validation, cutoff = wpbench._date_blocks(dates, outer)
        base = dict(game_gid=np.arange(1, 21), game_dates=dates, row_game=row_game,
            gid=(row_game + 1), y=game_y[row_game].astype(float),
            t_min=np.tile(np.arange(10), 20), outer_train=outer,
            inner_train=inner, validation=validation, validation_cutoff=cutoff,
            split_method="date")
        seen_calibration = []
        def stack(_base, games):
            return {"raw": np.column_stack([row_game / 50., -row_game / 100.]),
                    "train": games[row_game]}
        def calibration(p, y, gids):
            seen_calibration.append(set(gids.tolist()))
            return {"intercept": float(y.mean()) / 10, "slope": .9, "ece10": 0.0}
        def summary(p, y, gids, t_min, **kwargs):
            return wpbench._basic_metrics(p, y, gids)
        with tempfile.TemporaryDirectory() as td:
            dataset = os.path.join(td, "states.npz")
            with open(dataset, "wb") as fh:
                fh.write(b"fixture")
            with mock.patch.object(wpbench, "_load_base", return_value=base), \
                    mock.patch.object(wpbench, "_stacked_arrays", side_effect=stack), \
                    mock.patch.object(wpbench, "_fit_method", return_value={}), \
                    mock.patch.object(wpbench, "_predict_method", side_effect=lambda model, raw, t: .5 + raw[:, 0] / 10), \
                    mock.patch.object(wpbench, "_calibration_diagnostics", side_effect=calibration), \
                    mock.patch.object(wpbench, "_summary", side_effect=summary), \
                    mock.patch.object(wpbench.wpgam, "_fit_calibration_intercept", side_effect=AssertionError("training intercept forbidden")):
                result = wpbench.run(dataset, os.path.join(td, "result.json"),
                    bootstrap=10, quick=True, exposure_path=os.path.join(td, "exposure.json"))
            fit1, cal1, _ = wpbench._date_blocks(dates, inner)
            fit2, cal2, _ = wpbench._date_blocks(dates, outer)
            allowed = [set(np.where(cal1)[0] + 1), set(np.where(cal2)[0] + 1)]
            self.assertTrue(all(gids in allowed for gids in seen_calibration))
            self.assertEqual(len(seen_calibration), 12)  # five methods + prior, twice
            self.assertEqual(result["holdout_status"], "consumed_development_only")


if __name__ == "__main__":
    unittest.main()
