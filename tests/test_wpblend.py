import json
import os
import tempfile
import unittest

import numpy as np

from lol_ticker import wpblend, wpgam


class BlendFitTests(unittest.TestCase):
    def test_probability_blend_prefers_better_model(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        data = wpblend._data(
            market=[0.8, 0.7, 0.3, 0.2],
            model=[0.1, 0.2, 0.8, 0.9],
            y=y, gids=np.arange(4), t_min=np.full(4, 10.0),
            dates=["2026-01-01"] * 4)
        fitted = wpblend._fit_probability_blend(data, {})
        pred = wpblend._predict_candidate(fitted, data)
        self.assertLess(fitted["market_weight"], 0.01)
        self.assertLess(np.mean((pred - y) ** 2),
                        np.mean((data["market"] - y) ** 2))

    def test_chronological_split_keeps_games_whole(self):
        gids = np.repeat(np.arange(6), 3)
        dates = np.repeat([
            "2026-01-01", "2026-01-02", "2026-01-03",
            "2026-01-04", "2026-01-05", "2026-01-06",
        ], 3)
        data = wpblend._data(
            np.full(18, 0.5), np.full(18, 0.5), np.tile([0, 1, 1], 6),
            gids, np.tile([5.0, 10.0, 15.0], 6), dates)
        train, test, cutoff = wpblend._chron_split(data, holdout=1 / 3)
        self.assertEqual(cutoff, "2026-01-05")
        self.assertLess(sorted(train["date"])[-1], sorted(test["date"])[0])
        self.assertFalse(set(train["gid"]) & set(test["gid"]))

    def test_time_logit_prediction_is_finite_and_bounded(self):
        data = wpblend._data(
            market=[0.01, 0.4, 0.99], model=[0.2, 0.6, 0.8], y=[0, 1, 1],
            gids=[1, 2, 3], t_min=[1.0, 18.0, 50.0], dates=[""] * 3)
        model = {
            "family": "time_logit_residual",
            "intercepts": np.zeros(len(wpblend.TIME_KNOTS)),
            "weights": np.full(len(wpblend.TIME_KNOTS), 0.5),
            "knots": wpblend.TIME_KNOTS,
        }
        pred = wpblend._predict_candidate(model, data)
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue(np.all((pred > 0.0) & (pred < 1.0)))

    def test_logit_stack_can_optimize_brier_directly(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        data = wpblend._data(
            market=[0.35, 0.45, 0.55, 0.65],
            model=[0.10, 0.20, 0.80, 0.90], y=y, gids=np.arange(4),
            t_min=np.full(4, 10.0), dates=[""] * 4)
        fitted = wpblend._fit_logit_stack(data, {"l2": 0.1, "loss": "brier"})
        pred = wpblend._predict_candidate(fitted, data)
        self.assertTrue(np.isfinite(fitted["beta"]).all())
        self.assertLess(np.mean((pred - y) ** 2), 0.1)


class LiveBlendTests(unittest.TestCase):
    def _artifact(self, both):
        return {
            "kind": "wpblend_live_v1",
            "base_model_kind": wpgam.MODEL_KIND,
            "market_lead_s": 45,
            "polymarket": {"family": "logit_stack", "beta": [0.0, 1.0, 0.5]},
            "kalshi": {"family": "logit_stack", "beta": [0.0, 0.5, 0.5]},
            "both": both,
        }

    def _predict(self, artifact, after_draft=False):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blend.json")
            with open(path, "w") as fh:
                json.dump(artifact, fh)
            return wpblend.predict_live(
                0.55, 20.0, {"polymarket": 0.60, "kalshi": 0.58}, path=path,
                after_draft=after_draft)

    def test_live_artifact_can_select_best_two_way_blend(self):
        out = self._predict(self._artifact({
            "family": "logit_stack", "beta": [0.0, 0.5, 0.5],
            "uses": "kalshi", "source_order": ["model", "polymarket", "kalshi"],
        }))
        self.assertEqual(out["source"], "model+kalshi")
        self.assertAlmostEqual(out["p_blue"], out["platforms"]["kalshi"])
        self.assertEqual(out["market_lead_s"], 45)

    def test_live_artifact_can_use_three_way_stack(self):
        beta = [0.1, 0.4, 0.3, 0.5]
        out = self._predict(self._artifact({
            "family": "three_way_stack", "beta": beta, "uses": "both",
            "source_order": ["model", "polymarket", "kalshi"],
        }))
        expected = wpgam._sigmoid(
            beta[0] + beta[1] * wpblend._logit(0.55)
            + beta[2] * wpblend._logit(0.60)
            + beta[3] * wpblend._logit(0.58))
        self.assertEqual(out["source"], "model+polymarket+kalshi")
        self.assertAlmostEqual(out["p_blue"], float(expected))

    def test_live_artifact_has_separate_post_draft_fit(self):
        artifact = self._artifact({
            "family": "logit_stack", "beta": [0.0, 0.5, 0.5],
            "uses": "kalshi", "source_order": ["model", "polymarket", "kalshi"],
        })
        artifact["after_draft_market_offset_s"] = 120
        artifact["after_draft"] = {
            "polymarket": {"family": "market"},
            "kalshi": {"family": "model"},
            "both": {"family": "model", "uses": "kalshi",
                     "source_order": ["model", "polymarket", "kalshi"]},
        }
        out = self._predict(artifact, after_draft=True)
        self.assertTrue(out["after_draft"])
        self.assertEqual(out["market_lead_s"], 120)
        self.assertAlmostEqual(out["platforms"]["polymarket"], 0.60)
        self.assertAlmostEqual(out["platforms"]["kalshi"], 0.55)
        self.assertEqual(out["source"], "model")


if __name__ == "__main__":
    unittest.main()
