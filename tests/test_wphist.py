import inspect
import os
import tempfile
import unittest

import numpy as np

from lol_ticker import wpgam, wphist


class HistoricalOddsTests(unittest.TestCase):
    @staticmethod
    def _base_model():
        npre = len(wpgam.PREGAME_FEATURES)
        nf = 2 + len(wpgam.STATE_FEATURES)
        team_beta = np.zeros(npre)
        team_beta[0] = 1.0
        return {
            "pregame": {
                "mean": np.zeros(npre), "std": np.ones(npre),
                "lo": np.full(npre, -10.0), "hi": np.full(npre, 10.0),
                "intercept": 0.0, "team_beta": team_beta,
                "champ_beta": np.zeros(0), "champ_names": np.array([], dtype=str),
            },
            "state": {
                "theta": np.zeros((nf + 1, len(wpgam.TIME_KNOTS))),
                "feature_names": np.array(
                    ["prior_team_logit", "prior_champ_logit"] + wpgam.STATE_FEATURES),
                "mean": np.zeros(nf), "std": np.ones(nf),
                "lo": np.full(nf, -10.0), "hi": np.full(nf, 10.0),
                "knots": wpgam.TIME_KNOTS,
            },
        }

    @staticmethod
    def _teacher():
        nf = 2 + len(wpgam.STATE_FEATURES)
        theta = np.zeros((nf + 1, len(wpgam.TIME_KNOTS)))
        theta[0, :] = 0.4
        return {
            "theta": theta,
            "feature_names": np.array(
                ["prior_team_logit", "prior_champ_logit"] + wpgam.STATE_FEATURES),
            "mean": np.zeros(nf), "std": np.ones(nf),
            "lo": np.full(nf, -10.0), "hi": np.full(nf, 10.0),
            "knots": wpgam.TIME_KNOTS,
        }

    def test_live_contract_has_no_market_argument(self):
        self.assertNotIn("markets", inspect.signature(wphist.predict_live).parameters)
        with tempfile.TemporaryDirectory() as td:
            base_path = os.path.join(td, "base.npz")
            artifact_path = os.path.join(td, "historical.npz")
            wpgam.save_model(self._base_model(), base_path, {"test": True})
            wphist._save_artifact(
                self._teacher(), 0.5, True, {"uses_live_odds": False},
                artifact_path, base_path)
            out = wphist.predict_live(
                0.55, {"t_min": 12.0, "elo_oe": 0.1},
                path=artifact_path, base_path=base_path)
            self.assertFalse(out["uses_live_odds"])
            self.assertEqual(out["source"], "model+historical-odds")
            self.assertGreater(out["p_blue"], 0.55)

    def test_outcome_gate_can_reject_historical_teacher(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = np.array([0.1, 0.2, 0.8, 0.9])
        worse_teacher = 1.0 - model
        alpha = wphist._fit_mix_weight(model, worse_teacher, y, np.arange(4))
        self.assertLess(alpha, 1e-4)


if __name__ == "__main__":
    unittest.main()
