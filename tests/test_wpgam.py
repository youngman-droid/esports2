import os
import tempfile
import unittest

import numpy as np

from lol_ticker import wpgam, wpx


class TimeBasisTests(unittest.TestCase):
    def test_hat_basis_is_partition_of_unity(self):
        t = np.array([-5, 0, 5, 10, 17, 45, 80], dtype=float)
        b = wpgam.time_basis(t)
        self.assertTrue(np.all(b >= 0))
        np.testing.assert_allclose(b.sum(axis=1), 1.0)
        np.testing.assert_allclose(b[1], [1, 0, 0, 0, 0])
        np.testing.assert_allclose(b[-1], [0, 0, 0, 0, 1])

    def test_historical_interpolation_never_reads_future_minute(self):
        series = {0: 10.0, 1: 20.0, 2: 1000.0}
        self.assertEqual(wpx._interp(series, 119, float), 20.0)
        self.assertEqual(wpx._interp(series, 120, float), 1000.0)


class FeatureContractTests(unittest.TestCase):
    def test_historical_and_live_contract_match(self):
        state = {
            "t_min": 18.0, "gold_blue": 32000, "gold_red": 30500,
            "gold_diff_k": 1.5, "gold_diff_prev_k": 0.8,
            "gold_role": [0.6, -0.1, 0.5, 0.4, 0.1], "cs_diff_k": 0.7,
            "kills": 3, "towers": 2, "towers_blue": 7, "towers_red": 5,
            "dragons": 1, "barons": 1, "baron_active": -1,
            "elder_active": 1, "elder_buff": 1,
            "inhibs": 1, "elders": 0, "drag_blue": 3, "drag_red": 2,
            "inhib_blue": 1, "inhib_red": 0, "dead_blue": 1, "dead_red": 3,
            "items_done_diff": 2, "item_gold_diff_k": 1.2,
            "hp_pool": 0.9, "lvl_k": 0.6, "has_hp": 1,
            "elo_oe": 0.25, "pelo_oe": 0.1, "form_diff": 0.2,
        }
        old = wpx.live_vector(state, list(wpx.FEATURE_NAMES))[None, :]
        hist = wpgam.state_values_from_matrix(old, list(wpx.FEATURE_NAMES))[0]
        live = wpgam.state_values_from_live(state)
        np.testing.assert_allclose(hist, live)
        by_name = dict(zip(wpgam.STATE_FEATURES, live))
        self.assertEqual(by_name["dead_adv"], 2.0)
        self.assertEqual(by_name["baron_active"], -1.0)
        self.assertEqual(by_name["elder_active"], 1.0)
        self.assertEqual(by_name["dead_adv_sq"], 4.0)
        self.assertEqual(by_name["dead_count_sq_adv"], 8.0)
        self.assertEqual(by_name["dead_base_pressure"], 12.0)
        np.testing.assert_allclose(
            wpgam.pregame_values_from_matrix(old, list(wpx.FEATURE_NAMES))[0],
            wpgam.pregame_values_from_live(state),
        )


class ArtifactTests(unittest.TestCase):
    def _model(self):
        nchamp = 1
        nf = 2 + len(wpgam.STATE_FEATURES)
        theta = np.zeros((nf + 1, len(wpgam.TIME_KNOTS)))
        theta[1, :] = 1.0  # pregame team logit
        theta[2, :] = 1.0  # pregame champion logit
        theta[3, :] = 1.0  # gold advantage
        return {
            "pregame": {
                "mean": np.zeros(3), "std": np.ones(3),
                "lo": np.full(3, -10.0), "hi": np.full(3, 10.0),
                "intercept": 0.0, "team_beta": np.array([1.0, 0.0, 0.0]),
                "champ_beta": np.array([0.2]), "champ_names": np.array(["Ahri"]),
            },
            "state": {
                "theta": theta,
                "feature_names": np.array(["prior_team_logit", "prior_champ_logit"] + wpgam.STATE_FEATURES),
                "mean": np.zeros(nf), "std": np.ones(nf),
                "lo": np.full(nf, -10.0), "hi": np.full(nf, 10.0),
                "knots": wpgam.TIME_KNOTS,
            },
        }

    def test_artifact_round_trip_and_monotonic_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.npz")
            wpgam.save_model(self._model(), path, {"test": True})
            loaded = wpgam.load_model(path)
            self.assertTrue(loaded["meta"]["test"])
            base = {"t_min": 10, "gold_diff_k": 0, "gold_diff_prev_k": 0,
                    "gold_blue": 20000, "gold_red": 20000, "elo_oe": 0.1}
            p0 = wpgam.predict_live(base, path=path)["p_blue"]
            better = dict(base, gold_diff_k=1.0)
            p1 = wpgam.predict_live(better, path=path)["p_blue"]
            self.assertGreater(p1, p0)
            pc = wpgam.predict_live(base, blue_champs=["Ahri"], path=path)["p_blue"]
            self.assertGreater(pc, p0)

    def test_production_live_default_uses_blend_base_model(self):
        self.assertEqual(wpx.LIVE_MODEL_PATH, wpgam.MODEL_PATH)
        self.assertEqual(wpx.predict_live.__defaults__[-1], wpgam.MODEL_PATH)
        with np.load(wpx.LIVE_MODEL_PATH, allow_pickle=False) as artifact:
            self.assertEqual(str(artifact["kind"].item()), wpgam.MODEL_KIND)


class TemporalCalibrationTests(unittest.TestCase):
    def test_final_intercept_recenters_fixed_slope(self):
        logits = np.array([-1.0, -0.3, 0.2, 0.8, 1.4])
        y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        gids = np.arange(len(y))
        slope = 0.7
        intercept = wpgam._fit_calibration_intercept(logits, y, gids, slope)
        p = wpgam._sigmoid(intercept + slope * logits)
        self.assertAlmostEqual(float(p.mean()), float(y.mean()), places=8)

    def test_fit_arrays_uses_earlier_date_calibration_block(self):
        n_games, states_per_game = 150, 2
        n = n_games * states_per_game
        names = list(wpx.FEATURE_NAMES)
        idx = {name: i for i, name in enumerate(names)}
        game_y = (np.arange(n_games) % 2).astype(float)
        y = np.repeat(game_y, states_per_game)
        gid = np.repeat(np.arange(n_games), states_per_game)
        X = np.zeros((n, len(names)), dtype=float)
        X[:, idx["bias"]] = 1.0
        direction = 2.0 * y - 1.0
        X[:, idx["elo_oe"]] = 0.1 * direction
        X[:, idx["gold_k"]] = direction * np.tile([0.2, 1.0], n_games)
        t_s = np.tile([600, 1200], n_games)
        seq = np.full(n, -1)
        C = np.full((n, 10), -1, dtype=np.int16)
        game_dates = np.array([
            str(np.datetime64("2025-01-01") + np.timedelta64(i, "D"))
            for i in range(n_games)
        ])
        dates = np.repeat(game_dates, states_per_game)
        model = wpgam.fit_arrays(
            X, y, gid, t_s, seq, C, names, [], dates=dates,
            pregame_folds=3)
        self.assertEqual(model["state"]["calibration_method"],
                         "temporal_holdout")
        self.assertGreater(model["state"]["cal_slope"], 0.0)
        self.assertTrue(np.isfinite(model["state"]["theta"]).all())


if __name__ == "__main__":
    unittest.main()
