import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from lol_ticker import wpaudit, wpgam, wpx


class TimeBasisTests(unittest.TestCase):
    def test_hat_basis_is_partition_of_unity(self):
        t = np.array([-5, 0, 5, 10, 17, 45, 80], dtype=float)
        b = wpgam.time_basis(t)
        nk = len(wpgam.TIME_KNOTS)
        self.assertTrue(np.all(b >= 0))
        np.testing.assert_allclose(b.sum(axis=1), 1.0)
        np.testing.assert_allclose(b[1], np.eye(1, nk, 0)[0])
        np.testing.assert_allclose(b[-1], np.eye(1, nk, nk - 1)[0])

    def test_historical_interpolation_never_reads_future_minute(self):
        series = {0: 10.0, 1: 20.0, 2: 1000.0}
        self.assertEqual(wpx._interp(series, 119, float), 20.0)
        self.assertEqual(wpx._interp(series, 120, float), 1000.0)

    def test_historical_hp_observation_carries_exact_deaths(self):
        hpmap = {10: {"_clock_s": 600, "data": {
            "hpb": [0.0, 0.0, 0.5, 1.0, 1.0],
            "hpr": [0.0, 0.4, 0.8, 1.0, 1.0],
            "lvb": [10] * 5, "lvr": [9] * 5,
        }}}
        got = wpx._hp_observation(hpmap, 620)
        self.assertEqual(got["dead_blue"], 2)
        self.assertEqual(got["dead_red"], 1)
        self.assertEqual(got["features"][-1], 1.0)

    def test_rolling_date_masks_are_strict_and_keep_dates_whole(self):
        dates = np.array(["2026-01-%02d" % day for day in range(1, 11)
                          for _ in range(2)])
        folds = wpgam._rolling_date_masks(
            dates, windows=2, min_train_fraction=0.5)
        self.assertEqual(len(folds), 2)
        for train, test, start, end in folds:
            self.assertFalse(np.any(train & test))
            self.assertTrue(np.all(dates[train] < start))
            self.assertTrue(np.all((dates[test] >= start) & (dates[test] <= end)))
            for date in np.unique(dates):
                idx = dates == date
                self.assertTrue(train[idx].all() or test[idx].all()
                                or (~train[idx] & ~test[idx]).all())


class FeatureContractTests(unittest.TestCase):
    def test_rare_discrete_events_survive_scaling_and_fit(self):
        names = wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES
        raw = np.zeros((2000, len(names)))
        y = (np.arange(2000) % 2).astype(float)
        for name in ("d_elder", "elder_active"):
            raw[-4:, names.index(name)] = [-1, 1, -1, 1]
        model = wpgam.fit_state_model(raw, y, np.arange(len(y)),
                                      np.full(len(y), 35.0))
        probe = np.zeros((3, len(names)))
        for name in ("d_elder", "elder_active"):
            j = names.index(name)
            self.assertLess(model["lo"][j], 0)
            self.assertGreater(model["hi"][j], 0)
            probe[:, j] = [-1, 0, 1]
        p = wpgam.predict_state(model, probe, [35, 35, 35])
        self.assertTrue(np.all(np.diff(p) > 0.001))

    def test_no_observed_variation_is_erased_by_percentile_clipping(self):
        raw = np.zeros((1000, 1)); raw[-1] = 3.0
        mean, std, lo, hi = wpgam._scale_fit(raw)
        self.assertGreater(hi[0], lo[0])
        self.assertTrue(np.isfinite(wpgam._scale_apply(raw, mean, std, lo, hi)).all())

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
            "elo_gg": 0.3, "t_since_kill_min": 3.5,
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
        self.assertAlmostEqual(by_name["gold_rel"], 1.5 / 62.5)
        self.assertEqual(by_name["t_since_kill"], 3.5)
        np.testing.assert_allclose(
            wpgam.pregame_values_from_matrix(old, list(wpx.FEATURE_NAMES))[0],
            wpgam.pregame_values_from_live(state),
        )

    def test_missing_golgg_elo_falls_back_to_oe_elo(self):
        pre = wpgam.pregame_values_from_live({"elo_oe": 0.25, "elo_gg": None})
        by_name = dict(zip(wpgam.PREGAME_FEATURES, pre))
        self.assertEqual(by_name["elo_gg"], 0.25)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        # These tests exercise the incumbent/legacy stack independently of
        # whichever production bundle is active on the developer's machine.
        pointer = mock.patch("lol_ticker.wpcombined_prod.load_active", return_value=None)
        pointer.start()
        self.addCleanup(pointer.stop)

    def _model(self):
        npre = len(wpgam.PREGAME_FEATURES)
        n_inputs = len(wpgam.PRIOR_INPUTS)
        nf = n_inputs + len(wpgam.STATE_FEATURES)
        theta = np.zeros((nf + 1, len(wpgam.TIME_KNOTS)))
        theta[1, :] = 1.0  # pregame team logit
        theta[2, :] = 1.0  # pregame champion logit
        theta[3, :] = 1.0  # champion state score
        theta[1 + n_inputs, :] = 1.0  # gold advantage
        team_beta = np.zeros(npre)
        team_beta[0] = 1.0
        return {
            "pregame": {
                "mean": np.zeros(npre), "std": np.ones(npre),
                "lo": np.full(npre, -10.0), "hi": np.full(npre, 10.0),
                "intercept": 0.0, "team_beta": team_beta,
                "champ_beta": np.array([0.2]), "champ_names": np.array(["Ahri"]),
            },
            "state": {
                "theta": theta,
                "feature_names": np.array(wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES),
                "mean": np.zeros(nf), "std": np.ones(nf),
                "lo": np.full(nf, -10.0), "hi": np.full(nf, 10.0),
                "knots": wpgam.TIME_KNOTS,
            },
            "champ_state": {"beta": np.array([0.3]), "cap_min": 15.0, "l2": 800.0},
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
            out = wpgam.predict_live(base, blue_champs=["Ahri"], path=path)
            self.assertGreater(out["p_blue"], p0)
            # the in-game champion channel contributes on top of the pregame one
            self.assertGreater(out["lo_champ_state"], 0.0)
            enemy = wpgam.predict_live(base, red_champs=["Ahri"], path=path)
            self.assertLess(enemy["lo_champ_state"], 0.0)
            self.assertEqual(out["reliability"], "reduced")
            self.assertIn("hp_unavailable", out["input_warnings"])

    def test_player_gold_is_monotone_at_clipping_boundaries(self):
        rng = np.random.default_rng(904)
        model = self._model()
        state = model["state"]
        state.update(cal_slope=0.9, cal_intercept=0.)
        for i, name in enumerate(state["feature_names"]):
            if name in wpgam.MONOTONE_FEATURES:
                state["theta"][i+1] = rng.uniform(0, 2, len(wpgam.TIME_KNOTS))
        j = list(state["feature_names"]).index("gold_rel")
        state["lo"][j], state["hi"][j] = -.13, .14
        raw = rng.normal(size=(500, len(state["feature_names"])))
        gold_j = list(state["feature_names"]).index("gold_k")
        total = np.full(len(raw), 10.)
        raw[:, j] = raw[:, gold_j] / total
        audit = wpaudit.audit_gold(state, raw, rng.uniform(0, 80, len(raw)), total)
        self.assertTrue(audit["passed"], audit)
        # The recorded red-support failure: other roles must stay constant.
        state_input = dict(t_min=2., gold_blue=2865., gold_red=4084.,
                           gold_diff_k=-1.219, gold_diff_prev_k=0.,
                           gold_role=[-.318, -.172, -.096, .094, -.727])
        model["kind"] = wpgam.MODEL_KIND
        before = wpgam.predict_live_model(model, state_input, rounded=False)["p_blue"]
        changed = dict(state_input, gold_red=5084., gold_diff_k=-2.219,
                       gold_role=[-.318, -.172, -.096, .094, -1.727])
        after = wpgam.predict_live_model(model, changed, rounded=False)["p_blue"]
        self.assertLessEqual(after, before)

    def test_old_artifact_keeps_its_feature_semantics_and_kind(self):
        model = wpgam.load_model()
        if model["kind"] != wpgam.LEGACY_MODEL_KIND:
            self.skipTest("v8 deployment has been replaced")
        state = dict(t_min=35., gold_blue=55000., gold_red=55000.,
                     gold_diff_k=0., has_hp=1., drag_blue=4, drag_red=4)
        out = wpgam.predict_live_model(model, state)
        self.assertEqual(out["model_kind"], wpgam.LEGACY_MODEL_KIND)
        raw = wpgam.state_values_from_live(state, wpgam.LEGACY_STATE_FEATURES)
        self.assertEqual(len(raw), 25)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.npz")
            wpgam.save_model(model, path)
            restored = wpgam.load_model(path)
            self.assertEqual(restored["kind"], model["kind"])
            self.assertEqual(wpgam.predict_live(state, path=path)["p_blue"], out["p_blue"])

    def test_series_score_alone_does_not_count_as_a_team_prior(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.npz")
            wpgam.save_model(self._model(), path, {})
            state = {"t_min": 10, "gold_diff_k": 0, "gold_blue": 20000,
                     "gold_red": 20000, "series_diff": 0.0}
            out = wpgam.predict_live(state, path=path)
            self.assertIn("pregame_priors_unavailable", out["input_warnings"])
            rated = wpgam.predict_live(dict(state, elo_gg=0.1), path=path)
            self.assertNotIn("pregame_priors_unavailable", rated["input_warnings"])

    def test_production_live_default_uses_blend_base_model(self):
        self.assertEqual(wpx.LIVE_MODEL_PATH, wpgam.MODEL_PATH)
        self.assertEqual(wpx.predict_live.__defaults__[-2], wpgam.MODEL_PATH)
        with np.load(wpx.LIVE_MODEL_PATH, allow_pickle=False) as artifact:
            self.assertIn(str(artifact["kind"].item()), wpgam.SUPPORTED_MODEL_KINDS)

    def test_production_live_blend_lies_between_components(self):
        state = {"t_min": 22.0, "gold_diff_k": 2.5, "gold_diff_prev_k": 2.0,
                 "gold_blue": 42000, "gold_red": 39500, "kills": 4, "towers": 2,
                 "towers_blue": 5, "towers_red": 3, "elo_oe": 0.2}
        with mock.patch.object(
                wpx, "load_live_stack",
                return_value={"deployed": True, "w_gam": 0.45,
                              "stack_sha256": "a" * 64}), \
                mock.patch.object(
                    wpx, "_predict_live_legacy",
                    return_value={"p_blue": 0.65, "unknown_champions": []}):
            out = wpx.predict_live(state)
        self.assertIn("p_gam", out)
        self.assertIn("p_legacy", out)
        lo = min(out["p_gam"], out["p_legacy"]) - 1e-9
        hi = max(out["p_gam"], out["p_legacy"]) + 1e-9
        self.assertTrue(lo <= out["p_blue"] <= hi)
        self.assertEqual(out["blend_w_gam"], 0.45)
        gam_only = wpx.predict_live(state, blend=False)
        self.assertEqual(gam_only["p_blue"], out["p_gam"])

    def test_production_falls_back_to_gam_without_promoted_stack(self):
        with mock.patch.object(
                wpx, "load_live_stack",
                return_value={"deployed": False, "w_gam": 1.0,
                              "reason": "gate rejected"}):
            out = wpx.predict_live({"t_min": 10.0})
        self.assertNotIn("p_legacy", out)
        self.assertEqual(out["stack_reason"], "gate rejected")

    def test_production_never_substitutes_legacy_when_gam_is_missing(self):
        with mock.patch.object(wpx.os.path, "exists", return_value=False), \
                mock.patch.object(wpx, "_predict_live_legacy") as legacy:
            with self.assertRaises(FileNotFoundError):
                wpx.predict_live({"t_min": 10.0})
        legacy.assert_not_called()


class TemporalCalibrationTests(unittest.TestCase):
    def test_insufficient_temporal_history_leaves_calibration_at_identity(self):
        n = 12
        names = list(wpx.FEATURE_NAMES)
        X = np.zeros((n, len(names)))
        X[:, names.index("bias")] = 1.
        X[:, names.index("t")] = .5
        y = (np.arange(n) % 2).astype(float)
        C = np.full((n, 10), -1, dtype=int)
        dates = np.array([str(np.datetime64("2025-01-01") + np.timedelta64(i, "D"))
                          for i in range(n)])
        for history in (None, dates):
            with self.subTest(dated=history is not None):
                result = wpgam.fit_arrays(X, y, np.arange(n), np.full(n, 900.),
                    np.full(n, -1), C, names, [], dates=history,
                    pregame_folds=2, champ_state_folds=2)
            self.assertEqual(result["state"]["cal_intercept"], 0.)
            self.assertEqual(result["state"]["cal_slope"], 1.)
            self.assertEqual(result["state"]["calibration_method"],
                             "identity_insufficient_temporal_data")

    def test_stacked_folds_are_past_only_and_cover_complete_dates(self):
        dates = np.repeat(np.array([str(np.datetime64("2025-01-01") +
                                       np.timedelta64(i, "D")) for i in range(80)]), 3)
        coverage = np.zeros(len(dates), dtype=int)
        for train, test in wpgam._stack_folds(np.arange(len(dates)), dates=dates):
            coverage += test
            if train.any() and test.any():
                self.assertLess(max(dates[train]), min(dates[test]))
        np.testing.assert_array_equal(coverage, 1)

    def test_calibration_labels_cannot_change_its_input_logits(self):
        rng = np.random.default_rng(18)
        n = 150
        names = list(wpx.FEATURE_NAMES)
        X = np.zeros((n, len(names)))
        X[:, names.index("bias")] = 1.
        X[:, names.index("t")] = .5
        X[:, names.index("gold_k")] = rng.normal(size=n)
        C = np.full((n, 10), -1, dtype=int)
        C[:, 0] = np.arange(n) % 2
        y = rng.integers(0, 2, size=n).astype(float)
        dates = np.array([str(np.datetime64("2025-01-01") + np.timedelta64(i, "D"))
                          for i in range(n)])
        logits = []
        def calibrate(z, *_args):
            logits.append(z.copy())
            return .17, .91
        for labels in (y, np.r_[y[:120], 1 - y[120:]]):
            with mock.patch.object(wpgam, "_fit_platt_logits", side_effect=calibrate):
                result = wpgam.fit_arrays(X, labels, np.arange(n), np.full(n, 900.),
                    np.full(n, -1), C, names, ["Ahri", "Ashe"], dates=dates,
                    pregame_folds=2, champ_state_folds=2)
            self.assertEqual(result["state"]["cal_intercept"], .17)
        self.assertEqual(len(logits), 2)
        np.testing.assert_allclose(logits[0], logits[1], atol=1e-10, rtol=0)

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
