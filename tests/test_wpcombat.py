import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from lol_ticker import wpcombat as combat, wpgam


class CombatObservationTests(unittest.TestCase):
    def fixture(self):
        return dict(hpb=[1.] * 5, hpr=[1.] * 5,
                    gdb=[8000, 9000, 11000, 14000, 5000],
                    gdr=[7000, 9000, 10000, 12000, 4000],
                    lvb=[13, 12, 14, 14, 10], lvr=[12, 12, 13, 13, 10])

    def live_fixture(self):
        md, frame = {}, {}
        for side, offset in (("blue", 0), ("red", 5)):
            md[side + "TeamMetadata"] = dict(participantMetadata=[
                dict(participantId=i + offset + 1, role=role)
                for i, role in enumerate(("top", "jungle", "mid", "bottom", "support"))])
            frame[side + "Team"] = dict(participants=[
                dict(participantId=i + offset + 1, currentHealth=100 * (i + 1),
                     maxHealth=1000, totalGold=4000 + (i + offset) * 1000, level=10 + i)
                for i in range(5)])
        return md, frame

    def test_dead_adc_and_support_keep_role_and_living_resource_identity(self):
        adc, support = self.fixture(), self.fixture()
        adc["hpb"][3] = 0.
        support["hpb"][4] = 0.
        a, s = (combat.observation_features(data) for data in (adc, support))
        self.assertTrue(a["available"])
        by_name = dict(zip(a["names"], a["features"]))
        self.assertEqual(by_name["alive_bot"], -1.)
        self.assertEqual(by_name["hp_bot"], -1.)
        self.assertEqual(by_name["living_gold_bot_k"], -12.)
        self.assertEqual(by_name["living_level_bot"], -13.)
        self.assertEqual(a["raw"]["alive"], [1., 1., 1., 0., 1.] + [1.] * 5)
        self.assertNotEqual(a["features"], s["features"])
        self.assertEqual(sum(a["features"][:5]), sum(s["features"][:5]))

    def test_side_swap_negates_every_feature_and_preserves_known_masks(self):
        data = self.fixture()
        data["hpb"] = [0., .2, .5, .8, 1.]
        data["hpr"] = [.5, 0., .7, .8, .9]
        reverse = {key[:-1] + ("r" if key[-1] == "b" else "b"): value for key, value in data.items()}
        left, right = map(combat.observation_features, (data, reverse))
        assert_array_equal(left["features"], -np.asarray(right["features"]))
        self.assertEqual(left["known"], right["known"])

    def test_optional_resources_are_independently_gated(self):
        data = self.fixture()
        data["hpb"][3] = 0.
        data.pop("gdb")
        obs = combat.observation_features(data)
        self.assertTrue(obs["available"])
        self.assertEqual(obs["known"], [True] * 10 + [False] * 5 + [True] * 5)
        self.assertEqual(obs["features"][10:15], [0.] * 5)
        self.assertEqual(obs["features"][3], -1.)
        data["lvb"][0] = 13.5
        obs = combat.observation_features(data)
        self.assertEqual(obs["known"], [True] * 10 + [False] * 10)
        self.assertEqual(obs["features"][15:], [0.] * 5)
        self.assertEqual(obs["excluded_inputs"], ["items"])

    def test_incomplete_or_invalid_health_never_becomes_full_health(self):
        for value in (None, [], [0.] * 4, [0., 0., 0., 0., None],
                      [0., 0., 0., 0., np.nan], [0., 0., 0., 0., np.inf],
                      [0., 0., 0., 0., -1.], [0., 0., 0., 0., 1.01],
                      [0., 0., 0., 0., True], np.array(1.), np.ones((5, 1))):
            data = self.fixture()
            data["hpb"] = value
            obs = combat.observation_features(data)
            self.assertFalse(obs["available"], str(value))
            self.assertEqual(obs["features"], [0.] * 20)
            self.assertEqual(obs["known"], [False] * 20)
            json.dumps(obs, allow_nan=False)

    def test_age_gate_is_metadata_only_and_inclusive(self):
        data = self.fixture()
        early, last = [combat.observation_features(data, age_upper_s=age) for age in (0., 90.)]
        self.assertEqual(early["features"], last["features"])
        self.assertEqual(last["age_upper_s"], 90.)
        for age in (90.001, -1., None, float("nan"), float("inf")):
            obs = combat.observation_features(data, age_upper_s=age)
            self.assertFalse(obs["available"])
            self.assertEqual(obs["features"], [0.] * 20)
            json.dumps(obs, allow_nan=False)
        with self.assertRaises(ValueError):
            combat.observation_features(data, max_age_s=-1.)

    def test_live_participant_and_metadata_permutations_join_by_role_and_id(self):
        md, frame = self.live_fixture()
        # Metadata can also assign roles to different valid side IDs.
        perm = [2, 0, 4, 1, 3]
        for i, participant in enumerate(md["blueTeamMetadata"]["participantMetadata"]):
            participant["participantId"] = perm[i] + 1
        md["blueTeamMetadata"]["participantMetadata"].reverse()
        frame["blueTeam"]["participants"].reverse()
        frame["redTeam"]["participants"] = [frame["redTeam"]["participants"][i] for i in perm]
        out = combat.live_observation(md, frame)
        self.assertTrue(out["available"])
        self.assertEqual(out["pidb"], [3, 1, 5, 2, 4])
        self.assertEqual(out["pidr"], [6, 7, 8, 9, 10])
        assert_allclose(out["hpb"], [.3, .1, .5, .2, .4], atol=0.)
        self.assertEqual(out["gdb"], [6000., 4000., 8000., 5000., 7000.])
        self.assertTrue(combat.observation_features(out)["available"])
        json.dumps(out, allow_nan=False)

    def test_live_wrong_side_duplicate_ids_and_bad_roles_reject(self):
        cases = [
            ("frame", "participantId", 6),
            ("frame", "participantId", 2),
            ("metadata", "participantId", 6),
            ("metadata", "participantId", 2),
            ("metadata", "role", "support"),
            ("metadata", "role", "unknown"),
            ("metadata", "role", None),
            ("frame", "currentHealth", -1),
            ("frame", "currentHealth", 1001),
            ("frame", "currentHealth", True),
            ("frame", "maxHealth", 0),
        ]
        for target, key, value in cases:
            md, frame = self.live_fixture()
            participant = (md["blueTeamMetadata"]["participantMetadata"][0] if target == "metadata"
                           else frame["blueTeam"]["participants"][0])
            participant[key] = value
            obs = combat.live_observation(md, frame)
            self.assertFalse(obs["available"], (target, key, value))
            self.assertEqual(combat.observation_features(obs)["features"], [0.] * 20)

    def test_live_optional_missing_gold_preserves_health_and_levels(self):
        md, frame = self.live_fixture()
        frame["blueTeam"]["participants"][0].pop("totalGold")
        out = combat.observation_features(combat.live_observation(md, frame))
        self.assertTrue(out["available"])
        self.assertEqual(out["known"], [True] * 10 + [False] * 5 + [True] * 5)


class CombatResidualTests(unittest.TestCase):
    def model(self):
        return dict(kind=combat.KIND, feature_names=list(combat.FEATURE_NAMES),
                    time_knots=wpgam.TIME_KNOTS.tolist(), input_contract=dict(combat.INPUT_CONTRACT),
                    spec=dict(l2=800., smooth=70.), scale=np.ones(20),
                    coefficients=np.linspace(.01, .2, 20 * len(wpgam.TIME_KNOTS)),
                    converged=True, iterations=1, objective=1.)

    def test_zero_missing_correction_returns_exact_original_probability(self):
        p = np.array([0., 1., .5, 1e-30, .123456789012345])
        t = np.array([0., 1., 15., 40., 100.])
        out = combat.predict(self.model(), np.zeros((len(p), 20)), p, t)
        assert_array_equal(out, p)
        assert_array_equal(combat.correction(self.model(), np.zeros((len(p), 20)), t), np.zeros(len(p)))

    def test_oriented_extra_and_complemented_baseline_are_side_symmetric(self):
        rng = np.random.default_rng(19)
        extra = rng.uniform(-1., 1., (40, 20))
        p = rng.uniform(.05, .95, 40)
        t = np.linspace(0., 100., 40)
        left = combat.predict(self.model(), extra, p, t)
        right = combat.predict(self.model(), -extra, 1. - p, t)
        assert_allclose(left + right, 1., atol=1e-15, rtol=0.)

    def test_all_player_gold_health_and_revive_partial_effects_are_monotone(self):
        model = self.model()
        fixture = CombatObservationTests().fixture()
        fixture["hpb"] = [0., .1, .4, .7, .9]
        fixture["hpr"] = [0., .1, .4, .7, .9]
        t = np.array([0., 8., 19., 35., 60., 100.])

        def probability(data):
            extra = np.tile(combat.observation_features(data)["features"], (len(t), 1))
            return combat.predict(model, extra, np.full(len(t), .5), t)

        before = probability(fixture)
        for side, direction in (("b", 1), ("r", -1)):
            for role in range(5):
                for delta in (1., 1000., 30000.):
                    more = copy.deepcopy(fixture)
                    more["gd" + side][role] += delta
                    self.assertTrue(np.all(direction * (probability(more) - before) >= 0))
                for health in (.001, .3, .6, 1.):
                    more = copy.deepcopy(fixture)
                    more["hp" + side][role] = max(more["hp" + side][role], health)
                    self.assertTrue(np.all(direction * (probability(more) - before) >= 0))
                more = copy.deepcopy(fixture)
                more["lv" + side][role] += 1
                self.assertTrue(np.all(direction * (probability(more) - before) >= 0))

    def test_penalty_gradient(self):
        rng = np.random.default_rng(8)
        values = rng.uniform(.02, .2, 20 * len(wpgam.TIME_KNOTS))
        _, gradient = combat._penalty(values, 800., 70.)
        for j in rng.choice(len(values), 20, replace=False):
            plus, minus = values.copy(), values.copy()
            plus[j] += 1e-6
            minus[j] -= 1e-6
            measured = (combat._penalty(plus, 800., 70.)[0] - combat._penalty(minus, 800., 70.)[0]) / 2e-6
            self.assertAlmostEqual(measured, gradient[j], places=5)

    def test_fit_freezes_offset_game_balances_scale_and_roundtrips(self):
        rng = np.random.default_rng(9)
        n = 300
        extra = rng.uniform(-4., 4., (n, 20))
        p = np.full(n, .5)
        gids = np.r_[np.repeat(np.arange(50), 2), np.repeat(np.arange(50, 60), 20)]
        y = np.repeat(rng.integers(0, 2, 60), np.r_[np.full(50, 2), np.full(10, 20)])
        t = np.linspace(0., 70., n)
        extra[:, 0] = (2 * y - 1.) * 2.
        original = p.copy()
        fitted = combat.fit(extra, p, y, gids, t)
        assert_array_equal(p, original)
        expected_scale = np.maximum(np.sqrt(np.average(extra ** 2, axis=0,
                                                      weights=wpgam._game_balanced_weights(gids))), 1.)
        assert_allclose(fitted["scale"], expected_scale, atol=0., rtol=0.)
        self.assertGreater(fitted["coefficients"].max(), 0.)
        self.assertGreaterEqual(fitted["coefficients"].min(), 0.)
        out = combat.predict(fitted, extra, p, t)
        self.assertLess(np.mean((out - y) ** 2), .25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combat.npz"
            combat.save(fitted, path)
            restored = combat.load(path)
            assert_array_equal(combat.predict(restored, extra, p, t), out)
        reverse_fit = combat.fit(-extra, 1. - p, 1. - y, gids, t)
        assert_array_equal(fitted["scale"], reverse_fit["scale"])
        assert_allclose(fitted["coefficients"], reverse_fit["coefficients"], atol=1e-12, rtol=0.)

    def test_artifact_and_bad_inference_contracts_reject(self):
        for key, bad in (("coefficients", np.full(120, -.01)), ("scale", np.zeros(20)),
                         ("feature_names", combat.FEATURE_NAMES[:-1]), ("converged", False)):
            model = self.model()
            model[key] = bad
            with self.assertRaises(ValueError):
                combat.validate(model)
        for extra, p, t in ((np.zeros((1, 19)), [.5], [10.]),
                            (np.full((1, 20), np.nan), [.5], [10.]),
                            (np.zeros((1, 20)), [1.1], [10.]),
                            (np.zeros((1, 20)), [.5], [-1.])):
            with self.assertRaises(ValueError):
                combat.predict(self.model(), extra, p, t)
        with self.assertRaises(ValueError):
            combat.fit(np.empty((0, 20)), [], [], [], [])

    def test_inactive_fit_rows_keep_full_population_weights_and_constant_loss(self):
        extra = np.zeros((6, 20))
        extra[:2, 0] = [3., -3.]
        p, y = np.full(6, .5), np.array([1, 0, 1, 1, 0, 0])
        gids, t = np.array([1, 2, 1, 1, 2, 2]), np.full(6, 20.)
        model = combat.fit(extra, p, y, gids, t)
        self.assertEqual(model["training_rows"], 6)
        self.assertEqual(model["active_rows"], 2)
        weights = wpgam._game_balanced_weights(gids)
        delta = combat.correction(model, extra, t)
        loss = np.dot(weights, np.logaddexp(0., delta) - y * delta)
        penalty, _ = combat._penalty(model["coefficients"], 800., 70.)
        self.assertAlmostEqual(model["objective"], loss + penalty, places=12)
        assert_array_equal(combat.predict(model, extra, p, t)[2:], p[2:])
        empty = combat.fit(np.zeros((6, 20)), p, y, gids, t)
        self.assertEqual(empty["active_rows"], 0)
        assert_array_equal(empty["coefficients"], np.zeros(120))
        self.assertAlmostEqual(empty["objective"], 6 * np.log(2), places=12)


if __name__ == "__main__":
    unittest.main()
