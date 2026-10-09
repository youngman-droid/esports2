import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from lol_ticker import wpresource as resource, wpgam


class ResourceTests(unittest.TestCase):
    def fixture(self, mode="smooth", n=60):
        rng = np.random.default_rng(42)
        gold = rng.uniform(500, 25000, size=(n, 10))
        C = np.tile(np.arange(10), (n, 1))
        C[:, 5] = 0  # Same role/champion must pool across sides.
        raw = np.zeros((n, len(resource.INPUT_NAMES)))
        t = np.linspace(0, 75, n)
        model = dict(kind=resource.KIND, mode=mode,
            groups=np.array([(i, -1) for i in range(5)] + [(0, 0), (1, 1)]),
            gold_cap=np.full(5, 20000.), gold_scale=np.full(5, 4000.),
            scale=(np.zeros(len(resource.CORE_NAMES)), np.ones(len(resource.CORE_NAMES)),
                   np.full(len(resource.CORE_NAMES), -3.), np.full(len(resource.CORE_NAMES), 3.)),
            spec=dict(l2=24., smooth=70., shrink=800.),
            converged=True, iterations=1, objective=0., calibration=dict(intercept=0., slope=1.))
        width = resource._design(model, raw, gold, C, t).shape[1]
        coefficients = np.zeros(width)
        nc = (len(resource.CORE_NAMES) + 1) * len(wpgam.TIME_KNOTS)
        coefficients[nc:] = rng.uniform(.01, .7, size=width - nc)
        model["coefficients"] = coefficients
        return model, raw, gold, C, t

    def test_total_gold_response_and_negative_champion_deviation(self):
        for mode in ("constant", "smooth"):
            model, raw, gold, C, t = self.fixture(mode)
            nk = len(wpgam.TIME_KNOTS)
            nc = (len(resource.CORE_NAMES) + 1) * nk
            if mode == "constant":
                model["coefficients"][nc] = .5
                model["coefficients"][nc + 5] = .1
            else:
                model["coefficients"][nc:nc + nk] = .5
                model["coefficients"][nc + 5 * nk:nc + 6 * nk] = .1
            before = resource.predict(model, raw, gold, C, t)
            for slot in range(10):
                for amount in (100., 1000., 30000.):
                    more = gold.copy(); more[:, slot] += amount
                    after = resource.predict(model, raw, more, C, t)
                    direction = 1 if slot < 5 else -1
                    self.assertTrue(np.all(direction * (after - before) >= -1e-14))
            # This champion's total slope can be below its role slope without
            # a negative derivative, including when other gold inputs clip.
            self.assertLess(model["coefficients"][nc + (5 if mode == "constant" else 5 * nk)],
                            model["coefficients"][nc])

    def test_resources_share_sides_and_unknowns_use_role(self):
        for mode in ("constant", "smooth"):
            model, raw, gold, C, t = self.fixture(mode)
            p = resource.predict(model, raw, gold, C, t)
            reverse = resource.predict(model, raw, np.c_[gold[:, 5:], gold[:, :5]],
                                       np.c_[C[:, 5:], C[:, :5]], t)
            assert_allclose(p + reverse, 1., atol=1e-14)
            C[:, 0] = 999
            unseen = resource.predict(model, raw, gold, C, t)
            C[:, 0] = -1
            assert_allclose(unseen, resource.predict(model, raw, gold, C, t), atol=0)

    def test_penalty_gradient(self):
        rng = np.random.default_rng(1)
        for mode in ("constant", "smooth"):
            model, *_ = self.fixture(mode)
            vector = rng.uniform(.05, .2, len(model["coefficients"]))
            _, grad = resource._penalty(model, vector)
            for i in rng.choice(len(vector), 30, replace=False):
                plus, minus = vector.copy(), vector.copy()
                plus[i] += 1e-6; minus[i] -= 1e-6
                measured = (resource._penalty(model, plus)[0] - resource._penalty(model, minus)[0]) / 2e-6
                self.assertAlmostEqual(measured, grad[i], places=5)

    def test_group_support_counts_games_and_pools_both_sides(self):
        C = np.full((100, 10), -1)
        C[:50, 0] = 7; C[50:, 5] = 7
        self.assertEqual(len(resource._groups(C, np.zeros(100), minimum=2)), 5)
        groups = resource._groups(C, np.arange(100), minimum=2)
        self.assertEqual(groups.tolist(), [[i, -1] for i in range(5)] + [[0, 7]])

    def test_fit_and_serialized_predictions(self):
        rng = np.random.default_rng(4)
        _, raw, gold, C, t = self.fixture(n=300)
        gids = np.repeat(np.arange(100), 3)
        # Independent-game labels and repeated states, not 300 fictitious games.
        y = np.repeat(rng.binomial(1, .5, size=100), 3)
        raw[:, resource.INPUT_NAMES.index("gold_k")] = (gold[:, :5].sum(1) - gold[:, 5:].sum(1)) / 1000
        for mode in ("role", "constant", "smooth"):
            fitted = resource.fit(raw, gold, C, y, gids, t, dict(mode=mode, min_games=5))
            p = resource.predict(fitted, raw, gold, C, t)
            self.assertTrue(np.isfinite(p).all())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.npz"
                resource.save(fitted, path)
                restored = resource.load(path)
                assert_allclose(resource.predict(restored, raw, gold, C, t), p, rtol=0, atol=0)

    def test_missing_gold_and_negative_calibration_rejected(self):
        model, raw, gold, C, t = self.fixture()
        broken = gold.copy(); broken[0, 0] = np.nan
        with self.assertRaises(ValueError):
            resource.predict(model, raw, broken, C, t)
        model["calibration"]["slope"] = -.1
        with self.assertRaises(ValueError):
            resource.predict(model, raw, gold, C, t)

    def test_artifact_bounds_and_contract_are_enforced(self):
        model, _, _, _, _ = self.fixture()
        model.update(input_names=list(resource.INPUT_NAMES), time_knots=wpgam.TIME_KNOTS.tolist())
        resource.validate(model)
        model["coefficients"][-1] = -.01
        with self.assertRaisesRegex(ValueError, "monotonicity"):
            resource.validate(model)
        model["coefficients"][-1] = .01
        model["input_names"] = model["input_names"][:-1]
        with self.assertRaisesRegex(ValueError, "contract"):
            resource.validate(model)


if __name__ == "__main__":
    unittest.main()
