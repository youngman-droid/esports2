import types
import unittest

import numpy as np

from lol_ticker import wpgam, wpx
from research import wpx_resource_audit as audit


class ResourceSupplementaryAuditTests(unittest.TestCase):
    def test_missing_hp_and_death_reencode_every_input_view(self):
        names = wpx.FEATURE_NAMES
        idx = {n: i for i, n in enumerate(names)}
        X = np.zeros((2, len(names)))
        X[:, idx["has_hp"]] = 1
        X[:, idx["hp_pool"]] = 2
        X[:, idx["lvl_k"]] = 3
        X[:, idx["hp_low_r"]] = 2
        X[:, idx["towers_blue"]] = 7
        prior = np.full((2, len(wpgam.PRIOR_INPUTS)), .2)
        old = types.SimpleNamespace(state_values_from_matrix=lambda values, columns:
            wpgam.state_values_from_matrix(values, columns, feature_names=wpgam.LEGACY_STATE_FEATURES))
        raw = np.c_[prior, wpgam.state_values_from_matrix(X, names)]
        part = dict(raw=raw, old_raw=np.c_[prior, old.state_values_from_matrix(X, names)],
                    rich_raw=np.c_[raw, np.ones((2, 14))])
        changed = audit.remove_hp(X, names)
        changed[:, idx["dead_red"]] = 3
        encoded = audit.encode_state(part, changed, names, old)
        index = {n: i for i, n in enumerate(wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES)}
        for name, expected in (("dead_adv", 3), ("dead_adv_sq", 9),
                               ("dead_count_sq_adv", 9), ("dead_base_pressure", 6),
                               ("hp_pool", 0), ("lvl_k", 0), ("has_hp", 0)):
            np.testing.assert_array_equal(encoded["raw"][:, index[name]], expected)
        np.testing.assert_array_equal(encoded["rich_raw"][:, :raw.shape[1]], encoded["raw"])
        np.testing.assert_array_equal(encoded["raw"][:, :3], prior)
        np.testing.assert_array_equal(X[:, idx["hp_pool"]], 2)

    def test_independent_calibration_root_satisfies_convex_optimality(self):
        p = np.array([.05, .15, .3, .45, .6, .75, .9, .95] * 4)
        y = np.array([0, 0, 1, 0, 1, 1, 0, 1] * 4)
        gids = np.repeat(np.arange(8), 4)
        for method in ("temperature", "platt"):
            result = audit.calibration_audit(p, y, gids, method, dict(intercept=0, slope=1), p, y, gids)
            self.assertLess(result["refit_normalized_kkt"], 1e-10)
            self.assertGreaterEqual(result["objective_improvement"], 0)
            beta = result["independent_calibration"]
            self.assertTrue(.05 <= beta["slope"] <= 5)
            self.assertTrue(-5 <= beta["intercept"] <= 5)
            repeated = audit.calibration_audit(p, y, gids, method, beta, p, y, gids)
            self.assertLess(repeated["max_coefficient_difference"], 1e-12)


if __name__ == "__main__":
    unittest.main()
