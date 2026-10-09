import tempfile
import unittest
from pathlib import Path

import numpy as np

from lol_ticker import wpgam, wpbench
from scripts import promote_combination as package


class CombinationPackagingTests(unittest.TestCase):
    def fixture(self):
        width = len(wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES)
        pre_width = len(wpgam.PREGAME_FEATURES)
        priors = dict(train=np.ones(4, dtype=bool), champ_state_beta=np.array([.2, -.1]),
            pregame=dict(mean=np.zeros(pre_width), std=np.ones(pre_width),
                         lo=np.full(pre_width, -10.), hi=np.full(pre_width, 10.),
                         intercept=.3, team_beta=np.arange(pre_width) / 10.,
                         champ_beta=np.array([.1, -.2]), champ_names=np.array(["A", "B"])))
        state = dict(feature_names=np.array(wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES),
                     mean=np.zeros(width), std=np.ones(width), lo=np.full(width, -10.),
                     hi=np.full(width, 10.), theta=np.full((width + 1, len(wpgam.TIME_KNOTS)), .05),
                     knots=wpgam.TIME_KNOTS, optimizer_success=True, cal_intercept=0., cal_slope=1.)
        return priors, state

    def test_package_preserves_raw_core_and_fitted_upstream_without_refit(self):
        priors, state = self.fixture()
        base = package.reconstruct_base(priors, state)
        raw = np.zeros((4, len(state["feature_names"])))
        raw[:, 0] = np.arange(4)
        expected = wpgam.predict_state(state, raw, [0., 5., 20., 40.])
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "base.npz")
            wpgam.save_model(base, path)
            restored = wpgam.load_model(path)
            np.testing.assert_array_equal(expected, wpgam.predict_state(restored["state"], raw, [0., 5., 20., 40.]))
            np.testing.assert_array_equal(restored["pregame"]["champ_beta"], priors["pregame"]["champ_beta"])
            np.testing.assert_array_equal(restored["champ_state"]["beta"], priors["champ_state_beta"])
        base["state"]["theta"][0, 0] = 9.
        self.assertEqual(state["theta"][0, 0], .05)

    def test_calibration_cannot_be_applied_before_joint_correction(self):
        priors, state = self.fixture()
        state["cal_slope"] = .8
        with self.assertRaisesRegex(ValueError, "uncalibrated"):
            package.reconstruct_base(priors, state)
        state["cal_slope"] = 1.
        state["calibration_probability_clip"] = [1e-5, 1.-1e-5]
        with self.assertRaisesRegex(ValueError, "no probability clipping"):
            package.reconstruct_base(priors, state)

    def test_shared_platt_probability_clip_including_saturated_logits(self):
        p = np.array([0., 1e-8, .25, .75, 1.-1e-8, 1.])
        calibration = dict(intercept=.05213831277613384, slope=.9851490833507187)
        clipped = np.clip(p, 1e-5, 1.-1e-5)
        expected = wpgam._sigmoid(calibration["intercept"] + calibration["slope"] * np.log(clipped/(1.-clipped)))
        np.testing.assert_array_equal(wpbench._apply_platt(p, calibration), expected)

    def test_reproduction_fails_closed_on_probability_drift(self):
        self.assertEqual(package.assert_match([.4], [.4], what="same"), 0.)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            package.assert_match([.4], [.4001], what="changed")
        with self.assertRaisesRegex(ValueError, "finiteness"):
            package.assert_match([np.nan], [.4], what="nonfinite")


if __name__ == "__main__":
    unittest.main()
