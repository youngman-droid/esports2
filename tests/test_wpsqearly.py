import tempfile
import unittest
from pathlib import Path

import numpy as np

from lol_ticker import wpsqearly as sq


class FakeScorer:
    patches = ["16.8", "16.9"]
    def score(self, patch, blue, red):
        return dict(score=0., coverage=1., matchup=0., synergy=0.)


class SqEarlyTests(unittest.TestCase):
    def test_valid_zero_is_covered_and_only_prior_patch_is_named(self):
        got = sq.capture("16.9", ["a"+str(i) for i in range(5)], ["b"+str(i) for i in range(5)], scorer=FakeScorer())
        self.assertTrue(got["available"])
        self.assertEqual(got["pair_score"], 0)
        self.assertEqual(got["source_patches"], ["16.8"])

    def test_live_release_suffix_never_admits_same_patch_cells(self):
        class PatchScorer(FakeScorer):
            patches = ["16.15", "16.16"]
            def score(self, patch, blue, red):
                from lol_ticker import sqpairs
                value = sum(i+1 for i, p in enumerate(self.patches) if sqpairs.pnum(p) < sqpairs.pnum(patch))
                return dict(score=float(value), coverage=1., matchup=float(value), synergy=0.)
        a, b = ["a"+str(i) for i in range(5)], ["b"+str(i) for i in range(5)]
        short = sq.capture("16.16", a, b, scorer=PatchScorer())
        full = sq.capture("16.16.1", a, b, scorer=PatchScorer())
        self.assertEqual(full["pair_score"], short["pair_score"])
        self.assertEqual(full["pair_score"], 1.)
        self.assertEqual(full["source_patches"], ["16.15"])
        self.assertEqual(full["patch"], "16.16")

    def test_duplicate_champions_and_missing_draft_fail_closed(self):
        self.assertFalse(sq.capture("16.9", ["a"]*5, ["b"]*5, scorer=FakeScorer())["available"])

    def test_own_curve_is_zero_at_every_late_minute_and_missing(self):
        model = sq.fit(np.zeros(4), np.ones(4, bool), np.full(4, .5), [0, 1, 0, 1], [1, 2, 3, 4], [0, 5, 10, 15])
        model["slopes"] = np.array([.2, .15, .1, .05, 0])
        t = np.array([0, 2, 5, 12, 19, 20, 25, 999])
        p = np.array([0., .5, .8, .4, .5, .1, 1., .3])
        got = sq.predict(model, np.ones(len(t)), np.ones(len(t), bool), p, t)
        np.testing.assert_array_equal(got[t >= 20], p[t >= 20])
        np.testing.assert_array_equal(sq.predict(model, np.ones(len(t)), np.zeros(len(t), bool), p, t), p)
        values = sq.correction(model, np.ones(len(t)), np.ones(len(t), bool), t)
        self.assertTrue(np.all(np.diff(values) <= 0))

    def test_fit_constraints_and_serialization(self):
        rng = np.random.default_rng(4)
        score = rng.normal(0, .25, 120); t = np.tile(np.arange(0, 24), 5)
        target = rng.binomial(1, 1/(1+np.exp(-score)))
        model = sq.fit(score, np.ones(120, bool), np.full(120, .5), target, np.arange(120), t)
        self.assertTrue(np.all(np.diff(model["slopes"]) <= 1e-8))
        self.assertEqual(model["slopes"][-1], 0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"candidate.npz"; sq.save(model, p); loaded = sq.load(p)
            np.testing.assert_array_equal(sq.predict(model, score, np.ones(120, bool), np.full(120, .5), t), sq.predict(loaded, score, np.ones(120, bool), np.full(120, .5), t))

    def test_invalid_clock_and_coverage_are_rejected(self):
        with self.assertRaises(ValueError):
            sq.fit([1], [1], [.5], [1], [1], [0])
        with self.assertRaises(ValueError):
            sq.fit([1], [True], [.5], [1], [1], [-1])


if __name__ == "__main__":
    unittest.main()
