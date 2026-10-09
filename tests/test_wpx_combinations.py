import unittest

import numpy as np

from research import wpx_combinations as runner


class CombinationRunnerTests(unittest.TestCase):
    def test_all_nonempty_subsets_once(self):
        subsets = runner.combinations()
        self.assertEqual(len(subsets), 31)
        self.assertEqual(len(set(subsets)), 31)
        self.assertEqual(subsets[-1], runner.FAMILIES)
        self.assertEqual(sum(len(x) == 2 for x in subsets), 10)

    def test_paired_clusters_and_simultaneous_bands(self):
        gids = np.repeat(np.arange(40), 2)
        y = np.repeat(np.arange(40) % 2, 2)
        baseline = np.full(len(gids), .5)
        good = baseline + np.where(y == 1, .05, -.05)
        bad = 1-good
        series = {g: 'match'+str(g//2) for g in range(40)}
        result = runner.joint_intervals({'good': good, 'bad': bad, 'exact': baseline}, baseline,
                                        y, gids, series, draws=200)
        self.assertEqual(result['good']['clusters'], 20)
        self.assertLess(result['good']['simultaneous_ci95'][1], 0)
        self.assertGreater(result['bad']['simultaneous_ci95'][0], 0)
        self.assertEqual(result['exact']['simultaneous_ci95'], [0., 0.])
        self.assertEqual(result, runner.joint_intervals({'good': good, 'bad': bad, 'exact': baseline}, baseline,
                                                       y, gids, series, draws=200))

    def test_selection_requires_stability_and_logloss(self):
        gids = np.arange(20)
        y = gids % 2
        ref = np.full(20, .5)
        good = ref + np.where(y == 1, .1, -.1)
        series = {int(g): str(g) for g in gids}
        part = dict(predictions={'good': good, 'unstable': good}, baseline=ref, y=y, gid=gids)
        report = dict(candidates={'good': {'paired': {'delta_brier': -.01}},
                                  'unstable': {'paired': {'delta_brier': .01}}})
        result = runner.pool([part], [report], series)
        self.assertEqual(result['eligible'], ['good'])
        self.assertEqual(result['selected'], 'good')


if __name__ == '__main__':
    unittest.main()
