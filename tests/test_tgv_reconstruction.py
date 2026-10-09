import math
from pathlib import Path
import unittest

from lol_ticker.tgv_reconstruction import (PublicDraftModel, sigmoid, verify_daily,
    recover_two_point_curve, saturating_count)

ROOT=Path(__file__).resolve().parents[1]/'data/tgv/20260916'


class ReconstructionTests(unittest.TestCase):
    def test_logistic_extremes(self):
        self.assertEqual(sigmoid(0),.5)
        self.assertEqual(sigmoid(1000),1)
        self.assertEqual(sigmoid(-1000),0)

    def test_curve_recovery_uses_two_points_and_generalizes(self):
        a,k=.079,1.04
        ra,rk=recover_two_point_curve(saturating_count(1,a,k),saturating_count(2,a,k))
        for n in (0,3,7,100,1000):
            self.assertAlmostEqual(saturating_count(n,ra,rk),saturating_count(n,a,k),places=14)
        with self.assertRaises(ValueError): saturating_count(-1,ra,rk)

    @unittest.skipUnless((ROOT/'challenge.json').exists(),'requires scraped public evidence')
    def test_published_component_parity(self):
        report=verify_daily(ROOT)
        self.assertLess(max(report['independent_component_errors'].values()),1e-12)
        self.assertLess(report['review_logit_error'],1e-12)
        self.assertLess(report['engine_logit_error'],2e-8)

    @unittest.skipUnless((ROOT/'current-model-catalog.json').exists(),'requires scraped public evidence')
    def test_missing_composition_cannot_silently_become_full_prediction(self):
        model=PublicDraftModel(ROOT)
        with self.assertRaises(TypeError):
            model.score('16.18',[1,2,3,4,5],[6,7,8,9,10])
        with self.assertRaises(ValueError):
            model.components('16.18',[1]*5,[2]*5)


if __name__=='__main__': unittest.main()
