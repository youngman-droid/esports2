import unittest
import numpy as np
from scripts.wpx_market_v9 import past_prices


class HistoricalPriceTests(unittest.TestCase):
    def test_no_future_fallback_and_exact_timestamp_is_available(self):
        p,a=past_prices([(100,.2),(120,.7)],np.array([90,100,110,120]))
        self.assertTrue(np.isnan(p[0]))
        np.testing.assert_allclose(p[1:],[.2,.2,.7])
        np.testing.assert_allclose(a[1:],[0,10,0])

    def test_stale_invalid_and_duplicate_prices(self):
        p,a=past_prices([(100,.2),(100,.4),(120,np.nan),(121,2)],np.array([100,160,161]))
        np.testing.assert_allclose(p[:2],[.3,.3])
        self.assertTrue(np.isnan(p[2]))
        np.testing.assert_allclose(a[:2],[0,60])
        missing,_=past_prices([],np.array([100]))
        self.assertTrue(np.isnan(missing[0]))


if __name__=='__main__':unittest.main()
