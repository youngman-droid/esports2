from datetime import date
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lol_ticker import wpadapt as wa
from research.wpx_adapt import search_specs


class TemporalTests(unittest.TestCase):
    def test_windows_and_decay_keep_game_weighting_and_exclude_same_day(self):
        gids=np.array([1,1,1,2,2,3])
        dates=np.array(['2026-01-01']*3+['2026-01-11']*2+['2026-01-21'])
        keep,w=wa.temporal_weights(gids,dates,'2026-01-21',window_days=30,half_life=10)
        self.assertFalse(keep[-1])
        self.assertAlmostEqual(w[:3].sum()/w[3:].sum(),.5)
        keep,_=wa.temporal_weights(gids,dates,'2026-01-21',window_days=15)
        np.testing.assert_array_equal(keep,[False,False,False,True,True,False])

    def test_complete_days_and_earlier_calibration(self):
        dates=np.repeat(np.arange('2024-01-01','2024-07-01',dtype='datetime64[D]').astype(str),2)
        outer=dates<'2024-06-01'
        folds,final=wa.split_blocks(dates,outer,3)
        seen=set()
        for block in folds+[final]:
            self.assertLess(max(dates[block['fit']]),min(dates[block['calibration']]))
            self.assertLess(max(dates[block['calibration']]),min(dates[block['validation']]))
            self.assertEqual((np.datetime64(block['validation_start'])-
                              np.datetime64(block['fit_end'])).astype(int),28)
            for d in np.unique(dates):
                for key in ('fit','calibration','validation'):
                    self.assertTrue(np.all(block[key][dates==d]) or not np.any(block[key][dates==d]))
            indices=set(np.flatnonzero(block['validation']))
            self.assertFalse(seen & indices);seen |= indices


class CategoryTests(unittest.TestCase):
    def test_unseen_role_champions_does_not_depend_on_model_family(self):
        train=np.full((2,33),'A',dtype=object);val=train.copy()
        val[0,3]='B'
        np.testing.assert_array_equal(wa.unseen_role_champions(train,val),[True,False])

    def test_future_and_rare_categories_fall_back_without_state_count_leak(self):
        cats=np.array([['A']*33]*21,dtype=object)
        cats[-1,:]='B'
        gids=np.array([1]*20+[2])
        maps=wa.fit_categories(cats,gids,min_games=2)
        self.assertIn('A',maps[0])
        self.assertEqual(maps[13],{}) # 20 states still represent one game
        unseen=np.array([['C']*33],dtype=object)
        np.testing.assert_array_equal(wa.encode_categories(unseen,maps),-np.ones((1,33)))

    def test_observed_patch_age_is_causal_and_versions_are_numeric(self):
        dates=['2026-01-01','2026-01-03','2026-02-01']
        patches=['16.2','16.2','16.10'];C=np.zeros((3,10),int)
        meta,cats=wa.calendar_metadata(dates,patches,['LCK CL Spring 2026']*3,C)
        np.testing.assert_allclose(meta[:,1],[2,2,10])
        np.testing.assert_allclose(meta[:,2],[0,2/30,0])
        self.assertEqual(cats[0,12],'LCK CL')
        prior,_=wa.calendar_metadata(dates[:2],patches[:2],['LCK CL Spring 2026']*2,C[:2])
        np.testing.assert_array_equal(prior,meta[:2])

    def test_sparse_resource_terms_use_side_sign_and_unknown_is_zero(self):
        maps=[{'A':0} for _ in range(33)]
        encoded=np.zeros((2,33),int);encoded[1]=-1
        gold=np.ones((2,10));gold[:,5:]=-1
        S,penalties,positive=wa.sparse_context(encoded,maps,gold,np.array([15.,15.]))
        self.assertEqual(S[1].nnz,0)
        self.assertTrue(np.all(penalties>0))
        self.assertEqual(int(positive.sum()),20)


class ModelTests(unittest.TestCase):
    def test_rich_gam_handles_unknown_champions_and_resource_slopes(self):
        rng=np.random.default_rng(8);n=80
        raw=rng.normal(size=(n,len(wa.CORE_NAMES)+len(wa.EXTRA_NAMES)))
        cats=np.full((n,33),'A',dtype=object)
        y=(raw[:,0]+rng.normal(size=n)>0).astype(float)
        gids=np.repeat(np.arange(20),4);t=np.tile([5,15,25,35],20)
        model=wa.fit_gam(raw,cats,y,gids,t,np.ones(n),dict(features='rich',min_category_games=2))
        p=wa.predict(model,raw,cats,t)
        increased=raw.copy();increased[:,len(wa.CORE_NAMES)]+=.1
        self.assertTrue(np.all(wa.predict(model,increased,cats,t)>=p-1e-10))
        cats[:]='new'
        self.assertTrue(np.isfinite(wa.predict(model,raw,cats,t)).all())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);wa.save_model(model,path,dict(slope=1.,intercept=0.))
            np.testing.assert_allclose(wa.predict(model,raw,cats,t),wa.predict(wa.load_model(path),raw,cats,t))

    def test_core_gam_fit_and_artifact_roundtrip(self):
        rng=np.random.default_rng(2);n=100
        raw=rng.normal(size=(n,len(wa.CORE_NAMES)+len(wa.EXTRA_NAMES)))
        cats=np.full((n,33),'A',dtype=object)
        y=(raw[:,0]+rng.normal(size=n)>0).astype(float)
        gids=np.repeat(np.arange(20),5);t=np.tile([0,10,20,30,40],20)
        model=wa.fit_gam(raw,cats,y,gids,t,np.ones(n),dict(features='core'))
        p=wa.predict(model,raw,cats,t)
        self.assertTrue(np.isfinite(p).all())
        self.assertLess(np.mean((p-y)**2),.25)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            wa.save_model(model,path,dict(slope=.8,intercept=.1))
            np.testing.assert_allclose(p,wa.predict(wa.load_model(path),raw,cats,t))
            from lol_ticker import wpbench
            np.testing.assert_allclose(wpbench._apply_platt(p,dict(slope=.8,intercept=.1)),
                                       wa.predict_calibrated(wa.load_model(path),raw,cats,t))

    def test_search_depth_and_advanced_constraint_parameters_are_compatible(self):
        specs=search_specs()
        self.assertEqual(specs,search_specs())
        for s in specs:
            if s['family']!='lightgbm':continue
            if s['max_depth']>0:self.assertLessEqual(s['num_leaves'],2**s['max_depth'])
            if s['monotone']:self.assertEqual(s['feature_fraction'],1.)
            self.assertGreater(s['max_rounds'],s['early_stopping_rounds'])


if __name__=='__main__':unittest.main()
