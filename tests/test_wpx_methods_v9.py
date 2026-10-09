import importlib
import unittest
from unittest import mock

import numpy as np

from lol_ticker import wpbench, wpgam, wpadapt as wa
from scripts import wpx_altmodels as alt, wpx_methods_v9 as methods


class RepairedMethodTests(unittest.TestCase):
    def test_importing_alternatives_does_not_replace_shared_benchmark(self):
        fit, predict, families = wpbench._fit_method, wpbench._predict_method, wpbench.FAMILIES
        importlib.reload(alt)
        self.assertIs(wpbench._fit_method, fit)
        self.assertIs(wpbench._predict_method, predict)
        self.assertEqual(wpbench.FAMILIES, families)

    def test_spline_scaler_preserves_unseen_discrete_support(self):
        raw = np.zeros((1000,len(wa.CORE_NAMES)))
        _,info = alt._spline_fit(raw,4)
        j = wa.CORE_NAMES.index("elder_active")
        lo,hi = info["scale"][2:]
        self.assertLessEqual(lo[j],-1)
        self.assertGreaterEqual(hi[j],1)
        probe = raw[:2].copy(); probe[:,j]=[-1,1]
        self.assertFalse(np.array_equal(*alt._spline_apply(probe,info)))

    def test_chronological_state_stack_cannot_read_scored_labels(self):
        n=150
        dates=np.array([str(np.datetime64("2025-01-01")+np.timedelta64(i,"D")) for i in range(n)])
        part=dict(raw=np.zeros((n,len(wa.CORE_NAMES))),gid=np.arange(n),date=dates,
                  t=np.full(n,15.),y=(np.arange(n)%2).astype(float))
        def fit(raw,y,gids,t):
            return dict(score=float(np.mean(y)))
        def predict(model,raw,t,components=False):
            return None,np.full(len(raw),model["score"])
        folds=list(wpgam._stack_folds(part["gid"],k=5,dates=dates))
        first_scored=min(np.flatnonzero(te)[0] for tr,te in folds if tr.any())
        altered=dict(part,y=part["y"].copy()); altered["y"][first_scored:]=1.
        with mock.patch.object(wpgam,"fit_state_model",side_effect=fit), \
                mock.patch.object(wpgam,"predict_state",side_effect=predict):
            before,_=methods.chronological_gam_scores(part)
            after,_=methods.chronological_gam_scores(altered)
        _,first_test=next((tr,te) for tr,te in folds if tr.any())
        np.testing.assert_array_equal(before[first_test],after[first_test])

    def test_metrics_balance_games_not_number_of_snapshots(self):
        p=np.array([.1,.9,.9,.9]);y=np.array([0.,0.,0.,0.]);gid=np.array([1,2,2,2])
        got=methods.metrics(p,y,gid)
        self.assertAlmostEqual(got["brier_game"],(.01+.81)/2)
        expected=wpbench._basic_metrics(p,y,gid)
        self.assertAlmostEqual(got["logloss_game"],expected["logloss_game"])

    def test_gold_audit_changes_one_player_and_consistent_totals(self):
        n=10;raw=np.zeros((n,len(wa.CORE_NAMES)+len(wa.EXTRA_NAMES)))
        raw[:,len(wa.CORE_NAMES):len(wa.CORE_NAMES)+10]=np.array([1]*5+[-1]*5)
        raw[:,len(wa.CORE_NAMES)+10]=1
        part=dict(raw=raw,cats=np.zeros((n,len(wa.CAT_NAMES))),t=np.full(n,20.))
        def predictor(part):
            x=part["raw"]
            slots=x[:,len(wa.CORE_NAMES):len(wa.CORE_NAMES)+10]
            delta=slots.sum(axis=1)*10
            np.testing.assert_allclose(x[:,wa.CORE_NAMES.index("gold_k")],delta,atol=1e-12)
            np.testing.assert_allclose(x[:,wa.CORE_NAMES.index("gold_rel")],delta/(abs(slots).sum(axis=1)*10),atol=1e-12)
            return wpgam._sigmoid(delta)
        result=methods.gold_audit(predictor,part)
        self.assertTrue(result["passed"])
        self.assertEqual(result["comparisons"],200)


if __name__=="__main__":
    unittest.main()
