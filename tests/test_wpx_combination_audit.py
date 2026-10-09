from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from lol_ticker import wpcombat, wpcombined, wpgam, wptrend
from research import wpx_combination_audit as shape


class CombinationShapeTests(unittest.TestCase):
    def fixtures(self):
        n = 4
        names = ["gold_k", "gold_mom", "gold_top", "gold_jng", "gold_mid", "gold_bot", "gold_sup",
                 "gold_rel", "hp_pool", "dead_adv", "dead_adv_sq", "dead_count_sq_adv", "dead_base_pressure"]
        core_names = dict(zip(names, range(len(names))))
        gold = np.full((n, 10), 1000.)
        players = np.zeros((n, 10, 3))
        players[:, :, 0] = .5
        players[:, 0, 0] = 0.
        players[:, :, 1] = gold
        players[:, :, 2] = 8.
        combat = []
        for observation in players:
            source = {prefix+side: observation[start:start+5, channel].tolist()
                      for channel, prefix in enumerate(("hp", "gd", "lv"))
                      for side, start in (("b", 0), ("r", 5))}
            combat.append(wpcombat.observation_features(source)["features"])
        features = {"combat": dict(extra=np.asarray(combat), known=np.ones((n, 20), dtype=bool),
                                   available=np.ones(n, dtype=bool), raw=players),
                    "trend": dict(extra=np.zeros((n, 66)), known=np.zeros((n, 66), dtype=bool),
                                  available=np.ones(n, dtype=bool)),
                    "sq": dict(pair_score=np.array([-.2, -.1, .1, .2]), available=np.ones(n, dtype=bool))}
        for j, name in enumerate(wptrend.FEATURE_NAMES):
            if name.startswith(("lead_change_", "gold_change_")) and name.endswith(("120s_k", "300s_k")):
                features["trend"]["known"][:, j] = True
        core = np.zeros((n, len(names)))
        core[:, core_names["hp_pool"]] = -.5
        effects = wpgam._death_features(np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))
        for key, effect in zip(("dead_adv", "dead_adv_sq", "dead_count_sq_adv", "dead_base_pressure"), effects):
            core[:, core_names[key]] = effect
        rows = dict(gid=np.arange(n), t_min=np.array([2., 8., 15., 26.]), hp_age_upper_s=np.full(n, 60.),
                    names=["towers_blue", "towers_red", "inhib_blue", "inhib_red"], X=np.zeros((n, 4)))
        specs = {"combat": dict(feature_names=wpcombat.FEATURE_NAMES, time_knots=[0., 10., 20., 30.]),
                 "trend": dict(feature_names=wptrend.FEATURE_NAMES, time_knots=[0., 10., 20., 30.]),
                 "sq": dict(feature_names=["pair_score"], time_knots=[0., 5., 10., 15., 20.],
                            monotone_decay=True, zero_after_min=20., scale=.25, bounds=(0., 2.5), l2=300.)}
        model = wpcombined.fit(shape._inputs(features), np.full(n, .5), np.zeros(n), rows["gid"], rows["t_min"], specs=specs)
        return model, dict(feature_names=names), core, rows, features, gold, core_names

    def core_predict(self, model, value, t):
        names = {name: j for j, name in enumerate(model["feature_names"])}
        return wpgam._sigmoid(.02*value[:, names["gold_k"]]+.02*value[:, names["hp_pool"]]
                             +.02*value[:, names["dead_adv"]])

    def test_current_gold_keeps_aged_combat_fixed_and_updates_only_known_trends(self):
        _, _, core, _, features, gold, names = self.fixtures()
        j = wptrend.FEATURE_NAMES.index("gold_change_top_120s_k")
        features["trend"]["known"][0, j] = False
        value, changed = shape._current_gold(core, features, gold, names, 0, 1000.)
        np.testing.assert_array_equal(changed["combat"]["extra"], features["combat"]["extra"])
        np.testing.assert_array_equal(value[:, names["gold_k"]], 1.)
        np.testing.assert_allclose(value[:, names["gold_rel"]], 1./11.)
        np.testing.assert_array_equal(changed["trend"]["extra"][:, j], [0., 1., 1., 1.])
        np.testing.assert_array_equal(features["trend"]["extra"], 0.)

    def test_archived_gold_intervention_respects_living_and_known_resources(self):
        _, _, _, _, features, _, _ = self.fixtures()
        dead = shape._observed_gold(features, 0, 1000.)
        np.testing.assert_array_equal(dead["combat"]["extra"], features["combat"]["extra"])
        j = wpcombat.FEATURE_NAMES.index("living_gold_jng_k")
        features["combat"]["known"][0, j] = False
        changed = shape._observed_gold(features, 1, 1000.)
        np.testing.assert_array_equal(changed["combat"]["extra"][:, j], [0., 1., 1., 1.])

    def test_hp_revival_updates_core_and_combat_without_inventing_optional_level(self):
        _, _, core, rows, features, _, names = self.fixtures()
        features["combat"]["known"][:, 15:] = False
        features["combat"]["raw"][:, :, 2] = np.nan
        value, changed = shape._observed_hp(core, features, rows["X"], dict(zip(rows["names"], range(4))), names, 0, 1.)
        np.testing.assert_array_equal(value[:, names["hp_pool"]], .5)
        np.testing.assert_array_equal(value[:, names["dead_adv"]], 0.)
        np.testing.assert_array_equal(changed["combat"]["extra"][:, 0], 0.)
        np.testing.assert_array_equal(changed["combat"]["extra"][:, 10], 0.)
        np.testing.assert_array_equal(changed["combat"]["extra"][:, 15:], 0.)

    def test_complete_audit_and_empty_population(self):
        model, core_model, core, rows, features, gold, _ = self.fixtures()
        with patch.object(shape.wpgam, "predict_state", side_effect=self.core_predict):
            report = shape.audit(model, core_model, core, {"intercept": 0., "slope": 1.},
                                 rows, features, gold, np.ones(4, dtype=bool), sample_size=4)
        self.assertEqual(report["comparisons"], 240)
        self.assertEqual(report["reversals"], 0)
        self.assertTrue(report["sq_late_zero_exact"])
        empty = shape.audit(model, core_model, core, {"intercept": 0., "slope": 1.},
                            rows, features, gold, np.zeros(4, dtype=bool))
        self.assertEqual(empty["comparisons"], 0)

    def test_current_gold_reversal_fails(self):
        model, core_model, core, rows, features, gold, _ = self.fixtures()
        for block in model["blocks"]:
            block["coefficients"][:] = 0.
        with patch.object(shape.wpgam, "predict_state", side_effect=lambda m, v, t: wpgam._sigmoid(-v[:, 0])):
            with self.assertRaisesRegex(ValueError, "physical audit failed"):
                shape.audit(model, core_model, core, {"intercept": 0., "slope": 1.}, rows, features,
                            gold, np.ones(4, dtype=bool), sample_size=4)

    def test_synchronous_health_is_not_silently_audited_as_carried_sample(self):
        model, core_model, core, rows, features, gold, _ = self.fixtures()
        j = wptrend.FEATURE_NAMES.index("hp_change_top_120s")
        features["trend"]["known"][:, j] = True
        with patch.object(shape.wpgam, "predict_state", side_effect=self.core_predict):
            with self.assertRaisesRegex(ValueError, "synchronous health"):
                shape.audit(model, core_model, core, {"intercept": 0., "slope": 1.}, rows, features,
                            gold, np.ones(4, dtype=bool), sample_size=4)

    def test_sq_input_missingness_and_invalid_population_are_explicit(self):
        model, core_model, core, rows, features, gold, _ = self.fixtures()
        features["sq"]["available"][1] = False
        self.assertEqual(shape._inputs(features)["sq"][1, 0], 0.)
        with self.assertRaisesRegex(ValueError, "row population"):
            shape.audit(model, core_model, core, {"intercept": 0., "slope": 1.}, rows, features,
                        gold, np.ones(4), sample_size=4)


if __name__ == "__main__":
    unittest.main()
