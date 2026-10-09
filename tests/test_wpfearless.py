import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from lol_ticker import wpfearless as fearless


class FearlessTests(unittest.TestCase):
    def fixture(self, mode="both_teams", game_num=2):
        picks = {side: {role: side + "current" + role for role in fearless.ROLES} for side in ("blue", "red")}
        players = {side: {role: side + "player" + role for role in fearless.ROLES} for side in ("blue", "red")}
        context = dict(series_id="S", tournament_id="Exact event", phase="playoffs", game_num=game_num,
                       blue_team_id="B", red_team_id="A", history_complete=True, picks=picks, players=players)
        prior = [dict(series_id="S", tournament_id="Exact event", game_num=n, completed_ts=30 + n,
                      blue_team_id="A", red_team_id="B",
                      picks={side: {r: side + "prior" + str(n) + r for r in fearless.ROLES} for side in ("blue", "red")})
                 for n in range(1, game_num)]
        rules = dict(rule_id="test explicit rule", tournament_id="Exact event", phases=["playoffs"],
                     effective_from_ts=0, effective_until_ts=200, source_published_ts=0,
                     source_url="https://lolesports.com/en-US/news/lol-esports-in-2025",
                     mode=mode, reset_before_games=[], max_games=5, verified=True)
        records = []
        for side in ("blue", "red"):
            for role in fearless.ROLES:
                # Both players know a current champion and one previous pick;
                # blue's first-game team was red, verifying stable-team joins.
                for champion in (picks[side][role], "redprior1" + role if side == "blue" else "blueprior1" + role):
                    for i in range(6):
                        records.append(dict(game_id=side + role + champion + str(i), player_id=players[side][role],
                                            role=role, champion=champion, completed_ts=10 + i))
        history = fearless.build_player_pools(records, as_of_ts=25, identity_namespace="riot_esports_player_id")
        return context, prior, rules, history

    def test_both_team_removals_and_player_pool_pressure(self):
        context, prior, rules, history = self.fixture()
        out = fearless.features(context, prior, rules, history, draft_ts=100)
        self.assertTrue(out["complete"])
        self.assertEqual(len(out["blocked"]["blue"]), 10)
        self.assertEqual(out["blocked"]["blue"], out["blocked"]["red"])
        self.assertEqual(out["features"], [0.] * 20)
        for pool in out["player_pools"].values():
            self.assertEqual(len(pool["supported_pool"]), 2)
            self.assertEqual(len(pool["remaining_pool"]), 1)
        json.dumps(out, allow_nan=False)

    def test_own_team_rule_follows_stable_team_ids_across_side_swap(self):
        context, prior, rules, history = self.fixture(mode="own_team")
        out = fearless.features(context, prior, rules, history, draft_ts=100)
        self.assertTrue(out["complete"])
        self.assertEqual(out["blocked"]["blue"], sorted("redprior1" + r for r in fearless.ROLES))
        self.assertEqual(out["blocked"]["red"], sorted("blueprior1" + r for r in fearless.ROLES))

    def test_side_swap_negates_features_and_pool_support_remains_role_specific(self):
        context, prior, rules, history = self.fixture()
        history["pools"]["blueplayertop"]["top"]["unusedsupported"] = 6
        left = fearless.features(context, prior, rules, history, draft_ts=100)
        reverse = copy.deepcopy(context)
        reverse["blue_team_id"], reverse["red_team_id"] = context["red_team_id"], context["blue_team_id"]
        reverse["picks"] = dict(blue=context["picks"]["red"], red=context["picks"]["blue"])
        reverse["players"] = dict(blue=context["players"]["red"], red=context["players"]["blue"])
        right = fearless.features(reverse, prior, rules, history, draft_ts=100)
        assert_array_equal(left["features"], -np.asarray(right["features"]))
        self.assertGreater(left["features"][0], 0.)

    def test_unknown_rule_wrong_tournament_phase_and_future_evidence_zero(self):
        context, prior, rules, history = self.fixture()
        for key, value in (("verified", False), ("tournament_id", "Different event"),
                           ("phases", ["regular"]), ("source_published_ts", 101),
                           ("source_url", "https://example.com/rules"), ("mode", "guess")):
            bad = dict(rules)
            bad[key] = value
            out = fearless.features(context, prior, bad, history, draft_ts=100)
            self.assertFalse(out["available"], (key, value))
            self.assertEqual(out["features"], [0.] * 20)

    def test_gap_wrong_series_future_game_and_contradictory_draft_zero(self):
        context, prior, rules, history = self.fixture()
        cases = [[], [dict(prior[0], game_num=2)], [dict(prior[0], series_id="Other")],
                 [dict(prior[0], completed_ts=100)], [dict(prior[0], red_team_id="Other")],
                 [dict(prior[0], game_num=None)]]
        for bad in cases:
            self.assertFalse(fearless.features(context, bad, rules, history, draft_ts=100)["available"])
        context["picks"]["blue"]["top"] = prior[0]["picks"]["red"]["top"]
        out = fearless.features(context, prior, rules, history, draft_ts=100)
        self.assertEqual(out["reason"], "current_draft_contradicts_rules")

    def test_reset_variant_is_explicit_and_validates_old_prefix_rules(self):
        context, prior, rules, history = self.fixture(game_num=3)
        rules["reset_before_games"] = [3]
        context["picks"]["blue"]["top"] = prior[0]["picks"]["red"]["top"]
        out = fearless.features(context, prior, rules, history, draft_ts=100)
        self.assertTrue(out["available"])
        self.assertEqual(out["blocked"], dict(blue=[], red=[]))
        prior[1]["picks"]["blue"]["top"] = prior[0]["picks"]["blue"]["top"]
        out = fearless.features(context, prior, rules, history, draft_ts=100)
        self.assertEqual(out["reason"], "prior_draft_contradicts_rules")

    def test_missing_player_role_support_is_zero_for_that_role_pair_only(self):
        context, prior, rules, history = self.fixture()
        history["pools"].pop("blueplayertop")
        out = fearless.features(context, prior, rules, history, draft_ts=100)
        self.assertTrue(out["available"])
        self.assertFalse(out["complete"])
        self.assertEqual(out["known"], [False, True, True, True, True] * 4)
        self.assertEqual(out["features"][::5], [0.] * 4)
        history["as_of_ts"] = 101
        out = fearless.features(context, prior, rules, history, draft_ts=100)
        self.assertFalse(out["available"])
        self.assertEqual(out["features"], [0.] * 20)

    def test_history_excludes_current_future_and_duplicate_appearances(self):
        record = dict(game_id="G", player_id="P", role="top", champion="Aatrox", completed_ts=10)
        history = fearless.build_player_pools([record, record, dict(record, game_id="Future", completed_ts=20)], as_of_ts=20)
        self.assertEqual(history["records"], 1)
        self.assertEqual(history["excluded_records"], 1)
        self.assertEqual(history["pools"]["P"]["top"], dict(aatrox=1))
        with self.assertRaises(ValueError):
            fearless.build_player_pools([record, dict(record, champion="Ahri")], as_of_ts=20)

    def test_capture_uses_actual_metadata_and_checks_local_game_identity(self):
        context, prior, rules, history = self.fixture()
        md = {}
        for side, offset in (("blue", 0), ("red", 5)):
            md[side + "TeamMetadata"] = dict(esportsTeamId=context[side + "_team_id"],
                participantMetadata=[dict(participantId=offset + i + 1, role=role,
                    championId=context["picks"][side][role], esportsPlayerId=context["players"][side][role])
                    for i, role in enumerate(fearless.ROLES)][::-1])
        bundle = dict(context=context, prior_games=prior, rules=rules, player_history=history)
        self.assertTrue(fearless.capture(md, bundle, as_of_ts=100)["complete"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "series").mkdir()
            (root / "series" / "S.json").write_text(json.dumps(bundle))
            self.assertTrue(fearless.capture(md, dict(series_id="S", game_num=2), as_of_ts=100, source_root=root)["complete"])
            self.assertFalse(fearless.capture(md, dict(series_id="S", game_num=3), as_of_ts=100, source_root=root)["available"])
        md["blueTeamMetadata"]["esportsTeamId"] = "Wrong"
        self.assertFalse(fearless.capture(md, bundle, as_of_ts=100)["available"])

    def test_candidate_pool_advantage_is_positive_and_missing_is_exact_baseline(self):
        rng = np.random.default_rng(64)
        extra = rng.normal(size=(60, len(fearless.FEATURE_NAMES)))
        y = rng.integers(0, 2, 60)
        extra[:, 0] = 2. * y - 1.
        baseline = np.full(60, .5)
        model = fearless.fit(extra, baseline, y, np.arange(60))
        self.assertTrue(np.all(model["coefficients"] >= 0.))
        before = fearless.predict(model, extra, baseline)
        more = extra.copy()
        more[:, 0] += 1.
        self.assertTrue(np.all(fearless.predict(model, more, baseline) >= before))
        assert_array_equal(fearless.predict(model, np.zeros_like(extra), baseline), baseline)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fearless.npz"
            fearless.save(model, path)
            assert_array_equal(fearless.predict(fearless.load(path), extra, baseline), before)

    def test_malformed_rule_and_player_pool_optional_fields_fail_closed(self):
        context, prior, rules, history = self.fixture()
        for key, value in (("mode", []), ("source_url", [])):
            bad = dict(rules)
            bad[key] = value
            self.assertFalse(fearless.features(context, prior, bad, history, draft_ts=100)["available"])
        history["pools"]["blueplayertop"] = ["bad"]
        out = fearless.features(context, prior, rules, history, draft_ts=100)
        self.assertEqual(out["known"], [False, True, True, True, True] * 4)
        json.dumps(out, allow_nan=False)

    def test_verified_series_removals_survive_missing_player_history_with_zero_correction(self):
        context, prior, rules, _ = self.fixture()
        out = fearless.features(context, prior, rules, None, draft_ts=100)
        self.assertTrue(out["series_available"])
        self.assertFalse(out["available"])
        self.assertEqual(len(out["blocked"]["blue"]), 10)
        self.assertEqual(out["features"], [0.] * 20)

    def test_primary_event_profiles_keep_bo3_and_bo5_phase_limits_distinct(self):
        self.assertEqual(fearless.FIRST_STAND_2025_RULE_TEMPLATE["phases"], ["knockout"])
        self.assertEqual(fearless.FIRST_STAND_2025_RULE_TEMPLATE["max_games"], 5)
        self.assertEqual(fearless.FIRST_STAND_2025_ROUND_ROBIN_RULE_TEMPLATE["phases"], ["round_robin"])
        self.assertEqual(fearless.FIRST_STAND_2025_ROUND_ROBIN_RULE_TEMPLATE["max_games"], 3)


if __name__ == "__main__":
    unittest.main()
