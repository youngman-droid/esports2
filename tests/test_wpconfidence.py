import copy
import json
import unittest

from lol_ticker import wpconfidence as confidence

NOW = 1700000000.


def fixture():
    source = dict(available=True, source="oe_games+oe_ratings", game_id=1,
                  date=NOW-86400, players=["a", "b", "c", "d", "e"])
    return dict(elo_oe=1., pelo_oe=.5, elo_blue=1800., elo_red=1400.,
        pelo_blue=1700., pelo_red=1500., prior_availability=dict(elo_oe=True, pelo_oe=True),
        prior_provenance=dict(oe=dict(blue=source, red=dict(source, game_id=2))),
        roster={s: dict(available=True, matched=5, reference_game_id=g,
                       reference_source="oe_games+oe_ratings") for s, g in (("blue", 1), ("red", 2))})


class ConfidenceTests(unittest.TestCase):
    def test_capture_is_candidate_only_and_does_not_modify_priors(self):
        priors = fixture(); original = copy.deepcopy(priors)
        got = confidence.capture(priors, as_of_ts=NOW, teams=["A", "B"])
        self.assertEqual(priors, original)
        self.assertTrue(got["available"])
        self.assertFalse(got["production_applied"])
        self.assertLess(got["adjusted_priors"]["elo_oe"], priors["elo_oe"])
        json.dumps(got, allow_nan=False)

    def test_future_source_and_missing_metadata_preserve_baseline(self):
        for value in (None, NOW+1):
            p = fixture(); p["prior_provenance"]["oe"]["blue"]["date"] = value
            got = confidence.capture(p, as_of_ts=NOW)
            self.assertEqual(got["adjusted_priors"]["elo_oe"], p["elo_oe"])
            self.assertFalse(got["channels"]["elo_oe"]["available"])

    def test_absent_support_is_unknown_not_zero(self):
        got = confidence.capture(fixture(), as_of_ts=NOW)
        side = got["channels"]["elo_oe"]["sides"]["blue"]
        self.assertFalse(side["support_available"])
        self.assertGreater(side["confidence"], .9)

    def test_future_count_cannot_be_attached_to_an_older_rating(self):
        p = fixture(); source = p["prior_provenance"]["oe"]["blue"]
        source.update(sample_games=40, sample_window_days=120, sample_count_as_of_ts=NOW-86400)
        good = confidence.capture(p, as_of_ts=NOW)
        self.assertTrue(good["channels"]["elo_oe"]["sides"]["blue"]["support_available"])
        for timestamp in (NOW+1, NOW-10):
            source["sample_count_as_of_ts"] = timestamp
            bad = confidence.capture(p, as_of_ts=NOW)
            self.assertEqual(bad["adjusted_priors"]["elo_oe"], p["elo_oe"])
            self.assertEqual(bad["channels"]["elo_oe"]["sides"]["blue"]["reason"], "uncertified_or_future_sample_support")

    def test_wrong_reference_roster_cannot_reduce_confidence(self):
        p = fixture(); p["roster"]["blue"].update(matched=0, reference_game_id=999)
        got = confidence.capture(p, as_of_ts=NOW)
        side = got["channels"]["elo_oe"]["sides"]["blue"]
        self.assertNotIn("continuity", side["factors"])

    def test_resolved_identity_and_future_support_are_checked(self):
        p = fixture()
        p["pelo_adjustment"] = dict(blue=dict(applied=True, resolved_players=[
            dict(matched_name="a", live_name="A", games=20, last_ts=NOW+1)]))
        got = confidence.capture(p, as_of_ts=NOW)
        self.assertEqual(got["adjusted_priors"]["pelo_oe"], p["pelo_oe"])
        self.assertEqual(got["channels"]["pelo_oe"]["sides"]["blue"]["reason"], "future_source_date")
        p["pelo_adjustment"]["blue"]["resolved_players"] *= 2
        self.assertEqual(confidence.capture(p, as_of_ts=NOW)["channels"]["pelo_oe"]["sides"]["blue"]["reason"], "ambiguous_player_identity")

    def test_adjusted_lineup_does_not_use_old_roster_continuity(self):
        p = fixture(); p["roster"]["blue"]["matched"] = 0
        p["pelo_adjustment"] = dict(blue=dict(applied=True, resolved_players=[
            dict(matched_name=str(i), live_name=str(i), games=1000, last_ts=NOW) for i in range(5)]))
        side = confidence.capture(p, as_of_ts=NOW)["channels"]["pelo_oe"]["sides"]["blue"]
        self.assertNotIn("continuity", side["factors"])
        self.assertNotIn("age", side["factors"])
        self.assertGreater(side["confidence"], .95)

    def test_adjusted_lineup_without_resolved_identities_is_unknown(self):
        p = fixture(); p["pelo_adjustment"] = dict(blue=dict(applied=True))
        got = confidence.capture(p, as_of_ts=NOW)
        self.assertEqual(got["adjusted_priors"]["pelo_oe"], p["pelo_oe"])
        self.assertEqual(got["channels"]["pelo_oe"]["sides"]["blue"]["reason"], "missing_adjusted_player_identities")

    def test_mismatched_side_values_cannot_change_channel(self):
        p = fixture(); p["elo_blue"] = 9999
        got = confidence.capture(p, as_of_ts=NOW)
        self.assertEqual(got["adjusted_priors"]["elo_oe"], p["elo_oe"])
        self.assertEqual(got["channels"]["elo_oe"]["reason"], "side_rating_differential_mismatch")

    def test_side_reversal_reverses_candidate_difference(self):
        p = fixture(); got = confidence.capture(p, as_of_ts=NOW)["adjusted_priors"]["elo_oe"]
        q = fixture(); q["elo_blue"], q["elo_red"] = p["elo_red"], p["elo_blue"]; q["elo_oe"] *= -1
        q["prior_provenance"]["oe"]["blue"], q["prior_provenance"]["oe"]["red"] = p["prior_provenance"]["oe"]["red"], p["prior_provenance"]["oe"]["blue"]
        q["roster"]["blue"], q["roster"]["red"] = p["roster"]["red"], p["roster"]["blue"]
        self.assertAlmostEqual(confidence.capture(q, as_of_ts=NOW)["adjusted_priors"]["elo_oe"], -got)


if __name__ == "__main__":
    unittest.main()
