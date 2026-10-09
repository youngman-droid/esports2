import unittest

from scripts import fearless_pools as pools
from lol_ticker import wpfearless


class OfflinePoolsTests(unittest.TestCase):
    def fixture(self, day):
        return dict(id=day, day=day,
                    **{side: dict(picks={r: side + r for r in wpfearless.ROLES},
                                  players={r: "oe:player:" + side + r for r in wpfearless.ROLES})
                       for side in ("blue", "red")})

    def test_day_upper_bounds_boundary_exclusion_and_foreign_identity_are_explicit(self):
        games = [self.fixture("2026-09-01"), self.fixture("2026-09-02")]
        result = pools.build(games, "2026-09-03")
        self.assertEqual(result["records"], 10)
        self.assertEqual(result["excluded_records"], 10)
        self.assertEqual(result["identity_namespace"], "oracle_elixir_player_id")
        self.assertFalse(result["live_identity_bridge_available"])
        self.assertEqual(result["max_input_date"], "2026-09-02")

    def test_fresh_outcomes_are_rejected(self):
        with self.assertRaises(ValueError):
            pools.build([self.fixture("2026-09-03")], "2026-09-04")


if __name__ == "__main__":
    unittest.main()
