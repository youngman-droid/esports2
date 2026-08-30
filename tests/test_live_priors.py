import unittest
from unittest import mock

from lol_ticker import draft, live


class _Conn:
    """Serves the OE-ratings query first, then the gol.gg query."""

    def __init__(self, oe_rows, gg_rows=()):
        self.row_sets = [list(oe_rows), list(gg_rows)]
        self.calls = 0

    def execute(self, _query):
        self.calls += 1
        self._pending = self.row_sets[min(self.calls - 1, 1) % 2]
        return self

    def fetchall(self):
        return self._pending


_OE_ROW = {
    "blue_team": "Alpha", "red_team": "Beta", "winner": "Alpha",
    "date_utc": 1, "elo_blue": 1500.0, "elo_red": 1500.0,
    "pelo_blue": 1500.0, "pelo_red": 1500.0,
}
_GG_ROW = {
    "blue_team": "Alpha", "red_team": "Beta", "winner_side": "blue",
    "elo_blue_pre": 1500.0, "elo_red_pre": 1500.0,
}


class TeamPriorTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(live.team_priors, "_cache"):
            del live.team_priors._cache

    def test_latest_pregame_rating_is_advanced_by_known_result(self):
        conn = _Conn([_OE_ROW], [_GG_ROW])
        out = live.team_priors(conn, ["Alpha", "Beta"])
        self.assertTrue(out["found"])
        self.assertAlmostEqual(out["elo_blue"], 1515.0)
        self.assertAlmostEqual(out["elo_red"], 1485.0)
        self.assertAlmostEqual(out["elo_oe"], 30.0 / 400.0)
        self.assertAlmostEqual(out["pelo_oe"], 24.0 / 400.0)
        self.assertEqual(out["form_diff"], 1.0)
        self.assertAlmostEqual(out["elo_gg"], 30.0 / 400.0)

    def test_golgg_elo_carries_priors_while_oe_source_is_stale(self):
        conn = _Conn([], [_GG_ROW])
        out = live.team_priors(conn, ["Alpha", "Beta"])
        self.assertTrue(out["found"])
        self.assertFalse(out["oe_found"])
        self.assertEqual(out["elo_oe"], 0.0)
        self.assertEqual(out["form_diff"], 0.0)
        self.assertAlmostEqual(out["elo_gg"], 30.0 / 400.0)

    def test_cache_has_a_bounded_lifetime_shape(self):
        conn = _Conn([_OE_ROW], [_GG_ROW])
        live.team_priors(conn, ["Alpha", "Beta"])
        live.team_priors(conn, ["Alpha", "Beta"])
        self.assertEqual(conn.calls, 2)
        self.assertIn("at", live.team_priors._cache)


class FrameStateTests(unittest.TestCase):
    @staticmethod
    def _team(side, dead):
        participants = []
        for i in range(5):
            participants.append({
                "participantId": i + 1 + (5 if side == "red" else 0),
                "totalGold": 8000 + i * 100,
                "currentHealth": 0 if i < dead else 1000,
                "maxHealth": 1000,
                "level": 16,
                "creepScore": 200,
            })
        return {
            "participants": participants, "totalGold": 41000,
            "totalKills": 10, "towers": 5, "inhibitors": 0,
            "barons": 0, "dragons": [],
        }

    def test_frame_deaths_come_from_window_health_not_details(self):
        frame = {
            "rfc460Timestamp": "2026-08-29T00:40:00Z",
            "blueTeam": self._team("blue", dead=4),
            "redTeam": self._team("red", dead=0),
        }
        # A real details participant has items, but no currentHealth.
        details = {"participants": [
            {"participantId": i, "items": [1001]} for i in range(1, 11)
        ]}
        state = live._frame_state(
            {}, frame, None, details, {1001: 300},
            live._ts("2026-08-29T00:00:00Z"),
        )
        self.assertEqual(state["dead_blue"], 4)
        self.assertEqual(state["dead_red"], 0)
        self.assertEqual(state["hp_low_b"], 4.0)

    def test_baron_timer_starts_at_180_and_active_means_timer_positive(self):
        frame = {
            "blueTeam": self._team("blue", dead=0),
            "redTeam": self._team("red", dead=0),
        }
        prev = {"baron_counts": [0, 0], "last_baron_clock": [None, None]}
        frame["redTeam"]["barons"] = 1
        self.assertEqual(live._update_baron_state(prev, frame, 100.0), [0.0, 180.0])
        self.assertEqual(live._update_baron_state(prev, frame, 279.0), [0.0, 1.0])
        self.assertEqual(live._update_baron_state(prev, frame, 280.0), [0.0, 0.0])

    def test_elder_timer_starts_at_150_and_elder_does_not_create_soul(self):
        frame = {
            "rfc460Timestamp": "2026-08-29T00:40:00Z",
            "blueTeam": self._team("blue", dead=0),
            "redTeam": self._team("red", dead=0),
        }
        frame["blueTeam"]["dragons"] = ["ocean", "cloud", "infernal", "elder"]
        prev = {"elder_counts": [0, 0], "last_elder_clock": [None, None]}
        self.assertEqual(live._update_elder_state(prev, frame, 100.0), [150.0, 0.0])
        self.assertEqual(live._update_elder_state(prev, frame, 249.0), [1.0, 0.0])
        self.assertEqual(live._update_elder_state(prev, frame, 250.0), [0.0, 0.0])
        state = live._frame_state(
            {}, frame, None, None, {}, live._ts("2026-08-29T00:00:00Z"))
        self.assertEqual(state["drag_blue"], 3)
        self.assertEqual(state["elders"], 1)
        self.assertFalse(state["soul_blue"])


class TeamNameTests(unittest.TestCase):
    def test_sponsor_aliases_fold_to_canonical_names(self):
        self.assertEqual(draft.norm_team("Cloud9 Kia"), "cloud9")
        self.assertEqual(draft.norm_team("Cloud9"), "cloud9")
        self.assertEqual(draft.norm_team("Team Liquid Alienware"), "liquid")
        # 2026-08-29 audit: feed name on the left, stored name on the right
        self.assertEqual(draft.norm_team("NONGSHIM RED FORCE"),
                         draft.norm_team("Nongshim RedForce"))
        self.assertEqual(draft.norm_team("Relove Deep Cross Gaming"),
                         draft.norm_team("Deep Cross Gaming"))
        self.assertEqual(draft.norm_team("Beijing JDG Esports"),
                         draft.norm_team("JD Gaming"))
        self.assertEqual(draft.norm_team("AG.AL"),
                         draft.norm_team("Anyone's Legend"))
        self.assertEqual(draft.norm_team("Saigon Warrior"),
                         draft.norm_team("Saigon Warriors"))
        self.assertEqual(draft.norm_team("TP.HCM SN CyberCore Esports"),
                         draft.norm_team("SN CyberCore Esports"))
        self.assertEqual(draft.norm_team("Brod & Friends"),
                         draft.norm_team("Brod n Friends"))
        self.assertEqual(draft.norm_team("KaBuM! Eports"),
                         draft.norm_team("KaBuM! Esports"))
        self.assertEqual(draft.norm_team("DK Challengers"),
                         draft.norm_team("Dplus KIA Challengers"))

    def test_academy_rosters_stay_distinct(self):
        self.assertNotEqual(draft.norm_team("LYON Academy"),
                            draft.norm_team("LYON"))
        self.assertNotEqual(draft.norm_team("NONGSHIM RED FORCE"),
                            draft.norm_team("Nongshim RedForce Academy"))
        self.assertNotEqual(draft.norm_team("NS Challengers"),
                            draft.norm_team("Nongshim RedForce"))
        self.assertNotEqual(draft.norm_team("CTBC Flying Oyster Academy"),
                            draft.norm_team("CTBC Flying Oyster"))


class _SeqConn:
    """Serves one prepared row set per execute() call, in order."""

    def __init__(self, results):
        self.results = list(results)

    def execute(self, _query, _params=None):
        self._current = self.results.pop(0)
        return self

    def fetchall(self):
        return self._current

    def __iter__(self):
        return iter(self._current)


class RosterAwarenessTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(live._reference_roster, "_cache"):
            del live._reference_roster._cache

    def test_tagged_summoner_names_match_reference_roster(self):
        matched, new, missing = live.match_lineup(
            ["Thanatos", "Oddie", "Saint", "Hena", "Lyonz"],
            ["LYON Thanatos", "LYON Oddie", "LYON Saint", "LYON Hena", "LYON Lyonz"])
        self.assertEqual((len(matched), new, missing), (5, [], []))

    def test_substitution_is_flagged(self):
        matched, new, missing = live.match_lineup(
            ["Blaber", "Berserker", "Jojopyun", "Thanatos", "VULCAN"],
            ["C9 Blaber", "C9 Berserker", "C9 Jojopyun", "C9 Thanatos", "C9 Zven"])
        self.assertEqual(new, ["C9 Zven"])
        self.assertEqual(missing, ["VULCAN"])

    def test_tagless_join_matches_by_suffix_but_short_names_do_not(self):
        matched, new, missing = live.match_lineup(["Faker"], ["T1Faker"])
        self.assertEqual((new, missing), ([], []))
        matched, new, missing = live.match_lineup(["Bo"], ["XYZ Rambo"])
        self.assertEqual(new, ["XYZ Rambo"])
        self.assertEqual(missing, ["Bo"])

    def test_roster_check_reports_per_side_changes(self):
        games = [{"game_id": 7, "blue_team": "Alpha", "red_team": "Beta",
                  "date": "2026-08-20"}]
        alpha = [{"player": p} for p in ("A1", "A2", "A3", "A4", "A5")]
        beta = [{"player": p} for p in ("B1", "B2", "B3", "B4", "B5")]
        conn = _SeqConn([games, alpha, beta])
        out = live.roster_check(
            conn, ["Alpha", "Beta"],
            ["ALP A1", "ALP A2", "ALP A3", "ALP A4", "ALP Sub9"],
            ["BET B1", "BET B2", "BET B3", "BET B4", "BET B5"])
        self.assertTrue(out["blue"]["changed"])
        self.assertEqual(out["blue"]["new"], ["ALP Sub9"])
        self.assertEqual(out["blue"]["missing"], ["A5"])
        self.assertEqual(out["blue"]["reference_game_id"], 7)
        self.assertFalse(out["red"]["changed"])
        self.assertEqual(out["red"]["matched"], 5)


class MarketResolutionTests(unittest.TestCase):
    def test_sponsor_alias_and_terminal_map_title_resolve_both_exchanges(self):
        self.assertEqual(draft.norm_team("Team Liquid Alienware"), "liquid")
        kalshi = [
            {"ticker": "TL-M1", "title": "Team Liquid wins map 1",
             "yes_sub_title": "Team Liquid"},
            {"ticker": "SR-M1", "title": "Shopify Rebellion wins map 1",
             "yes_sub_title": "Shopify Rebellion"},
        ]
        polymarket = [{"markets": [{
            "question": "LoL: Team Liquid vs Shopify Rebellion - Game 1 Winner",
            "outcomes": '["Team Liquid", "Shopify Rebellion"]',
            "clobTokenIds": '["tl-token", "sr-token"]',
            "outcomePrices": '["0.6", "0.4"]',
            "slug": "lol-tl2-sr-game1", "closed": False,
            "acceptingOrders": True,
        }]}]
        with mock.patch.object(live, "_kalshi_markets", return_value=kalshi), \
                mock.patch.object(live, "_gamma_events", return_value=polymarket):
            resolved = live.resolve_markets(
                ["Team Liquid Alienware", "Shopify Rebellion"], 1)
        self.assertEqual(resolved["kalshi"], {
            "liquid": "TL-M1", "shopify rebellion": "SR-M1"})
        self.assertEqual(resolved["pm"], {
            "liquid": "tl-token", "shopify rebellion": "sr-token"})
        self.assertTrue(resolved["complete"])


if __name__ == "__main__":
    unittest.main()
