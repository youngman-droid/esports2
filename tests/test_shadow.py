import hashlib
import unittest

from lol_ticker import shadow


class ProtocolTests(unittest.TestCase):
    def test_protocol_id_is_content_addressed(self):
        cfg = {"version": "x", "started_at": "2026-01-01T00:00:00Z"}
        self.assertEqual(shadow._protocol_id(cfg), shadow._protocol_id(dict(cfg)))
        self.assertNotEqual(shadow._protocol_id(cfg),
                            shadow._protocol_id({**cfg, "version": "y"}))

    def test_stale_and_settled_quotes_are_not_scoreable(self):
        markets = {
            "polymarket": {"p_blue": 0.6, "stale": True},
            "kalshi": {"p_blue": 0.55, "settled": False},
        }
        self.assertEqual(shadow._valid_market_quotes(markets), {"kalshi": 0.55})

    def test_feed_side_orientation_swaps_schedule_order(self):
        game = {"teams": ["Red", "Blue"], "team_ids": ["r", "b"],
                "wins": [1, 0]}
        got = shadow._orient_game(game, {"blue_team_id": "b"})
        self.assertEqual(got["teams"], ["Blue", "Red"])
        self.assertEqual(got["wins"], [0, 1])

    def test_outcome_is_reoriented_to_recorded_blue_team(self):
        prediction = {"blue_team": "Team B", "red_team": "Team A"}
        got = shadow._winner_from_teams(
            prediction, blue_team="Team A", red_team="Team B", winner_side="red")
        self.assertEqual(got, 1)


class ProspectiveScoreTests(unittest.TestCase):
    @staticmethod
    def _row(game, minute, outcome, market, blend):
        return {
            "game_id": game, "game_start_ts": 1000 + int(game[-1]),
            "minute": minute, "blue_win": outcome,
            "model_p": 0.5,
            "polymarket_p": market, "polymarket_blend_p": blend,
            "kalshi_p": None, "kalshi_blend_p": None,
            "polymarket_lead_s": 45.0, "kalshi_lead_s": None,
            "model_sha256": "a" * 64, "blend_sha256": "b" * 64,
            "legacy_component_sha256": "c" * 64,
            "stack_sha256": "d" * 64, "live_blend_w_gam": 0.45,
            "code_revision": "e" * 40,
        }

    def test_score_is_game_balanced_and_paired_on_identical_rows(self):
        rows = [
            self._row("g1", 1, 1, 0.6, 0.8),
            self._row("g1", 2, 1, 0.6, 0.8),
            self._row("g2", 1, 0, 0.4, 0.2),
        ]
        got = shadow.score_rows(rows, bootstrap=100)
        pm = got["platforms"]["polymarket"]
        self.assertEqual(pm["games"], 2)
        self.assertEqual(pm["states"], 3)
        self.assertAlmostEqual(pm["scores"]["market"]["brier_game"], 0.16)
        self.assertAlmostEqual(pm["scores"]["blend"]["brier_game"], 0.04)
        self.assertAlmostEqual(pm["paired"]["market_minus_blend"], 0.12)
        self.assertEqual(pm["median_market_lead_s"], 45.0)
        self.assertFalse(pm["confirmatory_ready"])

    def test_platform_cohort_requires_market(self):
        rows = [self._row("g1", 1, 1, None, None)]
        got = shadow.score_rows(rows, bootstrap=0)
        self.assertEqual(got["platforms"]["polymarket"]["games"], 0)

    def test_standalone_model_is_scored_when_historical_blend_is_rejected(self):
        rows = [self._row("g1", 1, 1, 0.6, None)]
        got = shadow.score_rows(rows, bootstrap=0)["platforms"]["polymarket"]
        self.assertEqual(got["games"], 1)
        self.assertEqual(got["paired"]["forecast"], "model")
        self.assertNotIn("blend", got["scores"])

    def test_scores_are_separated_by_exact_stack_version(self):
        old = self._row("g1", 1, 1, 0.6, None)
        new = self._row("g2", 1, 0, 0.4, None)
        new["stack_sha256"] = "f" * 64
        new["legacy_component_sha256"] = "9" * 64
        got = shadow.score_rows([old, new], bootstrap=0)
        revision_key = hashlib.sha256(("e" * 40).encode()).hexdigest()[:12]
        old_key = "d" * 12 + "@" + revision_key
        new_key = "f" * 12 + "@" + revision_key
        self.assertEqual(got["artifact_versions"], {old_key: 1, new_key: 1})
        self.assertEqual(set(got["by_artifact_version"]), {old_key, new_key})
        self.assertEqual(
            got["by_artifact_version"][new_key]["legacy_component_sha256"],
            "9" * 64)

    def test_legacy_rows_are_marked_as_partial_versions(self):
        row = self._row("g1", 1, 1, 0.6, None)
        row.pop("stack_sha256")
        self.assertTrue(shadow._version_key(row).startswith("partial-"))

    def test_dirty_revisions_on_same_commit_get_distinct_cohorts(self):
        first = self._row("g1", 1, 1, 0.6, None)
        second = dict(first)
        first["code_revision"] = "a" * 40 + "+worktree.1111111111111111"
        second["code_revision"] = "a" * 40 + "+worktree.2222222222222222"
        self.assertNotEqual(shadow._version_key(first),
                            shadow._version_key(second))


if __name__ == "__main__":
    unittest.main()
