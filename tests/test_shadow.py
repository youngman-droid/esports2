import hashlib
import json
import unittest
from unittest import mock

from lol_ticker import live, shadow


class _MarketsConn:
    """Serves stored ``markets`` JSON rows for _quoted_markets."""

    def __init__(self, rows):
        self.rows = rows

    def execute(self, _query, _params=None):
        return self

    def fetchall(self):
        return [{"markets": m} for m in self.rows]


class ExchangeSettlementTests(unittest.TestCase):
    prediction = {"game_id": "g1", "game_start_ts": 100,
                  "blue_team": "T1", "red_team": "Gen.G Esports"}
    markets = {
        "kalshi": {"source": "map", "detail": {
            "t1": {"ticker": "KXLOLMAP-1-T1", "mid": 0.6},
            "gen g": {"ticker": "KXLOLMAP-1-GEN", "mid": 0.4}}},
        "polymarket": {"source": "map", "detail": {
            "t1": {"slug": "t1-vs-geng-game-1", "token_id": "a"},
            "gen g": {"slug": "t1-vs-geng-game-1", "token_id": "b"}}},
    }

    def test_quoted_markets_are_read_back_from_rows(self):
        tickers, slugs = shadow._quoted_markets(_MarketsConn([self.markets, {}]), self.prediction)
        self.assertEqual(set(tickers), {"KXLOLMAP-1-T1", "KXLOLMAP-1-GEN"})
        self.assertEqual(tickers["KXLOLMAP-1-GEN"], ("gen g", "map"))
        self.assertEqual(slugs, {"t1-vs-geng-game-1": "map"})

    def test_settlement_is_oriented_to_recorded_blue_team(self):
        self.assertEqual(shadow._settlement_winner(self.prediction, "t1", True), 1)
        self.assertEqual(shadow._settlement_winner(self.prediction, "t1", False), 0)
        self.assertEqual(shadow._settlement_winner(self.prediction, "gen g", True), 0)
        self.assertIsNone(shadow._settlement_winner(self.prediction, "dk", True))

    def test_kalshi_result_resolves_the_game(self):
        def fake_get(url, params=None, **_kw):
            if "KXLOLMAP-1-T1" in url:
                return {"market": {"result": "no", "status": "finalized"}}
            raise AssertionError("unexpected call %s" % url)
        with mock.patch.object(live, "_get", side_effect=fake_get):
            got = shadow._exchange_settlement(_MarketsConn([self.markets]), self.prediction)
        self.assertEqual(got[0], 0)
        self.assertEqual(got[1], "kalshi_settlement")
        self.assertEqual(got[2]["ticker"], "KXLOLMAP-1-T1")

    def test_unsettled_kalshi_falls_through_to_polymarket_resolution(self):
        def fake_get(url, params=None, **_kw):
            if "kalshi" in url:
                return {"market": {"result": "", "status": "active"}}
            return [{"closed": True, "outcomes": json.dumps(["T1", "Gen.G"]),
                     "outcomePrices": json.dumps(["1", "0"]),
                     "umaResolutionStatus": "resolved"}]
        with mock.patch.object(live, "_get", side_effect=fake_get):
            got = shadow._exchange_settlement(_MarketsConn([self.markets]), self.prediction)
        self.assertEqual(got[0], 1)
        self.assertEqual(got[1], "polymarket_settlement")

    def test_open_markets_leave_the_game_pending(self):
        def fake_get(url, params=None, **_kw):
            if "kalshi" in url:
                return {"market": {"result": "", "status": "active"}}
            return [{"closed": False, "outcomes": json.dumps(["T1", "Gen.G"]),
                     "outcomePrices": json.dumps(["0.7", "0.3"])}]
        with mock.patch.object(live, "_get", side_effect=fake_get):
            self.assertIsNone(shadow._exchange_settlement(
                _MarketsConn([self.markets]), self.prediction))

    def test_network_errors_do_not_resolve(self):
        with mock.patch.object(live, "_get", side_effect=OSError("down")):
            self.assertIsNone(shadow._exchange_settlement(
                _MarketsConn([self.markets]), self.prediction))


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

    def test_primary_cohort_excludes_rows_beyond_the_lead_window(self):
        inside = self._row("g1", 1, 1, 0.6, None)
        stale = self._row("g2", 1, 0, 0.9, None)
        stale["polymarket_lead_s"] = shadow.MAX_MARKET_LEAD_S + 300.0
        unknown = self._row("g3", 1, 0, 0.9, None)
        unknown["polymarket_lead_s"] = None
        got = shadow.score_rows([inside, stale, unknown], bootstrap=0)
        pm = got["platforms"]["polymarket"]
        self.assertEqual(pm["states"], 1)
        self.assertEqual(pm["excluded_lead_rows"], 2)
        self.assertEqual(pm["max_market_lead_s"], shadow.MAX_MARKET_LEAD_S)
        self.assertEqual(pm["coverage"]["resolved_model_rows"], 3)
        self.assertEqual(pm["coverage"]["eligible_quote_rows"], 1)
        self.assertEqual(pm["coverage"]["eligible_games_needed"], 99)
        self.assertEqual(pm["all_leads"]["states"], 3)
        self.assertNotIn("all_leads", pm["all_leads"])
        # an ungated score keeps every quoted row
        every = shadow.score_rows([inside, stale], bootstrap=0, max_lead_s=None)
        self.assertEqual(every["platforms"]["polymarket"]["states"], 2)

    def test_lead_gate_is_part_of_the_registered_protocol(self):
        cfg = shadow._protocol_config()
        self.assertEqual(cfg["primary_market_lead_s"], shadow.MAX_MARKET_LEAD_S)
        self.assertIn("settlement", cfg["outcome_sources"])

    def test_standalone_model_is_scored_when_historical_blend_is_rejected(self):
        rows = [self._row("g1", 1, 1, 0.6, None)]
        got = shadow.score_rows(rows, bootstrap=0)["platforms"]["polymarket"]
        self.assertEqual(got["games"], 1)
        self.assertEqual(got["paired"]["forecast"], "model")
        self.assertNotIn("blend", got["scores"])

    def test_mixed_deployments_keep_all_rows_and_use_recorded_forecasts(self):
        model = self._row("g1", 1, 1, 0.6, None)
        model.update(recommended_p=0.5, recommended_source="model")
        blend = self._row("g2", 1, 0, 0.4, 0.2)
        blend.update(recommended_p=0.1, recommended_source="historical_odds_blend")
        later = dict(blend, minute=2, recommended_p=0.0)
        got = shadow.score_rows([model, blend, later], bootstrap=0)["platforms"]["polymarket"]
        self.assertEqual((got["games"], got["states"]), (2, 3))
        self.assertEqual(got["paired"]["forecast"], "recommended")
        self.assertAlmostEqual(got["scores"]["recommended"]["brier_game"], 0.1275)
        self.assertAlmostEqual(got["scores"]["model"]["brier_game"], 0.25)
        self.assertAlmostEqual(got["paired"]["market_minus_forecast"], 0.0325)
        self.assertEqual(got["coverage"]["eligible_quote_rows"], 3)

    def test_legacy_mixed_rows_fall_back_to_blend_or_model(self):
        rows = [self._row("g1", 1, 1, 0.6, None),
                self._row("g2", 1, 0, 0.4, 0.2)]
        got = shadow.score_rows(rows, bootstrap=0)["platforms"]["polymarket"]
        self.assertEqual(got["games"], 2)
        self.assertEqual(got["paired"]["forecast"], "recommended")
        self.assertAlmostEqual(got["scores"]["recommended"]["brier_game"], 0.145)

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


class MarketValidityTests(unittest.TestCase):
    def test_rows_quoting_another_meeting_are_dropped_from_cohorts(self):
        start = live.ticker_time("KXLOLMAP-26SEP011230BRTMEA-1-BRT") + 1200
        good = {"game_start_ts": start, "markets": {"kalshi": {"detail": {
            "brute": {"ticker": "KXLOLMAP-26SEP011230BRTMEA-1-BRT"}}}}}
        bad = {"game_start_ts": start, "markets": {"kalshi": {"detail": {
            "brute": {"ticker": "KXLOLMAP-26SEP031000BRTNBS-1-BRT"}}}}}
        self.assertTrue(shadow._market_event_ok(good, "kalshi"))
        self.assertFalse(shadow._market_event_ok(bad, "kalshi"))
        self.assertTrue(shadow._market_event_ok({"game_start_ts": start, "markets": {}}, "polymarket"))

    def test_slug_date_guard(self):
        start = live.ticker_time("KXLOLMAP-26SEP011230BRTMEA-1-BRT") + 1200
        self.assertTrue(shadow._slug_date_ok("lol-brt-mea-2026-09-01-game1", start))
        self.assertFalse(shadow._slug_date_ok("lol-brt-nbs-2026-09-05-game1", start))
        self.assertTrue(shadow._slug_date_ok("no-date-here", start))


class ConfirmatoryFreezeTests(unittest.TestCase):
    def test_freeze_uses_scoring_eligibility_and_first_valid_capture_order(self):
        start = live.ticker_time("KXLOLMAP-26SEP011230BRTMEA-1-BRT")
        for platform, detail in (
                ("polymarket", {"slug": "lol-brt-nbs-2026-09-05-game1"}),
                ("kalshi", {"ticker": "KXLOLMAP-26SEP031000BRTNBS-1-BRT"})):
            with self.subTest(platform=platform):
                def row(game, minute=1, lead=45.0, quote=0.6):
                    r = ProspectiveScoreTests._row(game, minute, 1, None, None)
                    r.update(game_start_ts=start)
                    r[platform + "_p"] = quote
                    r[platform + "_lead_s"] = lead
                    return r

                invalid = row("g1")
                invalid["markets"] = {platform: {"detail": {"brute": detail}}}
                rows = [invalid, row("g2"), row("g2", minute=2),
                        row("g3", lead=500.0), row("g4", quote=None),
                        row("g5"), row("g6")]
                conn = mock.MagicMock()
                conn.execute.return_value.fetchall.return_value = []
                with mock.patch.object(shadow, "CONFIRMATORY_GAMES", 2):
                    frozen = shadow._freeze_confirmatory_games(conn, "test", rows)
                    scored = shadow.score_rows(rows, bootstrap=0)["platforms"][platform]
                self.assertEqual(frozen[platform], [("g2", start), ("g5", start)])
                self.assertEqual((scored["games"], scored["states"]), (3, 4))
                inserted = conn.cursor.return_value.__enter__.return_value.executemany
                self.assertEqual(inserted.call_args.args[1], [
                    ("test", platform, 1, "g2", start),
                    ("test", platform, 2, "g5", start)])
                for call in conn.execute.call_args_list:
                    self.assertNotIn("shadow_predictions", call.args[0])
                with mock.patch.object(shadow, "CONFIRMATORY_GAMES", 4):
                    self.assertEqual(
                        shadow._freeze_confirmatory_games(conn, "test", rows)[platform], [])

    def test_existing_frozen_membership_is_preserved_when_quotes_become_invalid(self):
        conn = mock.MagicMock()
        conn.execute.return_value.fetchall.side_effect = [
            [{"game_id": "old", "game_start_ts": 1000}], []]
        frozen = shadow._freeze_confirmatory_games(conn, "test", [])
        self.assertEqual(frozen["polymarket"], [("old", 1000)])
        conn.cursor.assert_not_called()

    def test_invalid_frozen_quotes_do_not_produce_a_confirmatory_result(self):
        row = ProspectiveScoreTests._row("g1", 1, 1, 0.6, None)
        row.update(game_start_ts=live.ticker_time("KXLOLMAP-26SEP011230BRTMEA-1-BRT"),
                   markets={"polymarket": {"detail": {
                       "brute": {"slug": "lol-brt-nbs-2026-09-05-game1"}}}})
        conn = mock.MagicMock()
        conn.execute.return_value.fetchall.side_effect = [
            [row], [{"game_id": row["game_id"], "game_start_ts": row["game_start_ts"]}], []]
        protocol = {"protocol_id": "test", "config": {"primary_market_lead_s": 90.0}}
        with mock.patch.object(shadow, "_protocol_by_id", return_value=protocol), \
                mock.patch.object(shadow, "_expose_resolved_history", return_value={}), \
                mock.patch.object(shadow, "_refresh_candidate_scores", return_value={}), \
                mock.patch.object(shadow, "CONFIRMATORY_GAMES", 1):
            got = shadow.score(conn, bootstrap=0, output_path=None)["platforms"]["polymarket"]
        self.assertFalse(got["confirmatory_ready"])
        self.assertNotIn("confirmatory", got)
        self.assertFalse(got.get("confirmatory_frozen", False))
        conn.cursor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
