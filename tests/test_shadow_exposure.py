import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lol_ticker import shadow, wpexposure, wpcandidate


class ShadowExposureTests(unittest.TestCase):
    def test_adopts_other_protocol_history_and_rows_without_quotes_before_scoring(self):
        start = int(dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc).timestamp())
        current = {"game_id": "current", "game_start_ts": start, "blue_win": 1,
                   "polymarket_p": None, "kalshi_p": None}
        old = {"game_id": "old", "game_start_ts": start - 86400}
        conn = mock.MagicMock()
        conn.execute.return_value.fetchall.side_effect = [
            [current], [old], [{"esports_game_id": "current", "golgg_game_id": 2},
                              {"esports_game_id": "old", "golgg_game_id": 1}]]
        protocol = {"protocol_id": "test", "config": {"primary_market_lead_s": 90.0}}
        calls = []
        def exposed(*args, **kwargs):
            calls.append("exposure")
            self.assertEqual(args[0], [2, 1])
            self.assertEqual(args[1], ["2026-09-10", "2026-09-09"])
            self.assertEqual(kwargs["feed_game_ids"], ["current", "old"])
            return {"canonical_games_added": 2, "unlinked_games_added": 0}
        def scored(*args, **kwargs):
            self.assertEqual(calls, ["exposure"])
            calls.append("score")
            return {"platforms": {}}
        with mock.patch.object(shadow, "_protocol_by_id", return_value=protocol), \
                mock.patch.object(shadow, "_table_exists", return_value=True), \
                mock.patch.object(shadow, "_freeze_confirmatory_games", return_value={}), \
                mock.patch.object(shadow, "_refresh_candidate_scores", return_value={}), \
                mock.patch.object(wpexposure, "expose", side_effect=exposed), \
                mock.patch.object(shadow, "score_rows", side_effect=scored):
            result = shadow.score(conn, output_path=None, bootstrap=0)
        self.assertEqual(result["outcome_exposure"]["resolved_attempts_adopted"], 2)
        self.assertEqual(calls, ["exposure", "score"])
        history_sql = conn.execute.call_args_list[1].args[0]
        self.assertNotIn("protocol_id=%s", history_sql)
        self.assertIn("p.captured_at < o.recorded_at", history_sql)

    def test_unlinked_history_is_retained_and_repeat_adoption_is_idempotent(self):
        start = int(dt.datetime(2026, 9, 9, tzinfo=dt.timezone.utc).timestamp())
        conn = mock.MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            {"game_id": "unlinked", "game_start_ts": start},
            {"game_id": "unlinked", "game_start_ts": start}]
        real_expose = wpexposure.expose
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "exposure.json")
            def exposed(*args, **kwargs):
                return real_expose(*args, **kwargs, path=path)
            with mock.patch.object(shadow, "_table_exists", return_value=False), \
                    mock.patch.object(wpexposure, "expose", side_effect=exposed):
                first = shadow._expose_resolved_history(conn, {"protocol_id": "p", "config": {}}, [])
                second = shadow._expose_resolved_history(conn, {"protocol_id": "p", "config": {}}, [])
            inventory = wpexposure.load(path)
        self.assertEqual(first["unlinked_games_added"], 1)
        self.assertEqual(second["unlinked_games_added"], 0)
        self.assertEqual(inventory["consumed_through"], "2026-09-09")
        self.assertIn("unlinked", inventory["unlinked_feed_games"])
        self.assertEqual(len(inventory["history"]), 1)

    def test_exposure_failure_prevents_score_and_output_publication(self):
        conn = mock.MagicMock(); conn.execute.return_value.fetchall.return_value = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with mock.patch.object(shadow, "_protocol_by_id", return_value={"protocol_id": "p", "config": {}}), \
                    mock.patch.object(shadow, "_expose_resolved_history", side_effect=OSError("ledger write failed")), \
                    mock.patch.object(shadow, "_freeze_confirmatory_games") as freeze, \
                    mock.patch.object(shadow, "score_rows") as score:
                with self.assertRaises(OSError):
                    shadow.score(conn, output_path=str(path))
            self.assertFalse(path.exists())
            freeze.assert_not_called(); score.assert_not_called()

    def test_empty_history_has_no_exposure_write(self):
        conn = mock.MagicMock(); conn.execute.return_value.fetchall.return_value = []
        with mock.patch.object(wpexposure, "expose") as expose:
            result = shadow._expose_resolved_history(conn, {"protocol_id": "p", "config": {}}, [])
        expose.assert_not_called()
        self.assertEqual(result["resolved_attempts_adopted"], 0)

    def test_common_score_refreshes_candidate_sidecar_after_incumbent_publication(self):
        import json
        conn = mock.MagicMock(); conn.execute.return_value.fetchall.return_value = []
        protocol = {"protocol_id": "p", "config": {}}
        with tempfile.TemporaryDirectory() as directory:
            incumbent = Path(directory) / "incumbent.json"
            def candidate_score(*args, **kwargs):
                self.assertTrue(incumbent.exists())
                self.assertEqual(json.loads(incumbent.read_text())["protocol_id"], "p")
                return {"scored_at": "2026-09-12", "candidates": []}
            with mock.patch.object(shadow, "_protocol_by_id", return_value=protocol), \
                    mock.patch.object(shadow, "_expose_resolved_history", return_value={}), \
                    mock.patch.object(shadow, "_freeze_confirmatory_games", return_value={}), \
                    mock.patch.object(shadow, "score_rows", return_value={"platforms": {}}), \
                    mock.patch.object(wpcandidate, "score", side_effect=candidate_score):
                result = shadow.score(conn, output_path=str(incumbent), bootstrap=0)
            sidecar = Path(directory) / "shadow_candidate_score.json"
            self.assertEqual(json.loads(sidecar.read_text())["candidates"], [])
        self.assertEqual(result["candidate_score_refresh"]["status"], "ok")

    def test_candidate_metric_failure_preserves_previous_sidecar(self):
        conn = mock.MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "candidate.json"; sidecar.write_text("previous")
            with mock.patch.object(wpcandidate, "score", side_effect=RuntimeError("metric failed")), \
                    self.assertLogs("shadow", level="ERROR"):
                result = shadow._refresh_candidate_scores(conn, 0, str(sidecar))
            self.assertEqual(sidecar.read_text(), "previous")
        self.assertEqual(result["status"], "failed")
        conn.rollback.assert_called_once()

    def test_candidate_exposure_failure_propagates_without_publication(self):
        conn = mock.MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "candidate.json"
            with mock.patch.object(wpcandidate, "score", side_effect=wpcandidate.OutcomeExposureError("write failed")):
                with self.assertRaises(wpcandidate.OutcomeExposureError):
                    shadow._refresh_candidate_scores(conn, 0, str(sidecar))
            self.assertFalse(sidecar.exists())
        conn.rollback.assert_called_once()

    def test_explicit_historical_protocol_does_not_inspect_current_candidates(self):
        conn = mock.MagicMock(); conn.execute.return_value.fetchall.return_value = []
        with mock.patch.object(shadow, "_protocol_by_id", return_value={"protocol_id": "old", "config": {}}), \
                mock.patch.object(shadow, "_expose_resolved_history", return_value={}), \
                mock.patch.object(shadow, "_freeze_confirmatory_games", return_value={}), \
                mock.patch.object(shadow, "score_rows", return_value={"platforms": {}}), \
                mock.patch.object(shadow, "_refresh_candidate_scores") as refresh:
            shadow.score(conn, bootstrap=0, output_path=None, protocol_id="old")
        refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
