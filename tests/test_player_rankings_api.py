"""Ranking refresh and HTTP failure behavior without a database or socket."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Importing the incumbent dashboard starts its market refresh daemon. These
# route tests use no market data and must not trigger that background network IO.
with mock.patch("threading.Thread.start"):
    from lol_ticker import dashboard, player_rankings


def _artifact(name="First", score=1.25):
    return {
        "meta": {"as_of": "2026-10-04", "source": "oracle_csv"},
        "players": [{"player_id": "oe:player:first", "name": name,
                     "impact": score, "history": [score]}],
    }


class RankingsArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ratings.json"
        for patcher in (
                mock.patch.object(player_rankings, "ARTIFACT_PATH", self.path),
                mock.patch.object(player_rankings, "_cache_key", None),
                mock.patch.object(player_rankings, "_cache_payload", None),
                mock.patch.object(dashboard, "_db", side_effect=AssertionError("DB accessed")),
                mock.patch.object(dashboard.db, "connect", side_effect=AssertionError("DB opened"))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _replace(self, payload):
        replacement = self.path.with_suffix(".next")
        replacement.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(replacement, self.path)

    def test_atomic_replacement_refreshes_even_with_equal_size_and_mtime(self):
        self._replace(_artifact())
        original = self.path.stat()
        self.assertEqual(player_rankings.ratings()["players"][0]["name"], "First")
        self._replace(_artifact("Other"))
        os.utime(self.path, ns=(original.st_atime_ns, original.st_mtime_ns))
        updated = self.path.stat()
        self.assertEqual(updated.st_size, original.st_size)
        self.assertEqual(updated.st_mtime_ns, original.st_mtime_ns)
        self.assertNotEqual(updated.st_ino, original.st_ino)
        self.assertEqual(player_rankings.ratings()["players"][0]["name"], "Other")

    def test_returned_payload_cannot_mutate_the_cached_artifact(self):
        expected = _artifact()
        self._replace(expected)
        returned = player_rankings.ratings()
        returned["meta"]["source"] = "changed"
        returned["players"][0]["history"].append(999)
        returned["players"].clear()
        self.assertEqual(player_rankings.ratings(), expected)

    def test_missing_or_deleted_artifact_is_unavailable_and_recovers(self):
        with self.assertRaises(player_rankings.RankingsUnavailable):
            player_rankings.ratings()
        self._replace(_artifact())
        self.assertEqual(player_rankings.ratings(), _artifact())
        self.path.unlink()
        with self.assertRaises(player_rankings.RankingsUnavailable):
            player_rankings.ratings()
        self._replace(_artifact("Other"))
        self.assertEqual(player_rankings.ratings(), _artifact("Other"))

    def test_invalid_replacement_never_serves_a_previously_cached_snapshot(self):
        for invalid in (None, [], {}, {"players": {}}, {"players": []},
                        {"players": [], "meta": []}):
            with self.subTest(invalid=invalid):
                self._replace(_artifact())
                self.assertEqual(player_rankings.ratings(), _artifact())
                self._replace(invalid)
                with self.assertRaises(player_rankings.RankingsUnavailable):
                    player_rankings.ratings()
        self._replace(_artifact("Other"))
        self.assertEqual(player_rankings.ratings(), _artifact("Other"))

    def test_malformed_json_replacement_is_unavailable(self):
        self._replace(_artifact())
        player_rankings.ratings()
        self.path.write_text('{"players": [', encoding="utf-8")
        with self.assertRaises(player_rankings.RankingsUnavailable) as caught:
            player_rankings.ratings()
        self.assertIn("python3 -m lol_ticker.player_ratings build", str(caught.exception))

    def test_nonfinite_constants_and_overflow_numbers_are_unavailable(self):
        for number in ("NaN", "Infinity", "-Infinity", "1e309", "-1e309"):
            with self.subTest(number=number):
                self._replace(_artifact())
                player_rankings.ratings()
                self.path.write_text('{"meta":{},"players":[{"impact":' + number + '}]}',
                                     encoding="utf-8")
                with self.assertRaises(player_rankings.RankingsUnavailable):
                    player_rankings.ratings()


class RankingsHttpTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ratings.json"
        for patcher in (
                mock.patch.object(player_rankings, "ARTIFACT_PATH", self.path),
                mock.patch.object(player_rankings, "_cache_key", None),
                mock.patch.object(player_rankings, "_cache_payload", None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.db_getter = mock.patch.object(dashboard, "_db", side_effect=AssertionError("DB accessed"))
        self.db_connect = mock.patch.object(dashboard.db, "connect", side_effect=AssertionError("DB opened"))
        self.no_db = self.db_getter.start()
        self.no_connect = self.db_connect.start()
        self.addCleanup(self.db_getter.stop)
        self.addCleanup(self.db_connect.stop)

    def tearDown(self):
        self.no_db.assert_not_called()
        self.no_connect.assert_not_called()

    def _get(self, path):
        # Exercise the real handler and routes; only transport is replaced.
        handler = object.__new__(dashboard.Handler)
        handler.path = path
        handler._send = mock.Mock()
        handler.do_GET()
        self.assertEqual(handler._send.call_count, 1)
        return handler._send.call_args.args

    def test_players_page_and_trailing_slash_serve_html_without_ratings_or_db(self):
        page = Path(self.directory.name) / "players.html"
        html = b"<!doctype html><title>World player impact</title>"
        page.write_bytes(html)
        with mock.patch.object(player_rankings, "PAGE_PATH", page):
            for path in ("/players", "/players/", "/players?role=mid"):
                with self.subTest(path=path):
                    self.assertEqual(self._get(path), (200, html, "text/html; charset=utf-8"))

    def test_ratings_endpoint_serves_artifact_and_refreshes_atomically(self):
        self.path.write_text(json.dumps(_artifact()), encoding="utf-8")
        status, body, content_type = self._get("/api/players/ratings?role=mid")
        self.assertEqual((status, content_type), (200, "application/json"))
        self.assertEqual(json.loads(body), _artifact())
        replacement = self.path.with_suffix(".next")
        replacement.write_text(json.dumps(_artifact("Other")), encoding="utf-8")
        os.replace(replacement, self.path)
        status, body, content_type = self._get("/api/players/ratings")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), _artifact("Other"))

    def test_missing_artifact_returns_actionable_503_without_opening_db(self):
        status, body, content_type = self._get("/api/players/ratings")
        self.assertEqual((status, content_type), (503, "application/json"))
        self.assertIn("python3 -m lol_ticker.player_ratings build", json.loads(body)["error"])

    def test_invalid_replacement_returns_503_instead_of_stale_ratings(self):
        self.path.write_text(json.dumps(_artifact()), encoding="utf-8")
        self.assertEqual(self._get("/api/players/ratings")[0], 200)
        self.path.write_text('{"players":[]}', encoding="utf-8")
        status, body, content_type = self._get("/api/players/ratings")
        self.assertEqual((status, content_type), (503, "application/json"))
        self.assertEqual(set(json.loads(body)), {"error"})

    def test_nonfinite_artifact_returns_503_without_opening_db(self):
        for number in ("NaN", "Infinity", "-Infinity", "1e309"):
            with self.subTest(number=number):
                self.path.write_text('{"meta":{},"players":[{"impact":' + number + '}]}',
                                     encoding="utf-8")
                status, body, content_type = self._get("/api/players/ratings")
                self.assertEqual((status, content_type), (503, "application/json"))
                self.assertEqual(set(json.loads(body)), {"error"})

    def test_early_archive_endpoint_never_opens_match_database(self):
        payload={"meta":{"start":"2010-09-30"},"matches":[],"games":[]}
        self.path.with_name("early_archive.json").write_text(json.dumps(payload))
        status,body,kind=self._get("/api/players/archive")
        self.assertEqual((status,kind),(200,"application/json"))
        self.assertEqual(json.loads(body),payload)

    def test_missing_early_archive_returns_503(self):
        self.assertEqual(self._get("/api/players/archive")[0],503)

    def test_early_archive_page_is_served_without_model_or_db(self):
        page=Path(self.directory.name)/"players.html"
        archive=page.with_name("player_archive.html")
        archive.write_bytes(b"<!doctype html><title>Early archive</title>")
        with mock.patch.object(player_rankings,"PAGE_PATH",page):
            for url in ("/players/archive","/players/archive/"):
                self.assertEqual(self._get(url),(200,archive.read_bytes(),"text/html; charset=utf-8"))

    def test_json_response_sets_length_and_browser_access_headers(self):
        handler = object.__new__(dashboard.Handler)
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        handler.wfile = io.BytesIO()
        body = json.dumps(_artifact()).encode("utf-8")
        handler._send(200, body, "application/json")
        handler.send_response.assert_called_once_with(200)
        self.assertIn(mock.call("Content-Length", str(len(body))), handler.send_header.call_args_list)
        self.assertIn(mock.call("Content-Type", "application/json"), handler.send_header.call_args_list)
        self.assertIn(mock.call("Access-Control-Allow-Origin", "*"), handler.send_header.call_args_list)
        self.assertEqual(handler.wfile.getvalue(), body)


if __name__ == "__main__":
    unittest.main()
