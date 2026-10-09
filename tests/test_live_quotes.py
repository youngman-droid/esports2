import threading
import unittest
from unittest import mock

from lol_ticker import live, live_quotes, shadow


class QuoteCacheTests(unittest.TestCase):
    game = {"game_id": "one", "teams": ["Alpha", "Beta"], "number": 1}

    def test_optional_blend_failure_does_not_disable_model_recording(self):
        with mock.patch.object(shadow.wphist, "load_artifact", side_effect=ValueError("old contract")):
            versions = shadow._artifact_versions()
        self.assertEqual(versions["blend_error"], "old contract")
        self.assertIsNotNone(versions["model_sha256"])
        self.assertIsNone(versions["blend_kind"])

    def test_recorder_provenance_uses_its_loaded_revision(self):
        with mock.patch.object(shadow, "_RECORDER_SOURCE_REVISION", "loaded-source"), \
                mock.patch.object(shadow, "_code_revision") as disk_revision:
            versions = shadow._artifact_versions()
        self.assertEqual(versions["code_revision"], "loaded-source")
        disk_revision.assert_not_called()

    def test_slow_lookup_does_not_block_capture_or_queue_unbounded_jobs(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def fetch(*args, **kwargs):
            entered.set()
            release.wait(2)
            finished.set()
            return {}
        cache = live_quotes.QuoteCache(fetch=fetch, max_workers=1)
        try:
            self.assertTrue(cache.request(self.game, 1000))
            self.assertTrue(entered.wait(1))
            self.assertEqual(cache.snapshot(self.game, 1000), {})
            self.assertFalse(cache.request(self.game, 1000))
            self.assertFalse(cache.request(dict(self.game, game_id="two"), 2000))
            self.assertFalse(finished.is_set())
        finally:
            release.set()
            self.assertTrue(finished.wait(1))

    def test_expired_unknown_future_and_other_attempt_quotes_are_missing(self):
        cache = live_quotes.QuoteCache(clock=lambda: 100.)
        key = cache.key(self.game, 1000)
        cache.entries[key] = (100., {
            "kalshi": {"p_blue": .6, "captured_ts_ms": 99000},
            "polymarket": {"p_blue": .6, "captured_ts_ms": 80000},
            "future": {"p_blue": .6, "captured_ts_ms": 101000},
            "unknown": {"p_blue": .6},
        })
        snapshot = cache.snapshot(self.game, 1000)
        self.assertEqual(list(snapshot), ["kalshi"])
        self.assertEqual(snapshot["kalshi"]["quote_age_s"], 1.)
        snapshot["kalshi"]["p_blue"] = .1
        self.assertEqual(cache.snapshot(self.game, 1000)["kalshi"]["p_blue"], .6)
        self.assertEqual(cache.snapshot(self.game, 2000), {})
        self.assertEqual(cache.snapshot(dict(self.game, teams=["Beta", "Alpha"]), 1000), {})

    def test_request_budget_is_shared_and_resets_after_error(self):
        with mock.patch.object(live.time, "monotonic", side_effect=[100., 100.25, 101.1]):
            with live.request_deadline(1):
                self.assertAlmostEqual(live._request_timeout(8), .75)
                with self.assertRaises(TimeoutError):
                    live._request_timeout(8)
        self.assertEqual(live._request_timeout(8), 8)

    def test_shadow_records_model_without_waiting_for_any_quote(self):
        conn = mock.Mock()
        estimate = dict(frames=[{"ts": 1990, "clock_s": 120, "p_blue": .6}],
                        game_start_ts=1000, feed_observed_at=1995,
                        upstream_feed_age_s=5.)
        versions = dict(model_kind="test", model_sha256="a", legacy_component_sha256=None,
                        stack_sha256="b", live_blend_w_gam=1., code_revision="c",
                        blend_sha256=None, blend_kind=None)
        cache = mock.Mock(); cache.snapshot.return_value = {}
        with mock.patch.object(live_quotes, "CACHE", cache), \
                mock.patch.object(live, "team_priors", return_value={}), \
                mock.patch.object(live, "estimate_series", return_value=estimate), \
                mock.patch.object(live, "market_prices") as blocking, \
                mock.patch.object(shadow, "_canonical_start", return_value=1000), \
                mock.patch.object(shadow, "_already_recorded", return_value=False), \
                mock.patch.object(shadow.wphist, "predict_live", return_value=None), \
                mock.patch.object(shadow.time, "time", return_value=2000):
            self.assertEqual(shadow._record_game(conn, {"protocol_id": "test"},
                                                 self.game, versions), 1)
        blocking.assert_not_called()
        args = conn.execute.call_args.args[1]
        self.assertIsNone(args[12])  # Polymarket remains missing, never backfilled.
        self.assertIsNone(args[13])
        timing = args[-2].obj["capture_timing"]
        self.assertEqual(timing["upstream_feed_age_s"], 5.)
        self.assertEqual(timing["processing_after_feed_s"], 5.)


if __name__ == "__main__":
    unittest.main()
