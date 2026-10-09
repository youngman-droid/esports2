import copy
from datetime import datetime, timedelta, timezone
import json
import os
import tempfile
import unittest

import numpy as np

from lol_ticker import wpdeploy, wpexposure


class ExposureTests(unittest.TestCase):
    def test_registry_union_cannot_relabel_other_experiment_outcomes(self):
        with tempfile.TemporaryDirectory() as td:
            sources = []
            for name, cutoff in zip(wpexposure.LEGACY_NAMES, ["2026-09-02", "2026-08-24"]):
                path = os.path.join(td, name)
                value = {"kind": "wpx_evaluation_registry_v1", "consumed_through": cutoff,
                         "history": [{"action": "earlier_inspection", "preserve": [1, 2]}]}
                with open(path, "w") as fh:
                    json.dump(value, fh)
                sources.append((path, wpexposure.sha256(path)))
            gids = np.array([10, 11, 12, 13, 14, 15])
            dates = np.array(["2026-08-01", "2026-08-10", "2026-08-24",
                              "2026-08-25", "2026-09-02", "2026-09-03"])
            path = os.path.join(td, "outcome_exposure.json")
            inventory = wpexposure.migrate(gids, dates, path=path)
            np.testing.assert_array_equal(wpexposure.fresh_mask(inventory, gids, dates),
                                           [False, False, False, False, False, True])
            # The historical runner uses the same union and excludes September 3.
            train, test, fresh, _ = wpdeploy._development_split(
                gids, dates, sources[1][0], exposure_path=path)
            self.assertFalse((train | test)[-1])
            self.assertTrue(test[-2])
            for source, before in sources:
                self.assertEqual(wpexposure.sha256(source), before)
            self.assertEqual(len(inventory["history"]), 2)
            wpexposure.migrate(gids, dates, path=path)
            self.assertEqual(len(wpexposure.load(path)["history"]), 2)
            # Canonical identity survives a date correction beyond the watermark.
            self.assertFalse(wpexposure.fresh_mask(inventory, [13], ["2026-09-11"])[0])

    def test_freeze_is_immutable_and_reservation_burns_before_scoring(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "exposure.json")
            today = datetime.now(timezone.utc).date()
            provenance = {k: "a" * 64 for k in ["candidate_sha256", "incumbent_sha256",
                "input_contract_sha256", "source_revision", "training_cutoff"]}
            rule = {"metric": "game_balanced_brier", "bootstrap_unit": "series_date",
                "endpoint": "fixed_date", "end_date": str(today + timedelta(days=30)),
                "minimum_games": 2, "precision_target": .01, "min_improvement": 0.0}
            plan = wpexposure.freeze_plan("test", provenance, {"leagues": "all"}, rule,
                                            str(today), path=path)
            self.assertEqual(plan, wpexposure.freeze_plan("test", provenance,
                {"leagues": "all"}, rule, str(today), path=path))
            changed = dict(rule, minimum_games=3)
            with self.assertRaisesRegex(ValueError, "already frozen"):
                wpexposure.freeze_plan("test", provenance, {"leagues": "all"}, changed,
                                          str(today), path=path)
            with self.assertRaisesRegex(ValueError, "endpoint has not closed"):
                wpexposure.reserve(plan["plan_sha256"], [1, 2],
                    [str(today + timedelta(days=1))] * 2, path=path)
            # Simulate a genuinely old registered endpoint, without outcomes.
            inventory = wpexposure.load(path)
            plan["start_after"] = "2020-01-01"
            plan["rule"]["end_date"] = "2020-01-31"
            old_hash = plan["plan_sha256"]
            plan["plan_sha256"] = wpexposure.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
            del inventory["plans"][old_hash]
            inventory["plans"][plan["plan_sha256"]] = plan
            with open(path, "w") as fh:
                json.dump(inventory, fh)
            wpexposure.reserve(plan["plan_sha256"], [1, 2], ["2020-01-10"] * 2, path=path)
            self.assertFalse(wpexposure.fresh_mask(wpexposure.load(path), [1], ["2099-01-01"])[0])
            with self.assertRaisesRegex(ValueError, "already reserved"):
                wpexposure.reserve(plan["plan_sha256"], [3, 4], ["2020-01-11"] * 2, path=path)

    def test_forward_exposures_preserve_unlinked_identity_and_quarantine_dates(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "exposure.json")
            got = wpexposure.expose([12, None], ["2026-09-02", "2026-09-03"],
                "candidate", "plan", path=path, feed_game_ids=["feed-12", "feed-13"])
            self.assertEqual(got["canonical_games_added"], 1)
            inventory = wpexposure.load(path)
            self.assertIn("feed-13", inventory["unlinked_feed_games"])
            np.testing.assert_array_equal(wpexposure.fresh_mask(inventory,
                [12, 13, 14], ["2026-09-04", "2026-09-03", "2026-09-04"]),
                [False, False, True])
            before = wpexposure.sha256(path)
            wpexposure.expose([12, None], ["2026-09-02", "2026-09-03"],
                "candidate", "plan", path=path, feed_game_ids=["feed-12", "feed-13"])
            self.assertEqual(before, wpexposure.sha256(path))


class ClusterTests(unittest.TestCase):
    def test_series_date_groups_maps_and_falls_back_for_missing_match(self):
        got = wpexposure.cluster_labels([1, 2, 3, 4, 5],
            ["2026-01-01"] * 3 + ["2026-01-02"] * 2, [10, 10, 11, 12, None])
        self.assertEqual(got[0], got[1])
        self.assertNotEqual(got[1], got[2])
        self.assertEqual(got[3], got[4])
        result = wpexposure.paired_interval([.8] * 5, [.7] * 5, [1.] * 5,
                                            [1, 2, 3, 4, 5], got, bootstrap=50)
        self.assertEqual(result["clusters"], 3)
        self.assertEqual(result["games"], 5)
        self.assertAlmostEqual(result["candidate_minus_incumbent"], -.05)

    def test_single_series_cannot_pass_even_with_many_maps(self):
        gate = wpdeploy.paired_gate([.6] * 20, [.9] * 20, [1.] * 20,
            np.arange(1, 21), clusters=["same series"] * 20, bootstrap=100)
        self.assertFalse(gate["passed"])
        self.assertIsNone(gate["ci95"])

    def test_resampling_retains_equal_game_not_equal_cluster_weight(self):
        result = wpexposure.paired_interval([0., 0., 0., 1.], [1., 1., 1., 0.],
            [1.] * 4, [1, 2, 3, 4], ["A", "A", "A", "B"], bootstrap=50)
        self.assertAlmostEqual(result["candidate_minus_incumbent"], .5)


if __name__ == "__main__":
    unittest.main()
