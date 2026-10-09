import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from lol_ticker import wpcomposition as composition
from research import wpx_combination_composition as adapter


class ArchiveCompositionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = self.root / "inputs.npz"
        (self.root / "catalogs").mkdir()
        self.names = np.array(["Aatrox", "Amumu", "Ahri", "Ashe", "Alistar",
                               "Camille", "Corki", "Diana", "Draven", "Elise"])
        self.values = dict(gid=np.array([20, 20, 10, 10]), t=np.array([0, 60, 0, 60]),
            seq=np.array([-1, 2, -1, -1]), date=np.array(["2026-05-01"] * 4),
            patch=np.array(["16.9", "16.9", "16.9.1", "16.9.1"]),
            C=np.array([np.arange(10), np.arange(10), np.r_[np.arange(5, 10), np.arange(5)],
                        np.r_[np.arange(5, 10), np.arange(5)]]), champ_names=self.names)
        source = dict(version="16.9.1", data={champ: dict(name=champ,
            tags=["Tank"] if i < 3 else ["Fighter"],
            stats=dict(attackrange=125 + i * 25, hpperlevel=80 + i, armorperlevel=3 + i / 10))
            for i, champ in enumerate(self.names)})
        self.day_open = datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
        self.catalog = composition.catalog_from_payload(source,
            source_url="https://ddragon.leagueoflegends.com/cdn/16.9.1/data/en_US/champion.json",
            available_from_ts=self.day_open - 1)
        self.catalog_path = self.root / "catalogs" / "16.9.1.json"
        self.write_inputs()
        self.write_catalog()

    def write_inputs(self):
        np.savez(self.dataset, **self.values)
        fixed = self.values["seq"] < 0
        self.rows = {key: self.values[key][fixed].copy() for key in ("gid", "t", "date")}

    def write_catalog(self):
        self.catalog_path.write_text(json.dumps(self.catalog))

    def collect(self):
        return adapter.collect(self.dataset, self.rows, self.root)

    def test_alignment_role_order_sign_broadcast_and_explicit_static_scope(self):
        capture = self.collect()
        self.assertEqual(capture["names"].tolist(), composition.STATIC_NAMES)
        self.assertEqual(capture["extra"].shape, (3, 4))
        assert_array_equal(capture["gid"], [20, 10, 10])
        assert_array_equal(capture["t"], [0, 0, 60])
        assert_array_equal(capture["known"], np.ones((3, 4), dtype=bool))
        assert_array_equal(capture["extra"][1:], np.tile(-capture["extra"][0], (2, 1)))
        self.assertEqual(capture["extra"][0, 0], 3)
        self.assertEqual(capture["extra"][0, 1], -125)
        self.assertEqual(capture["provenance"]["coverage"]["covered_games"], 2)
        self.assertEqual(capture["provenance"]["coverage"]["covered_states"], 3)
        self.assertEqual(capture["provenance"]["sources"][str(self.catalog_path)], adapter.sha(self.catalog_path))
        self.assertTrue(capture["provenance"]["utc_day_gate"])
        json.dumps(capture["provenance"], allow_nan=False)

    def test_day_open_is_conservative_and_boundary_is_inclusive(self):
        self.catalog["available_from_ts"] = self.day_open + 1
        self.write_catalog()
        self.assertFalse(self.collect()["available"].any())
        self.catalog["available_from_ts"] = self.day_open
        self.write_catalog()
        self.assertTrue(self.collect()["available"].all())

    def test_future_missing_wrong_release_or_tampered_sources_stay_zero(self):
        self.catalog_path.unlink()
        missing = self.collect()
        assert_array_equal(missing["extra"], np.zeros((3, 4)))
        self.assertFalse(missing["available"].any())
        self.catalog["data"]["Aatrox"]["stats"]["attackrange"] = 999
        self.write_catalog()
        self.assertFalse(self.collect()["available"].any())
        self.catalog_path.write_text("invalid JSON")
        self.assertFalse(self.collect()["available"].any())

    def test_each_static_family_requires_all_ten_known_champions(self):
        source = copy.deepcopy(self.catalog)
        source["data"]["Aatrox"]["stats"].pop("hpperlevel")
        self.catalog = composition.catalog_from_payload(dict(version=source["version"], data=source["data"]),
            source_url=source["source_url"], available_from_ts=source["available_from_ts"])
        self.write_catalog()
        capture = self.collect()
        self.assertTrue(capture["available"].all())
        assert_array_equal(capture["known"][:, 2], [False] * 3)
        assert_array_equal(capture["extra"][:, 2], [0] * 3)
        self.assertTrue(capture["known"][:, [0, 1, 3]].all())

    def test_row_identity_mismatch_and_contradictory_drafts_reject(self):
        for key in ("gid", "t", "date"):
            saved = self.rows[key].copy()
            self.rows[key] = self.rows[key][::-1]
            if np.array_equal(saved, self.rows[key]):
                self.rows[key][0] = "2026-05-02" if key == "date" else 999
            with self.assertRaisesRegex(ValueError, "alignment"):
                self.collect()
            self.rows[key] = saved
        self.values["C"][3, 0] = 0
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.collect()

    def test_unconsumed_dates_and_invalid_or_duplicate_champions_are_rejected_or_unavailable(self):
        self.values["date"][:] = "2026-09-03"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "consumed"):
            self.collect()
        self.values["date"][:] = "2026-05-01"
        self.values["C"][:] = 0
        self.write_inputs()
        self.assertFalse(self.collect()["available"].any())
        self.values["C"][:] = -1
        self.write_inputs()
        self.assertFalse(self.collect()["available"].any())


if __name__ == "__main__":
    unittest.main()
