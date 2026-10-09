import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from lol_ticker import wpcomposition as comp


class CompositionTests(unittest.TestCase):
    def fixture(self):
        blue = dict(zip(comp.ROLES, ["Aatrox", "Amumu", "Ahri", "Ashe", "Alistar"]))
        red = dict(zip(comp.ROLES, ["Camille", "Corki", "Diana", "Draven", "Elise"]))
        data = {}
        annotations = {}
        for i, champ in enumerate(list(blue.values()) + list(red.values())):
            data[champ] = dict(name=champ, tags=["Tank"] if i % 3 == 0 else ["Fighter"],
                               stats=dict(attackrange=125 + 25 * i, hpperlevel=80 + i, armorperlevel=3 + i / 10))
            traits = {}
            for trait in ("engage", "disengage", "waveclear", "late_scaling", "primary_damage"):
                value = ("physical", "magic", "mixed")[i % 3] if trait == "primary_damage" else bool(i % 2)
                traits[trait] = dict(value=value, review_status="reviewed", reviewer="fixture reviewer",
                                     evidence_urls=["https://www.leagueoflegends.com/en-us/champions/" + champ.lower() + "/"],
                                     source_available_from_ts=50)
            annotations[comp.champion_key(champ)] = dict(version="16.9.1", traits=traits)
        catalog = comp.catalog_from_payload(dict(version="16.9.1", data=data),
                                             source_url="https://ddragon.leagueoflegends.com/cdn/16.9.1/data/en_US/champion.json",
                                             available_from_ts=50)
        return blue, red, catalog, annotations

    def test_static_proxies_are_literal_and_annotation_families_are_explicit(self):
        blue, red, catalog, annotations = self.fixture()
        out = comp.features(blue, red, "16.09", catalog=catalog, annotations=annotations, as_of_ts=100)
        self.assertTrue(out["complete"])
        self.assertTrue(all(out["known"]))
        values = dict(zip(out["names"], out["features"]))
        self.assertEqual(values["frontline_tank_tag_count"], 0.)
        self.assertEqual(values["mean_attack_range"], -125.)
        self.assertEqual(values["hp_growth_sum"], -25.)
        self.assertAlmostEqual(values["armor_growth_sum"], -2.5)
        self.assertEqual(values["engage_count"], -1.)
        self.assertTrue(all(n == 10 for n in out["coverage"].values()))
        json.dumps(out, allow_nan=False)

    def test_swapping_sides_negates_all_features_with_same_known_mask(self):
        blue, red, catalog, annotations = self.fixture()
        left = comp.features(blue, red, "16.9.1", catalog=catalog, annotations=annotations, as_of_ts=100)
        right = comp.features(red, blue, "16.9.1", catalog=catalog, annotations=annotations, as_of_ts=100)
        assert_array_equal(left["features"], -np.asarray(right["features"]))
        self.assertEqual(left["known"], right["known"])

    def test_static_data_never_invents_engage_damage_or_scaling(self):
        blue, red, catalog, _ = self.fixture()
        out = comp.features(blue, red, "16.9", catalog=catalog, as_of_ts=100)
        self.assertTrue(out["available"])
        self.assertFalse(out["complete"])
        self.assertEqual(out["known"], [True] * 4 + [False] * 8)
        self.assertEqual(out["features"][4:], [0.] * 8)

    def test_wrong_patch_future_source_and_tampered_catalog_are_zero(self):
        blue, red, catalog, _ = self.fixture()
        for patch, ts, mutate in (("16.10", 100, False), ("16.9.2", 100, False),
                                  ("16.9", 49, False), ("16.9", 100, True)):
            source = copy.deepcopy(catalog)
            if mutate:
                source["data"]["Aatrox"]["stats"]["attackrange"] = 999
            out = comp.features(blue, red, patch, catalog=source, as_of_ts=ts)
            self.assertFalse(out["available"])
            self.assertEqual(out["features"], [0.] * len(comp.FEATURE_NAMES))

    def test_incomplete_traits_are_family_gated(self):
        blue, red, catalog, annotations = self.fixture()
        annotations["aatrox"]["traits"]["engage"]["source_available_from_ts"] = 101
        out = comp.features(blue, red, "16.9", catalog=catalog, annotations=annotations, as_of_ts=100)
        i = comp.FEATURE_NAMES.index("engage_count")
        self.assertFalse(out["known"][i])
        self.assertEqual(out["coverage"]["engage_count"], 9)
        self.assertEqual(out["features"][i], 0.)
        self.assertTrue(out["known"][comp.FEATURE_NAMES.index("waveclear_count")])

    def test_missing_duplicate_drafts_and_nonreviewed_or_nonprimary_labels_reject(self):
        blue, red, catalog, annotations = self.fixture()
        incomplete = dict(blue)
        incomplete.pop("sup")
        self.assertFalse(comp.features(incomplete, red, "16.9", catalog=catalog, as_of_ts=100)["available"])
        duplicate = dict(red)
        duplicate["sup"] = blue["top"]
        self.assertFalse(comp.features(blue, duplicate, "16.9", catalog=catalog, as_of_ts=100)["available"])
        annotations["aatrox"]["traits"]["engage"]["review_status"] = "pending"
        annotations["ahri"]["traits"]["waveclear"]["evidence_urls"] = ["https://example.com/guess"]
        annotations["ashe"]["traits"]["primary_damage"]["value"] = ["magic"]
        out = comp.features(blue, red, "16.9", catalog=catalog, annotations=annotations, as_of_ts=100)
        for name in ("engage_count", "waveclear_count", "damage_type_balance"):
            self.assertFalse(out["known"][comp.FEATURE_NAMES.index(name)])

    def test_metadata_join_permutations_and_local_capture(self):
        blue, red, catalog, annotations = self.fixture()
        md = {}
        for side, champs, offset in (("blue", blue, 0), ("red", red, 5)):
            md[side + "TeamMetadata"] = dict(participantMetadata=[
                dict(participantId=i + offset + 1, role=r, championId=champs[r])
                for i, r in enumerate(comp.ROLES)][::-1])
        direct = comp.from_metadata(md, "16.9", catalog=catalog, annotations=annotations, as_of_ts=100)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "catalogs").mkdir()
            (root / "annotations").mkdir()
            (root / "catalogs" / "16.9.1.json").write_text(json.dumps(catalog))
            (root / "annotations" / "16.9.1.json").write_text(json.dumps(annotations))
            captured = comp.capture(md, "16.09", as_of_ts=100, source_root=root)
            self.assertEqual(direct["features"], captured["features"])
            self.assertTrue(captured["complete"])
        md["blueTeamMetadata"]["participantMetadata"][0]["participantId"] = 6
        self.assertFalse(comp.from_metadata(md, "16.9", catalog=catalog, as_of_ts=100)["available"])

    def test_mode_specific_shared_display_names_cannot_overwrite_base_stats(self):
        blue, red, catalog, _ = self.fixture()
        before = comp.features(blue, red, "16.9", catalog=catalog, as_of_ts=100)
        data = copy.deepcopy(catalog["data"])
        data["Jade_Ahri"] = dict(name="Ahri", tags=["Tank"],
                                 stats=dict(attackrange=9999, hpperlevel=9999, armorperlevel=9999))
        mixed = comp.catalog_from_payload(dict(version="16.9.1", data=data),
                                          source_url=catalog["source_url"], available_from_ts=50)
        after = comp.features(blue, red, "16.9", catalog=mixed, as_of_ts=100)
        self.assertEqual(before["features"], after["features"])

    def test_candidate_offset_roundtrip_missing_fallback_and_contract(self):
        rng = np.random.default_rng(71)
        extra = rng.normal(size=(60, len(comp.FEATURE_NAMES)))
        y = rng.integers(0, 2, 60)
        extra[:, 0] = 2. * y - 1.
        baseline = np.full(60, .5)
        model = comp.fit(extra, baseline, y, np.arange(60))
        p = comp.predict(model, extra, baseline)
        assert_array_equal(comp.predict(model, np.zeros_like(extra), baseline), baseline)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "composition.npz"
            comp.save(model, path)
            assert_array_equal(comp.predict(comp.load(path), extra, baseline), p)
        model["candidate_kind"] = "wrong_candidate"
        with self.assertRaises(ValueError):
            comp.predict(model, extra, baseline)

    def test_malformed_optional_records_are_gated_and_json_safe(self):
        blue, red, catalog, annotations = self.fixture()
        annotations["aatrox"]["traits"] = ["bad"]
        out = comp.features(blue, red, "16.9", catalog=catalog, annotations=annotations, as_of_ts=100)
        self.assertTrue(out["available"])
        self.assertEqual(out["features"][4:], [0.] * 8)
        data = copy.deepcopy(catalog["data"])
        data["Aatrox"]["stats"] = ["bad"]
        malformed = comp.catalog_from_payload(dict(version="16.9.1", data=data),
                                              source_url=catalog["source_url"], available_from_ts=50)
        out = comp.features(blue, red, "16.9", catalog=malformed, annotations=annotations, as_of_ts=100)
        self.assertTrue(out["known"][0])
        self.assertEqual(out["known"][1:4], [False] * 3)
        annotations["ahri"]["bad_unused_field"] = float("nan")
        out = comp.features(blue, red, "16.9", catalog=malformed, annotations=annotations, as_of_ts=100)
        json.dumps(out, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
