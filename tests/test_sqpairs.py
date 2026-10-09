import json
import os
import tempfile
import unittest
from unittest import mock

from lol_ticker import sqpairs

LANE = sqpairs.LANES
CHAMPS = {"a%d" % i: 100 + i for i in range(5)}
CHAMPS.update({"b%d" % i: 200 + i for i in range(5)})


def _write_patch(root, patch, d2_match, d2_syn, n=100000):
    pdir = os.path.join(root, patch); os.makedirs(pdir)
    for name, cid in CHAMPS.items():
        lane = LANE[int(name[1])]
        for vs in LANE[LANE.index(lane):]:
            foes = [c for k, c in CHAMPS.items() if LANE[int(k[1])] == vs and k[0] != name[0]]
            sign = 1 if name[0] == "a" else -1
            body = {"stats": {"lane": lane, "vsLane": vs, "cid": cid},
                    "counters": [{"cid": f, "n": n, "d2": sign * d2_match} for f in foes]}
            with open(os.path.join(pdir, "%s-%s-vs-%s.json" % (lane, name, vs)), "w") as f:
                json.dump(body, f)
        mates = {}
        for k, c in CHAMPS.items():
            if k[0] == name[0] and k != name:
                mates.setdefault(LANE[int(k[1])], []).append([c, 50, 0, d2_syn if name[0] == "a" else 0.0, 1, n])
        with open(os.path.join(pdir, "%s-%s-team.json" % (lane, name)), "w") as f:
            json.dump({"team_h": ["id", "wr", "d1", "d2", "pr", "n"], "team": mates}, f)


class SqPairsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = os.path.join(self.tmp.name, "lolalytics")
        _write_patch(root, "16.8", 1.0, 0.5)
        _write_patch(root, "16.9", 3.0, 0.5)      # must NOT leak into 16.9 scores
        self.path = os.path.join(self.tmp.name, "tables.npz")
        with mock.patch.object(sqpairs, "ROOT", root):
            sqpairs.build_tables(self.path)
        self.s = sqpairs.Scorer(self.path)
        self.a = ["a%d" % i for i in range(5)]; self.b = ["b%d" % i for i in range(5)]

    def tearDown(self):
        self.tmp.cleanup()

    def test_prior_patches_only(self):
        self.assertIsNone(self.s.score("16.08", self.a, self.b))     # nothing earlier than 16.8
        r9 = self.s.score("16.09", self.a, self.b)                   # OE spelling; uses 16.8 only
        shrink = 100000 / (100000 + self.s.k[0])
        self.assertAlmostEqual(r9["matchup"], 25 * 0.04 * 1.0 * shrink, places=9)
        r10 = self.s.score("16.10", self.a, self.b)                  # pools 16.8 + 16.9 -> mean d2 = 2
        self.assertAlmostEqual(r10["matchup"], 25 * 0.04 * 2.0 * 200000 / (200000 + self.s.k[0]), places=9)
        self.assertEqual(r9["coverage"], 1.0)

    def test_side_swap_negates(self):
        f, g = self.s.score("16.9", self.a, self.b), self.s.score("16.9", self.b, self.a)
        self.assertAlmostEqual(f["score"], -g["score"], places=12)
        self.assertGreater(f["synergy"], 0)

    def test_unknown_inputs_score_zero(self):
        with mock.patch.object(sqpairs, "_scorer", self.s):
            self.assertEqual(sqpairs.draft_score("16.9", ["nobody"] + self.a[1:], self.b), 0.0)
            self.assertEqual(sqpairs.draft_score("", self.a, self.b), 0.0)
            self.assertNotEqual(sqpairs.draft_score("16.9", self.a, self.b), 0.0)

    def test_champion_keys(self):
        self.assertEqual(sqpairs.key("Nunu & Willump"), "nunu")
        self.assertEqual(sqpairs.key("Renata Glasc"), "renata")
        self.assertEqual(sqpairs.key("K'Sante"), "ksante")
        self.assertEqual(sqpairs.canon("16.09"), "16.9")


if __name__ == "__main__":
    unittest.main()
