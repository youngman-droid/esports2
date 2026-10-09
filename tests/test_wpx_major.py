import unittest
import numpy as np
from scripts.wpx_major import competition_group, filter_arrays


class MajorCohortTests(unittest.TestCase):
    def test_membership_follows_season_not_current_name(self):
        for name in ('PCS Spring 2024','VCS Summer 2024','LLA Opening 2024',
                     'LTA North 2025 Split 1','LCP 2026 Split 1','CBLOL Cup 2026',
                     'LCK 2025 Road to MSI','Worlds 2025 Play-In','2026 First Stand'):
            self.assertIsNotNone(competition_group(name),name)
        for name in ('PCS 2025 Split 1','VCS 2026 Spring','CBLOL Academy Split 1 Playoffs 2024',
                     'LCK CL 2026 Rounds 1-2','LJL Spring 2024','LCO Split 1 2024',
                     'EMEA Masters 2026 Winter','Esports World Cup 2026','LCP Promotion 2026'):
            self.assertIsNone(competition_group(name),name)
        with self.assertRaises(ValueError):
            competition_group('LCK Spring 2027')

    def test_filter_preserves_row_alignment_and_vocabularies(self):
        a={k:np.arange(4) for k in ('X','y','gid','t','seq','date','patch','pm','ks','C')}
        a.update(league=np.array(['LCK Cup 2026','LCK CL 2026 Kickoff']*2),
                 names=np.array(['one','two','three','four']),champ_names=np.array(['A']))
        got,mask,_=filter_arrays(a)
        for k in set(a)-{'names','champ_names'}:
            np.testing.assert_array_equal(got[k],a[k][[0,2]])
        np.testing.assert_array_equal(got['names'],a['names'])
        a['gid']=np.array([1,1,2,2])
        with self.assertRaisesRegex(ValueError,'within a game'):
            filter_arrays(a)
