import copy
import unittest
import numpy as np
from research.draft_comfort_compare import history_features, signed_draft, ROLES


def game(i,day):
    g=dict(id=str(i), day=day, patch='16.1', blue_team='A', red_team='B', y=1)
    for side in ('blue','red'):
        g[side]=dict(players={r:side+r for r in ROLES},
                     picks={r:side+r for r in ROLES}, bans=[])
    return g


class DraftComfortTests(unittest.TestCase):
    def test_same_day_and_future_outcomes_cannot_change_features(self):
        games=[game(1,'2026-01-01'),game(2,'2026-01-01'),game(3,'2026-01-02')]
        a=history_features(games)
        changed=copy.deepcopy(games); changed[0]['y']=0; changed[2]['y']=0
        b=history_features(changed)
        for gid in ('1','2'):
            self.assertEqual(a[gid][0],b[gid][0])
            np.testing.assert_array_equal(a[gid][1],b[gid][1])
        self.assertNotEqual(a['3'][0],b['3'][0])

    def test_draft_reversal_negates_every_feature(self):
        g=game(1,'2026-01-01'); g['blue']['bans']=['a']; g['red']['bans']=['b']
        h=copy.deepcopy(g); h['blue'],h['red']=h['red'],h['blue']
        self.assertEqual(signed_draft(g),{k:-v for k,v in signed_draft(h).items()})

    def test_cold_start_and_role_history(self):
        g=game(1,'2026-01-01'); h=game(2,'2026-01-02')
        h['blue']['players']['top']='new_player'
        result=history_features([g,h])
        np.testing.assert_array_equal(result['1'][1],np.zeros(3))
        self.assertEqual(result['2'][2][0],0)
        self.assertEqual(result['2'][2][1],1)


if __name__=='__main__': unittest.main()
