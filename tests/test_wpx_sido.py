"""Temporal and phase-boundary checks for the offline performance-prior screen."""
from datetime import date
import unittest

import numpy as np

from scripts import wpx_sido as sido


def roster(game_id):
    return [dict(game_id=game_id, player_id=slot, slot=slot,
                 side='blue' if slot < 5 else 'red',
                 role=str(slot % 5), champion=f'c{slot}',
                 g0=500, g7=2500, g15=5000, g25=9000,
                 damage='20000', taken='10000', gold=12000)
            for slot in range(10)]


class SidoScreenTests(unittest.TestCase):
    def test_same_month_and_future_targets_cannot_change_current_rating(self):
        games = [dict(game_id=i, date=d) for i,d in enumerate(
            [date(2025,2,1), date(2025,2,28), date(2025,3,1)])]
        rows = [r for g in games for r in roster(g['game_id'])]
        features = [{sido.player_key(r): 1.0} for r in roster(9)] * 2
        targets = np.array([2.0]*5 + [1.0]*5 + [3.0]*5 + [1.0]*5)
        dates = np.array([date(2025,1,10).toordinal()]*10 + [date(2025,2,1).toordinal()]*10)
        original, _, _ = sido.monthly_ratings(features, targets, dates, games, rows, min_rows=1)
        targets[10:15] = 2000
        changed, _, _ = sido.monthly_ratings(features, targets, dates, games, rows, min_rows=1)
        self.assertEqual(original[0], changed[0])
        self.assertEqual(original[1], changed[1])
        self.assertEqual(original[0], original[1])
        self.assertNotEqual(original[2], changed[2])

    def test_gold_targets_are_increments_and_short_games_are_excluded(self):
        game = dict(game_id=1, date=date(2025,1,10), duration_s=1300,
                    elo_blue_pre=1500, elo_red_pre=1500)
        f,y,_ = sido.build_panel([game], roster(1), phase=(7,15))
        self.assertEqual(len(y), 10)
        np.testing.assert_allclose(y, 2500/8/100)
        self.assertEqual(f[0]['start_own_gold'], 2.5)
        _,late,_ = sido.build_panel([game], roster(1), phase=(15,25))
        self.assertEqual(len(late), 0)

    def test_missing_gold_is_not_zero_performance(self):
        game = dict(game_id=1, date=date(2025,1,10), duration_s=1800,
                    elo_blue_pre=1500, elo_red_pre=1500)
        rows = roster(1)
        rows[0]['g7'] = None
        _,y,_ = sido.build_panel([game], rows, phase=(0,7))
        self.assertEqual(len(y), 0)


if __name__ == '__main__':
    unittest.main()
