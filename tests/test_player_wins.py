from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.special import expit
from lol_ticker import player_ratings as ratings
from test_player_ratings import game, rows, write_csv


class WinRatingsTests(unittest.TestCase):
    def test_neutral_scale(self):
        self.assertEqual(ratings.wins_added(0), 0)
        self.assertAlmostEqual(ratings.wins_added(np.log(.55/.45)), 5)
        self.assertAlmostEqual(ratings.wins_added(-.3), -ratings.wins_added(.3))

    def test_outcomes_drive_fit_and_gold_is_irrelevant(self):
        games = [game(i) for i in range(80)]
        cutoff = games[-1].played_at
        fit = ratings.fit_win_ratings(games, cutoff)
        changed = [replace(g, gold_margin=-g.gold_margin, players=tuple(replace(p, total_gold=1, gold_at15=None) for p in g.players)) for g in games]
        same = ratings.fit_win_ratings(changed, cutoff)
        np.testing.assert_allclose(fit.coefficients, same.coefficients)
        reversed_fit = ratings.fit_win_ratings([replace(g, blue_win=not g.blue_win) for g in games], cutoff)
        np.testing.assert_allclose(fit.coefficients, -reversed_fit.coefficients, atol=1e-7)
        self.assertGreater(fit.coefficients[fit.index['A:mid']], 0)
        # Permanently shared lineups receive equal credit, not invented role differentiation.
        self.assertAlmostEqual(fit.coefficients[fit.index['A:top']], fit.coefficients[fit.index['A:mid']])
        predictions = expit(ratings._predict(fit, games))
        self.assertTrue(np.all((predictions >= .5) == np.array([g.blue_win for g in games])))

    def test_future_games_rejected(self):
        games = [game(i) for i in range(3)]
        with self.assertRaises(ValueError):
            ratings.fit_win_ratings(games, games[0].played_at)

    def test_missing_gold_does_not_exclude_win_games(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'games.csv'
            values = rows([game(i) for i in range(40)])
            for row in values:
                row['totalgold'] = ''
                row['goldat15'] = ''
            write_csv(path, values)
            self.assertEqual(len(ratings.read_games([path])[0]), 0)
            self.assertEqual(len(ratings.read_games([path], require_gold=False)[0]), 40)
            payload = ratings.build_ratings([path], outcome='wins', history_months=0, validate=False)
            self.assertEqual(payload['meta']['unit'], 'wins added / 100 games')
            for player in payload['players']:
                self.assertNotIn('lane_prior', player)
                self.assertLess(player['ci_low'], player['impact'])
                self.assertGreater(player['ci_high'], player['impact'])
                self.assertGreaterEqual(player['ci_low'], -50)
                self.assertLessEqual(player['ci_high'], 50)
                self.assertAlmostEqual(player['impact'], ratings.wins_added(player['win_log_odds']), places=3)

    def test_frozen_holdout_reports_win_probability_metrics(self):
        games = [game(i) for i in range(130)]
        validation = ratings.win_validation(games, games[-1].played_at, holdout_days=30)
        self.assertTrue(validation['available'])
        self.assertLess(validation['training_end'], validation['holdout_start'])
        self.assertEqual(set(validation['metrics']), {'win_rapm','team_logistic','side_only','coin_flip'})
        self.assertLess(validation['metrics']['win_rapm']['brier_score'], .25)
        self.assertAlmostEqual(validation['metrics']['coin_flip']['log_loss'], np.log(2), places=6)

if __name__ == '__main__':
    unittest.main()
