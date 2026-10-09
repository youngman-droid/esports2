import copy
import unittest

from research import wpx_postdraft_bets as replay


def market_pair(platform='kalshi', start=None):
    start = replay.PM_FEE_CHANGE if start is None else start
    pred = dict(gid=10, p=.65, y=1, date='2026-07-10', league='LPL 2026 Split 3')
    rows = []
    for side, team, won, price in [('blue', 'Alpha', 1, .6), ('red', 'Beta', 0, .3)]:
        row = dict(platform=platform, prediction=pred, team_side=side, team=team,
                   outcome=team, won=won, game_start=start, blue_team='Alpha',
                   red_team='Beta', market_id=side, result='yes' if won else 'no',
                   condition_id='condition', event_id='event',
                   market_meta=dict(clobTokenIds=['blue', 'red'], outcomePrices=['1', '0'],
                                    feesEnabled=True, feeSchedule=dict(exponent=1, takerOnly=True),
                                    feeType='sports_fees_v3'),
                   quote=dict(ts=start+120, price=price,
                              bid=dict(close_dollars=str(price-.01)),
                              ask=dict(close_dollars=str(price))))
        rows.append(row)
    return rows


class PostdraftBetTests(unittest.TestCase):
    def test_selects_highest_return_and_uses_yes_ask(self):
        rows = market_pair()
        rows[0]['quote']['price'] = .1  # Candle trade price must not replace ask.
        bet, reason = replay.select_bet(rows)
        self.assertIsNone(reason)
        self.assertEqual(bet['side'], 'red')
        self.assertEqual(bet['entry_price'], .3)
        self.assertFalse(bet['won'])

    def test_both_quotes_must_be_recent_and_never_future(self):
        for age in (-1, 61):
            rows = market_pair()
            rows[0]['quote']['ts'] = rows[0]['game_start']+120-age
            self.assertEqual(replay.select_bet(rows), (None, 'missing_or_stale_quote'))
        rows = market_pair()
        rows[0]['quote']['ts'] -= 60
        self.assertIsNotNone(replay.select_bet(rows)[0])

    def test_rejects_wrong_market_team_and_settlement(self):
        cases = [('event_id', 'other', 'different_markets'),
                 ('outcome', 'Other team', 'market_team_mismatch'),
                 ('result', 'no', 'outcome_conflict')]
        for key, value, reason in cases:
            rows = market_pair()
            rows[0][key] = value
            self.assertEqual(replay.select_bet(rows), (None, reason))
        rows = market_pair('polymarket')
        rows[0]['market_meta']['outcomePrices'] = ['0', '1']
        self.assertEqual(replay.select_bet(rows), (None, 'outcome_conflict'))

    def test_historical_polymarket_fee_change_and_prefee_selection(self):
        for target, expected in [(replay.PM_FEE_CHANGE-1, .03), (replay.PM_FEE_CHANGE, .05)]:
            rows = market_pair('polymarket', target-120)
            bet, reason = replay.select_bet(rows)
            self.assertIsNone(reason)
            self.assertEqual(bet['fee_rate'], expected)
        rows = market_pair('polymarket')
        rows[0]['quote']['price'] = .649
        rows[1]['quote']['price'] = .36
        bet, reason = replay.select_bet(rows)
        self.assertIsNone(reason)  # Edge is positive before fees, negative after fees.
        self.assertEqual(bet['side'], 'blue')
        rows[0]['market_meta']['feesEnabled'] = False
        self.assertEqual(replay.select_bet(rows), (None, 'unsupported_fee_metadata'))

    def test_principal_payout_and_fee_precision(self):
        bet = dict(entry_price=.63, fee_rate=.07, won=True,
                   start=replay.KS_PRECISION_CHANGE-121)
        old = replay.summarize([bet], 'kalshi', 10)
        self.assertEqual(old['fees'], .26)
        self.assertAlmostEqual(old['net_profit'], 10/.63-10-.26)
        new_bet = dict(bet, start=replay.KS_PRECISION_CHANGE-120)
        new = replay.summarize([new_bet], 'kalshi', 10)
        self.assertEqual(new['fees'], .259)
        self.assertEqual(new['cash_outlay'], 10.259)
        loss = copy.deepcopy(new_bet)
        loss.update(won=False, fee_rate=.03)
        pm = replay.summarize([loss], 'polymarket', 20)
        self.assertEqual(pm['fees'], .222)
        self.assertEqual(pm['net_profit'], -20.222)
        combined = replay.summarize([bet, new_bet], 'kalshi', 10)
        self.assertAlmostEqual(combined['net_profit'], old['net_profit']+new['net_profit'])


if __name__ == '__main__':
    unittest.main()
