"""Extend the frozen postdraft replay to maps without gameplay timelines.

This runner adds a new league to the previously frozen ledger. It neither
trains a model nor changes earlier bets. Database access is read-only.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lol_ticker import config, draft, wppostdraft
from scripts.wpx_major import competition_group

PM_FEE_CHANGE = int(datetime(2026, 7, 10, tzinfo=timezone.utc).timestamp())
KS_PRECISION_CHANGE = int(datetime(2026, 5, 28, tzinfo=timezone.utc).timestamp())
DEFAULT_OUTPUT = ROOT/'data/wpx/postdraft_lpl_2026-09-12'
DEFAULT_PREVIOUS = ROOT/'data/wpx/postdraft_betting_2026-09-12/major_leagues/with_fees/bet_ledger.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def parsed(value):
    return json.loads(value) if isinstance(value, str) else value


def settlement(row):
    if row['platform'] == 'kalshi':
        return {'yes': 1, 'no': 0}.get(row['result'])
    try:
        raw = row['market_meta']
        tokens = parsed(raw['clobTokenIds'])
        value = float(parsed(raw['outcomePrices'])[tokens.index(row['market_id'])])
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    return int(value) if value in (0, 1) else None


def quote_field(row, field):
    raw = row['quote'].get(field) or {}
    value = raw.get('close_dollars')
    if value is not None:
        return float(value)
    return float(raw['close'])/100 if raw.get('close') is not None else None


def select_bet(rows):
    """Same native YES-ask / token-price strategy as the earlier replay."""
    if len(rows) != 2 or {r['team_side'] for r in rows} != {'blue','red'}:
        return None, 'missing_or_duplicate_sides'
    platform = rows[0]['platform']
    key = 'condition_id' if platform == 'polymarket' else 'event_id'
    if len({r[key] for r in rows}) != 1 or not rows[0][key]:
        return None, 'different_markets'
    pred = rows[0]['prediction']
    if any(r['prediction'] != pred for r in rows):
        return None, 'prediction_conflict'
    offers = []
    for row in rows:
        if draft.norm_team(row['outcome']) != draft.norm_team(row['team']):
            return None, 'market_team_mismatch'
        expected = pred['y'] if row['team_side']=='blue' else 1-pred['y']
        if row['won'] != expected or settlement(row) != expected:
            return None, 'outcome_conflict'
        q = row.get('quote')
        target = row['game_start']+120
        if not q or not 0 <= target-q['ts'] <= 60:
            return None, 'missing_or_stale_quote'
        if platform == 'kalshi':
            bid, ask = quote_field(row,'bid'), quote_field(row,'ask')
            if bid is None or ask is None or not 0 <= bid <= ask <= 1:
                return None, 'invalid_quote'
            cost, rate = ask, .07
        else:
            cost = q['price']
            m = row['market_meta']
            if (m.get('feesEnabled') is not True or
                    m.get('feeSchedule',{}).get('exponent') != 1 or
                    m.get('feeSchedule',{}).get('takerOnly') is not True or
                    m.get('feeType') not in ('sports_fees_v2','sports_fees_v3')):
                return None, 'unsupported_fee_metadata'
            rate = .03 if target < PM_FEE_CHANGE else .05
        if cost is None or not math.isfinite(cost) or not 0 < cost < 1:
            return None, 'terminal_or_invalid_price'
        prob = pred['p'] if row['team_side']=='blue' else 1-pred['p']
        if prob > cost:
            offers.append((prob/cost-1, row['team_side'], prob, cost, rate, row))
    if not offers:
        return None, 'no_positive_edge'
    _, side, prob, cost, rate, row = max(offers, key=lambda r:(r[0],r[1]))
    return dict(gid=pred['gid'], date=pred['date'], start=row['game_start'],
                blue_team=row['blue_team'], red_team=row['red_team'], league=pred['league'],
                competition_group=competition_group(pred['league']), side=side,
                model_probability=prob, entry_price=cost, edge=prob-cost, fee_rate=rate,
                won=bool(settlement(row)), market_id=row['market_id'],
                quote_age_seconds=row['game_start']+120-row['quote']['ts'],
                cluster=pred['date']+'|'+'|'.join(sorted([row['blue_team'],row['red_team']]))), None


def summarize(rows, platform, stake):
    stake = Decimal(stake)
    before, fees = Decimal(0), Decimal(0)
    for row in rows:
        p, rate = Decimal(str(row['entry_price'])), Decimal(str(row['fee_rate']))
        fee = stake*rate*(1-p)
        if platform == 'kalshi':
            precision = Decimal('.01') if row['start']+120 < KS_PRECISION_CHANGE else Decimal('.0001')
            fee = fee.quantize(Decimal('.000001'), rounding=ROUND_CEILING).quantize(precision, rounding=ROUND_CEILING)
        else:
            fee = fee.quantize(Decimal('.00001'), rounding=ROUND_HALF_UP)
        before += (stake/p if row['won'] else Decimal(0))-stake
        fees += fee
    principal = len(rows)*stake
    return dict(bets=len(rows), wins=sum(r['won'] for r in rows), losses=sum(not r['won'] for r in rows),
                principal=float(principal), fees=float(fees), cash_outlay=float(principal+fees),
                before_fees=float(before), net_profit=float(before-fees),
                roi=float((before-fees)/(principal+fees)) if rows else None)


def read_quotes(conn, predictions):
    records = conn.execute('''SELECT g.game_id AS gid,g.blue_team,g.red_team,d.*,
        m.event_id,m.condition_id,m.result,m.status,m.title,m.outcome,m.raw AS market_meta
        FROM golgg_games g JOIN draft_deltas d ON d.oe_game_id=g.oe_game_id
        JOIN markets m USING(platform,market_id)
        WHERE g.game_id=ANY(%s) AND d.post_p IS NOT NULL
        ORDER BY g.game_id,d.platform,d.market_id''',(list(predictions),)).fetchall()
    rows = []
    for record in records:
        row = dict(record)
        team = draft.norm_team(row['team'])
        if team == draft.norm_team(row['blue_team']):
            row['team_side'] = 'blue'
        elif team == draft.norm_team(row['red_team']):
            row['team_side'] = 'red'
        else:
            continue
        row['prediction'] = predictions[row['gid']]
        target = row['game_start']+120
        if row['platform']=='polymarket':
            quote = conn.execute('''SELECT extract(epoch FROM ts)::double precision AS ts,price
                FROM price_points WHERE platform='polymarket' AND market_id=%s AND ts<=to_timestamp(%s)
                ORDER BY ts DESC,fidelity LIMIT 1''',(row['market_id'],target)).fetchone()
        else:
            quote = conn.execute('''SELECT extract(epoch FROM ts)::double precision AS ts,close AS price,
                raw->'yes_bid' AS bid,raw->'yes_ask' AS ask FROM candles
                WHERE platform='kalshi' AND market_id=%s AND close IS NOT NULL AND ts<=to_timestamp(%s)
                ORDER BY ts DESC,period_min LIMIT 1''',(row['market_id'],target)).fetchone()
        row['quote'] = dict(quote) if quote else None
        # Keep only public fields needed to reproduce settlement/fees.
        row['market_meta'] = {k:v for k,v in row['market_meta'].items()
                              if k in ('clobTokenIds','outcomePrices','feesEnabled','feeSchedule','feeType')}
        rows.append(row)
    return rows


def main(output=DEFAULT_OUTPUT, previous=DEFAULT_PREVIOUS, league='LPL'):
    output, previous = Path(output), Path(previous)
    recovery = json.loads((output/'recovery.json').read_text())
    model_path = output/'recovered_v8.npz'
    if recovery.get('accepted') is not True or sha(model_path)!=recovery['artifact_sha256']:
        raise ValueError('An accepted, unchanged recovered model is required')
    source_info = recovery['identity']['inputs']['source']
    source_path = ROOT/source_info['path']
    if sha(source_path)!=source_info['sha256']:
        raise ValueError('Recovered inference source changed')
    spec = importlib.util.spec_from_file_location('lol_ticker._postdraft_v8',source_path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    model = old.load_model(str(model_path))
    cutoff = recovery['identity']['training_cutoff_exclusive']
    before = '2026-09-03'
    plan = dict(recovery_sha256=sha(output/'recovery.json'), model_sha256=sha(model_path),
                source_sha256=sha(source_path), previous_ledger_sha256=sha(previous),
                builder_sha256=sha(wppostdraft.__file__), runner_sha256=sha(__file__),
                league=league, after=cutoff, before=before,
                prediction='Completed draft, pregame ratings and earlier series results; neutral t=0 gameplay state',
                quote_rule='Same start+120s, both-side <=60s-old quotes and matched settlement',
                selection='Same positive pre-fee value rule; fee charged afterward',
                sizing='10/20/100 dollar principal per bet plus historical taker fees; continuous shares',
                no_retraining=True, previous_bets_unchanged=True)
    plan_path = output/'betting_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Changed replay inputs; use a fresh output directory')
    write_json(plan_path,plan)
    dataset = output/'postdraft_inputs.npz'
    with psycopg.connect(config.PG_DSN,row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        if not dataset.exists():
            wppostdraft.build(conn,dataset,before=before,after=cutoff,league_prefix=league+' ',
                             champion_names=model['pregame']['champ_names'])
        manifest = json.loads(Path(str(dataset)+'.manifest.json').read_text())
        if manifest.get('sha256')!=sha(dataset):
            raise ValueError('Postdraft dataset hash mismatch')
        with np.load(dataset,allow_pickle=False) as data:
            if (np.any(data['t']!=0) or np.any(data['seq']!=-1) or
                    np.any(data['date']<cutoff) or np.any(data['date']>=before)):
                raise ValueError('Inference rows violate frozen postdraft/date contract')
            np.testing.assert_array_equal(data['champ_names'], model['pregame']['champ_names'])
            raw = np.column_stack([old.prior_values_from_matrix(model,data['X'],list(data['names']),data['C']),
                                   old.state_values_from_matrix(data['X'],list(data['names']))])
            probability = old.predict_state(model['state'],raw,data['t']/60)
            predictions = {int(g):dict(gid=int(g),p=float(p),y=int(y),date=str(date),league=str(lg))
                           for g,p,y,date,lg in zip(data['gid'],probability,data['y'],data['date'],data['league'])}
        write_json(output/'predictions.json',predictions)
        quotes_path = output/'quoted_markets.json'
        if quotes_path.exists():
            records = json.loads(quotes_path.read_text())
            if any(r['prediction']!=predictions[r['gid']] for r in records):
                raise ValueError('Cached market records use different predictions')
        else:
            records = read_quotes(conn,predictions)
            write_json(quotes_path,records)
    grouped = defaultdict(list)
    for row in records:
        grouped[row['platform'],row['gid']].append(row)
    new = {p:[] for p in ('kalshi','polymarket')}
    exclusions = {p:Counter() for p in new}
    for (platform,gid),rows in grouped.items():
        bet, reason = select_bet(rows)
        if bet is not None:
            new[platform].append(bet)
        else:
            exclusions[platform][reason] += 1
    original = json.loads(previous.read_text())
    merged = {}
    for platform in new:
        if {r['gid'] for r in original[platform]} & {r['gid'] for r in new[platform]}:
            raise ValueError('New league overlaps previous bets')
        merged[platform] = sorted(original[platform]+new[platform],key=lambda r:(r['start'],r['gid']))
    result = dict(plan=plan, dataset_sha256=sha(dataset), quotes_sha256=sha(quotes_path),
                  dataset_games=len(predictions), market_games={p:sum(k[0]==p for k in grouped) for p in new},
                  exclusions={p:dict(v) for p,v in exclusions.items()}, results={})
    for stake in (10,20,100):
        result['results'][str(stake)] = {}
        for name, ledgers in [('added_league',new),('all_majors_including_msi',merged),
                              ('domestic_only',{p:[r for r in rs if r['competition_group']!='International'] for p,rs in merged.items()})]:
            result['results'][str(stake)][name] = {p:summarize(rs,p,stake) for p,rs in ledgers.items()}
    write_json(output/'lpl_bet_ledger.json',new)
    write_json(output/'combined_bet_ledger.json',merged)
    write_json(output/'betting_results.json',result)
    print(json.dumps({k:result[k] for k in ('dataset_games','market_games','exclusions','results')},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--previous-ledger',type=Path,default=DEFAULT_PREVIOUS)
    parser.add_argument('--league',default='LPL')
    args=parser.parse_args()
    main(args.output,args.previous_ledger,args.league)
