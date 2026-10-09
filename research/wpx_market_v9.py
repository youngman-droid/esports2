"""Pair saved v9 predictions with strictly earlier stored market observations.

Read-only historical diagnostic. Price histories mix trades, candle/history
prices and book midpoints; they are not proof of executable trading returns.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import psycopg
from psycopg.rows import dict_row

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lol_ticker import align,config,shadow,wpgam
from research.wpx_methods_v9 import dump,metrics,paired

log=logging.getLogger('market_v9')
LEADS=(0,45,195)
MAX_AGE=60.


def past_prices(points, targets, max_age=MAX_AGE):
    """No future fallback; average duplicate timestamps before lookup."""
    points=np.asarray(points,dtype=float)
    price=np.full(len(targets),np.nan);age=price.copy()
    if not points.size:return price,age
    valid=np.isfinite(points).all(axis=1)&(points[:,1]>=0)&(points[:,1]<=1)
    points=points[valid]
    if not len(points):return price,age
    ts,inv=np.unique(points[:,0],return_inverse=True)
    pp=np.bincount(inv,weights=points[:,1])/np.bincount(inv)
    ix=np.searchsorted(ts,targets,side='right')-1
    safe=np.maximum(ix,0);a=targets-ts[safe]
    ok=(ix>=0)&(a>=0)&(a<=max_age)
    price[ok]=pp[safe[ok]];age[ok]=a[ok]
    return price,age


def main(methods, dataset, resources, output):
    started=time.time();output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((methods/'plan.json').read_text())
    assert hashlib.sha256(dataset.read_bytes()).hexdigest()==plan['dataset_sha256']
    with np.load(dataset,allow_pickle=True) as d:
        first,train,_=wpgam._date_split(d)
        game_gid=d['gid'][first];date_map=dict(zip(game_gid.astype(int),d['date'][first].astype(str)))
        wanted=game_gid[~train].astype(int)
        mask=(d['seq']<0)&np.isin(d['gid'],wanted)
        gid,y,t=d['gid'][mask],d['y'][mask],d['t'][mask]
    predictions={}
    for mode in ['frozen','rolling']:
        with np.load(methods/(mode+'_predictions.npz')) as d:
            np.testing.assert_array_equal(d['gid'],gid);np.testing.assert_array_equal(d['y'],y)
            predictions[mode]={k:d[k].copy() for k in d.files if k not in ['gid','y']}
    with np.load(resources,allow_pickle=False) as d:
        clusters=dict(zip(d['series_gid'].astype(int),d['series_key'].astype(str)))
    experiment=dict(dataset_sha256=plan['dataset_sha256'],methods_plan_sha256=hashlib.sha256((methods/'plan.json').read_bytes()).hexdigest(),
        prediction_hashes={m:hashlib.sha256((methods/(m+'_predictions.npz')).read_bytes()).hexdigest() for m in predictions},
        lookup_offsets_s=LEADS,maximum_observation_age_s=MAX_AGE,minimum_alignment_quality=.4,
        source='align.odds_series: trades, history/candle prices, book midpoints',
        timestamp_rule='last timestamp <= lookup; reject stale/missing, average duplicate timestamps',
        game_window='minute >= 1 and lookup <= reconstructed game end; per-market pauses respected',
        outputs='each-offset coverage plus common rows across offsets; all saved models, no selection/refitting',
        limitations='Retrospective market-derived clock alignment; mixed price sources; not executable returns or fresh confirmation',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(align.__file__)]},
        protected_hashes=plan['protected_hashes'])
    dump(output/'plan.json',experiment)
    sums={p:np.zeros((len(LEADS),len(gid))) for p in ['polymarket','kalshi']}
    ages={p:v.copy() for p,v in sums.items()};counts={p:v.copy() for p,v in sums.items()}
    ix_by_game={int(g):np.flatnonzero(gid==g) for g in wanted}
    with psycopg.connect(config.PG_DSN,row_factory=dict_row,options='-c default_transaction_read_only=on') as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        rows=conn.execute('''SELECT game_id,platform,market_id,team_side,start_wall,end_wall,pauses,duration_s
            FROM game_alignment WHERE quality >= .4 AND game_id=ANY(%s)
            AND platform IN ('polymarket','kalshi') ORDER BY game_id,platform,market_id''',(wanted.tolist(),)).fetchall()
        for n,row in enumerate(rows,1):
            ix=ix_by_game[row['game_id']];seconds=t[ix]
            points=align.odds_series(conn,row['platform'],row['market_id'],row['start_wall']-600,row['end_wall'])
            if len(points)<10:continue
            pauses=row['pauses']
            if not isinstance(pauses,list):pauses=json.loads(pauses or '[]')
            pause=np.array([sum(p['length_s'] for p in pauses if p['game_time_s']<=s) for s in seconds])
            state_wall=row['start_wall']+seconds+pause
            for j,lead in enumerate(LEADS):
                lookup=state_wall+lead
                p,age=past_prices(points,lookup)
                keep=np.isfinite(p)&(seconds>=60)&(seconds<=row['duration_s'])&(lookup<=row['end_wall'])
                if row['team_side']=='red':p=1-p
                elif row['team_side']!='blue':continue
                jj=ix[keep];platform=row['platform']
                sums[platform][j,jj]+=p[keep];ages[platform][j,jj]+=age[keep];counts[platform][j,jj]+=1
            if n%500==0:log.info('Read %d/%d market alignments',n,len(rows))
        # Read-only live-ledger snapshot; do not enroll or freeze games.
        live={}
        protocols=conn.execute('''SELECT protocol_id,config FROM shadow_protocols
            WHERE protocol_id LIKE 'shadow_v11_%%' OR protocol_id LIKE 'shadow_v12_%%'
            ORDER BY created_at''').fetchall()
        for protocol in protocols:
            rr=conn.execute('''SELECT p.*,o.blue_win FROM shadow_predictions p JOIN shadow_outcomes o
                USING(game_id,game_start_ts) WHERE p.protocol_id=%s AND o.status='resolved'
                AND p.captured_at<o.recorded_at ORDER BY p.captured_at''',(protocol['protocol_id'],)).fetchall()
            scored=shadow.score_rows([dict(r) for r in rr],bootstrap=2000,
                                     max_lead_s=protocol['config'].get('primary_market_lead_s'))
            live[protocol['protocol_id']]=dict(rows=len(rr),platforms=scored['platforms'])
    dump(output/'live_ledgers.json',live)
    report={};arrays={}
    for platform in sums:
        present=counts[platform]>0
        market=np.full_like(sums[platform],np.nan);age=market.copy()
        market[present]=sums[platform][present]/counts[platform][present]
        age[present]=ages[platform][present]/counts[platform][present]
        arrays[platform]=market;arrays[platform+'_age_s']=age
        common=present.all(axis=0)
        report[platform]={}
        for j,lead in enumerate(LEADS):
            report[platform][str(lead)]={}
            for cohort,keep in [('available',present[j]),('common',common)]:
                if not keep.any():continue
                q=market[j,keep];yy=y[keep];gg=gid[keep]
                result=dict(market=metrics(q,yy,gg),median_price_age_s=float(np.median(age[j,keep])),
                    median_effective_market_lead_s=float(np.median(lead-age[j,keep])),models={})
                for mode,models in predictions.items():
                    result['models'][mode]={name:dict(metrics=metrics(p[keep],yy,gg),
                        versus_market=paired(p[keep],q,yy,gg,clusters),
                        date_sensitivity=paired(p[keep],q,yy,gg,date_map)) for name,p in models.items()}
                report[platform][str(lead)][cohort]=result
        log.info('%s coverage %s',platform,{k:v['available']['market']['games'] for k,v in report[platform].items()})
    np.savez_compressed(output/'paired_prices.npz',gid=gid,y=y,t=t,**arrays)
    dump(output/'results.json',report)
    for path,sha in plan['protected_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    dump(output/'completion.json',dict(seconds=time.time()-started,market_alignments=len(rows),production_changed=False))
    log.info('Completed market comparison in %.1fs',time.time()-started)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(message)s')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',type=Path,default=Path(wpgam.OUT_DIR)/'methods_v9')
    parser.add_argument('--dataset',type=Path,default=Path(wpgam.OUT_DIR)/'states.npz')
    parser.add_argument('--resources',type=Path,default=Path(wpgam.OUT_DIR)/'adapt_cache/408b0e7b37ff020354920067/resources.npz')
    parser.add_argument('--output',type=Path,default=Path(wpgam.OUT_DIR)/'market_v9')
    args=parser.parse_args();main(args.methods,args.dataset,args.resources,args.output)
