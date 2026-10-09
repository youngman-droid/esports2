"""Season/patch/champion-aware GAM and LightGBM search, with temporal isolation.

Run with the optional LightGBM experiment environment. All outputs are research
artifacts. No production paths, registry updates, or database writes are used.
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
from lol_ticker import config,wpadapt as wa,wpbench,wpgam

log=logging.getLogger('adapt')


def search_specs(trials=24,seed=20260904):
    rng=np.random.default_rng(seed)
    specs=[dict(id='gam_baseline',family='gam',features='core',window_days=None,half_life=None,
                l2=24.,smooth=70.,context_l2=800.)]
    for i,(window,half,reg) in enumerate([(None,None,800.),(None,90,800.),(None,180,800.),
                                        (365,None,800.),(180,None,800.),(365,90,800.),
                                        (None,180,300.),(None,180,2000.)]):
        specs.append(dict(id=f'gam_rich_{i}',family='gam',features='rich',window_days=window,
                          half_life=half,l2=24. if reg<=800 else 60.,
                          smooth=70. if reg<=800 else 150.,context_l2=reg,min_category_games=10))
    specs.append(dict(id='gam_core_decay',family='gam',features='core',window_days=None,
                      half_life=180,l2=24.,smooth=70.,context_l2=800.))
    for i in range(trials):
        choose=lambda vals: vals[int(rng.integers(len(vals)))]
        leaves=choose([7,15,31,63]); depth=choose([4,6,8,-1])
        if depth>0: leaves=min(leaves,2**depth-1)
        mono=(i%2==0)
        specs.append(dict(
            id=f'lightgbm_{i:02}',family='lightgbm',features='core' if i==0 else 'rich',
            window_days=choose([None,180,365,730]),half_life=choose([None,90,180,365]),
            min_category_games=choose([5,10,25]),monotone=mono,
            learning_rate=choose([.015,.03,.06]),num_leaves=leaves,max_depth=depth,
            min_data_in_leaf=choose([100,300,700,1500]),min_sum_hessian_in_leaf=choose([1.,10.,30.]),
            lambda_l1=choose([0.,1.,5.]),lambda_l2=choose([5.,20.,60.,150.]),
            min_gain_to_split=choose([0.,.05,.2]),max_bin=choose([63,127,255]),
            feature_fraction=1. if mono else choose([.7,.85,1.]),
            bagging_fraction=choose([.7,.85,1.]),bagging_freq=1,
            cat_smooth=choose([10.,50.,150.]),cat_l2=choose([10.,50.,150.]),
            min_data_per_group=choose([100,300,700]),max_cat_threshold=choose([16,32]),
            path_smooth=choose([0.,10.,50.]),extra_trees=(i%6==5),
            max_rounds=1200,early_stopping_rounds=80,seed=43))
    # Matched core/rich controls use the same reasonable tree settings and
    # unrestricted history; search candidates above explore additional choices.
    for rich in (False,True):
        specs.append(dict(id='lgb_control_rich' if rich else 'lgb_control_core',
            family='lightgbm',features='rich' if rich else 'core',
            window_days=None,half_life=None,min_category_games=10,monotone=False,
            learning_rate=.03,num_leaves=15,max_depth=6,min_data_in_leaf=200,
            min_sum_hessian_in_leaf=1.,lambda_l1=0.,lambda_l2=15.,
            min_gain_to_split=0.,max_bin=255,feature_fraction=1.,bagging_fraction=1.,
            bagging_freq=0,cat_smooth=50.,cat_l2=50.,min_data_per_group=100,
            max_cat_threshold=32,path_smooth=0.,extra_trees=False,
            max_rounds=1200,early_stopping_rounds=80,seed=43))
    return specs


def load_extra(base,dataset,cache):
    target=cache/'resources.npz'
    with np.load(dataset,allow_pickle=True) as d:
        first=wpgam._game_rows(d['gid'])
        meta,cats=wa.calendar_metadata(base['game_dates'],d['patch'][first],d['league'][first],base['game_C'])
        patches=d['patch'][first].astype(str)
        competitions=cats[:,12].astype(str)
    if target.exists():
        with np.load(target,allow_pickle=False) as d:
            gold=d['gold'];series=dict(zip(d['series_gid'].astype(int),d['series_key'].astype(str)))
    else:
        gold=np.full((len(base['gid']),10),np.nan,dtype=np.float64)
        keys=base['gid'].astype(np.int64)*1000 + np.floor(base['t_min']).astype(np.int64)
        if len(np.unique(keys))!=len(keys): raise ValueError('Duplicate fixed-minute states')
        order=np.argsort(keys);sorted_keys=keys[order]
        series={}
        with psycopg.connect(config.PG_DSN,row_factory=dict_row,
                              options='-c default_transaction_read_only=on') as conn:
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            rows=conn.execute('SELECT game_id,date,match_id FROM golgg_games WHERE game_id=ANY(%s)',
                              (base['game_gid'].astype(int).tolist(),)).fetchall()
            for r in rows: series[r['game_id']]=f"{r['date']}:{r['match_id'] or r['game_id']}"
            with conn.cursor(name='adapt_gold') as cur:
                cur.execute('''SELECT game_id,minute,array_agg(gold ORDER BY slot) gold
                    FROM golgg_timeline WHERE game_id=ANY(%s)
                    GROUP BY game_id,minute HAVING count(*)=10 AND count(DISTINCT slot)=10
                    AND count(gold)=10''',(base['game_gid'].astype(int).tolist(),))
                while True:
                    rows=cur.fetchmany(10000)
                    if not rows: break
                    kk=np.array([r['game_id']*1000+r['minute'] for r in rows],dtype=np.int64)
                    jj=np.searchsorted(sorted_keys,kk)
                    valid=jj<len(sorted_keys)
                    valid[valid]&=sorted_keys[jj[valid]]==kk[valid]
                    found=np.flatnonzero(valid)
                    gold[order[jj[found]]]=np.array([rows[i]['gold'] for i in found],dtype=float)
        np.savez_compressed(target,gold=gold,series_gid=np.array(list(series)),series_key=np.array(list(series.values())))
    present=np.isfinite(gold).all(axis=1)
    if np.any(gold[present]<0): raise ValueError('Negative gold snapshot')
    # Validate side/slot alignment against the established training contract.
    fixed_X=base['raw_X'][base['raw_fixed']]
    idx={str(n):i for i,n in enumerate(base['names'])}
    delta=(gold[:,:5]-gold[:,5:])/1000.
    mismatch=np.zeros(len(gold),dtype=bool)
    for i,role in enumerate(('top','jng','mid','bot','sup')):
        mismatch|=present & (np.abs(delta[:,i]-fixed_X[:,idx['gold_'+role]])>.002)
    if mismatch.any():
        raise ValueError(f'{mismatch.sum()} resource snapshots disagree with existing causal state rows')
    signed=np.nan_to_num(gold/10000.)*np.array([1]*5+[-1]*5)
    extra=np.column_stack([signed,present,meta[base['row_game']]])
    return dict(extra=extra,cats=cats,patches=patches,competitions=competitions,
                series=series,present=present,gold=gold)


def stage_arrays(base,extra,block,cache):
    key=hashlib.sha256(base['game_gid'][block['fit']].tobytes()).hexdigest()[:16]
    path=cache/f'priors_{key}.npy'
    if path.exists(): raw=np.load(path)
    else:
        log.info('Fitting shared prior stack before %s',block['fit_end'])
        raw=wpbench._stacked_arrays(base,block['fit'])['raw']
        np.save(path,raw)
    raw=np.column_stack([raw,extra['extra']])
    rows=base['row_game']
    parts={}
    for name in ('fit','calibration','validation'):
        mask=block[name][rows]
        parts[name]=dict(raw=raw[mask],cats=extra['cats'][rows[mask]],y=base['y'][mask],
                         gid=base['gid'][mask],t=base['t_min'][mask],
                         date=base['game_dates'][rows[mask]],mask=mask)
    return parts


def fit_candidate(spec,parts,cutoff):
    fit=parts['fit'];cal=parts['calibration']
    keep,weights=wa.temporal_weights(fit['gid'],fit['date'],cutoff,
                                   spec.get('window_days'),spec.get('half_life'))
    args=[fit[k][keep] for k in ('raw','cats','y','gid','t')]
    clean={k:v for k,v in spec.items() if k!='id'}
    if spec['family']=='gam':model=wa.fit_gam(*args,weights,clean)
    else:model=wa.fit_lgb(*args,weights,clean,cal)
    return model,int(len(np.unique(fit['gid'][keep])))


def evaluate_candidate(spec,parts,cutoff):
    started=time.time()
    model,games=fit_candidate(spec,parts,cutoff)
    cal,val=parts['calibration'],parts['validation']
    cp=wa.predict(model,cal['raw'],cal['cats'],cal['t'])
    vp=wa.predict(model,val['raw'],val['cats'],val['t'])
    methods={}
    for method in ('none','temperature','platt'):
        calibration=wa.calibrate(cp,cal['y'],cal['gid'],method)
        p=wpbench._apply_platt(vp,calibration)
        methods[method]=dict(metrics=wpbench._basic_metrics(p,val['y'],val['gid']),calibration=calibration)
    result=dict(methods=methods,training_games=games,validation_games=len(np.unique(val['gid'])),
                best_iteration=model.get('best_iteration'),converged=model['converged'],
                seconds=round(time.time()-started,2))
    return result,model,vp


def average_score(records,method):
    # Equal weight to temporal blocks, not to the busiest league calendar.
    return float(np.mean([r['methods'][method]['metrics']['brier_game'] for r in records]))


def summarize_slices(p,baseline,base,extra,mask):
    rows=base['row_game'][mask]; y=base['y'][mask];gid=base['gid'][mask]
    dates=base['game_dates'][rows]
    groups={'patch':extra['patches'][rows], 'season':np.array([d[:4] for d in dates]),
            'competition':extra['competitions'][rows],
            'phase':np.where(base['t_min'][mask]<15,'early',np.where(base['t_min'][mask]<30,'mid','late')),
            'patch_first_3_observed_days':(extra['extra'][mask,-1]<=3/30.).astype(str),
            'season_first_30_calendar_days':np.array([
                (np.datetime64(d)-np.datetime64(d[:4]+'-01-01')).astype(int)<30 for d in dates]).astype(str),
            'has_player_gold':extra['present'][mask].astype(str)}
    result={}
    for name,values in groups.items():
        result[name]={}
        for value in np.unique(values):
            sel=values==value
            if len(np.unique(gid[sel]))<15:continue
            result[name][str(value)]={**wpbench._basic_metrics(p[sel],y[sel],gid[sel]),
                **wa.series_interval(p[sel],baseline[sel],y[sel],gid[sel],extra['series'],draws=500)}
    return result


def audit_gold(model,calibration,val,seed=71):
    """Sample coherent gold changes, including allocation, share and momentum.

    This is a diagnostic, not the complete production monotonicity contract.
    """
    rng=np.random.default_rng(seed)
    eligible=np.flatnonzero(val['raw'][:,len(wa.CORE_NAMES)+10]>0)
    ix=rng.choice(eligible,min(500,len(eligible)),replace=False)
    raw=val['raw'][ix].copy();cats=val['cats'][ix];t=val['t'][ix]
    before=wpbench._apply_platt(wa.predict(model,raw,cats,t),calibration)
    names={n:i for i,n in enumerate(wa.CORE_NAMES)}
    violations=[]
    for slot in range(10):
        changed=raw.copy();sign=1 if slot<5 else -1
        changed[:,len(wa.CORE_NAMES)+slot]+=sign*.1
        changed[:,names['gold_k']]+=sign
        changed[:,names['gold_mom']]+=sign
        changed[:,names['gold_'+wpgam.GOLD_ROLES[slot%5]]]+=sign
        gold=changed[:,len(wa.CORE_NAMES):len(wa.CORE_NAMES)+10]
        blue=gold[:,:5].sum(axis=1)*10;red=-gold[:,5:].sum(axis=1)*10
        # Match the contract's fractional relative gold definition.
        changed[:,names['gold_rel']]=(blue-red)/np.maximum(blue+red,1)
        after=wpbench._apply_platt(wa.predict(model,changed,cats,t),calibration)
        diff=sign*(after-before)
        violations.append(dict(slot=slot,violations=int((diff < -1e-7).sum()),
                               worst_oriented_change=float(diff.min()) if len(diff) else None))
    return dict(states=len(ix),per_slot=violations,
                passed=all(v['violations']==0 for v in violations))


def audit_development(dataset,output):
    """Inspect patch/season stability of frozen finalists, without reselection."""
    plan=json.loads((output/'search_plan.json').read_text())
    selected=json.loads((output/'selection.json').read_text())
    fingerprint=hashlib.sha256(dataset.read_bytes()).hexdigest()
    if fingerprint!=plan['dataset_sha256']:raise ValueError('Dataset changed since selection')
    codehash=hashlib.sha256(Path(wpgam.__file__).read_bytes()+Path(wpbench.__file__).read_bytes()).hexdigest()
    cache=Path(wpgam.OUT_DIR)/'adapt_cache'/(fingerprint[:12]+codehash[:12])
    base=wpbench._load_base(str(dataset));extra=load_extra(base,dataset,cache)
    blocks,_=wa.split_blocks(base['game_dates'],base['outer_train'],len(plan['blocks']))
    baseline=next(r for r in selected['ranking'] if r['id']=='gam_baseline')
    chosen={r['id']:r for r in [baseline,*selected['winners'].values()]}
    results=[]
    for i,block in enumerate(blocks):
        parts=stage_arrays(base,extra,block,cache);predictions={}
        for label,choice in chosen.items():
            spec=next(s for s in plan['specs'] if s['id']==label)
            record,model,vp=evaluate_candidate(spec,parts,block['fit_end'])
            calibration=record['methods'][choice['calibration']]['calibration']
            predictions[label]=wpbench._apply_platt(vp,calibration)
            del model
        reference=predictions['gam_baseline'];val=parts['validation']
        results.append(dict(block=i+1,start=block['validation_start'],end=block['validation_end'],
            candidates={label:dict(metrics=wpbench._basic_metrics(p,val['y'],val['gid']),
                paired=wa.series_interval(p,reference,val['y'],val['gid'],extra['series']),
                slices=summarize_slices(p,reference,base,extra,val['mask']))
                for label,p in predictions.items()}))
        log.info('Completed frozen-finalist development audit %d',i+1)
    (output/'development_audit.json').write_text(json.dumps(results,indent=2)+'\n')


def rolling_replay(dataset,output):
    """Monthly refits with frozen settings and an earlier 28-day calibration block.

    Earlier test-period games may train later months, just as in deployment.
    No hyperparameter or model-family selection uses these replay results.
    """
    plan=json.loads((output/'search_plan.json').read_text())
    selected=json.loads((output/'selection.json').read_text())
    fingerprint=hashlib.sha256(dataset.read_bytes()).hexdigest()
    if fingerprint!=plan['dataset_sha256']:raise ValueError('Dataset changed since selection')
    codehash=hashlib.sha256(Path(wpgam.__file__).read_bytes()+Path(wpbench.__file__).read_bytes()).hexdigest()
    cache=Path(wpgam.OUT_DIR)/'adapt_cache'/(fingerprint[:12]+codehash[:12])
    base=wpbench._load_base(str(dataset));extra=load_extra(base,dataset,cache)
    baseline=next(r for r in selected['ranking'] if r['id']=='gam_baseline')
    chosen={r['id']:r for r in [baseline,*selected['winners'].values()]}
    dates=base['game_dates'];test=~base['outer_train'];all_mask=test[base['row_game']]
    predictions={label:np.full(len(base['y']),np.nan) for label in chosen};months=[]
    for month in sorted(set(d[:7] for d in dates[test])):
        start=month+'-01';end=str(np.datetime64(month,'M')+1)+'-01'
        cal_start=str(np.datetime64(start)-np.timedelta64(28,'D'))
        block=dict(fit=dates<cal_start,calibration=(dates>=cal_start)&(dates<start),
                   validation=(dates>=start)&(dates<end)&test,
                   fit_end=cal_start,validation_start=start,validation_end=end)
        parts=stage_arrays(base,extra,block,cache);val=parts['validation'];entries={}
        for label,choice in chosen.items():
            spec=next(s for s in plan['specs'] if s['id']==label)
            record,model,vp=evaluate_candidate(spec,parts,block['fit_end'])
            calibration=record['methods'][choice['calibration']]['calibration']
            pp=wpbench._apply_platt(vp,calibration)
            predictions[label][val['mask']]=pp
            entries[label]=dict(metrics=wpbench._basic_metrics(pp,val['y'],val['gid']),
                                best_iteration=model.get('best_iteration'),training_games=record['training_games'])
            del model
        months.append(dict(month=month,calibration_start=cal_start,candidates=entries))
        log.info('Monthly replay %s: %s',month,{k:round(v['metrics']['brier_game'],6) for k,v in entries.items()})
    pp={k:v[all_mask] for k,v in predictions.items()}
    if not all(np.isfinite(p).all() for p in pp.values()):raise ValueError('Incomplete monthly replay')
    yy,gg=base['y'][all_mask],base['gid'][all_mask];reference=pp['gam_baseline']
    result=dict(selection_frozen=selected['selected'],cadence='calendar month',calibration_days=28,
                months=months,production_changed=False,
                candidates={k:dict(metrics=wpbench._basic_metrics(p,yy,gg),
                    paired=wa.series_interval(p,reference,yy,gg,extra['series']),
                    slices=summarize_slices(p,reference,base,extra,all_mask)) for k,p in pp.items()})
    (output/'rolling_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'rolling_predictions.npz',gid=gg,y=yy,**pp)


def run(dataset,output,trials=24,folds=3):
    import lightgbm
    output.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(dataset.read_bytes()).hexdigest()
    codehash=hashlib.sha256(Path(wpgam.__file__).read_bytes()+Path(wpbench.__file__).read_bytes()).hexdigest()
    cache=Path(wpgam.OUT_DIR)/'adapt_cache'/(fingerprint[:12]+codehash[:12]);cache.mkdir(parents=True,exist_ok=True)
    registry=json.loads((Path(wpgam.OUT_DIR)/'evaluation_registry.json').read_text())
    base=wpbench._load_base(str(dataset))
    if base['split_method']!='date' or max(base['game_dates'])>registry['consumed_through']:
        raise ValueError('Fresh dates are reserved for the promotion gate; use a consumed dated dataset')
    blocks,final=wa.split_blocks(base['game_dates'],base['outer_train'],folds)
    specs=search_specs(trials)
    implementation=hashlib.sha256(Path(wa.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    plan=dict(dataset_sha256=fingerprint,implementation_sha256=implementation,
              library_version=lightgbm.__version__,specs=specs,
              selection='First-block screening, then mean Brier across all development blocks; calibration selected on development only',
              calibration_days=28,
              blocks=[{k:v for k,v in b.items() if isinstance(v,str)} for b in blocks],
              final={k:v for k,v in final.items() if isinstance(v,str)},
              production_changed=False,registry_consumed_through=registry['consumed_through'],
              long_memory_priors='Shared stacked priors keep all earlier fit-block history; window/decay affect state learners',
              fixed_or_deferred=dict(objective='binary log loss',boosting='gbdt',device='cpu',
                  class_weights='none; preserve probability prevalence',dart='deferred; no standard early stopping',
                  seeds='43 during selection; 73 validation sensitivity for selected tree',
                  constraints='advanced where enabled; categories unconstrained',
                  row_sampling='LightGBM bagging samples states; total training weights remain game balanced'))
    plan_path=output/'search_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Existing output has a different plan; select a new output directory')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    extra=load_extra(base,dataset,cache)
    log.info('Causal gold coverage %.2f%%; %d games, %d states',extra['present'].mean()*100,len(base['game_gid']),len(base['y']))
    records={s['id']:[] for s in specs};active=specs
    for fold,block in enumerate(blocks):
        parts=stage_arrays(base,extra,block,cache)
        log.info('Development fold %d: %s to %s',fold+1,block['validation_start'],block['validation_end'])
        for spec in active:
            file=output/f"trial_{spec['id']}_fold{fold}.json"
            if file.exists(): result=json.loads(file.read_text())
            else:
                result,model,_=evaluate_candidate(spec,parts,block['fit_end'])
                del model
                file.write_text(json.dumps(result,indent=2)+'\n')
            records[spec['id']].append(result)
            best=min(v['metrics']['brier_game'] for v in result['methods'].values())
            log.info('fold=%d %-20s Brier=%.6f rounds=%s (%ss)',fold+1,spec['id'],best,result['best_iteration'],result['seconds'])
        if fold==0:
            retain={'gam_baseline','gam_core_decay','gam_rich_0','lgb_control_core','lgb_control_rich'}
            for family,n in (('gam',3),('lightgbm',4)):
                eligible=[s for s in specs if s['family']==family and s['id']!='gam_baseline']
                eligible.sort(key=lambda s:min(average_score(records[s['id']],m) for m in ('none','temperature','platt')))
                retain.update(s['id'] for s in eligible[:n])
            active=[s for s in specs if s['id'] in retain]
        del parts
    ranked=[]
    for spec in active:
        method=min(('none','temperature','platt'),key=lambda m:average_score(records[spec['id']],m))
        ranked.append(dict(id=spec['id'],family=spec['family'],calibration=method,
                           mean_brier=average_score(records[spec['id']],method)))
    ranked.sort(key=lambda r:r['mean_brier'])
    winners={family:next(r for r in ranked if r['family']==family) for family in ('gam','lightgbm')}
    baseline=next(r for r in ranked if r['id']=='gam_baseline')
    seed_spec=dict(next(s for s in specs if s['id']==winners['lightgbm']['id']),seed=73)
    seed_parts=stage_arrays(base,extra,blocks[-1],cache)
    seed_result,seed_model,_=evaluate_candidate(seed_spec,seed_parts,blocks[-1]['fit_end'])
    del seed_parts,seed_model
    selection=dict(ranking=ranked,winners=winners,selected=ranked[0],records=records,
                   seed_sensitivity=dict(seed=73,candidate=seed_spec['id'],last_development_block=seed_result))
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    # All configuration choices are frozen before final diagnostic inference.
    final_parts=stage_arrays(base,extra,final,cache)
    predictions={};models={};calibrations={};metrics={}
    for chosen in {v['id']:v for v in [baseline,*winners.values()]}.values():
        spec=next(s for s in specs if s['id']==chosen['id'])
        model,games=fit_candidate(spec,final_parts,final['fit_end'])
        cal,val=final_parts['calibration'],final_parts['validation']
        cp=wa.predict(model,cal['raw'],cal['cats'],cal['t'])
        calibration=wa.calibrate(cp,cal['y'],cal['gid'],chosen['calibration'])
        vp=wpbench._apply_platt(wa.predict(model,val['raw'],val['cats'],val['t']),calibration)
        label=chosen['id'];predictions[label]=vp;models[label]=model;calibrations[label]=calibration
        metrics[label]={**wpbench._basic_metrics(vp,val['y'],val['gid']),
                        'training_games':games,'best_iteration':model.get('best_iteration'),
                        'calibration_method':chosen['calibration'],'calibration':calibration}
        wa.save_model(model,output/label,calibration)
        restored=wa.load_model(output/label)
        np.testing.assert_allclose(wa.predict(model,val['raw'][:100],val['cats'][:100],val['t'][:100]),
                                   wa.predict(restored,val['raw'][:100],val['cats'][:100],val['t'][:100]),rtol=1e-10,atol=1e-10)
        log.info('Final %-20s Brier=%.6f',label,metrics[label]['brier_game'])
    val=final_parts['validation'];reference=predictions['gam_baseline']
    for label,p in predictions.items():
        metrics[label]['paired']=wa.series_interval(p,reference,val['y'],val['gid'],extra['series'])
        metrics[label]['slices']=summarize_slices(p,reference,base,extra,val['mask'])
        spec=next(s for s in specs if s['id']==label)
        fit=final_parts['fit']
        keep,_=wa.temporal_weights(fit['gid'],fit['date'],final['fit_end'],
                                  spec.get('window_days'),spec.get('half_life'))
        unseen=wa.unseen_role_champions(fit['cats'][keep],val['cats'])
        metrics[label]['unseen_champion_games']=int(len(np.unique(val['gid'][unseen])))
        if unseen.any():metrics[label]['unseen_champion_metrics']=wpbench._basic_metrics(p[unseen],val['y'][unseen],val['gid'][unseen])
        metrics[label]['gold_audit']=audit_gold(models[label],calibrations[label],val)
    result=dict(plan=plan,selection=ranked,selected=ranked[0],final=metrics,
                test_games=int(final['validation'].sum()),test_states=len(val['y']),
                gold_coverage=float(extra['present'].mean()),production_changed=False,
                limitation='Retrospective diagnostic, not a fresh promotion gate. Physical gold audit is sampled, not complete.')
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'predictions.npz',gid=val['gid'],y=val['y'],**predictions)
    print(json.dumps({k:result[k] for k in ('selected','test_games','test_states')},indent=2))
    for k,v in metrics.items():print(k,v['brier_game'],v['paired'],'gold_audit',v['gold_audit']['passed'])


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(message)s')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path(wpgam.OUT_DIR)/'states.npz')
    p.add_argument('--output',type=Path,default=Path(wpgam.OUT_DIR)/'adapt_v2')
    p.add_argument('--trials',type=int,default=24)
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--audit-development',action='store_true')
    p.add_argument('--rolling',action='store_true')
    a=p.parse_args()
    if a.audit_development:audit_development(a.dataset,a.output)
    elif a.rolling:rolling_replay(a.dataset,a.output)
    else:run(a.dataset,a.output,a.trials,a.folds)
