"""Experimental calendar-aware win-probability models and temporal validation.

This module has no production dispatch or promotion side effects. Both model
families consume the same resource and categorical information. Their ways of
representing interactions differ: sparse, pooled GAM terms versus native trees.
"""
from collections import Counter
import json
import re

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from . import wpgam, wpbench

CORE_NAMES = wpgam.PRIOR_INPUTS + wpgam.STATE_FEATURES
EXTRA_NAMES = [f'gold_slot_{i}' for i in range(10)] + [
    'has_player_gold', 'season_number', 'patch_number', 'patch_observed_age']
CAT_NAMES = ([f'champ_{i}' for i in range(10)] + ['season', 'patch', 'competition'] +
             [f'champ_season_{i}' for i in range(10)] +
             [f'champ_patch_{i}' for i in range(10)])


def normalize_competition(name):
    """Strip calendar/stage labels, not distinctions such as LCK versus LCK CL."""
    value = re.sub(r'\b20\d{2}\b', '', str(name))
    value = re.sub(r'\b(Spring|Summer|Winter|Fall|Playoffs?|Play-In|Main Event|'
                   r'Rounds?\s*[\d-]+|Split\s*\d+|Finals?)\b', '', value, flags=re.I)
    return ' '.join(value.split()) or 'unknown'


def calendar_metadata(dates, patches, competitions, C):
    dates = np.asarray(dates).astype('datetime64[D]')
    patches = np.asarray(patches).astype(str)
    season, number = np.zeros(len(dates)), np.zeros(len(dates))
    for i, value in enumerate(patches):
        match = re.fullmatch(r'(\d+)\.(\d+)(?:\.\d+)?', value)
        if match:
            season[i], number[i] = map(int, match.groups())
    age = np.zeros(len(dates))
    # An observed first date is known at that date. No release dates or labels
    # are inferred from future outcomes. Report this as observed patch age.
    for p in np.unique(patches):
        mask = patches == p
        age[mask] = (dates[mask] - dates[mask].min()).astype(float)
    years = dates.astype('datetime64[Y]').astype(str)
    cats = np.empty((len(dates), len(CAT_NAMES)), dtype=object)
    cats[:, :10] = np.asarray(C).astype(str)
    cats[:, 10] = years
    cats[:, 11] = patches
    cats[:, 12] = [normalize_competition(v) for v in competitions]
    for j in range(10):
        cats[:, 13+j] = [f'{c}@{s}' for c,s in zip(C[:, j], years)]
        cats[:, 23+j] = [f'{c}@{p}' for c,p in zip(C[:, j], patches)]
    return np.column_stack([season, number, np.minimum(age, 60)/30.]), cats


def split_blocks(game_dates, outer_train, folds=3, calibration_days=28):
    """Disjoint development blocks; each has an earlier calibration block."""
    dates = np.asarray(game_dates).astype(str)
    train_dates = np.sort(dates[outer_train])
    boundaries = [train_dates[min(len(train_dates)-1, int(q*len(train_dates)))]
                  for q in np.linspace(.5, 1., folds+1)[:-1]]
    test_start = min(dates[~outer_train])
    boundaries.append(test_start)
    blocks = []
    for start,end in zip(boundaries[:-1], boundaries[1:]):
        cal_start = str(np.datetime64(start)-np.timedelta64(calibration_days,'D'))
        blocks.append(dict(fit=dates < cal_start,
                           calibration=(dates >= cal_start) & (dates < start),
                           validation=(dates >= start) & (dates < end),
                           fit_end=cal_start, validation_start=start, validation_end=end))
    cal_start = str(np.datetime64(test_start)-np.timedelta64(calibration_days,'D'))
    final = dict(fit=dates < cal_start,
                 calibration=(dates >= cal_start) & outer_train,
                 validation=~outer_train, fit_end=cal_start,
                 validation_start=test_start, validation_end=str(max(dates)))
    for block in blocks + [final]:
        if not all(block[k].any() for k in ('fit','calibration','validation')):
            raise ValueError('Empty temporal block')
        assert max(dates[block['fit']]) < min(dates[block['calibration']])
        assert max(dates[block['calibration']]) < min(dates[block['validation']])
    return blocks, final


def temporal_weights(gids, dates, cutoff, window_days=None, half_life=None):
    age = (np.datetime64(cutoff, 'D') - np.asarray(dates).astype('datetime64[D]')).astype(float)
    keep = age > 0
    if window_days:
        keep &= age <= window_days
    if not keep.any():
        raise ValueError('No strictly earlier training rows')
    weights = wpgam._game_balanced_weights(np.asarray(gids)[keep])
    if half_life:
        weights *= np.exp2(-age[keep]/half_life)
    weights /= weights.mean()
    return keep, weights


def fit_categories(cats, gids, min_games=10):
    """Fit category vocabularies on games, never on repeated state counts."""
    _, first = np.unique(gids, return_index=True)
    maps = []
    for j in range(cats.shape[1]):
        counts = Counter(str(v) for v in cats[first, j])
        minimum = min_games if j >= 13 else 1
        values = sorted(k for k,n in counts.items() if n >= minimum and k != '-1'
                        and not k.startswith('-1@'))
        maps.append({v:i for i,v in enumerate(values)})
    return maps


def encode_categories(cats, maps):
    out = np.empty(cats.shape, dtype=np.int32)
    for j,mapping in enumerate(maps):
        out[:, j] = [mapping.get(str(v), -1) for v in cats[:, j]]
    return out


def unseen_role_champions(training_cats, validation_cats):
    """Novel champion/slot combinations, including for models without categories."""
    unseen=np.zeros(len(validation_cats),dtype=bool)
    for j in range(10):
        known=set(str(v) for v in training_cats[:,j] if str(v)!='-1')
        unseen |= np.array([str(v) not in known for v in validation_cats[:,j]])
    return unseen


def sparse_context(encoded, maps, gold, time):
    """Pooled category effects plus champion × resources × in-game time.

    Gold is signed by side. Resource slopes are constrained nonnegative in the
    GAM so this block alone cannot punish a player's additional gold.
    """
    n = len(encoded)
    row_parts, col_parts, values, penalties, signs = [], [], [], [], []
    offset = 0
    ramp = np.clip(time/30., 0, 2)
    for j,mapping in enumerate(maps):
        mask = encoded[:, j] >= 0
        rows = np.flatnonzero(mask)
        width = len(mapping)
        row_parts.append(rows); col_parts.append(encoded[mask,j]+offset)
        values.append(np.ones(len(rows)))
        penalties.extend([3. if j >= 23 else 2. if j >= 13 else 1.] * width)
        signs.extend([False]*width)
        offset += width
        if j < 10:
            for v, constrained in ((ramp, False), (gold[:, j], True),
                                   (gold[:, j]*ramp, True)):
                row_parts.append(rows); col_parts.append(encoded[mask,j]+offset)
                values.append(v[mask])
                penalties.extend([2.]*width); signs.extend([constrained]*width)
                offset += width
    mat = sparse.csr_matrix((np.concatenate(values),
                             (np.concatenate(row_parts), np.concatenate(col_parts))),
                            shape=(n, offset))
    return mat, np.array(penalties), np.array(signs)


def fit_gam(raw, cats, y, gids, t, weights, spec):
    enriched = spec.get('features', 'rich') == 'rich'
    names = CORE_NAMES + (EXTRA_NAMES if enriched else [])
    raw = raw[:, :len(names)]
    scale = wpgam._scale_fit(raw, feature_names=names)
    Z = wpgam._scale_apply(raw, *scale)
    B = wpgam.time_basis(t)
    maps = fit_categories(cats, gids, spec.get('min_category_games',10)) if enriched else []
    if enriched:
        S, penalties, positive = sparse_context(encode_categories(cats,maps), maps,
                                                raw[:,len(CORE_NAMES):len(CORE_NAMES)+10], t)
    else:
        S, penalties, positive = sparse.csr_matrix((len(raw),0)), np.zeros(0), np.zeros(0,dtype=bool)
    nf, nk = Z.shape[1], B.shape[1]
    nt = (nf+1)*nk
    mono = wpgam.MONOTONE_FEATURES | {f'gold_slot_{i}' for i in range(10)}
    bounds = [(-10.,10.)]*nk
    for name in names:
        bounds.extend([(0.,10.) if name in mono else (-10.,10.)]*nk)
    bounds.extend([(0.,5.) if p else (-5.,5.) for p in positive])
    l2, smooth, context_l2 = spec.get('l2',24.), spec.get('smooth',70.), spec.get('context_l2',800.)

    def objective(flat):
        theta, beta = flat[:nt].reshape(nf+1,nk), flat[nt:]
        effects = np.einsum('ik,fk->if',B,theta[1:],optimize=True)
        eta = np.einsum('ik,k->i',B,theta[0],optimize=True) + np.sum(Z*effects,axis=1) + S @ beta
        resid = weights*(wpgam._sigmoid(eta)-y)
        loss = np.sum(weights*(np.logaddexp(0,eta)-y*eta))
        grad = np.empty_like(theta)
        grad[0] = np.einsum('ik,i->k',B,resid,optimize=True)
        grad[1:] = np.einsum('if,ik->fk',Z*resid[:,None],B,optimize=True)
        loss += .5*l2*np.sum(theta[1:]**2)
        grad[1:] += l2*theta[1:]
        delta = np.diff(theta,axis=1)
        loss += .5*smooth*np.sum(delta**2)
        grad[:,:-1] -= smooth*delta; grad[:,1:] += smooth*delta
        loss += .5*context_l2*np.sum(penalties*beta**2)
        gb = S.T @ resid + context_l2*penalties*beta
        return float(loss), np.r_[grad.ravel(),gb]

    initial = np.zeros(nt+S.shape[1])
    p = np.clip(np.average(y,weights=weights),1e-5,1-1e-5)
    initial[:nk] = np.log(p/(1-p))
    result = minimize(objective, initial, jac=True, bounds=bounds, method='L-BFGS-B',
                      options=dict(maxiter=350,ftol=1e-8,gtol=1e-4,maxcor=15))
    if not result.success:
        result = minimize(objective, result.x, jac=True, bounds=bounds, method='L-BFGS-B',
                          options=dict(maxiter=700,ftol=1e-8,gtol=1e-4,maxcor=20))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'GAM optimizer failed: {result.message}')
    return dict(family='gam', scale=scale, maps=maps, theta=result.x[:nt].reshape(nf+1,nk),
                beta=result.x[nt:], width=len(names), spec=spec, iterations=int(result.nit),
                converged=True)


def lgb_design(raw, cats, scale, maps, rich):
    width = len(CORE_NAMES) + (len(EXTRA_NAMES) if rich else 0)
    Z = wpgam._scale_apply(raw[:,:width],*scale)
    if rich:
        encoded=encode_categories(cats,maps).astype(float)
        encoded[encoded<0]=np.nan
        return np.column_stack([Z, encoded])
    return Z


def fit_lgb(raw,cats,y,gids,t,weights,spec,cal):
    import lightgbm as lgb
    rich = spec.get('features','rich') == 'rich'
    names = CORE_NAMES + (EXTRA_NAMES if rich else [])
    scale = wpgam._scale_fit(raw[:,:len(names)], feature_names=names)
    maps = fit_categories(cats,gids,spec.get('min_category_games',10)) if rich else []
    X = np.column_stack([t/30.,lgb_design(raw,cats,scale,maps,rich)])
    cx = np.column_stack([cal['t']/30.,lgb_design(cal['raw'],cal['cats'],scale,maps,rich)])
    categorical = list(range(1+len(names),X.shape[1]))
    parameters = {k:v for k,v in spec.items() if k not in {
        'family','features','window_days','half_life','min_category_games','monotone',
        'early_stopping_rounds','max_rounds','calibration'}}
    parameters.update(objective='binary', metric='None', verbosity=-1,
                      num_threads=4, deterministic=True,force_col_wise=True,
                      feature_pre_filter=False)
    if spec.get('monotone',False):
        mono = wpgam.MONOTONE_FEATURES | {f'gold_slot_{i}' for i in range(10)}
        parameters['monotone_constraints'] = [0]+[int(n in mono) for n in names]+[0]*len(maps)
        parameters['monotone_constraints_method'] = 'advanced'
        parameters['feature_fraction'] = 1.  # advanced method requires all features
    cw = wpgam._game_balanced_weights(cal['gid'])
    train = lgb.Dataset(X,label=y,weight=weights,categorical_feature=categorical or 'auto')
    valid = lgb.Dataset(cx,label=cal['y'],weight=cw,reference=train)
    def brier(p,d):
        return 'game_brier',float(np.average((p-d.get_label())**2,weights=d.get_weight())),False
    model = lgb.train(parameters,train,num_boost_round=spec.get('max_rounds',1200),
                      valid_sets=[valid],feval=brier,
                      callbacks=[lgb.early_stopping(spec.get('early_stopping_rounds',80),
                                                  first_metric_only=True,verbose=False)])
    return dict(family='lightgbm',model=model,scale=scale,maps=maps,rich=rich,spec=spec,
                best_iteration=int(model.best_iteration),converged=True)


def predict(model, raw, cats, t):
    if model['family'] == 'lightgbm':
        X = np.column_stack([t/30.,lgb_design(raw,cats,model['scale'],model['maps'],model['rich'])])
        p = model['model'].predict(X,num_iteration=model['best_iteration'])
    else:
        Z = wpgam._scale_apply(raw[:,:model['width']],*model['scale'])
        B = wpgam.time_basis(t)
        theta = model['theta']
        eta = np.einsum('ik,k->i',B,theta[0],optimize=True) + np.sum(
            Z*np.einsum('ik,fk->if',B,theta[1:],optimize=True),axis=1)
        if model['maps']:
            S,_,_ = sparse_context(encode_categories(cats,model['maps']),model['maps'],
                                   raw[:,len(CORE_NAMES):len(CORE_NAMES)+10],t)
            eta += S @ model['beta']
        p = wpgam._sigmoid(eta)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise FloatingPointError('Nonfinite/out-of-range probability')
    return p


def calibrate(p, y, gids, method):
    if method == 'none':
        return dict(intercept=0.,slope=1.)
    if method == 'temperature':
        logits=np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1))
        beta=wpgam._fit_static_logit(logits[:,None],y,np.array([.25]),
                                    bounds=[(.05,5.)],weights=wpgam._game_balanced_weights(gids))
        return dict(intercept=0.,slope=float(beta[0]))
    if method != 'platt':
        raise ValueError('Unknown calibration method')
    return wpbench._calibration_diagnostics(p,y,gids)


def series_interval(p, baseline, y, gids, series, draws=2000):
    ug,delta = wpbench._per_game((p-y)**2-(baseline-y)**2,gids)
    groups = np.array([series[int(g)] for g in ug])
    _, inv=np.unique(groups,return_inverse=True)
    sums,counts=np.bincount(inv,weights=delta),np.bincount(inv)
    rng=np.random.default_rng(240304873)
    sample=rng.integers(len(sums),size=(draws,len(sums)))
    values=sums[sample].sum(axis=1)/counts[sample].sum(axis=1)
    return dict(delta_brier=float(delta.mean()),ci95=np.quantile(values,[.025,.975]).tolist(),
                games=len(ug),series=len(sums))


def save_model(model, directory, calibration):
    directory.mkdir(parents=True,exist_ok=True)
    meta={k:model[k] for k in ('family','spec','maps')}
    meta.update(calibration=calibration,kind='experimental_meta_adaptive_v1',
                core_names=CORE_NAMES,extra_names=EXTRA_NAMES,categorical_names=CAT_NAMES)
    arrays={f'scale_{i}':v for i,v in enumerate(model['scale'])}
    if model['family']=='gam':
        arrays.update(theta=model['theta'],beta=model['beta'])
        meta['width']=model['width']
    else:
        model['model'].save_model(str(directory/'trees.txt'),num_iteration=model['best_iteration'])
        meta.update(rich=model['rich'],best_iteration=model['best_iteration'])
    np.savez_compressed(directory/'arrays.npz',**arrays)
    (directory/'model.json').write_text(json.dumps(meta,indent=2)+'\n')


def load_model(directory):
    meta=json.loads((directory/'model.json').read_text())
    if (meta['kind']!='experimental_meta_adaptive_v1' or meta['core_names']!=CORE_NAMES
            or meta['extra_names']!=EXTRA_NAMES or meta['categorical_names']!=CAT_NAMES):
        raise ValueError('Incompatible experimental feature contract')
    with np.load(directory/'arrays.npz',allow_pickle=False) as d:
        model=dict(meta,scale=tuple(d[f'scale_{i}'].copy() for i in range(4)))
        if meta['family']=='gam':
            model.update(theta=d['theta'].copy(),beta=d['beta'].copy())
    if meta['family']=='lightgbm':
        import lightgbm as lgb
        model['model']=lgb.Booster(model_file=str(directory/'trees.txt'))
    return model


def predict_calibrated(model,raw,cats,t):
    """Use the calibration packaged with a saved experimental model."""
    return wpbench._apply_platt(predict(model,raw,cats,t),
                                model.get('calibration',dict(intercept=0.,slope=1.)))
