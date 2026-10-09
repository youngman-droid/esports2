"""Reevaluate alternative families with repaired features and past-only calibration.

All settings are selected on three earlier blocks, then frozen for the later
block and monthly replay. Existing diagnostic outcomes are not fresh evidence.
This runner cannot deploy models or update the evaluation registry.
"""
import argparse
import gc
import hashlib
import json
import logging
import pickle
import platform
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpadapt as wa, wpaudit, wpbench, wpgam
from research import wpx_altmodels as alt
from research.wpx_adapt import load_extra

log = logging.getLogger("methods_v9")
METHODS = ("none", "temperature", "platt")
REFERENCE = "constrained_gam"
MONOTONE = (REFERENCE, "spline_gam", "monotone_hist_boost", "gam_boost_stack",
            "monotone_lightgbm", "rich_gam", "monotone_rich_lightgbm")


def specifications():
    gam = [dict(l2=24., smooth=70.), dict(l2=12., smooth=35.)]
    trees = [dict(max_leaf_nodes=15, min_samples_leaf=150, l2=10.),
             dict(max_leaf_nodes=31, min_samples_leaf=200, l2=15.)]
    lgb = [dict(num_leaves=n, min_data_in_leaf=m, lambda_l2=l2, rounds=400)
           for n, m, l2 in ((7, 400, 30.), (15, 200, 15.), (31, 400, 30.))]
    return {
        REFERENCE: gam[:1], "unconstrained_gam": gam,
        "ridge_logit": [dict(C=.03), dict(C=.3)],
        "hist_boost": trees, "monotone_hist_boost": trees,
        "spline_gam": alt.specs()["spline_gam"],
        "mlp": [alt.specs()["mlp"][0], alt.specs()["mlp"][2]],
        "rff_logit": alt.specs()["rff_logit"],
        "gam_boost_stack": alt.specs()["gam_boost_stack"],
        "lightgbm": lgb, "monotone_lightgbm": lgb,
        "rich_gam": [dict(features="rich", l2=24., smooth=70., context_l2=c)
                     for c in (800., 2000.)],
        "rich_lightgbm": lgb[:2], "monotone_rich_lightgbm": lgb[:2],
    }


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def per_game(values, gids):
    ug, inv = np.unique(gids, return_inverse=True)
    return ug, np.bincount(inv, weights=values) / np.bincount(inv)


def metrics(p, y, gids):
    p, y = np.asarray(p), np.asarray(y)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    loss = (p-y)**2
    pc = np.clip(p, 1e-6, 1-1e-6)
    ll = -(y*np.log(pc)+(1-y)*np.log1p(-pc))
    return dict(brier_game=float(per_game(loss,gids)[1].mean()),
                logloss_game=float(per_game(ll,gids)[1].mean()),
                brier_state=float(loss.mean()), logloss_state=float(ll.mean()),
                games=len(np.unique(gids)), states=len(y))


def paired(p, ref, y, gids, clusters, draws=2000):
    ug, delta = per_game((p-y)**2-(ref-y)**2,gids)
    keys = [clusters[int(g)] for g in ug]
    _, inv = np.unique(keys,return_inverse=True)
    sums, counts = np.bincount(inv,weights=delta), np.bincount(inv)
    rng = np.random.default_rng(240304873)
    ix = rng.integers(len(sums),size=(draws,len(sums)))
    boot = sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    return dict(delta_brier=float(delta.mean()), ci95=np.quantile(boot,[.025,.975]).tolist(),
                games=len(ug), clusters=len(sums))


def prepare(base, extra, block, directory):
    directory.mkdir(parents=True,exist_ok=True)
    path = directory / "core.npy"
    if not path.exists():
        log.info("Fitting shared chronological priors before %s",block["fit_end"])
        np.save(path,wpbench._stacked_arrays(base,block["fit"])["raw"])
    core = np.load(path,mmap_mode="r")
    parts = {}
    for name in ("fit", "calibration", "validation"):
        mask = block[name][base["row_game"]]
        rows = base["row_game"][mask]
        parts[name] = dict(raw=np.column_stack([core[mask],extra["extra"][mask]]),
            cats=extra["cats"][rows],y=base["y"][mask],gid=base["gid"][mask],
            t=base["t_min"][mask],date=base["game_dates"][rows],mask=mask)
    return parts


def chronological_gam_scores(part):
    """Every OOF state learner and its supplied priors precede scored dates."""
    raw = part["raw"][:,:len(wa.CORE_NAMES)]
    first = wpgam._game_rows(part["gid"])
    gids, dates = part["gid"][first], part["date"][first]
    scores = np.zeros(len(raw))
    for train, test in wpgam._stack_folds(gids,k=5,dates=dates):
        if not train.any():
            continue
        tr, te = np.isin(part["gid"],gids[train]), np.isin(part["gid"],gids[test])
        assert max(part["date"][tr]) < min(part["date"][te])
        sub = wpgam.fit_state_model(raw[tr],part["y"][tr],part["gid"][tr],part["t"][tr])
        scores[te] = wpgam.predict_state(sub,raw[te],part["t"][te],components=True)[1]
    full = wpgam.fit_state_model(raw,part["y"],part["gid"],part["t"])
    return scores, full


def lgb_design(raw, cats, t, scale, maps, rich):
    return np.column_stack([t/30.,wa.lgb_design(raw,cats,scale,maps,rich)])


def fit_method(family, spec, part, stack=None):
    raw, cats, y, gid, t = (part[k] for k in ("raw","cats","y","gid","t"))
    core = raw[:,:len(wa.CORE_NAMES)]
    weights = wpgam._game_balanced_weights(gid)
    if family == "rich_gam":
        return dict(adapter="rich_gam",model=wa.fit_gam(raw,cats,y,gid,t,weights,spec))
    if "lightgbm" in family:
        import lightgbm as lgb
        rich = "rich" in family
        names = wa.CORE_NAMES + (wa.EXTRA_NAMES if rich else [])
        scale = wpgam._scale_fit(raw[:,:len(names)],feature_names=names)
        maps = wa.fit_categories(cats,gid,min_games=10) if rich else []
        X = lgb_design(raw,cats,t,scale,maps,rich)
        params = dict(objective="binary",verbosity=-1,num_threads=4,seed=43,
            deterministic=True,force_col_wise=True,learning_rate=.03,
            feature_fraction=1.,bagging_fraction=1.,cat_smooth=50.,cat_l2=50.,
            min_data_per_group=100,**{k:v for k,v in spec.items() if k!="rounds"})
        if family.startswith("monotone"):
            mono = wpgam.MONOTONE_FEATURES | {f"gold_slot_{i}" for i in range(10)}
            params.update(monotone_constraints=[0]+[int(n in mono) for n in names]+[0]*len(maps),
                          monotone_constraints_method="advanced")
        train = lgb.Dataset(X,label=y,weight=weights,
            categorical_feature=list(range(1+len(names),X.shape[1])) or "auto")
        model = lgb.train(params,train,num_boost_round=spec["rounds"])
        return dict(adapter="lightgbm",model=model,scale=scale,maps=maps,rich=rich)
    if family == "gam_boost_stack":
        from sklearn.ensemble import HistGradientBoostingClassifier
        oof, gam = stack if stack is not None else chronological_gam_scores(part)
        scale = wpgam._scale_fit(core,feature_names=wa.CORE_NAMES)
        X = np.column_stack([np.clip(t,0,60)/45.,oof,wpgam._scale_apply(core,*scale)])
        model = HistGradientBoostingClassifier(loss="log_loss",learning_rate=spec["lr"],
            max_iter=spec["iters"],max_leaf_nodes=spec["leaves"],min_samples_leaf=spec["msl"],
            l2_regularization=spec["l2"],early_stopping=False,random_state=43,
            monotonic_cst=[0,1]+[int(n in wpgam.MONOTONE_FEATURES) for n in wa.CORE_NAMES])
        model.fit(X,y,sample_weight=weights)
        return dict(adapter="gam_boost_stack",model=model,gam=gam,scale=scale)
    return dict(adapter="core",model=alt.fit_method(family,spec,core,y,gid,t))


def predict(model, part):
    raw,cats,t = (part[k] for k in ("raw","cats","t"))
    core = raw[:,:len(wa.CORE_NAMES)]
    if model["adapter"] == "rich_gam":
        return wa.predict(model["model"],raw,cats,t)
    if model["adapter"] == "lightgbm":
        X = lgb_design(raw,cats,t,model["scale"],model["maps"],model["rich"])
        return model["model"].predict(X,num_threads=4)
    if model["adapter"] == "gam_boost_stack":
        score = wpgam.predict_state(model["gam"],core,t,components=True)[1]
        X = np.column_stack([np.clip(t,0,60)/45.,score,wpgam._scale_apply(core,*model["scale"])])
        return model["model"].predict_proba(X)[:,1]
    return alt.predict_method(model["model"],core,t)


def gold_audit(predictor, part, seed=904):
    eligible = np.flatnonzero(part["raw"][:,len(wa.CORE_NAMES)+10] > 0)
    ix = np.random.default_rng(seed).choice(eligible,min(5000,len(eligible)),replace=False)
    if not len(ix):
        return dict(passed=False,reason="no complete player gold states")
    probe = {k:part[k][ix] for k in ("raw","cats","t")}
    raw = probe["raw"]
    total = np.abs(raw[:,len(wa.CORE_NAMES):len(wa.CORE_NAMES)+10]).sum(axis=1)*10.
    before = predictor(probe)
    if not np.isfinite(before).all(): raise ValueError("Invalid gold-audit baseline")
    cases = []
    for amount in (.1,1.):
        for slot in range(10):
            changed = wpaudit.gold_perturbation(raw,wa.CORE_NAMES,total,slot,amount)
            sign = 1 if slot < 5 else -1
            changed[:,len(wa.CORE_NAMES)+slot] += sign*amount/10.
            after = predictor(dict(probe,raw=changed))
            if not np.isfinite(after).all(): raise ValueError("Invalid gold-audit prediction")
            delta = sign*(after-before)
            cases.append(dict(slot=slot,gold_added=amount*1000,
                violations=int(np.sum(delta < -1e-8)),worst_probability_change=float(delta.min())))
    violations = sum(c["violations"] for c in cases)
    return dict(passed=violations==0,states=len(ix),comparisons=len(ix)*20,
                violations=violations,per_slot=cases)


def run_stage(base, extra, block, directory, specs, choices=None, audit=False):
    parts = prepare(base,extra,block,directory)
    report, predictions, models = {}, {}, {}
    stack = None
    for family, candidates in specs.items():
        settings = [(choices[family]["index"],candidates[choices[family]["index"]])] if choices else list(enumerate(candidates))
        for i,spec in settings:
            key = family+"_"+str(i)
            path = directory/(key+".pkl")
            started = time.time()
            if path.exists():
                with path.open("rb") as f: model=pickle.load(f)
            else:
                log.info("%s fitting %s",directory.name,key)
                if family=="gam_boost_stack" and stack is None:
                    stack = chronological_gam_scores(parts["fit"])
                model = fit_method(family,spec,parts["fit"],stack)
                with path.open("wb") as f: pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)
            cp, vp = predict(model,parts["calibration"]), predict(model,parts["validation"])
            metrics(cp,parts["calibration"]["y"],parts["calibration"]["gid"])
            record = dict(family=family,index=i,spec=spec,methods={})
            methods = [choices[family]["calibration"]] if choices else METHODS
            for method in methods:
                c = wa.calibrate(cp,parts["calibration"]["y"],parts["calibration"]["gid"],method)
                p = wpbench._apply_platt(vp,c)
                record["methods"][method] = dict(calibration=c,metrics=metrics(p,parts["validation"]["y"],parts["validation"]["gid"]))
                predictions[family if choices else key+"_"+method] = p
                if choices: models[family] = (model,c)
            if audit:
                record["gold_audit"] = gold_audit(lambda part: wpbench._apply_platt(predict(model,part),c),parts["validation"])
            record["seconds"] = round(time.time()-started,2)
            report[key] = record
            dump(directory/"results.json",report)
            log.info("%s %s %s (%.1fs)",directory.name,key,
                {m:round(r["metrics"]["brier_game"],6) for m,r in record["methods"].items()},time.time()-started)
            del cp,vp
            if not choices: del model
            gc.collect()
    return report,predictions,models,parts


def select(records, specs):
    choices, ranking = {}, []
    for family, candidates in specs.items():
        options = []
        for i in range(len(candidates)):
            key = family+"_"+str(i)
            for method in METHODS:
                value = np.mean([r[key]["methods"][method]["metrics"]["brier_game"] for r in records])
                options.append(dict(family=family,index=i,calibration=method,mean_brier=float(value)))
        choices[family] = min(options,key=lambda x:(x["mean_brier"],x["index"],METHODS.index(x["calibration"])))
        ranking.extend(options)
    return choices, sorted(ranking,key=lambda x:x["mean_brier"])


def blend_weights(predictions, ys, gids, members):
    # Each development block has equal total weight, as in family selection.
    A = np.concatenate([np.column_stack([p[m] for m in members]) for p in predictions])
    y = np.concatenate(ys)
    weights = []
    for g in gids:
        w = wpgam._game_balanced_weights(g); weights.append(w/w.sum()/len(gids))
    weights = np.concatenate(weights)
    def objective(w):
        d=A@w-y
        return float(np.sum(weights*d*d)),2*A.T@(weights*d)
    result = minimize(objective,np.full(len(members),1/len(members)),jac=True,method="SLSQP",
        bounds=[(0,1)]*len(members),constraints=[dict(type="eq",fun=lambda w:w.sum()-1,jac=lambda w:np.ones(len(w)))],
        options=dict(ftol=1e-12,maxiter=1000))
    if not result.success: raise RuntimeError("Blend optimization failed")
    w=np.maximum(result.x,0); w/=w.sum()
    return dict(zip(members,w.tolist()))


def summary(predictions, base, extra, mask):
    y,gid,t = base["y"][mask],base["gid"][mask],base["t_min"][mask]
    ref = predictions[REFERENCE]
    date_clusters = dict(zip(base["game_gid"].astype(int),base["game_dates"]))
    state = base["state"][mask]
    sn = {name:i for i,name in enumerate(wpgam.STATE_FEATURES)}
    groups = dict(early=t<15,mid=(t>=15)&(t<25),late=t>=25,
        elder_active=state[:,sn["elder_active"]]!=0,
        elder_taken=state[:,sn["d_elder"]]!=0,
        inhib_open=state[:,sn["d_inhib"]]!=0,
        new_patch=extra["extra"][mask,-1]<=3/30.)
    out = {}
    for name,p in predictions.items():
        slices = {k:dict(metrics=metrics(p[m],y[m],gid[m]),
            paired=paired(p[m],ref[m],y[m],gid[m],extra["series"],draws=1000))
            for k,m in groups.items() if m.any()}
        out[name] = dict(metrics=metrics(p,y,gid),paired=paired(p,ref,y,gid,extra["series"]),
            date_paired=paired(p,ref,y,gid,date_clusters),
            calibration=wpbench._calibration_diagnostics(p,y,gid),slices=slices)
    return out


def main(dataset, output, resource_cache):
    import lightgbm, scipy, sklearn
    started=time.time()
    output.mkdir(parents=True,exist_ok=True)
    registry_path=Path(wpgam.OUT_DIR)/"evaluation_registry.json"
    protected=[registry_path,Path(wpgam.MODEL_PATH)]
    original_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    registry=json.loads(registry_path.read_text())
    base=wpbench._load_base(str(dataset))
    if (base["split_method"]!="date" or np.any(base["game_dates"]=="") or
            max(base["game_dates"])>registry["consumed_through"]):
        raise ValueError("Fresh outcomes are reserved for prospective comparison")
    specs=specifications()
    blocks,final=wa.split_blocks(base["game_dates"],base["outer_train"],folds=3)
    sources=[Path(m.__file__) for m in (wa,wpbench,wpgam,wpaudit,alt)]
    sources += [Path(__file__),Path(__file__).with_name("wpx_adapt.py")]
    plan=dict(kind="repaired_alternative_methods_v1",specifications=specs,calibration_methods=METHODS,
        calibration_days=28,selection="mean of three development game-balanced Brier scores",
        calibration="both coefficients fitted only on preceding calibration dates; no intercept recentering",
        tree_stopping="fixed rounds selected on development; calibration labels never select iterations",
        gam_stack="expanding date folds with past-only supplied priors; neutral initial block",
        source_hashes={str(p.relative_to(Path.cwd())):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        resources_sha256=hashlib.sha256((resource_cache/"resources.npz").read_bytes()).hexdigest(),
        protected_hashes=original_hashes,consumed_through=registry["consumed_through"],
        development=[{k:v for k,v in b.items() if isinstance(v,str)} for b in blocks],
        final={k:v for k,v in final.items() if isinstance(v,str)},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                      sklearn=sklearn.__version__,lightgbm=lightgbm.__version__),
        gold_audit=dict(states=5000,amounts=[100,1000],slots=list(range(10)),seed=904),
        diagnostics="already consumed outcomes; intervals are exploratory and not multiplicity-adjusted")
    # Canonicalize tuples before comparison with a resumed JSON plan.
    plan=json.loads(json.dumps(plan))
    if (output/"plan.json").exists() and json.loads((output/"plan.json").read_text())!=plan:
        raise ValueError("Plan changed; use a new output directory")
    dump(output/"plan.json",plan)
    for p in sources:
        dest=output/"source_snapshot"/p.relative_to(Path.cwd())
        dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(p.read_bytes())
    extra=load_extra(base,dataset,resource_cache)
    records,development_predictions,ys,gids=[],[],[],[]
    for i,block in enumerate(blocks):
        r,p,_,parts=run_stage(base,extra,block,output/f"development_{i+1}",specs)
        records.append(r); development_predictions.append(p)
        ys.append(parts["validation"]["y"]); gids.append(parts["validation"]["gid"])
        del parts; gc.collect()
    choices,ranking=select(records,specs)
    selected_predictions=[{f:p[f+"_"+str(c["index"])+"_"+c["calibration"]] for f,c in choices.items()}
                          for p in development_predictions]
    blends={"convex_ensemble":blend_weights(selected_predictions,ys,gids,list(specs)),
            "monotone_ensemble":blend_weights(selected_predictions,ys,gids,list(MONOTONE))}
    selection=dict(choices=choices,ranking=ranking,blend_weights=blends,
                   selected=min(choices.values(),key=lambda c:c["mean_brier"]))
    dump(output/"selection.json",selection)
    log.info("Settings frozen. Best single model on development: %s",selection["selected"])
    del development_predictions,selected_predictions
    r,p,models,parts=run_stage(base,extra,final,output/"frozen",specs,choices,audit=True)
    audits={record["family"]:record["gold_audit"] for record in r.values()}
    for label,weights in blends.items():
        p[label]=sum(weights[f]*p[f] for f in weights)
        def blended(part):
            return sum(w*wpbench._apply_platt(predict(models[f][0],part),models[f][1])
                       for f,w in weights.items() if w>1e-12)
        audits[label]=gold_audit(blended,parts["validation"])
    mask=parts["validation"]["mask"]
    frozen=summary(p,base,extra,mask)
    dump(output/"frozen.json",frozen); dump(output/"gold_audits.json",audits)
    np.savez_compressed(output/"frozen_predictions.npz",gid=base["gid"][mask],y=base["y"][mask],**p)
    del models,parts,p; gc.collect()
    replay={f:np.full(len(base["y"]),np.nan) for f in [*specs,*blends]}
    months=[]; dates=base["game_dates"]
    for month in sorted(set(d[:7] for d in dates[~base["outer_train"]])):
        start=month+"-01"; end=str(np.datetime64(month,"M")+1)+"-01"
        cutoff=str(np.datetime64(start)-np.timedelta64(28,"D"))
        block=dict(fit=dates<cutoff,calibration=(dates>=cutoff)&(dates<start),
            validation=(dates>=start)&(dates<end)&~base["outer_train"],
            fit_end=cutoff,validation_start=start,validation_end=end)
        mm=block["validation"][base["row_game"]]
        if start==final["validation_start"]:
            # Same fit and calibration dates: reuse the exact May predictions,
            # retaining the frozen stage's complete results and audits.
            with np.load(output/"frozen_predictions.npz") as saved:
                pp={f:saved[f][mm[mask]] for f in replay}
        else:
            rr,pp,models,parts=run_stage(base,extra,block,output/("replay_"+month),specs,choices)
            for label,weights in blends.items(): pp[label]=sum(weights[f]*pp[f] for f in weights)
            del models,parts
        for label,values in pp.items(): replay[label][mm]=values
        months.append(dict(month=month,metrics={f:metrics(v,base["y"][mm],base["gid"][mm]) for f,v in pp.items()}))
        dump(output/"months.json",months)
        del pp; gc.collect()
    replay={f:p[mask] for f,p in replay.items()}
    rolling=summary(replay,base,extra,mask)
    dump(output/"rolling.json",rolling)
    np.savez_compressed(output/"rolling_predictions.npz",gid=base["gid"][mask],y=base["y"][mask],**replay)
    unchanged={str(p):hashlib.sha256(p.read_bytes()).hexdigest()==original_hashes[str(p)] for p in protected}
    if not all(unchanged.values()): raise RuntimeError("Protected artifact changed during evaluation")
    dump(output/"completion.json",dict(seconds=round(time.time()-started,2),protected_unchanged=unchanged,
        fits=sum(len(json.loads(p.read_text())) for p in output.glob("*/results.json")),
        families=len(specs),production_changed=False))
    log.info("Completed %d families: %s",len(specs),{f:round(r["metrics"]["brier_game"],6) for f,r in rolling.items()})


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(name)s %(message)s")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path(wpgam.OUT_DIR)/"states.npz")
    parser.add_argument("--output",type=Path,default=Path(wpgam.OUT_DIR)/"methods_v9")
    parser.add_argument("--resource-cache",type=Path,required=True)
    args=parser.parse_args(); main(args.dataset,args.output,args.resource_cache)
