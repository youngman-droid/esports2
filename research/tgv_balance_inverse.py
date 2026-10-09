"""Fit the published squared-log damage-balance form to saved observations.

The unknown balance center is absorbed into the positive magic inputs. This
recovers a score parameterization, not raw physical channel units. Assuming
affine tankiness normalization allows its coefficients to be absorbed into Q.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tgv_minimax_graph import MinimaxGraph


def basis(n1, n2, positive=False):
    z = np.zeros((n1*n2, 1+n1+n2 if positive else n1+n2-1))
    z[:, 0] = 1
    for i in range(n1):
        for j in range(n2):
            if positive:
                z[i*n2+j, 1+i] = 1
                z[i*n2+j, 1+n1+j] = 1
            else:
                if i:
                    z[i*n2+j, i] = 1
                if j:
                    z[i*n2+j, n1+j-1] = 1
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--starts', type=int, default=16)
    parser.add_argument('--seed', type=int, default=6811)
    parser.add_argument('--holdout', default='')
    parser.add_argument('--refine-from')
    args = parser.parse_args()
    root = Path('data/tgv/20260916')
    data = np.load(root/'earlier-design.npz')
    known, keys = data['known'], data['keys']
    design = json.loads((root/'earlier-design-graph.json').read_text())
    groups, observations = design['groups'], design['observations']
    bi = np.array([groups[0].index(k[0])*len(groups[1])+groups[1].index(k[1]) for k in keys])
    ri = np.array([groups[2].index(k[2])*len(groups[3])+groups[3].index(k[3]) for k in keys])
    xb = basis(len(groups[0]), len(groups[1]))[bi]
    xr = basis(len(groups[2]), len(groups[3]))[ri]
    pb = basis(len(groups[0]), len(groups[1]), True)[bi]
    pr = basis(len(groups[2]), len(groups[3]), True)[ri]
    sizes = [xb.shape[1], xr.shape[1], xb.shape[1], xr.shape[1],
             pb.shape[1], pb.shape[1], pr.shape[1], pr.shape[1]]
    offsets = np.r_[0, np.cumsum(sizes)]
    slices = [slice(a, b) for a, b in zip(offsets[:-1], offsets[1:])]
    parameters = int(offsets[-1])
    graph = MinimaxGraph(len(keys), [(n['maximize'], n['children']) for n in design['nodes']],
                         [o['node'] for o in observations])
    challenge = json.loads((root/'challenge.json').read_text())['challenge']
    final = challenge['puzzle']['finalEvaluation']
    features = {f['key']: f for f in final['compositionFeatures']}
    reference = next(i for i, k in enumerate(keys) if list(k) == [1, 498, 799, 8])
    anchor_targets = np.array([features['adjustedDamageBalance']['values']['blue'],
                               features['adjustedDamageBalance']['values']['red'],
                               sum(f['effect'] for f in final['compositionFeatures'] if f['transform']=='linear')])
    targets = np.r_[[o['target'] for o in observations], anchor_targets]
    held = {tuple(map(int, s.split(':'))) for s in args.holdout.split(',') if s}
    train = np.r_[[(o['index'], o['championId']) not in held for o in observations], [True]*3]
    # The final breakdown reveals the canonical terminal value independently.
    assert (19, 8) not in held, 'Canonical breakdown would reveal this held-out score'

    def terminal(theta):
        ab, ar, qb, qr = [np.einsum('ni,i->n', z, theta[s])
                          for z, s in zip([xb, xr, xb, xr], slices[:4])]
        positive = [np.exp(theta[s]) for s in slices[4:]]
        mb, phb, mr, phr = [np.einsum('ni,i->n', z, v)
                            for z, v in zip([pb, pb, pr, pr], positive)]
        lb, lr = np.log(mb/phb), np.log(mr/phr)
        blue_effect, red_effect = qr*lb**2, qb*lr**2
        jblue = np.zeros((len(keys), parameters))
        jred = np.zeros_like(jblue)
        jbase = np.zeros_like(jblue)
        jbase[:, slices[0]], jbase[:, slices[1]] = xb, -xr
        jblue[:, slices[3]] = xr*lb[:, None]**2
        jred[:, slices[2]] = xb*lr[:, None]**2
        for which, sign, z, pos, denominator, factor in [
            (4, 1, pb, positive[0], mb, 2*qr*lb),
            (5, -1, pb, positive[1], phb, 2*qr*lb),
            (6, 1, pr, positive[2], mr, 2*qb*lr),
            (7, -1, pr, positive[3], phr, 2*qb*lr)]:
            dest = jblue if which < 6 else jred
            dest[:, slices[which]] = sign*z*pos[None, :]*(factor/denominator)[:, None]
        comp = ab-ar+blue_effect-red_effect
        anchors = np.array([blue_effect[reference], red_effect[reference], (ab-ar)[reference]])
        anchor_jac = np.array([jblue[reference], jred[reference], jbase[reference]])
        return comp, jbase+jblue-jred, anchors, anchor_jac

    def evaluate(theta):
        comp, jac, anchors, anchor_jac = terminal(theta)
        pred, active = graph.evaluate(known+comp)
        return np.r_[pred, anchors], np.vstack([jac[active], anchor_jac]), active

    last = [None, None]
    def cached(theta):
        if last[0] is None or not np.array_equal(theta, last[0]):
            last[:] = [theta.copy(), evaluate(theta)]
        return last[1]

    rng = np.random.default_rng(args.seed)
    check = rng.normal(0, .15, parameters)
    _, analytic, _ = evaluate(check)
    h, jac_error = 1e-6, 0.
    for col in [int(o) for o in offsets[:-1]]+[parameters-1]:
        plus, minus = check.copy(), check.copy()
        plus[col] += h
        minus[col] -= h
        numeric = (evaluate(plus)[0]-evaluate(minus)[0])/(2*h)
        jac_error = max(jac_error, float(np.max(np.abs(numeric-analytic[:, col]))))
    assert jac_error < 1e-6
    lower = np.r_[np.full(offsets[4], -2.), np.full(parameters-offsets[4], -6.)]
    upper = -lower
    initial_saved = None
    if args.refine_from:
        saved = json.loads(Path(args.refine_from).read_text())
        assert saved['parameterSizes'] == sizes
        assert held <= {(o['index'], o['championId']) for o in saved['observations'] if o.get('heldout')}
        initial_saved = np.array(saved['parameters'])
    best, runs = None, []
    start = time.time()
    for i in range(args.starts):
        theta = rng.normal(0, .08, parameters)
        theta[slices[0]][0] = anchor_targets[2]
        for side_slice in slices[2:4]:
            theta[side_slice] *= .2
            theta[side_slice][0] = -.1
        for side, effect in [(0, anchor_targets[0]), (1, anchor_targets[1])]:
            theta[slices[4+side*2]] += rng.choice([-1, 1])*np.sqrt(abs(effect)/.1)
        if initial_saved is not None:
            theta = initial_saved.copy()
            if i:
                theta[(i-1)//2] += .15 if i % 2 else -.15
                theta = np.clip(theta, lower+1e-8, upper-1e-8)
        result = least_squares(lambda t: (cached(t)[0]-targets)[train], theta,
                               jac=lambda t: cached(t)[1][train], bounds=(lower, upper),
                               max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        pred, _, active = evaluate(result.x)
        error = float(np.max(np.abs((pred-targets)[train])))
        runs.append(dict(start=i, trainingMaxError=error, nfev=result.nfev))
        if best is None or error < best[0]:
            best = error, result.x.copy(), pred.copy(), active.copy()
        print('Start', i, 'error', error, flush=True)
    error, theta, pred, active = best
    composition = terminal(theta)[0]
    for i, o in enumerate(observations):
        o.update(prediction=float(pred[i]), error=float(pred[i]-targets[i]),
                 heldout=not bool(train[i]), activeTerminal=keys[active[i]].tolist())
        o.pop('node')
    out = dict(designPrefix='earlier', formula='C=A_B-A_R+Q_R*log(M_B/P_B)^2-Q_B*log(M_R/P_R)^2',
               parameterSizes=sizes, parameters=theta.tolist(), parameterBlocks=['A_blue','A_red','Q_blue','Q_red','log_M_blue','log_P_blue','log_M_red','log_P_red'],
               seed=args.seed, refineFrom=args.refine_from, jacobianMaxError=jac_error,
               trainMaxError=error, heldoutMaxError=float(np.max(np.abs((pred-targets)[~train]))) if np.any(~train) else None,
               observations=observations, anchors=[dict(name=n, target=float(t), prediction=float(v)) for n,t,v in zip(['blueAdjustedBalance','redAdjustedBalance','linearComposition'],anchor_targets,pred[-3:])],
               terminalValues=[dict(picks=k.tolist(),known=float(known[i]),composition=float(composition[i])) for i,k in enumerate(keys)],
               runs=runs,seconds=time.time()-start,
               caveat='Source-motivated shape, local optimization, and fixed-roster domain only. Center absorbed into magic inputs; affine enemy-tankiness normalization assumed and absorbed into Q. Physical channel units and globally shared coefficients remain unidentified. Three breakdown anchors use double precision while move targets have engine rounding.')
    suffix = ('-holdout'+args.holdout.replace(':','_').replace(',','-') if held else '')+('-refined' if args.refine_from else '')
    output = root/f'balance-inverse{suffix}.json'
    output.write_text(json.dumps(out, indent=2))
    print('Saved', output, 'max error', error)


if __name__ == '__main__':
    main()
