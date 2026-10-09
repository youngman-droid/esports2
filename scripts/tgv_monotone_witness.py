"""Construct an observationally invisible terminal-score transformation.

The theorem applies to any exact compatible terminal table. A numerical witness
preserves a supplied table's predictions and residuals, whether or not that table
fits the observations exactly. It need not remain in the bilinear factor family.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tgv_minimax_graph import MinimaxGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fit')
    args = parser.parse_args()
    path = Path(args.fit)
    fit = json.loads(path.read_text())
    root = path.parent
    prefix = fit.get('designPrefix', 'ban')
    design = json.loads((root / f'{prefix}-design-graph.json').read_text())
    data = np.load(root / f'{prefix}-design.npz')
    lookup = {tuple(v['picks']): v for v in fit['terminalValues']}
    rows = [lookup[tuple(k)] for k in data['keys']]
    terminals = np.array([v['known'] + v['composition'] for v in rows])
    graph = MinimaxGraph(len(rows),
                         [(v['maximize'], v['children']) for v in design['nodes']],
                         [v['node'] for v in design['observations']])
    before, _ = graph.evaluate(terminals)
    targets = np.array([v['target'] for v in design['observations']])
    knots = np.unique(np.r_[before, targets])
    gaps = [(b - a, a, b) for a, b in zip(knots[:-1], knots[1:])
            if np.any((terminals > a) & (terminals < b))]
    width, a, b = max(gaps)
    amplitude = .2 * width

    def warp(z):
        z = np.asarray(z)
        inside = (z > a) & (z < b)
        result = z.copy()
        result[inside] += amplitude * np.sin(np.pi * (z[inside] - a) / width)
        return result

    changed = warp(terminals)
    after, _ = graph.evaluate(changed)
    assert np.array_equal(warp(knots), knots)
    assert np.array_equal(before, after)
    assert np.any(changed != terminals)
    observations = []
    for i, v in enumerate(design['observations']):
        observations.append(dict(index=v['index'], championId=v['championId'],
                                 target=float(targets[i]), prediction=float(after[i]),
                                 error=float(after[i] - targets[i])))
    result = dict(
        designPrefix=prefix, sourceFit=str(path),
        theorem='Increasing h commutes with every finite min and max. If h fixes all observed root scores, transforming every terminal by h preserves every observed root score.',
        formula='h(z)=z+amplitude*sin(pi*(z-a)/(b-a)) for a<z<b; h(z)=z otherwise',
        interval=[float(a), float(b)], amplitude=float(amplitude),
        minimumDerivative=1 - .2 * np.pi,
        observedTargetCount=len(set(targets)),
        changedTerminalCount=int(np.sum(changed != terminals)),
        maxTerminalChange=float(np.max(np.abs(changed - terminals))),
        maxPredictionChange=float(np.max(np.abs(after - before))),
        sourceMaxResidual=float(np.max(np.abs(before - targets))),
        transformedMaxResidual=float(np.max(np.abs(after - targets))),
        limitation='Unrestricted composition table witness on this fixed-roster subgame. The transformed table is not claimed to retain the source bilinear parameterization or TGV physical channel structure. It preserves the source residual; it does not turn an approximate fit into an exact one.',
        observations=observations,
        terminalValues=[dict(picks=v['picks'], known=v['known'],
                             composition=float(changed[i] - v['known']))
                        for i, v in enumerate(rows)])
    output = path.with_name(path.stem + '-monotone-witness.json')
    output.write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items()
           if k not in ['observations', 'terminalValues', 'theorem', 'limitation', 'formula']})


if __name__ == '__main__':
    main()
