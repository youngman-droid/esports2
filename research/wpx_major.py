"""Build a season-aware major-league cohort and run the complete repaired suite.

Only competition membership determines inclusion, never eventual qualification
or match outcome. Existing sequential Elo/form inputs remain historical inputs;
all fitted pregame, champion and state learners use the filtered cohort.
"""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpbench, wpgam
from research.wpx_adapt import load_extra
from research.wpx_methods_v9 import dump


def competition_group(name):
    """Direct Worlds-slot leagues in the dataset's 2024–2026 seasons."""
    name = str(name)
    years = re.findall(r'\b(20\d\d)\b', name)
    if len(years) != 1 or int(years[0]) not in (2024, 2025, 2026):
        raise ValueError(f'Unsupported competition season: {name}')
    year = int(years[0])
    if any(s in name.lower() for s in ('academy', 'promotion', 'qualifier', 'challenger')):
        return None
    if name.startswith('Worlds ') or name.startswith('MSI ') or 'Mid-Season Invitational' in name or 'First Stand' in name:
        return 'International'
    league = name.split()[0]
    if name.startswith('LCK CL '):
        return None
    eligible = {'LCK', 'LPL', 'LEC'}
    eligible |= {2024: {'LCS', 'CBLOL', 'PCS', 'VCS', 'LLA'},
                 2025: {'LTA', 'LCP'},
                 2026: {'LCS', 'CBLOL', 'LCP'}}[year]
    return league if league in eligible else None


def filter_arrays(arrays):
    groups = {str(s): competition_group(s) for s in np.unique(arrays['league'])}
    mask = np.array([groups[str(s)] is not None for s in arrays['league']])
    row_keys = {'X', 'y', 'gid', 't', 'seq', 'date', 'patch', 'league', 'pm', 'ks', 'C'}
    unknown = set(arrays) - row_keys - {'names', 'champ_names'}
    if unknown:
        raise ValueError(f'Unknown dataset fields: {unknown}')
    for key in row_keys:
        if len(arrays[key]) != len(mask):
            raise ValueError(f'Misaligned row field: {key}')
    # A game must never be split between included/excluded competitions.
    if set(arrays['gid'][mask]) & set(arrays['gid'][~mask]):
        raise ValueError('Inconsistent competition labels within a game')
    return {k: v[mask] if k in row_keys else v for k, v in arrays.items()}, mask, groups


def build(dataset, source_cache, output):
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(source_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        resources_sha256=hashlib.sha256((source_cache/'resources.npz').read_bytes()).hexdigest(),
        filter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest_path = output/'cohort.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['inputs'] != fingerprint:
            raise ValueError('Cohort inputs changed; use a new output directory')
        for rel, expected in saved['outputs'].items():
            if hashlib.sha256((output/rel).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Cohort artifact changed: {rel}')
        return
    with np.load(dataset, allow_pickle=True) as d:
        arrays, mask, groups = filter_arrays({k:d[k] for k in d.files})
        fixed_keep = mask[d['seq'] < 0]
        source_fixed_count = int((d['seq'] < 0).sum())
        source_games = len(np.unique(d['gid']))
    target = output/'states.npz'
    np.savez_compressed(target, **arrays)
    cache = output/'resources'; cache.mkdir(exist_ok=True)
    with np.load(source_cache/'resources.npz', allow_pickle=False) as r:
        if len(r['gold']) != source_fixed_count:
            raise ValueError('Source resource cache row count mismatch')
        keep = np.isin(r['series_gid'], arrays['gid'])
        np.savez_compressed(cache/'resources.npz', gold=r['gold'][fixed_keep],
            series_gid=r['series_gid'][keep], series_key=r['series_key'][keep])
    base = wpbench._load_base(str(target))
    load_extra(base, target, cache)  # verifies resource gold/role alignment
    first = wpgam._game_rows(arrays['gid'])
    manifest = dict(inputs=fingerprint, source_games=source_games,
        games=len(first), rows=len(mask[mask]), fixed_states=int(fixed_keep.sum()),
        groups=dict(Counter(groups[str(s)] for s in arrays['league'][first])),
        competitions=dict(Counter(map(str, arrays['league'][first]))),
        excluded_competitions=sorted(k for k,v in groups.items() if v is None),
        missing_current_leagues=sorted({'LCK','LPL','LEC','LCS','CBLOL','LCP'} - set(groups[str(s)] for s in arrays['league'][first])),
        evaluation_start=min(base['game_dates'][~base['outer_train']]),
        policy='Direct Worlds-slot leagues by season; Worlds/MSI/First Stand included. LJL/LCO indirect PCS pathways and other cups excluded.',
        prior_policy='Historical sequential Elo/form inputs retained; statistical prior, champion, calibration and state fits restricted to included games.',
        outputs={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (target,cache/'resources.npz')})
    dump(manifest_path, manifest)


def main(args):
    build(args.dataset, args.source_cache, args.output)
    if args.prepare_only:
        return
    common = ['--dataset',str(args.output/'states.npz')]
    for runner, folder in [('wpx_methods_v9.py','methods'), ('wpx_recency.py','recency'), ('wpx_market_v9.py','markets')]:
        done=args.output/(folder+'_complete.json')
        if done.exists():
            continue
        cmd=[sys.executable,str(Path(__file__).with_name(runner)),*common,'--output',str(args.output/folder)]
        if folder == 'markets':
            cmd += ['--methods',str(args.output/'methods'),'--resources',str(args.output/'resources/resources.npz')]
        else:
            cmd += ['--resource-cache',str(args.output/'resources')]
        subprocess.run(cmd,check=True)
        dump(done,dict(command=cmd,completed=True))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(message)s')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path(wpgam.OUT_DIR)/'states.npz')
    parser.add_argument('--source-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(wpgam.OUT_DIR)/'major_v9')
    parser.add_argument('--prepare-only',action='store_true')
    main(parser.parse_args())
