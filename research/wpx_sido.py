"""Offline SIDO-inspired performance-prior screen; never deploys a model.

Uses monthly, past-only Gaussian ridge player effects, not a reproduction of
SIDO's Bayesian posterior or its ally/enemy attribution. Tests whether these
features add to the existing pregame model on chronological game blocks.
Only already-consumed dates in the evaluation registry may be inspected.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import psycopg
from psycopg.rows import dict_row
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.feature_extraction import DictVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import config, wpbench, wpgam

log = logging.getLogger("sido")
PHASES = ((0, 7), (7, 15), (15, 25))
HALF_LIFE = 150.0
PLAYER_PENALTY = 60.0
CHAMP_PENALTY = 30.0


def load_panel(end_date):
    """Read a consistent snapshot, without invoking database schema helpers."""
    with psycopg.connect(config.PG_DSN, row_factory=dict_row,
                         options="-c default_transaction_read_only=on") as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        games = conn.execute("""
            SELECT game_id, date, match_id, duration_s, patch, trname,
                   elo_blue_pre, elo_red_pre
            FROM golgg_games WHERE date <= %s AND duration_s > 0
            ORDER BY date, game_id
        """, (end_date,)).fetchall()
        rows = conn.execute("""
            SELECT p.game_id, p.player_id, p.side, p.role, p.slot, p.champion,
                   p.gold, p.stats->>'Total damage to Champion' AS damage,
                   p.stats->>'Total damage taken' AS taken,
                   t.g0, t.g7, t.g15, t.g25
            FROM golgg_players p JOIN golgg_games g USING (game_id)
            LEFT JOIN (
                SELECT game_id, slot,
                    max(gold) FILTER (WHERE minute=0) AS g0,
                    max(gold) FILTER (WHERE minute=7) AS g7,
                    max(gold) FILTER (WHERE minute=15) AS g15,
                    max(gold) FILTER (WHERE minute=25) AS g25
                FROM golgg_timeline WHERE minute IN (0,7,15,25)
                GROUP BY game_id, slot
            ) t ON p.game_id=t.game_id AND p.slot=t.slot
            WHERE g.date <= %s ORDER BY g.date, p.game_id, p.slot
        """, (end_date,)).fetchall()
    return games, rows


def player_key(row):
    return f"player:{row['player_id']}:{row['role']}"


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def build_panel(games, rows, phase=None, contextual=True):
    """Historical responses and covariates; none are current-game predictors."""
    by_game = defaultdict(list)
    for row in rows:
        by_game[row['game_id']].append(row)
    features, targets, dates = [], [], []
    for game in games:
        roster = by_game[game['game_id']]
        # Complete roles ensure the team-state control is measured consistently.
        if len(roster) != 10 or len({(r['side'], r['role']) for r in roster}) != 10:
            continue
        if any(r['player_id'] is None or not r['champion'] for r in roster):
            continue
        if phase is not None:
            start, end = phase
            if game['duration_s'] < end * 60:
                continue
            if any(r[f'g{start}'] is None or r[f'g{end}'] is None or
                   r[f'g{end}'] < r[f'g{start}'] for r in roster):
                continue
            gold_start = {s: sum(r[f'g{start}'] for r in roster if r['side'] == s)
                          for s in ('blue', 'red')}
        for row in roster:
            other = 'red' if row['side'] == 'blue' else 'blue'
            own_elo = game[f"elo_{row['side']}_pre"] or 1500.0
            opp_elo = game[f'elo_{other}_pre'] or 1500.0
            f = {f"role:{row['role']}": 1.0, player_key(row): 1.0,
                 'elo_diff': (own_elo - opp_elo) / 400.0,
                 'blue': float(row['side'] == 'blue')}
            if contextual:
                opp = next(r for r in roster if r['side'] == other and r['role'] == row['role'])
                f[f"champ:{row['champion']}:{row['role']}"] = 1.0
                f[f"opponent_champ:{opp['champion']}:{row['role']}"] = 1.0
            if phase is not None:
                target = (row[f'g{end}'] - row[f'g{start}']) / (end-start) / 100.0
                f['start_own_gold'] = row[f'g{start}'] / 1000.0
                f['start_team_lead'] = (gold_start[row['side']] - gold_start[other]) / 1000.0
            else:
                damage, taken, gold = map(number, (row['damage'], row['taken'], row['gold']))
                if damage is None or taken is None or gold is None or taken <= 0:
                    continue
                minutes = game['duration_s'] / 60.0
                target = damage / minutes / 100.0
                f['taken_per_min'] = taken / minutes / 100.0
                f['gold_per_min'] = gold / minutes / 100.0
                f['duration'] = minutes / 30.0
            features.append(f)
            targets.append(target)
            dates.append(game['date'].toordinal())
    return features, np.asarray(targets), np.asarray(dates)


def monthly_ratings(features, targets, dates, games, rows, min_rows=2000):
    """Refit at month start; all games that month use the same older data.

    Vocabulary is restricted to each fit's training rows. Rookies receive zero.
    Same-day matches and later maps cannot leak through the player histories.
    """
    by_game = defaultdict(list)
    for r in rows:
        by_game[r['game_id']].append(r)
    by_month = defaultdict(list)
    for g in games:
        by_month[g['date'].replace(day=1)].append(g)
    out, reliability, stops = {}, {}, Counter()
    for month, group in sorted(by_month.items()):
        before = np.flatnonzero(dates < month.toordinal())
        beta_by_name, count = {}, Counter()
        if len(before) >= min_rows:
            vectorizer = DictVectorizer(sparse=True, dtype=np.float64)
            A = vectorizer.fit_transform([features[i] for i in before])
            names = vectorizer.get_feature_names_out()
            penalties = np.array([
                PLAYER_PENALTY if n.startswith('player:') else
                CHAMP_PENALTY if n.startswith(('champ:', 'opponent_champ:')) else
                0.1 if n.startswith('role:') else 10.0 for n in names])
            age = month.toordinal() - dates[before]
            sw = np.sqrt(np.exp2(-age / HALF_LIFE))
            augmented = sparse.vstack([A.multiply(sw[:, None]), sparse.diags(np.sqrt(penalties))], format='csr')
            rhs = np.r_[targets[before] * sw, np.zeros(A.shape[1])]
            result = lsqr(augmented, rhs, atol=1e-6, btol=1e-6, iter_lim=1000)
            stops[int(result[1])] += 1
            if result[1] not in (1, 2, 4, 5) or not np.isfinite(result[0]).all():
                raise RuntimeError(f'nonconverged monthly ridge: {month}, stop={result[1]}')
            beta_by_name = dict(zip(names, result[0]))
            for i in before:
                count.update(k for k in features[i] if k.startswith('player:'))
        for game in group:
            roster = by_game[game['game_id']]
            complete = len(roster) == 10 and all(r['player_id'] is not None for r in roster)
            out[game['game_id']] = sum(
                (1 if r['side'] == 'blue' else -1) * beta_by_name.get(player_key(r), 0.0)
                for r in roster) if complete else 0.0
            reliability[game['game_id']] = sum(count[player_key(r)] >= 10 for r in roster) if complete else 0
    return out, reliability, dict(stops)


def paired_interval(candidate, baseline, y, clusters, draws=2000):
    """Resample whole series, preserving the game-weighted estimand."""
    delta = (candidate-y)**2 - (baseline-y)**2
    _, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=delta)
    counts = np.bincount(inv)
    rng = np.random.default_rng(240304873)
    samples = rng.integers(len(sums), size=(draws, len(sums)))
    estimates = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return {'delta_brier': float(delta.mean()),
            'series_bootstrap_ci95': np.quantile(estimates, [0.025, 0.975]).tolist(),
            'series': len(sums)}


def run(dataset, output):
    registry_path = Path(wpgam.OUT_DIR) / 'evaluation_registry.json'
    registry = json.loads(registry_path.read_text())
    with np.load(dataset, allow_pickle=True) as d:
        first, outer, method = wpgam._date_split(d)
        dates = d['date'][first].astype(str)
        if method != 'date' or max(dates) > registry['consumed_through']:
            raise ValueError('Screen requires dated, already-consumed data; preserve fresh evaluation games.')
        gid, y, C = d['gid'][first], d['y'][first], d['C'][first]
        pre = wpgam.pregame_values_from_matrix(d['X'][first], list(d['names']))
        champs = list(d['champ_names'])
    inner, validation, val_cutoff = wpbench._date_blocks(dates, outer)
    games, rows = load_panel(max(dates))
    game_lookup = {g['game_id']: g for g in games}
    feature_values, reliability, coverage, solver_stops = {}, {}, {}, {}
    specs = [(f'gold_{a}_{b}', (a,b), True) for a,b in PHASES]
    specs += [('damage_fullmatch', None, True), ('gold_0_7_no_champ', (0,7), False)]
    for name, phase, contextual in specs:
        f, target, panel_dates = build_panel(games, rows, phase, contextual)
        ratings, known, stops = monthly_ratings(f, target, panel_dates, games, rows)
        feature_values[name] = np.array([ratings.get(int(g), 0.0) for g in gid])
        reliability[name] = np.array([known.get(int(g), 0) for g in gid])
        coverage[name] = {'player_observations': len(target),
                          'test_games_all_players_10_prior': int(np.sum(reliability[name][~outer] == 10))}
        solver_stops[name] = stops
        log.info('%s: %d observations; ratings ready', name, len(target))
    candidates = {
        'baseline': [],
        'early_gold_no_champ': ['gold_0_7_no_champ'],
        'early_gold_context': ['gold_0_7'],
        'phase_gold_context': ['gold_0_7', 'gold_7_15', 'gold_15_25'],
        'damage_context': ['damage_fullmatch'],
        'phase_gold_and_damage': ['gold_0_7', 'gold_7_15', 'gold_15_25', 'damage_fullmatch'],
    }
    validation_metrics, predictions = {}, {}
    def fit_predict(names, train, test):
        raw = np.column_stack([pre] + [feature_values[n] for n in names])
        model = wpgam.fit_pregame(raw[train], C[train], y[train], champs)
        p = wpgam._sigmoid(wpgam.predict_pregame(model, raw[test], C[test])[0])
        return p, model['team_beta'].tolist()
    for label, names in candidates.items():
        p, beta = fit_predict(names, inner, validation)
        validation_metrics[label] = {**wpbench._basic_metrics(p, y[validation], gid[validation]), 'beta': beta}
    # Freeze the best challenger using validation only, even if the baseline
    # wins. Its outer score is a diagnostic, never an override of selection.
    selected = min((k for k in candidates if k != 'baseline'),
                   key=lambda k: validation_metrics[k]['brier_game'])
    labels = list(dict.fromkeys(['baseline', selected]))
    test_metrics = {}
    for label in labels:
        p, beta = fit_predict(candidates[label], outer, ~outer)
        predictions[label] = p
        test_metrics[label] = {**wpbench._basic_metrics(p, y[~outer], gid[~outer]), 'beta': beta}
    # Scope clusters by date too: some imported single-game match IDs are reused.
    clusters = np.array([f"{game_lookup[int(g)]['date']}:{game_lookup[int(g)]['match_id'] or int(g)}" for g in gid[~outer]])
    paired = paired_interval(predictions[selected], predictions['baseline'], y[~outer], clusters)
    periods = {}
    for month in sorted(set(dates[~outer].astype('U7'))):
        mask = dates[~outer].astype('U7') == month
        periods[month] = {'games': int(mask.sum()), **paired_interval(
            predictions[selected][mask], predictions['baseline'][mask], y[~outer][mask], clusters[mask])}
    result = {
        'kind': 'sido_inspired_pregame_screen_v1', 'production_changed': False,
        'source': 'https://arxiv.org/html/2403.04873v1',
        'dataset_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        'registry_consumed_through': registry['consumed_through'],
        'scope': 'Post-draft pregame only; retrospective diagnostic, not a live-GAM or promotion test.',
        'method': {'rating_refit': 'month start, strictly earlier dates',
                   'half_life_days': HALF_LIFE, 'player_ridge': PLAYER_PENALTY,
                   'champion_ridge': CHAMP_PENALTY, 'same_month_updates': False,
                   'posterior_uncertainty': False, 'ally_enemy_effects': False,
                   'damage': 'Historical full-match DPM adjusted for taken/min, gold/min, duration, role, champions, Elo; not phase damage.'},
        'split': {'train_games': int(inner.sum()), 'validation_games': int(validation.sum()),
                  'test_games': int((~outer).sum()), 'validation_start': val_cutoff,
                  'test_start': min(dates[~outer]), 'test_end': max(dates[~outer])},
        'coverage': coverage, 'solver_stops': solver_stops,
        'validation': validation_metrics, 'best_challenger': selected,
        'selected': min(validation_metrics, key=lambda k: validation_metrics[k]['brier_game']),
        'selected_improves_validation': validation_metrics[selected]['brier_game'] < validation_metrics['baseline']['brier_game'],
        'test': test_metrics, 'paired': paired, 'test_by_month': periods,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(output.with_suffix('.npz'), gid=gid, date=dates,
                        feature_names=np.array(list(feature_values)),
                        features=np.column_stack(list(feature_values.values())),
                        reliability=np.column_stack(list(reliability.values())),
                        test_gid=gid[~outer], test_y=y[~outer],
                        baseline_p=predictions['baseline'], candidate_p=predictions[selected])
    print(json.dumps({k: result[k] for k in ('split','selected','best_challenger','test','paired','test_by_month')}, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(wpgam.OUT_DIR) / 'states.npz')
    parser.add_argument('--output', type=Path, default=Path(wpgam.OUT_DIR) / 'sido_screen.json')
    args = parser.parse_args()
    run(args.dataset, args.output)
