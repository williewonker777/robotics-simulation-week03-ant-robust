#!/usr/bin/env python3
"""Build offline slide data from published v28 results; no third-party imports."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'artifacts/combo_v28/results'
OUTPUT = ROOT / 'report/web/assets/data.js'
CATEGORIES = [
    ('flat_friction', '평지·저마찰', 4), ('boxes', '박스', 6),
    ('rough_wave', '요철·물결', 4), ('slope_stairs', '경사·계단', 5),
    ('obstacles', '장애물', 6), ('gaps_holes', '틈·구덩이', 3),
]
REFERENCES = ('ref_lim_f3a', 'flat_e0_d1', 'flat_e0_d0')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def recipe_rows(path):
    rows = read_csv(path)
    for row in rows:
        for key in row:
            if key not in ('recipe', 'label', 'seeds', 'worst_condition'):
                row[key] = float(row[key])
        row['category_means'] = {key: row['cat_' + key] for key, _, _ in CATEGORIES}
    return rows


def build_data(results=RESULTS):
    results = Path(results)
    summary = json.loads((results / 'summary.json').read_text(encoding='utf-8'))
    selection = recipe_rows(results / 'selection_recipes.csv')
    confirmation = recipe_rows(results / 'confirmation_recipes.csv')
    # Unrounded summary values avoid compounding CSV display rounding in effects.
    precise = {row['recipe']: row for row in summary['selection']['ranking']}
    by_recipe = {row['recipe']: row for row in selection}
    terrains = sorted(summary['selection']['main_effects']['terrain'])
    # Do not round the already rounded recipe CSV a second time: 4.25 is
    # displayed there, while the underlying three-seed boxes mean is 4.2473.
    baseline_returns = {'flat': {}, 'boxes_10': {}}
    baseline_ids = {f'flat_e0_d0_s{seed}' for seed in (42, 43, 44)}
    with gzip.open(results / 'selection_evaluations.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if (row['checkpoint_id'] in baseline_ids and row['terrain'] in baseline_returns
                    and row['friction'] == 1.0 and row['combine_mode'] == 'average'):
                values = baseline_returns[row['terrain']]
                if row['checkpoint_id'] in values:
                    raise ValueError('Duplicate baseline evaluation')
                values[row['checkpoint_id']] = row['return_mean']
    if any(set(values) != baseline_ids for values in baseline_returns.values()):
        raise ValueError('Missing baseline evaluation seeds')
    baseline_problem = {terrain: mean(values[key] for key in sorted(values))
                        for terrain, values in baseline_returns.items()}
    checkpoints = {row['checkpoint']: float(row['demo'])
                   for row in read_csv(results / 'selection_checkpoints.csv')}

    def seed_comparison(recipe, reference=None):
        seeds = [int(seed) for seed in by_recipe[recipe]['seeds'].split()]
        values = [{'seed': seed, 'mean': checkpoints[f'{recipe}_s{seed}']} for seed in seeds]
        if reference is not None:
            reference_seeds = [int(seed) for seed in by_recipe[reference]['seeds'].split()]
            if seeds != reference_seeds:
                raise ValueError(f'Unmatched seeds: {recipe} vs {reference}')
        return {'recipe': recipe, 'mean': by_recipe[recipe]['demo_mean'],
                'delta': (by_recipe[recipe]['demo_mean'] - by_recipe[reference]['demo_mean']
                          if reference else None),
                'wins': (sum(value['mean'] > checkpoints[f"{reference}_s{value['seed']}"]
                             for value in values) if reference else None),
                'pairs': len(values), 'seed_values': values}

    ladder_names = ('flat_e0_d0', 'sticklim_e0_d0', 'sticklim_e5_d0', 'sticklim_e5_d0+stock')
    ladder = [seed_comparison(recipe, ladder_names[index - 1] if index else None)
              for index, recipe in enumerate(ladder_names)]
    alternatives = []
    for recipe, reference in (('sticklim_e5_d1', 'sticklim_e5_d0'),
                              ('sticklim_e5_d0+recovery', 'sticklim_e5_d0+stock'),
                              ('mine_e5_d0', 'sticklim_e5_d0')):
        comparison = seed_comparison(recipe, reference)
        del comparison['seed_values']
        comparison.update(reference=reference, fall_rate=by_recipe[recipe]['fall_rate'],
                          reference_fall_rate=by_recipe[reference]['fall_rate'])
        alternatives.append(comparison)

    def paired(pairs):
        deltas = [precise[b]['demo_mean'] - precise[a]['demo_mean'] for a, b in pairs]
        return {'mean': mean(deltas), 'wins': sum(value > 0 for value in deltas),
                'pairs': len(deltas), 'deltas': deltas,
                'category_means': {key: mean(precise[b]['category_means'][key] -
                                             precise[a]['category_means'][key]
                                             for a, b in pairs) for key, _, _ in CATEGORIES}}

    entropy = paired([(f'{t}_e0_d{d}', f'{t}_e5_d{d}') for t in terrains for d in (0, 1)])
    randomization = paired([(f'{t}_e{e}_d0', f'{t}_e{e}_d1') for t in terrains for e in (0, 5)])
    terrain = paired([(f'flat_e{e}_d{d}', f'{t}_e{e}_d{d}')
                      for t in ('stick', 'lim', 'sticklim') for e in (0, 5) for d in (0, 1)])
    continuation = []
    recovery = []
    for name in sorted(precise):
        if name.endswith('+stock'):
            parent = name.split('+')[0]
            continuation.append({'recipe': name, 'parent': parent,
                                 'delta': precise[name]['demo_mean'] - precise[parent]['demo_mean']})
            recovered = parent + '+recovery'
            if recovered in precise:
                recovery.append({'recipe': recovered, 'stock': name,
                                 'delta': precise[recovered]['demo_mean'] - precise[name]['demo_mean'],
                                 'stock_fall_rate': precise[name]['fall_rate'],
                                 'recovery_fall_rate': precise[recovered]['fall_rate']})
    fresh = json.loads((results / 'fresh_process_check.json').read_text(encoding='utf-8'))
    parity = json.loads((results / 'official_parity.json').read_text(encoding='utf-8'))
    official = parity['post_fix_2026_10_03']['official']['submission_seed43_run1']
    runs = read_csv(results / 'training_runs.csv')
    stage1 = [row for row in runs if row['run'].startswith('v28s1_')]
    ranked_new = sorted((row for row in selection if not row['recipe'].startswith('ref_')
                         and row['recipe'] not in REFERENCES), key=lambda row: -row['demo_mean'])[:8]
    curve_sources = []
    curve_points = []
    for stage, suffix, offset in ((1, '', 0), (2, '+stock', 1000)):
        curves = []
        for seed in (42, 43, 44):
            relative = f'artifacts/combo_v28/runs/v28s{stage}_sticklim_e5_d0{suffix}_s{seed}/curves.csv'
            curve_sources.append(relative)
            curves.append({int(row['iteration']): float(row['mean_reward'])
                           for row in read_csv(results.parent.parent.parent / relative)})
        iterations = sorted(set.intersection(*(set(curve) for curve in curves)))
        for iteration in iterations:
            if iteration % 25 == 0 or iteration == iterations[-1]:
                values = [curve[iteration] for curve in curves]
                curve_points.append({'iteration': offset + iteration, 'mean': mean(values),
                                     'min': min(values), 'max': max(values)})
    return {
        'selection': selection, 'ranking': ranked_new + [by_recipe[key] for key in REFERENCES],
        'confirmation': confirmation, 'ladder': ladder, 'alternatives': alternatives,
        'baseline_problem': baseline_problem,
        'fresh': [row for row in fresh['recipes'] if row['kind'] == 'fresh_demo'],
        'official': {'mean': official[0], 'std': official[1]},
        'effects': {'entropy': entropy, 'randomization': randomization, 'terrain': terrain,
                    'continuation': continuation, 'recovery': recovery},
        'categories': [{'key': key, 'label': label, 'count': count} for key, label, count in CATEGORIES],
        'protocol': {'selection_seed': summary['selection_terrain_seed'],
                     'confirmation_seed': summary['confirmation_terrain_seed'],
                     'selection_evaluations': summary['selection']['results'],
                     'selection_checkpoints': summary['selection']['checkpoints_complete'],
                     'confirmation_evaluations': summary['confirmation']['results'],
                     'confirmation_checkpoints': summary['confirmation']['checkpoints_complete'],
                     'conditions': len(read_csv(results / 'selection_conditions.csv')[0]) - 2,
                     'new_policies': len(runs), 'new_stage1_policies': len(stage1),
                     'continuation_policies': len(runs) - len(stage1),
                     'stage1_iterations': sorted({int(row['iterations']) for row in stage1}),
                     'stage1_transitions': sorted({int(row['transitions']) for row in stage1}),
                     'all_training_sha_match': all(row['direct_read_sha_match'] == 'True' for row in runs)},
        'learning_curves': {'metric': 'mean_reward', 'seeds': [42, 43, 44], 'points': curve_points},
        'sources': ['artifacts/combo_v28/results/' + name for name in
                    ('selection_recipes.csv', 'confirmation_recipes.csv', 'summary.json',
                     'fresh_process_check.json', 'official_parity.json', 'training_runs.csv',
                     'selection_conditions.csv', 'selection_checkpoints.csv',
                     'selection_evaluations.jsonl.gz')] + curve_sources,
    }


def serialize_data(data):
    return '// Generated by scripts/build_web_slides_data.py; do not edit.\nwindow.SLIDES_DATA = ' + json.dumps(
        data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + ';\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialize_data(build_data()), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
