"""Audit the fixed v13 paired study; posture metrics are visited-state diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean

from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from week03_ant.posture_study import CONTROLLERS, HOLDOUTS, ROOT, SCHEMA, SOURCE_FILES, START_SHA, TASK, V5_SHA, evaluation_inputs, legacy_hashes
from week03_ant.posture_telemetry import GROUPS, SUM_FIELDS

_SPEC = importlib.util.spec_from_file_location('v13_frozen_history_summary', ROOT / 'scripts/summarize_history_hybrid.py')
HISTORY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(HISTORY)
BASE = HISTORY.BASE
FIELDS = tuple(key + '_sum' for key in (*SUM_FIELDS, 'foot_clearance', 'foot_deficit'))
COUNTS = ('steps', 'low_speed_steps', 'foot_samples')
FAMILIES = ('flat', 'rough', 'slope', 'stairs', 'waves', 'obstacles', 'stepping_stones')


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('posture sums must be finite numbers')
    return value


def _bounded(value, low, high, tolerance=1.e-5):
    if not low - tolerance <= value <= high + tolerance:
        raise ValueError('posture evidence outside physical/count bounds')


def _posture(data, rows):
    telemetry = data.get('posture_telemetry')
    if (not isinstance(telemetry, dict)
            or telemetry.get('pre_action_first_episode_only') is not True
            or telemetry.get('active_steps') != data['episode_lengths']
            or any(type(v) is not int for v in telemetry['active_steps'])):
        raise ValueError('posture telemetry must cover only active first-episode steps')
    for key in ('motion_steps', 'low_speed_steps'):
        if not isinstance(telemetry.get(key), list) or len(telemetry[key]) != len(rows):
            raise ValueError('missing first-episode motion occupancy')
    groups = telemetry.get('groups')
    if not isinstance(groups, dict) or set(groups) != set(GROUPS):
        raise ValueError('missing posture partitions')
    for group, values in groups.items():
        if not isinstance(values, dict) or set(values) != set((*FIELDS, *COUNTS)):
            raise ValueError('unexpected posture sum/count schema')
        if any(not isinstance(v, list) or len(v) != len(rows) for v in values.values()):
            raise ValueError('malformed posture arrays')
    for i, row in enumerate(rows):
        row['dt'] = data['condition']['dt']
        row['motion_steps'] = HISTORY._count(telemetry['motion_steps'][i], row['steps'])
        row['low_speed_steps'] = HISTORY._count(telemetry['low_speed_steps'][i], row['motion_steps'])
        row['posture'] = {}
        for group, values in groups.items():
            state = {key: vector[i] for key, vector in values.items()}
            steps = HISTORY._count(state['steps'], row['steps'])
            HISTORY._count(state['low_speed_steps'], steps)
            samples = HISTORY._count(state['foot_samples'], 4 * steps)
            for key in FIELDS:
                _finite(state[key])
                if not steps and state[key] != 0:
                    raise ValueError('empty posture group has nonzero sums')
            tol = 1.e-5 * max(1, steps)
            ranges = {'body_clearance': (-.49, 1.49), 'target_height': (.44, .58),
                      'absolute_body_error': (0., 1.14), 'body_cost': (0., 1.),
                      'foot_cost': (0., 1.), 'flat_speed_bonus': (0., 1.),
                      'reward': (-1.25, .5), 'local_coverage': (.9, 1.),
                      'front_coverage': (.9, 1.)}
            if group == 'clear':
                ranges['target_height'] = (.44, .454)
            elif group == 'rough':
                ranges['target_height'] = (.566, .58)
                ranges['flat_speed_bonus'] = (0., .1)
            elif group == 'intermediate':
                ranges['target_height'] = (.454, .566)
            for key, (low, high) in ranges.items():
                _bounded(state[key + '_sum'], low * steps, high * steps, tol)
            _bounded(state['foot_deficit_sum'], 0, samples, tol)
            _bounded(state['foot_cost_sum'], 0, state['foot_deficit_sum'] / 4, tol)
            if not samples and state['foot_clearance_sum'] != 0:
                raise ValueError('missing feet have nonzero clearance sums')
            if abs(state['body_clearance_sum'] - state['target_height_sum']) > state['absolute_body_error_sum'] + tol:
                raise ValueError('body error contradicts clearance/target sums')
            expected = -state['body_cost_sum'] - .25 * state['foot_cost_sum'] + .5 * state['flat_speed_bonus_sum']
            if not math.isclose(state['reward_sum'], expected, abs_tol=tol, rel_tol=1.e-6):
                raise ValueError('posture reward identity mismatch')
            row['posture'][group] = state
        valid = row['posture']['all_valid']
        if valid['steps'] > row['motion_steps'] or valid['low_speed_steps'] > row['low_speed_steps']:
            raise ValueError('valid posture occupancy exceeds known motion occupancy')
        if row['low_speed_steps'] - valid['low_speed_steps'] > row['motion_steps'] - valid['steps']:
            raise ValueError('unknown-scan low speed occupancy exceeds available samples')
        for key in (*FIELDS, *COUNTS):
            total = sum(row['posture'][group][key] for group in GROUPS if group != 'all_valid')
            actual = row['posture']['all_valid'][key]
            equal = actual == total if key in COUNTS else math.isclose(actual, total, abs_tol=1.e-7, rel_tol=1.e-9)
            if not equal:
                raise ValueError('clear/rough/intermediate do not partition valid samples')


def audit(data):
    controller = data.get('controller')
    if data.get('schema') != SCHEMA or controller not in CONTROLLERS or data.get('task') != TASK:
        raise ValueError('unexpected v13 task/schema/controller')
    mode = 'hybrid' if controller.startswith('history_') else ('v5' if controller == 'v5' else 'v10')
    if (data.get('mode') != mode or data.get('gate_config') != asdict(HistoryGateConfig())
            or data.get('posture_config') != asdict(PostureConfig())):
        raise ValueError('changed policy mode, frozen selector or posture configuration')
    if data.get('teacher_tensor_identity_verified') is not True or data.get('v5_sha256') != V5_SHA:
        raise ValueError('frozen teacher identity mismatch')
    if data.get('phase') not in ('smoke', 'holdout') or data.get('scenario') not in ('mixed', 'stones'):
        raise ValueError('invalid scoring phase/scenario')
    condition = data.get('condition', {})
    expected_seconds = 16 if data['scenario'] == 'mixed' else 64
    if (condition.get('seconds') != expected_seconds
            or condition.get('snapshot_seconds') != (8 if expected_seconds == 16 else 16)
            or data.get('difficulties') != [.2, .4, .6, .8, 1.]):
        raise ValueError('changed horizon/difficulty contract')
    HISTORY._hashes(data.get('initial_state_sha256'), HISTORY.INITIAL_KEYS)
    HISTORY._hashes({'checkpoint': data.get('checkpoint_sha256')})
    if data['phase'] == 'holdout':
        HISTORY._hashes({key: data.get(key) for key in (
            'experiment_freeze_sha256', 'evaluation_inputs_sha256', 'terrain_cache_manifest_sha256')})
        if (data.get('geometry_seed'), data.get('reset_seed')) not in HOLDOUTS:
            raise ValueError('changed holdout geometry/reset')
        if data.get('num_envs') != (175 if data['scenario'] == 'mixed' else 10):
            raise ValueError('changed holdout shape')
    rows = BASE.audit(dict(data, schema=HISTORY.BASE_SCHEMA))
    HISTORY._audit_routing(data)
    events = data.get('history_switch_events')
    if not isinstance(events, list) or len(events) != len(rows):
        raise ValueError('missing temporal switch evidence')
    for i, entries in enumerate(events):
        if not isinstance(entries, list) or len(entries) != data['episode_switch_count'][i]:
            raise ValueError('switch evidence count mismatch')
        previous = None
        for j, event in enumerate(entries):
            HISTORY._event(event, 'history', condition['dt'], previous)
            if (event['step'] != data['episode_switch_steps'][i][j]
                    or event['target_v10'] != data['episode_switch_to_v10'][i][j]):
                raise ValueError('switch evidence disagrees with first episode')
            previous = event['step']
    _posture(data, rows)
    return rows


def aggregate(rows):
    result = HISTORY.aggregate(rows)
    flat = [row for row in rows if row['family'] == 'flat']
    result['flat_mean_episode_speed'] = mean(row['distance'] / (row['steps'] * row['dt']) for row in flat) if flat else None
    result['flat_lane'] = sum(row['lane'] for row in flat)
    result['posture_active_steps'] = sum(row['steps'] for row in rows)
    result['motion_steps'] = sum(row['motion_steps'] for row in rows)
    result['low_speed_steps'] = sum(row['low_speed_steps'] for row in rows)
    result['unknown_motion_steps'] = result['posture_active_steps'] - result['motion_steps']
    result['motion_coverage'] = result['motion_steps'] / result['posture_active_steps'] if result['posture_active_steps'] else None
    result['low_speed_fraction_known_motion'] = result['low_speed_steps'] / result['motion_steps'] if result['motion_steps'] else None
    result['posture'] = {}
    for group in GROUPS:
        sums = {key: sum(row['posture'][group][key] for row in rows) for key in (*FIELDS, *COUNTS)}
        steps, samples = sums['steps'], sums['foot_samples']
        means = {key: sums[key + '_sum'] / steps if steps else None for key in SUM_FIELDS}
        means.update({key: sums[key + '_sum'] / samples if samples else None for key in ('foot_clearance', 'foot_deficit')})
        result['posture'][group] = dict(sums=sums, means=means,
            low_speed_fraction_valid=sums['low_speed_steps'] / steps if steps else None,
            foot_sample_coverage=samples / (4 * steps) if steps else None,
            fraction_active_steps=steps / result['posture_active_steps'] if result['posture_active_steps'] else None)
    return result


def improvement(candidate, control, *, hybrid=False, primary=True):
    def rate(group, key, denominator='n'):
        if group[denominator] <= 0:
            raise ValueError('empty comparison denominator')
        return Fraction(group[key], group[denominator])
    checks = {f'{key}_not_lower': rate(candidate, key) >= rate(control, key) for key in ('one', 'six')}
    checks.update({f'{key}_not_higher': rate(candidate, key) <= rate(control, key) for key in ('falls', 'lane')})
    checks['world_zero_including_flat'] = candidate['world'] + candidate['flat_world'] == 0
    strict_gain = (any(rate(candidate, key) > rate(control, key) for key in ('one', 'six'))
                   or any(rate(candidate, key) < rate(control, key) for key in ('falls', 'lane')))
    if primary:
        checks['flat_falls_not_higher'] = rate(candidate, 'flat_falls', 'flat_n') <= rate(control, 'flat_falls', 'flat_n')
        old, new = control['flat_mean_episode_speed'], candidate['flat_mean_episode_speed']
        if old is None or new is None:
            raise ValueError('missing flat speed evidence')
        checks['flat_speed_not_lower' if hybrid else 'flat_speed_strictly_higher'] = new >= old if hybrid else new > old
        if hybrid:
            checks['fixed_v5_flat_branch'] = candidate['flat_mean_v10_duty'] == control['flat_mean_v10_duty'] == 0
            checks['flat_lane_not_higher'] = rate(candidate, 'flat_lane', 'flat_n') <= rate(control, 'flat_lane', 'flat_n')
    if hybrid or not primary:
        checks['strict_terrain_gain'] = strict_gain
    result = {'passed': all(checks.values()), 'checks': checks}
    if hybrid and primary:
        result['flat_expected_identity_observed'] = all(candidate[key] == control[key] for key in (
            'flat_mean_episode_speed', 'flat_falls', 'flat_lane', 'flat_world', 'flat_mean_v10_duty'))
    return result


def _inside(name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('provenance paths must be repository-relative')
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('provenance path escapes repository')
    return path


def summarize(directory):
    directory = Path(directory)
    frozen_path = directory / 'frozen.json'
    frozen = json.loads(frozen_path.read_text())
    if frozen['holdouts'] != [list(pair) for pair in HOLDOUTS] or frozen['controllers'] != list(CONTROLLERS):
        raise ValueError('changed predeclared holdout/controller inventory')
    if frozen['gate_config'] != asdict(HistoryGateConfig()) or frozen['posture_config'] != asdict(PostureConfig()):
        raise ValueError('changed frozen configuration')
    HISTORY._hashes(frozen['source_sha256'], SOURCE_FILES)
    HISTORY._hashes(frozen['legacy']['sources'], count=72)
    HISTORY._hashes(frozen['legacy']['models'], count=4)
    if frozen['legacy'] != legacy_hashes():
        raise ValueError('legacy source/model inventory differs from canonical freeze')
    for name, digest in {**frozen['source_sha256'], **frozen['legacy']['sources'], **frozen['legacy']['models']}.items():
        if HISTORY.sha(_inside(name)) != digest:
            raise ValueError(f'frozen provenance bytes changed: {name}')
    HISTORY._hashes({'training_freeze': frozen['training_freeze_sha256']})
    if HISTORY.sha(directory / 'training_frozen.json') != frozen['training_freeze_sha256']:
        raise ValueError('training freeze digest mismatch')
    models = frozen['models']
    if set(models) != {'original', 'control', 'adaptive'} or models['original']['sha256'] != START_SHA:
        raise ValueError('invalid frozen model inventory')
    trained_path = directory / 'trained_models.json'
    HISTORY._hashes({'trained_models': frozen['trained_models_sha256']})
    if HISTORY.sha(trained_path) != frozen['trained_models_sha256']:
        raise ValueError('trained model manifest digest mismatch')
    trained = json.loads(trained_path.read_text())
    if (trained.get('training_freeze_sha256') != frozen['training_freeze_sha256']
            or type(trained.get('total_training_transitions')) is not int
            or trained['total_training_transitions'] != 65536000
            or trained.get('models') != {arm: models[arm] for arm in ('control', 'adaptive')}):
        raise ValueError('trained model manifest differs from declared paired training')
    for name, model in models.items():
        expected_keys = {'checkpoint', 'sha256'}
        if name != 'original':
            expected_keys |= {'source_checkpoint', 'iteration', 'transitions', 'initial_audit_sha256'}
        if set(model) != expected_keys:
            raise ValueError('invalid checkpoint declaration')
        HISTORY._hashes({'checkpoint': model['sha256']})
        if HISTORY.sha(_inside(model['checkpoint'])) != model['sha256']:
            raise ValueError('frozen checkpoint bytes changed')
        if name != 'original':
            if (type(model['iteration']) is not int or model['iteration'] != 249
                    or type(model['transitions']) is not int or model['transitions'] != 32768000):
                raise ValueError('trained model must be final249 with declared transition budget')
            _inside(model['source_checkpoint'])
            HISTORY._hashes({'initial_audit': model['initial_audit_sha256']})
            audit_path = directory / 'training' / f'{name}_initial.json'
            if HISTORY.sha(audit_path) != model['initial_audit_sha256']:
                raise ValueError('training initial audit digest mismatch')
            initial_audit = json.loads(audit_path.read_text())
            if initial_audit.get('arm') != name:
                raise ValueError('training initial audit arm mismatch')
    frozen_sha = HISTORY.sha(frozen_path)
    inputs = evaluation_inputs(directory)
    input_sha = HISTORY.sha(directory / 'evaluation_inputs.json')
    cache_sha = inputs['terrain_cache_manifest_sha256']
    HISTORY._hashes({'evaluation_inputs': input_sha, 'terrain_cache_manifest': cache_sha})
    result = {'files': [], 'provenance': {
        'experiment_freeze_sha256': frozen_sha,
        'training_freeze_sha256': frozen['training_freeze_sha256'],
        'trained_models_sha256': frozen['trained_models_sha256'],
        'evaluation_inputs_sha256': input_sha,
        'terrain_cache_manifest_sha256': cache_sha,
    }}
    for scenario, folder, envs, seconds in (('mixed', 'evaluations', 175, 16), ('stones', 'horizon', 10, 64)):
        groups, initial = defaultdict(list), {}
        expected_paths = {f'{c}__geometry{g}_reset{r}.json' for c in CONTROLLERS for g, r in HOLDOUTS}
        if {path.name for path in (directory / folder).glob('*.json')} != expected_paths:
            raise ValueError('missing or unexpected evaluation file')
        for controller in CONTROLLERS:
            model = models['original' if controller == 'v5' else controller.removeprefix('history_')]
            for geometry, reset in HOLDOUTS:
                path = directory / folder / f'{controller}__geometry{geometry}_reset{reset}.json'
                data = json.loads(path.read_text())
                expected = (controller, geometry, reset, scenario, envs, seconds, 'holdout')
                actual = (data.get('controller'), data.get('geometry_seed'), data.get('reset_seed'), data.get('scenario'),
                          data.get('num_envs'), data.get('condition', {}).get('seconds'), data.get('phase'))
                if actual != expected or data.get('checkpoint') != model['checkpoint'] or data.get('checkpoint_sha256') != model['sha256']:
                    raise ValueError('evaluation/model contract differs from freeze')
                if data.get('experiment_freeze_sha256') != frozen_sha:
                    raise ValueError('evaluation source/model freeze linkage mismatch')
                if (data.get('evaluation_inputs_sha256') != input_sha
                        or data.get('terrain_cache_manifest_sha256') != cache_sha):
                    raise ValueError('evaluation input/cache manifest linkage mismatch')
                if HISTORY._timestamp(data['started_utc']) <= HISTORY._timestamp(frozen['frozen_at']):
                    raise ValueError('holdout evaluation precedes freeze')
                rows = audit(data)
                expected_counts = ({(f, level): 5 for f in FAMILIES for level in range(5)}
                                   if scenario == 'mixed' else {('stepping_stones', 4): 10})
                if Counter((row['family'], row['level']) for row in rows) != expected_counts:
                    raise ValueError('family/difficulty inventory mismatch')
                HISTORY.check_initial(initial, (geometry, reset), data)
                groups[controller].extend(rows)
                result['files'].append({'path': str(path.relative_to(directory)), 'sha256': HISTORY.sha(path), 'episodes': len(rows)})
        section = {
            'groups': {c: aggregate(rows) for c, rows in groups.items()},
            'family': {c: {f: aggregate([r for r in rows if r['family'] == f]) for f in sorted({r['family'] for r in rows})}
                       for c, rows in groups.items()},
            'family_level': {c: {f'{f}/level{level}': aggregate([r for r in rows if (r['family'], r['level']) == (f, level)])
                                 for f, level in sorted({(r['family'], r['level']) for r in rows})} for c, rows in groups.items()},
        }
        section['actor_adaptive_vs_control'] = improvement(section['groups']['adaptive'], section['groups']['control'], primary=scenario == 'mixed')
        section['hybrid_adaptive_vs_control'] = improvement(section['groups']['history_adaptive'], section['groups']['history_control'], hybrid=True, primary=scenario == 'mixed')
        section['posture_mechanism'] = {}
        for label, candidate, control in (('actor', 'adaptive', 'control'),
                                           ('hybrid', 'history_adaptive', 'history_control')):
            diagnostics = {}
            for name, group, field in (('rough_target_error', 'rough', 'absolute_body_error'),
                                        ('clear_body_clearance', 'clear', 'body_clearance')):
                new = section['groups'][candidate]['posture'][group]
                old = section['groups'][control]['posture'][group]
                a, b = new['means'][field], old['means'][field]
                diagnostics[name] = {'adaptive': a, 'control': b, 'delta': a - b if a is not None and b is not None else None,
                                     'adaptive_samples': new['sums']['steps'], 'control_samples': old['sums']['steps']}
            if scenario == 'mixed':
                a = section['family'][candidate]['flat']['posture']['all_valid']['means']['body_clearance']
                b = section['family'][control]['flat']['posture']['all_valid']['means']['body_clearance']
                diagnostics['flat_family_body_clearance'] = {'adaptive': a, 'control': b,
                    'delta': a - b if a is not None and b is not None else None}
            section['posture_mechanism'][label] = diagnostics
        result[folder] = section
    if len(result['files']) != 28 or sum(item['episodes'] for item in result['files']) != 2590:
        raise ValueError('incomplete declared first-episode inventory')
    result['interpretation'] = ('One starting actor/fine-tuning seed and two maps: exploratory, not universal or statistically significant. '
        'Adaptive is compared with equal-budget continuation; old v10 and v5 are baselines. '
        '16s primary and 64s secondary remain separate; secondary cannot override primary failure. '
        'Posture means are conditional on visited valid states, not matched-state causal effects. '
        'Ideal ray depth and a kinematic swing proxy do not establish real-camera or robot safety.')
    return result


def markdown(result):
    lines = ['# v13 adaptive posture: matched continuation comparison', '', result['interpretation'], '']
    for folder, title in (('evaluations', 'Primary: 16s mixed terrain'), ('horizon', 'Secondary: 64s hardest stones')):
        section = result[folder]
        lines += [f'## {title}', '', '| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for controller in CONTROLLERS:
            d = section['groups'][controller]
            speed = '—' if d['flat_mean_episode_speed'] is None else f"{d['flat_mean_episode_speed']:.4f}"
            lines.append(f"| {controller} | {d['one']}/{d['n']} | {d['six']}/{d['n']} | {d['falls']}/{d['n']} | {d['lane']}/{d['n']} | {d['world'] + d['flat_world']} | {d['flat_falls']}/{d['flat_n']} | {speed} |")
        for key in ('actor_adaptive_vs_control', 'hybrid_adaptive_vs_control'):
            judgment = section[key]
            lines += ['', f"{key}: **{'PASS' if judgment['passed'] else 'FAIL'}**"]
            lines += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in judgment['checks'].items()]
        lines += ['', 'Visited-state posture mechanism (adaptive minus control; not causal):']
        for label, diagnostics in section['posture_mechanism'].items():
            values = ', '.join(f"{key}: {value['delta']:.4f}m" if value['delta'] is not None else f"{key}: unavailable"
                               for key, value in diagnostics.items())
            lines.append(f'- {label}: {values}')
        lines.append('')
    lines += ['Flat speed averages distance/(first-episode steps × dt), retaining falls. Frozen-v5 hybrid flat behavior is not evidence of learned posture.',
              'See JSON for all baselines, family/level outcomes, valid-state body/foot metrics, low-speed occupancy and coverage.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output-prefix', type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix or args.directory / 'summary'
    paths = [prefix.with_suffix(suffix) for suffix in ('.json', '.md')]
    if any(path.exists() for path in paths):
        raise FileExistsError('refusing to overwrite summary evidence')
    result = summarize(args.directory)
    for path, content in zip(paths, (json.dumps(result, indent=2, allow_nan=False) + '\n', markdown(result))):
        with path.open('x') as stream:
            stream.write(content)
