"""Post-experiment independent v24 verification, using Python's standard library only.

This verifier is deliberately OUTSIDE the frozen experimental source inventory.
It does not import the experiment's scoring, study, schedule or telemetry helpers.
The output pins this verifier's own bytes and explicitly limits its claims.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct

ROOT = Path(__file__).resolve().parents[1]
RUNS = ('seed51', 'seed52', 'seed53')
LOW_RUNS = ('low51', 'low52', 'low53')
CONTROLLERS = ('parent', *RUNS, *LOW_RUNS, 'history_parent',
               *(f'history_{run}' for run in (*RUNS, *LOW_RUNS)))
DEV_CONTROLLERS = ('parent', 'history_parent', *LOW_RUNS, *(f'history_{run}' for run in LOW_RUNS))
MAPS = ((119, 79), (120, 80))
FAMILIES = ('rough', 'slope', 'stairs', 'waves', 'obstacles', 'stepping_stones', 'flat')
PAIRS = tuple((prefix + low, prefix + high) for prefix in ('', 'history_') for low, high in zip(LOW_RUNS, RUNS)) + tuple(
    (prefix + run, prefix + 'parent') for runs in (LOW_RUNS, RUNS) for prefix in ('', 'history_') for run in runs)
TRANSITIONS = 32768000
PARENT_SHA = '1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc'
FLAT_FIELDS = (
    'episode_return', 'forward_distance', 'maximum_distance_m', 'episode_lengths',
    'episode_active_steps', 'episode_terminated', 'episode_out_of_lane', 'episode_world_exit',
    'episode_strict_one_tile_success', 'episode_strict_all_tiles_success',
    'episode_full_horizon_survival', 'first_hit_one_seconds', 'first_hit_six_seconds',
    'distance_at_snapshot_m', 'episode_v10_target_steps', 'episode_v10_duty',
    'episode_alpha_sum', 'episode_switch_count', 'episode_switch_steps',
    'episode_switch_to_v10', 'history_switch_events', 'episode_uncertain_steps',
    'episode_fall_within_switch_window', 'episode_max_action_jump_rms',
)

# Independently pinned from the frozen physical replay contract; report metadata
# cannot narrow this list or turn exact replay into a vacuous comparison.
PARITY_FIELDS = (*FLAT_FIELDS, 'initial_state_sha256', 'initial_prefix_sha256',
                 'initial_rng_sha256', 'condition', 'family_names', 'difficulties',
                 'family_indices', 'level_indices', 'aggregates', 'contact_telemetry',
                 'posture_telemetry')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite JSON number')
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique_object)
    finite(value)
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'nonrelative provenance path')
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'provenance path escapes repository')
    return path


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None, 'invalid SHA256')
    return value


def hashes(root, entries):
    require(isinstance(entries, dict) and bool(entries), 'empty hash inventory')
    for name, expected in entries.items():
        require(sha(inside(root, name)) == digest(expected), 'hash mismatch: ' + name)


def linked(root, record):
    path = inside(root, record['path'])
    require(sha(path) == digest(record['sha256']), 'linked evidence changed: ' + record['path'])
    return read(path)


def close(actual, expected, message, *, tolerance=1.e-12, absolute=None):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance if absolute is None else absolute), message)


def initialization(data, training=False):
    state = data['initial_state_sha256']
    require(set(state) == {'root_state', 'joint_pos', 'joint_vel', 'observations'}, 'initial state keys differ')
    for value in state.values():
        digest(value)
    digest(data['initial_prefix_sha256'])
    rng = data['rng_sha256' if training else 'initial_rng_sha256']
    require(set(rng) == {'cpu', 'cuda'}, 'initial RNG keys differ')
    for value in rng.values():
        digest(value)
    return state, data['initial_prefix_sha256'], rng


def raw_rows(data, *, controller, geometry, reset, scenario, development=False):
    """Recompute first-episode counts; no saved aggregate is used as input."""
    finite(data)
    seconds, n = data['condition']['seconds'], 175
    require(scenario == 'mixed' and type(seconds) is int and seconds in (16,64), 'undeclared window/scenario')
    if development:
        require((geometry, reset, scenario) == (51, 24, 'mixed'), 'development matrix differs')
        n = 35
        require(data['phase'] == 'smoke' and controller in DEV_CONTROLLERS, 'development phase differs')
    require((data['schema'], data['task'], data['phase'], data['controller'], data['geometry_seed'],
             data['reset_seed'], data['scenario'], data['num_envs']) ==
            ('week03_ant_lr_continuation_v24_window_v1', 'Week03-Ant-Contact-v16-Eval-v0', data['phase'] if development else 'holdout',
             controller, geometry, reset, scenario, n), 'raw evaluation identity differs')
    require(((geometry, reset) in MAPS or development) and controller in CONTROLLERS, 'undeclared map/controller')
    c = data['condition']
    require(c == dict(seconds=seconds, max_steps=seconds * 60, dt=1 / 60,
                      snapshot_seconds=8 if seconds == 16 else 16, one_threshold=13.1,
                      six_threshold=53.1, footprint_margin=1.1, observations=91,
                      expert_observations=91, v5_observations=60, actions=8), 'physical condition differs')
    require(data['family_names'] == list(FAMILIES) and data['difficulties'] == [.2, .4, .6, .8, 1.],
            'reporting family/level axes differ')
    require(data['command_mode'] == 'conditioned' and type(data['command_schema_version']) is int
            and data['command_schema_version'] == 1 and data['mode'] == ('hybrid' if controller.startswith('history_') else 'v10')
            and data['teacher_tensor_identity_verified'] is True and data['default_config_parity'] is True,
            'policy/teacher contract differs')
    require(type(data['new_training_transitions']) is int and data['new_training_transitions'] == 0
            and type(data['policy_training_transitions']) is int
            and data['policy_training_transitions'] == (0 if controller.removeprefix('history_') == 'parent' else TRANSITIONS),
            'evaluation vs model training accounting differs')
    key = controller.removeprefix('history_')
    expected_seed = None if key == 'parent' else int(key[-2:])
    require(data.get('training_seed') == expected_seed and (expected_seed is None or type(data['training_seed']) is int),
            'raw training seed differs')
    expected_lr = None if key == 'parent' else 1.e-5 if key in LOW_RUNS else 1.e-4
    require(data.get('lr_condition') == ('not_applicable' if key == 'parent' else 'low' if key in LOW_RUNS else 'high')
            and data.get('model_learning_rate') == expected_lr, 'raw LR provenance differs')
    initialization(data)
    arrays = set(FLAT_FIELDS) | {'family_indices', 'level_indices'}
    require(all(isinstance(data.get(k), list) and len(data[k]) == n for k in arrays), 'raw array shape differs')
    # Torch comparisons use float32 scalar thresholds, not decimal/double cutoffs.
    one, six = (struct.unpack('f', struct.pack('f', x))[0] for x in (13.1, 53.1))
    rows = []
    for i in range(n):
        distance, length = data['forward_distance'][i], data['episode_lengths'][i]
        require(type(distance) in (int, float) and math.isfinite(distance), 'invalid first-episode distance')
        require(type(length) is int and 1 <= length <= seconds * 60
                and type(data['episode_active_steps'][i]) is int and data['episode_active_steps'][i] == length,
                'invalid first-episode clock')
        flags = [data[k][i] for k in ('episode_terminated', 'episode_out_of_lane', 'episode_world_exit',
                 'episode_strict_one_tile_success', 'episode_strict_all_tiles_success', 'episode_full_horizon_survival')]
        require(all(type(flag) is bool for flag in flags), 'first-episode flags are not booleans')
        fall, lane, world = flags[:3]
        success = [not any(flags[:3]) and distance >= threshold for threshold in (one, six)]
        require(flags[3:5] == success and flags[5] == (length == seconds * 60 and not fall),
                'strict success/survival differs from first-episode evidence')
        family, level = data['family_indices'][i], data['level_indices'][i]
        require(type(family) is int and 0 <= family < 7 and type(level) is int and 0 <= level < 5,
                'invalid family/level index')
        snapshot = data['distance_at_snapshot_m'][i]
        require((snapshot is None) == (length * c['dt'] <= c['snapshot_seconds'] + 1.e-8),
                'auxiliary snapshot censoring differs')
        maximum = data['maximum_distance_m'][i]
        require(type(maximum) in (int, float) and maximum + 1.e-5 >= distance, 'maximum distance contradicts final')
        require(snapshot is None or (type(snapshot) in (int,float) and math.isfinite(snapshot) and snapshot<=maximum+1.e-5),
                'invalid auxiliary snapshot distance')
        for key, threshold in (('first_hit_one_seconds', one), ('first_hit_six_seconds', six)):
            hit = data[key][i]
            require((hit is not None) == (maximum >= threshold), 'first hit contradicts maximum distance')
            if hit is not None:
                require(type(hit) in (int, float) and 0 < hit <= length / 60 + 1.e-7, 'first hit outside first episode')
        one_hit, six_hit = data['first_hit_one_seconds'][i], data['first_hit_six_seconds'][i]
        require(six_hit is None or (one_hit is not None and one_hit <= six_hit),
                'first one-tile hit follows six-tile hit')
        rows.append(dict(family=FAMILIES[family], level=level, one=int(success[0]), six=int(success[1]),
                         falls=int(fall), lane=int(lane), world=int(world), survival=int(flags[5]),
                         speed=distance / (length * c['dt']), distance=distance, steps=length,
                         first_six_hit=data['first_hit_six_seconds'][i]))
    expected = {(family, level): 1 if development else 5 for family in FAMILIES for level in range(5)}
    require(Counter((r['family'], r['level']) for r in rows) == expected, 'raw family/level count differs')
    return rows


def counts(rows):
    rough = [r for r in rows if r['family'] != 'flat']
    flat = [r for r in rows if r['family'] == 'flat']
    values = dict(n=len(rough), flat_n=len(flat), completed_first_episodes=len(rows))
    values.update({k: sum(r[k] for r in rough) for k in ('one', 'six', 'falls', 'lane', 'world', 'survival')})
    values.update({f'flat_{k}': sum(r[k] for r in flat) for k in ('falls', 'lane', 'world')})
    values['flat_mean_episode_speed'] = statistics.mean(r['speed'] for r in flat) if flat else None
    return values


def judgment(candidate, reference, primary):
    a, b = counts(candidate), counts(reference)
    require(a['n'] == b['n'] and a['n'] > 0, 'unpaired rough gate denominator')
    checks = {}
    for metric in ('one', 'six'):
        checks[metric + '_not_lower'] = a[metric] >= b[metric]
    for metric in ('falls', 'lane'):
        checks[metric + '_not_higher'] = a[metric] <= b[metric]
    checks['world_zero_including_flat'] = a['world'] + a['flat_world'] == 0
    checks['strict_rough_improvement'] = (a['one'] > b['one'] or a['six'] > b['six']
                                           or a['falls'] < b['falls'] or a['lane'] < b['lane'])
    if primary:
        require(a['flat_n'] == b['flat_n'] and a['flat_n'] > 0, 'unpaired flat gate denominator')
        for metric in ('flat_falls', 'flat_lane'):
            checks[metric + '_not_higher'] = a[metric] <= b[metric]
        checks['flat_speed_not_lower'] = a['flat_mean_episode_speed'] >= b['flat_mean_episode_speed']
    require(len(candidate) == len(reference), 'paired row length differs')
    paired = Counter()
    for x, y in zip(candidate, reference):
        require((x['family'], x['level']) == (y['family'], y['level']), 'paired row ordering differs')
        if x['family'] != 'flat':
            paired[(x['six'], y['six'])] += 1
    return dict(passed=all(checks.values()), checks=checks,
                paired_six=dict(n=a['n'], gains=paired[1, 0], losses=paired[0, 1],
                                both_success=paired[1, 1], both_failure=paired[0, 0]))


def compare_group(rows, recorded):
    actual = counts(rows)
    for name, expected in actual.items():
        if isinstance(expected, float):
            close(recorded[name], expected, 'summary aggregate differs: ' + name)
        else:
            require(recorded[name] == expected and type(recorded[name]) is type(expected),
                    'summary aggregate differs: ' + name)


METRICS = ('one', 'six', 'falls', 'lane', 'world', 'flat_falls', 'flat_lane', 'flat_world', 'flat_mean_episode_speed')


def seed_diagnostics(aggregates):
    require(set(aggregates) == set(CONTROLLERS), 'seed range controller inventory differs')
    result = {}
    for prefix, label in (('', 'actor'), ('history_', 'history')):
        parent = aggregates[prefix + 'parent']
        for arm, runs in (('high', RUNS), ('low', LOW_RUNS)):
            require(all((aggregates[prefix + run]['n'], aggregates[prefix + run]['flat_n']) ==
                        (parent['n'], parent['flat_n']) for run in runs), 'parent duplicated or per-seed denominator differs')
            metrics = {}
            for key in METRICS:
                values = {run: aggregates[prefix + run][key] for run in runs}
                present = [v for v in values.values() if v is not None]
                require(len(present) in (0, 3), 'partially missing seed metric')
                metrics[key] = dict(values=values, parent=parent[key], minimum=min(present) if present else None,
                    maximum=max(present) if present else None, range=max(present)-min(present) if present else None,
                    parent_deltas={run: val-parent[key] if val is not None and parent[key] is not None else None
                                   for run, val in values.items()})
            result[label + '_' + arm] = dict(training_seeds=[51,52,53], rough_episodes_per_seed=parent['n'],
                                flat_episodes_per_seed=parent['flat_n'], parent_counted_once=True, metrics=metrics)
    return result


def matched_deltas(aggregates):
    result = {}
    for prefix, label in (('', 'actor'), ('history_', 'history')):
        result[label] = {}
        for seed in (51, 52, 53):
            low, high = aggregates[f'{prefix}low{seed}'], aggregates[f'{prefix}seed{seed}']
            result[label][str(seed)] = {key: low[key] - high[key] if low[key] is not None and high[key] is not None else None
                                       for key in METRICS}
    return result


def compare_diagnostics(section):
    require(section.get('seed_ranges') == seed_diagnostics(section['groups']), 'seed ranges/parent deltas differ')
    require(section.get('matched_low_minus_high') == matched_deltas(section['groups']), 'matched seed deltas differ')


def compare_section(groups, section, primary, *, breakdown=True):
    require(set(section['groups']) == set(CONTROLLERS), 'summary controller inventory differs')
    compare_diagnostics(section)
    for controller, rows in groups.items():
        compare_group(rows, section['groups'][controller])
    expected_gates = {f'{a}_vs_{b}': judgment(groups[a], groups[b], primary) for a, b in PAIRS}
    require(section['comparisons'] == expected_gates, 'summary gates/paired gains differ')
    if not breakdown:
        return
    families = sorted({r['family'] for rows in groups.values() for r in rows})
    levels = sorted({r['level'] for rows in groups.values() for r in rows})
    require(set(section['family']) == set(families) and set(section['level']) == {str(x) for x in levels},
            'summary family/level inventory differs')
    expected_family_levels = set()
    for family in families:
        selected = {name: [r for r in rows if r['family'] == family] for name, rows in groups.items()}
        family_section = section['family'][family]
        if family != 'flat':
            compare_section(selected, family_section, False, breakdown=False)
        else:
            compare_diagnostics(family_section)
            require(family_section['comparisons'] == {}, 'flat has no rough gate')
            for name, rows in selected.items():
                compare_group(rows, family_section['groups'][name])
        for level in levels:
            subset = {name: [r for r in rows if r['level'] == level] for name, rows in selected.items()}
            if not all(subset.values()):
                continue
            key = f'{family}/level{level}'
            expected_family_levels.add(key)
            target = section['family_level'][key]
            if family != 'flat':
                compare_section(subset, target, False, breakdown=False)
            else:
                compare_diagnostics(target)
                require(target['comparisons'] == {}, 'flat level has no rough gate')
                for name, rows in subset.items():
                    compare_group(rows, target['groups'][name])
    require(set(section['family_level']) == expected_family_levels, 'family/level cells differ')
    for level in levels:
        subset = {name: [r for r in rows if r['level'] == level] for name, rows in groups.items()}
        compare_section(subset, section['level'][str(level)], primary, breakdown=False)


def computed_section(groups, primary, breakdown=True):
    result = dict(groups={name: counts(rows) for name, rows in groups.items()},
                  comparisons={f'{a}_vs_{b}': judgment(groups[a], groups[b], primary) for a, b in PAIRS})
    result['seed_ranges'] = seed_diagnostics(result['groups'])
    result['matched_low_minus_high'] = matched_deltas(result['groups'])
    if not breakdown:
        return result
    result.update(family={}, family_level={}, level={})
    families = sorted({r['family'] for rows in groups.values() for r in rows})
    levels = sorted({r['level'] for rows in groups.values() for r in rows})
    for family in families:
        selected = {name: [r for r in rows if r['family'] == family] for name, rows in groups.items()}
        if family == 'flat':
            result['family'][family] = dict(groups={n: counts(v) for n, v in selected.items()}, comparisons={})
        else:
            result['family'][family] = computed_section(selected, False, False)
        for level in levels:
            subset = {name: [r for r in rows if r['level'] == level] for name, rows in selected.items()}
            if not all(subset.values()):
                continue
            result['family_level'][f'{family}/level{level}'] = (
                dict(groups={n: counts(v) for n, v in subset.items()}, comparisons={}) if family == 'flat'
                else computed_section(subset, False, False))
    for level in levels:
        subset = {name: [r for r in rows if r['level'] == level] for name, rows in groups.items()}
        result['level'][str(level)] = computed_section(subset, primary, False)
    for section in (*result['family'].values(), *result['family_level'].values()):
        section['seed_ranges'] = seed_diagnostics(section['groups'])
        section['matched_low_minus_high'] = matched_deltas(section['groups'])
    return result


def cache_hashes(cache, root):
    require(cache['tiles_per_geometry'] == 240 and len(cache['tiles']) == 240 * len(cache['geometries']),
            'terrain tile inventory differs')
    seen = set()
    for tile in cache['tiles']:
        key = tile['cache_key']
        require(re.fullmatch('[0-9a-f]{32}', key) is not None and key not in seen, 'invalid/duplicate cache tile')
        seen.add(key)
        require(set(tile['sha256']) == {'cfg.yaml', 'mesh.obj', 'origin.csv'}, 'cache file inventory differs')
        for filename, expected in tile['sha256'].items():
            require(sha(root / key / filename) == digest(expected), 'cached terrain bytes changed')


WINDOWS=('16','64')
PASSIVITY_HASH_KEYS=('root_state_sha256','joint_pos_sha256','joint_vel_sha256','episode_length_buf_sha256',
    'observations_sha256','cpu_rng_sha256','cuda_rng_sha256','policy_state_sha256','gate_state_sha256',
    'full_first_episode_active_sha256')
SOURCE_FILES=('src/week03_ant/lr_study_v24.py', 'src/week03_ant/tasks/lr_continuation_v24.py',
    'src/week03_ant/tasks/lr_continuation_v24_cfg.py', 'scripts/lr_continuation_v24.py',
    'scripts/run_lr_training_v24.py', 'tests/test_lr_training_v24.py', 'tests/test_lr_harness_v24.py',
    'docs/experiment_plans/lr_continuation_v24.md', 'scripts/evaluate_lr_v24.py', 'scripts/run_lr_eval_v24.py',
    'scripts/summarize_lr_v24.py', 'tests/test_lr_eval_v24.py')


def passivity(proof,window):
    require(proof.get('unchanged') is True and proof.get('before') == proof.get('after') and bool(proof.get('before')),
            'snapshot passivity equality differs')
    value=proof['before']
    require(set(value)==set(PASSIVITY_HASH_KEYS)|{'common_step_counter','policy_mode','policy_training'}
            and type(value['common_step_counter']) is int and value['common_step_counter']>=0
            and value['policy_mode']==window['mode'] and value['policy_training'] is False,
            'snapshot passivity inventory/policy differs')
    for key in PASSIVITY_HASH_KEYS:
        digest(value[key])


def censoring(data):
    n,seconds=data['num_envs'],data['condition']['seconds']
    captured=data.get('captured_step')
    require(type(data.get('window_seconds')) is int and data['window_seconds']==seconds
            and type(captured) is int and max(data['episode_lengths'])<=captured<=seconds*60,
            'window capture bounds differ')
    require(data.get('capture_reason')==('window_cutoff' if captured==seconds*60 else 'all_first_episodes_finished'),
            'window capture reason differs')
    for key in ('episode_physical_done','episode_physical_timeout','episode_window_censored'):
        require(isinstance(data.get(key),list) and len(data[key])==n and all(type(x) is bool for x in data[key]),
                'physical completion/censor flag shape differs')
    for i in range(n):
        done,timeout,censor,fall=(data[key][i] for key in ('episode_physical_done','episode_physical_timeout',
            'episode_window_censored','episode_terminated'))
        require(done != censor and timeout==(done and not fall) and (not fall or done)
                and censor==(seconds==16 and not done), 'physical done/timeout/censor semantics differ')
        require(not timeout or data['episode_lengths'][i]==3840, 'physical timeout is not 64s')
        require(not censor or data['episode_lengths'][i]==960, 'virtual cutoff clock differs')
        require(captured==seconds*60 or done, 'early capture invents an unfinished row')


def telemetry_arrays(value,n,prefix=''):
    require(isinstance(value,dict),'passive telemetry mapping missing')
    result={}
    for key,item in value.items():
        name=prefix+'/'+key
        if isinstance(item,dict):
            result.update(telemetry_arrays(item,n,name))
        elif isinstance(item,list):
            require(len(item)==n,'passive telemetry row shape differs')
            result[name]=item
    return result


def paired_windows(short,long):
    identity=('controller','geometry_seed','reset_seed','phase','num_envs','scenario','training_seed','mode',
        'checkpoint','checkpoint_sha256','initial_state_sha256','initial_prefix_sha256','initial_rng_sha256',
        'family_names','difficulties','family_indices','level_indices','lr_condition','model_learning_rate')
    require(all(key in short and short[key]==long.get(key) for key in identity),'cross-window row identity differs')
    require(short['captured_step']<=long['captured_step'],'window capture order differs')
    passive={}
    for key in ('contact_telemetry','posture_telemetry'):
        left=telemetry_arrays(short[key],short['num_envs'])
        right=telemetry_arrays(long[key],long['num_envs'])
        require(set(left)==set(right),'passive telemetry window inventory differs')
        passive[key]=(left,right)
    for i in range(short['num_envs']):
        for key in ('episode_lengths','maximum_distance_m','episode_terminated','episode_out_of_lane','episode_world_exit'):
            require(long[key][i]>=short[key][i],'first-episode prefix regressed')
        for key in ('episode_switch_steps','episode_switch_to_v10','history_switch_events'):
            require(long[key][i][:len(short[key][i])]==short[key][i],'history event prefix differs')
        for key in ('first_hit_one_seconds','first_hit_six_seconds'):
            require(short[key][i] is None or long[key][i]==short[key][i],'first-hit prefix differs')
            require(not (short[key][i] is None and long[key][i] is not None and long[key][i]<=16),
                    'missing early first-hit evidence')
        for left,right in passive.values():
            for key,values in left.items():
                if key.endswith(('steps','samples')):
                    require(right[key][i]>=values[i],'passive first-episode counts regressed')
                if short['episode_physical_done'][i]:
                    require(values[i]==right[key][i],'completed passive telemetry changed')
        if short['episode_physical_done'][i]:
            for key in FLAT_FIELDS:
                if key in ('distance_at_snapshot_m','episode_full_horizon_survival'):
                    continue
                require(short[key][i]==long[key][i],'completed first episode contaminated by reset')


def bundle_rows(bundle,*,development=False):
    finite(bundle)
    require(bundle.get('schema')=='week03_ant_lr_continuation_v24_bundle_v1'
            and type(bundle.get('instrumented')) is bool
            and bundle.get('physical_condition')==dict(seconds=64,max_steps=3840,dt=1/60),'physical bundle contract differs')
    require(bundle['instrumented'] is True, 'v24 has no uninstrumented run')
    expected=set(WINDOWS)
    require(set(bundle.get('windows',{}))==expected and set(bundle.get('snapshot_passivity',{}))==expected,
            'paired window/passivity inventory differs')
    if not bundle['instrumented']:
        require(development and bundle.get('phase')=='reference' and bundle.get('controller') in ('parent','history_parent'),
                'uninstrumented holdout or nonreference forbidden')
    else:
        require(bundle.get('phase')==('smoke' if development else 'holdout'),'paired phase differs')
    result={}
    for key,data in bundle['windows'].items():
        require(data['condition']['seconds']==int(key),'window duration identity differs')
        require(all(field in bundle and bundle[field]==data.get(field) for field in
                    ('controller','geometry_seed','reset_seed','phase','num_envs','scenario','lr_condition','model_learning_rate','checkpoint','checkpoint_sha256')),'outer/window identity differs')
        result[key]=raw_rows(data,controller=bundle['controller'],geometry=bundle['geometry_seed'],
                            reset=bundle['reset_seed'],scenario='mixed',development=development)
        censoring(data)
        for telemetry_key in ('contact_telemetry','posture_telemetry'):
            telemetry_arrays(data.get(telemetry_key),data['num_envs'])
            require(data[telemetry_key].get('active_steps')==data['episode_lengths'],'passive telemetry episode clocks differ')
        passivity(bundle['snapshot_passivity'][key],data)
        for i in range(data['num_envs']):
            events=data['history_switch_events'][i]
            require(isinstance(events,list) and len(events)==data['episode_switch_count'][i]
                    and len(events)==len(data['episode_switch_steps'][i])==len(data['episode_switch_to_v10'][i]),
                    'routing event count differs')
            previous=0
            for index,event in enumerate(events):
                require(type(event['step']) is int and previous<event['step']<=data['episode_lengths'][i]
                        and event['step']==data['episode_switch_steps'][i][index]
                        and type(event['target_v10']) is bool and event['target_v10']==data['episode_switch_to_v10'][i][index],
                        'routing event clock/target differs')
                previous=event['step']
    require(type(bundle.get('physical_steps_executed')) is int
            and bundle['physical_steps_executed']==bundle['windows']['64']['captured_step']
            and bundle.get('executed_first_episodes')==bundle['num_envs']
            and bundle.get('window_observations')==bundle['num_envs']*len(expected),'physical episode/window count differs')
    if bundle['instrumented']:
        paired_windows(bundle['windows']['16'],bundle['windows']['64'])
    return result


def duration_pairs(short,long):
    require(len(short)==len(long),'duration pair denominator differs')
    cells=dict(neither=0,late_gain=0,late_loss=0,both=0)
    reasons=dict(new_physical_termination=0,new_lane_exit=0,new_world_exit=0,final_distance_below_53_1=0)
    hits=0
    threshold=struct.unpack('f',struct.pack('f',53.1))[0]
    for a,b in zip(short,long):
        require((a['family'],a['level'])==(b['family'],b['level']),'duration pair row ordering differs')
        name={(0,0):'neither',(0,1):'late_gain',(1,0):'late_loss',(1,1):'both'}[a['six'],b['six']]
        cells[name]+=1
        if name=='late_loss':
            flags=(b['falls'] and not a['falls'],b['lane'] and not a['lane'],b['world'] and not a['world'],b['distance']<threshold)
            for flag,key in zip(flags,reasons):
                reasons[key]+=int(bool(flag))
        hits+=int(b['first_six_hit'] is not None and 16<b['first_six_hit']<=64)
    return dict(n=len(short),strict_six=cells,late_loss_flags=reasons,first_six_hit_in_16_64=hits,
                flags_overlap=True,threshold_hit_is_not_strict_success=True,strict_six_float32_threshold=threshold)


def duration_sections(short,long):
    result={'all':duration_pairs(short,long)}
    selectors=[('rough',lambda row:row['family']!='flat')]
    for family in FAMILIES:
        selectors.append((f'family/{family}',lambda row,f=family:row['family']==f))
        for level in range(5):
            selectors.append((f'family_level/{family}/{level}',lambda row,f=family,l=level:row['family']==f and row['level']==l))
    for level in range(5):
        selectors.append((f'level/{level}',lambda row,l=level:row['level']==l))
    for name,select in selectors:
        result[name]=duration_pairs([r for r in short if select(r)],[r for r in long if select(r)])
    return result


def flat_proof(data):
    indices=[i for i,family in enumerate(data['family_indices']) if family==6]
    return {'indices':indices,'levels':[data['level_indices'][i] for i in indices],
            **{key:[data[key][i] for i in indices] for key in FLAT_FIELDS}}


def parity(new,old):
    require(len(PARITY_FIELDS)==35 and all(key in new and key in old and new[key]==old[key] for key in PARITY_FIELDS),
            'exact 35-field physical parity differs')


def model_binding(data,models,plan_sha,reference_config):
    require(data['controller'] in CONTROLLERS, 'undeclared model controller')
    key=data['controller'].removeprefix('history_')
    model=models[key]
    require(data['checkpoint']==model['checkpoint'] and data['checkpoint_sha256']==model['sha256']
            and data['evaluation_plan_sha256']==plan_sha,'frozen model/plan binding differs')
    require(all(data[key]==value for key,value in reference_config.items()),'frozen physical gate/telemetry config differs')


def evaluator_equivalence(root=ROOT, candidate=None):
    """Independently compare whole AST, with four exact cosmetic string replacements."""
    original = ast.parse((root / 'scripts/evaluate_paired_horizon_v23.py').read_text())
    current = ast.parse((root / 'scripts/evaluate_lr_v24.py').read_text() if candidate is None else candidate)
    common = [node.value for node in ast.walk(current) if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'common' for t in node.targets)]
    require(len(common) == 1 and isinstance(common[0], ast.Dict), 'raw provenance dictionary changed')
    allowed_dictionary = common[0]
    allowed_expressions = {
        'lr_condition': '"not_applicable" if key == "parent" else ("low" if key.startswith("low") else "high")',
        'model_learning_rate': 'None if key == "parent" else (1.e-5 if key.startswith("low") else 1.e-4)'}
    additions = [(k.value, v) for k,v in zip(allowed_dictionary.keys, allowed_dictionary.values)
                 if isinstance(k,ast.Constant) and k.value in allowed_expressions]
    require(Counter(key for key, _ in additions)==Counter(allowed_expressions.keys()),
            'raw provenance key occurrence differs')
    require(all(ast.dump(value,include_attributes=False)==
            ast.dump(ast.parse(allowed_expressions[key],mode='eval').body,include_attributes=False)
            for key,value in additions), 'raw provenance expression changed')
    text_changes = {x.replace('v23', 'v24'): x for x in (
        'v23 is assigned to cuda:1', 'v23 evaluates final full-budget checkpoints only',
        'v23 allows only its predeclared historical checkpoints', '[v23] ')}
    class Normalize(ast.NodeTransformer):
        def visit_Module(self, node):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
            return self.generic_visit(node)
        def visit_ImportFrom(self, node):
            if node.module == 'week03_ant.lr_study_v24':
                node.module = 'week03_ant.paired_horizon_study_v23'
            return node
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = text_changes.get(node.value, node.value)
            return node
        def visit_Dict(self, node):
            pairs = [(k, v) for k, v in zip(node.keys, node.values)
                     if not (node is allowed_dictionary and isinstance(k, ast.Constant) and k.value in ('lr_condition', 'model_learning_rate'))]
            node.keys, node.values = [k for k, v in pairs], [v for k, v in pairs]
            return self.generic_visit(node)
    require(ast.dump(Normalize().visit(original), include_attributes=False) ==
            ast.dump(Normalize().visit(current), include_attributes=False), 'whole evaluator AST differs')
    return True


def development(root,directory,frozen,cache_root):
    proof=read(directory/'development.json')
    refs=read(directory/'historical_references.json')
    old_dir=root/'artifacts/terrain_demo/paired_horizon_v23'
    require(refs['v23_frozen_sha256']==sha(old_dir/'frozen.json')==frozen['v23_frozen_sha256'],
            'historical v23 binding differs')
    hashes(root,refs['sha256'])
    old_records=read(old_dir/'development.json')['records']
    require(refs['evaluation_records']==old_records, 'historical development inventory differs')
    old={record['controller']:linked(root,record) for record in old_records}
    require(set(old)=={'parent',*RUNS,'history_parent',*('history_'+r for r in RUNS)}, 'historical eight controllers differ')
    for record in old_records:
        require(refs['sha256'].get(record['path'])==record['sha256'], 'historical record pin differs')
    require(proof.get('passed') is True and proof['parity_fields']==list(PARITY_FIELDS)
            and proof['source_sha256']==frozen['source_sha256'] and proof['legacy']==frozen['legacy']
            and proof['models']==frozen['models'] and proof.get('evaluator_ast_equivalent') is True,
            'development frozen linkage differs')
    evaluator_equivalence(root)
    require([r['controller'] for r in proof['records']]==list(DEV_CONTROLLERS), 'development controller order differs')
    require(linked(root,proof['preparation'])['scored_episodes']==0,'development preparation scored')
    require(proof['terrain_cache']['geometries']==[51],'development cache geometry differs')
    cache_hashes(proof['terrain_cache'],cache_root)
    config={key:old['parent']['windows']['16'][key] for key in ('gate_config','posture_config','contact_config','v5_sha256')}
    require(config['v5_sha256']=='889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e','v5 teacher identity differs')
    first,flats,files=None,{},{}
    parity_checks=0
    for record in proof['records']:
        data=linked(root,record)
        rows=bundle_rows(data,development=True)
        require(data['controller']==record['controller'],'development raw controller substituted')
        require(datetime.fromisoformat(refs['created_utc'])<datetime.fromisoformat(data['started_utc']),
                'historical pin created after new rollout')
        files[inside(root,record['path'])]=data
        for window,raw in data['windows'].items():
            model_binding(raw,frozen['models'],sha(root/'docs/experiment_plans/lr_continuation_v24.md'),config)
            state=initialization(raw)
            require(first is None or state==first,'development initial/RNG not paired')
            first=state
            if data['controller'].startswith('history_'):
                flat=flat_proof(raw)
                require(window not in flats or flats[window]==flat,'development history flat branch differs')
                flats[window]=flat
            if data['controller'] in ('parent','history_parent'):
                parity(raw,old[data['controller']]['windows'][window])
                parity_checks+=1
        require(sum(len(value) for value in rows.values())==70,'development window count differs')
    require(parity_checks==4 and len(files)==8 and sum(x['executed_first_episodes'] for x in files.values())==280
            and sum(x['window_observations'] for x in files.values())==560,'development executed budget differs')
    return config,files


def command_ledger(root,directory,raw_files,development_files):
    work=root/'outputs/lr_continuation_v24_20260928'
    expected=[('parent',51,24,'prepare_development',work/'prepare_development.json','prepare_development')]
    for phase,controllers in (('smoke',DEV_CONTROLLERS),):
        expected.extend((c,51,24,phase,work/f'{phase}_{c}.json',f'{phase}_{c}') for c in controllers)
    expected.extend(('parent',g,r,'prepare',work/f'prepare_geometry{g}.json',f'prepare_geometry{g}') for g,r in MAPS)
    expected.extend((c,g,r,'holdout',directory/'evaluations'/f'{c}__geometry{g}_reset{r}.json',f'evaluations_{c}__geometry{g}_reset{r}') for c in CONTROLLERS for g,r in MAPS)
    pin=read(directory/'preholdout_ledger.json')
    path=inside(root,pin['path'])
    require(path==(directory/'commands.jsonl').resolve(),'wrong command ledger pinned')
    content=path.read_bytes()
    require(type(pin['bytes']) is int and pin['bytes']>0 and pin['lines']==11,'ledger prefix budget differs')
    prefix=content[:pin['bytes']]
    require(prefix.endswith(b'\n') and len(prefix.splitlines())==11 and hashlib.sha256(prefix).hexdigest()==pin['sha256'],
            'preholdout ledger prefix changed')
    entries=[json.loads(line,object_pairs_hook=unique_object) for line in content.splitlines()]
    require(len(entries)==len(expected)==39,'39 evaluation-command budget differs')
    logs=set()
    previous=None
    for index,(record,spec) in enumerate(zip(entries,expected)):
        c,g,r,phase,output,label=spec
        command=[str(root.parent/'run-python'),'scripts/evaluate_lr_v24.py','--headless','--device','cuda:1',
                 '--controller',c,'--geometry',str(g),'--seed',str(r),'--scenario','mixed','--seconds','64',
                 '--num_envs','35' if phase in ('prepare_development','reference','smoke') else '175',
                 '--phase',phase,'--output',str(output)]
        if phase=='reference':
            command.append('--no-prefixes')
        log=work/f'{label}.log'
        require(record['command']==command and record['label']==label and type(record['returncode']) is int and record['returncode']==0
                and inside(root,record['log'])==log.resolve() and log not in logs and sha(log)==record['log_sha256'],
                'command order/device/seed/no-retry/log contract differs')
        start,end=(datetime.fromisoformat(record[key]) for key in ('started_utc','finished_utc'))
        require(start<=end and (previous is None or previous<=start),'nonserial command chronology')
        previous=end
        logs.add(log)
        if phase.startswith('prepare'):
            require(read(output)['scored_episodes']==0,'preparation command scored')
        else:
            require(output.resolve() in (raw_files if phase=='holdout' else development_files),'command output not in verified inventory')
        if index>=11:
            require(record['evaluation_inputs_sha256']==sha(directory/'evaluation_inputs.json')
                    and record['terrain_cache_manifest_sha256']==sha(directory/'terrain_cache.json'),'scored command input pin differs')
    return len(entries)


def training_specs(root, directory, references):
    work=root/'outputs/lr_continuation_v24_20260928'
    python=str(root.parent/'run-python')
    output=work/'prepare_training_geometry110.json'
    specs=[('prepare_training_geometry110',[python,'scripts/evaluate_contact_v16.py','--headless','--device','cuda:1',
        '--controller','history_original','--geometry','110','--seed','51','--phase','prepare',
        '--scenario','mixed','--seconds','16','--num_envs','35','--output',str(output),'--checkpoint',
        str(root/'artifacts/terrain_demo/prior_v10/runs/v10_anchored_seed42/model_749.pt')])]
    runs=[('preflight',arm,51) for arm in ('high','low')]
    runs += [('capacity',arm,seed) for seed in (51,52,53) for arm in ('high','low')]
    runs += [('main','high',51),*(('main','low',seed) for seed in (51,52,53))]
    for phase,arm,seed in runs:
        name=f'{arm}{seed}'
        label=f'train_{name}' if phase=='main' else f'{phase}_{name}'
        base=directory/'training'/name if phase=='main' else work/label
        initial,schedule=Path(str(base)+'_initial.json'),Path(str(base)+'_schedule.json')
        envs,iters=(256,2) if phase=='preflight' else (4096,2) if phase=='capacity' else (4096,250)
        command=[python,'scripts/lr_continuation_v24.py','train','--arm','no_cost','--lr-condition',arm,
            '--training-seed',str(seed),'--audit-output',str(initial),'--schedule-output',str(schedule),
            '--ramp-steps','32' if phase=='preflight' else '4000','--headless','--device','cuda:1',
            '--num_envs',str(envs),'--max_iterations',str(iters),'--seed',str(seed),'--run_name',f'v24_{phase}_{name}',
            '--resume','--load_run',f'v24_{arm}_init','--checkpoint','model_0.pt',
            'env.scene.terrain.terrain_generator.seed=110','agent.device=cuda:1']
        expected=(directory/'expected_main'/f'{name}_initial.json' if phase=='main' else
                  inside(root,references['stages'][phase][f'seed{seed}']['initial']['path']) if arm=='high' else None)
        if expected is not None:
            command+=['--expected-initial',str(expected)]
        specs.append((label,command))
    return specs


def training_ledger(root,directory,frozen):
    """Validate commands and hash links, not the optimizer claims inside linked JSON."""
    path=directory/'training_commands.jsonl'
    trained=read(directory/'trained_models.json')
    require(trained['training_commands_sha256']==sha(path)==frozen['training_commands_sha256'],
            'completed training ledger pin differs')
    references=read(directory/'historical_references.json')
    entries=[json.loads(line,object_pairs_hook=unique_object) for line in path.read_text().splitlines()]
    specs=training_specs(root,directory,references)
    require(len(entries)==len(specs)==13,'13 training-command budget differs')
    previous=None
    for entry,(label,command) in zip(entries,specs):
        finite(entry)
        log=root/'outputs/lr_continuation_v24_20260928'/f'{label}.log'
        require(entry['label']==label and entry['command']==command and type(entry['returncode']) is int
                and entry['returncode']==0 and inside(root,entry['log'])==log.resolve()
                and sha(log)==entry['log_sha256'],'training command order/device/seed/log differs')
        start,end=(datetime.fromisoformat(entry[key]) for key in ('started_utc','finished_utc'))
        require(start<=end and (previous is None or previous<=start),'training command chronology differs')
        previous=end
    eval_first=json.loads((directory/'commands.jsonl').read_text().splitlines()[0],object_pairs_hook=unique_object)
    require(previous<datetime.fromisoformat(eval_first['started_utc']),'evaluation began before training finished')
    training=read(directory/'training_frozen.json')
    freeze=datetime.fromisoformat(training['frozen_at'])
    require(datetime.fromisoformat(entries[8]['finished_utc'])<=freeze<=datetime.fromisoformat(entries[9]['started_utc']),
            'all-source training freeze is not before full runs')
    require(trained['training_freeze_sha256']==sha(directory/'training_frozen.json')
            and trained['training_validation_sha256']==sha(directory/'training_validation.json'), 'training completion evidence link differs')
    for key,name in (('preflight_sha256','preflight.json'),('capacity_sha256','capacity.json'),
                     ('initial_shared_sha256','initial_shared.json'),('training_cache_sha256','training_cache.json'),
                     ('historical_references_sha256','historical_references.json')):
        require(training[key]==sha(directory/name),'training development link differs')
    hashes(root,training['expected_main_sha256'])
    require(set(training['initializers'])=={'high','low'},'initializer inventory differs')
    hashes(root,{row['checkpoint']:row['sha256'] for row in training['initializers'].values()})
    expected=dict(total_training_transitions=98304000,diagnostic_replay_transitions=32768000,
                  development_training_transitions=1605632,new_training_transitions=132677632)
    require(all(training[key]==value for key,value in expected.items()),'training transition budget differs')
    require(read(root/'outputs/lr_continuation_v24_20260928/prepare_training_geometry110.json')['scored_episodes']==0,
            'training preparation scored')
    return 13


def report_role(window, section):
    require(window in WINDOWS and section.get('role') == ('primary' if window == '16' else 'secondary descriptive'),
            'report window role differs')


def audit(root=ROOT,directory=None,cache_root=Path('/tmp/isaaclab/terrains'),*,compare_report=True):
    root=Path(root).resolve()
    directory=Path(directory or root/'artifacts/terrain_demo/lr_continuation_v24').resolve()
    summary=read(directory/'summary.json') if compare_report else None
    frozen=read(directory/'frozen.json')
    budget=dict(controllers=list(CONTROLLERS),holdouts=[list(pair) for pair in MAPS],windows=[16,64],physical_seconds=64,
                new_training_transitions=132677632,physical_rollouts=28,first_episodes=4900,window_observations=9800)
    require(all(frozen.get(key)==value for key,value in budget.items()),'frozen episode/window/training budget differs')
    require(set(frozen['source_sha256'])==set(SOURCE_FILES),'frozen twelve definition inventory differs')
    require(len(frozen['legacy']['sources'])==237 and len(frozen['legacy']['models'])==22,'legacy source/model inventory differs')
    for inventory in (frozen['source_sha256'],frozen['legacy']['sources'],frozen['legacy']['models']):
        hashes(root,inventory)
    old=read(root/'artifacts/terrain_demo/seed_continuation_v22/frozen.json')
    trained=read(directory/'trained_models.json')
    require(set(trained['models'])==set(LOW_RUNS) and frozen['models']=={**old['models'],**trained['models']}
            and frozen['models']['parent']['sha256']==PARENT_SHA,'all-seed model inventory differs')
    for name,model in frozen['models'].items():
        hashes(root,{model['checkpoint']:model['sha256']})
        if name!='parent':
            require(model['training_seed']==int(name[-2:]) and model['transitions']==32768000 and model['iteration']==249,
                    'training ancestry differs')
            expected_dir='lr_continuation_v24' if name in LOW_RUNS else 'seed_continuation_v22'
            require(model['checkpoint']==f'artifacts/terrain_demo/{expected_dir}/runs/{name}/model_249.pt', 'checkpoint path/arm differs')
            if name in LOW_RUNS:
                require(model['learning_rate']==1.e-5, 'new low model LR differs')
    for key,name in (('development_sha256','development.json'),('historical_references_sha256','historical_references.json'),
                     ('training_frozen_sha256','training_frozen.json'),('trained_models_sha256','trained_models.json')):
        require(frozen[key]==sha(directory/name),'frozen evidence changed')
    training=read(directory/'training_frozen.json')
    require(training['all_source_sha256']==frozen['source_sha256'] and training['legacy']==frozen['legacy']
            and training['source_sha256']=={key:frozen['source_sha256'][key] for key in SOURCE_FILES[:8]},
            'training/evaluation source freeze differs')
    config,devfiles=development(root,directory,frozen,Path(cache_root))
    inputs=read(directory/'evaluation_inputs.json')
    for key,name in (('experiment_freeze_sha256','frozen.json'),('terrain_cache_manifest_sha256','terrain_cache.json'),
                     ('preholdout_ledger_sha256','preholdout_ledger.json')):
        require(inputs[key]==sha(directory/name),'evaluation inputs changed')
    cache=read(directory/'terrain_cache.json')
    require(cache['cache']['geometries']==[119,120] and len(cache['preparation'])==2,'holdout cache inventory differs')
    cache_hashes(cache['cache'],Path(cache_root))
    for record in cache['preparation']:
        require(linked(root,record)['scored_episodes']==0,'holdout preparation scored')
    provenance={key:sha(directory/name) for key,name in (('experiment_freeze_sha256','frozen.json'),
        ('evaluation_inputs_sha256','evaluation_inputs.json'),('terrain_cache_manifest_sha256','terrain_cache.json'))}
    expected={f'{c}__geometry{g}_reset{r}.json' for c in CONTROLLERS for g,r in MAPS}
    require({p.name for p in (directory/'evaluations').glob('*.json')}==expected,'28bundle inventory differs')
    groups={window:defaultdict(list) for window in WINDOWS}
    maps={window:defaultdict(dict) for window in WINDOWS}
    initial,flats,files,raw_files={},{},[],{}
    for controller in CONTROLLERS:
        for geometry,reset in MAPS:
            path=directory/'evaluations'/f'{controller}__geometry{geometry}_reset{reset}.json'
            value=read(path)
            require(value['instrumented'] is True and (value['controller'],value['geometry_seed'],value['reset_seed'])==(controller,geometry,reset),
                    'physical output identity differs')
            rows=bundle_rows(value)
            raw_files[path.resolve()]=value
            for window,raw in value['windows'].items():
                require(all(raw[key]==digest for key,digest in provenance.items()),'raw scoring provenance differs')
                model_binding(raw,frozen['models'],sha(root/'docs/experiment_plans/lr_continuation_v24.md'),config)
                require(datetime.fromisoformat(inputs['created_utc'])<datetime.fromisoformat(raw['started_utc'])<=datetime.fromisoformat(raw['finished_utc']),
                        'raw scoring predates input freeze')
                state=initialization(raw)
                key=(geometry,reset)
                require(key not in initial or initial[key]==state,'controller/window initial states not paired')
                initial[key]=state
                if controller.startswith('history_'):
                    flat=flat_proof(raw)
                    pair=(window,geometry,reset)
                    require(pair not in flats or flats[pair]==flat,'history flat branches differ')
                    flats[pair]=flat
                groups[window][controller].extend(rows[window])
                maps[window][f'geometry{geometry}_reset{reset}'][controller]=rows[window]
            files.append(dict(path=str(path.relative_to(directory)),sha256=sha(path),episodes=175,window_observations=350))
    scores,paired={},{}
    for window in WINDOWS:
        scores[window]=computed_section(groups[window],True)
        scores[window]['per_map']={name:computed_section(selected,True) for name,selected in maps[window].items()}
        if summary is not None:
            report_role(window, summary['windows'][window])
            compare_section(groups[window],summary['windows'][window],True)
            require(set(summary['windows'][window]['per_map'])==set(maps[window]),'report map inventory differs')
            for name,selected in maps[window].items():
                compare_section(selected,summary['windows'][window]['per_map'][name],True)
    for controller in CONTROLLERS:
        paired[controller]=dict(pooled=duration_sections(groups['16'][controller],groups['64'][controller]),
            per_map={name:duration_sections(maps['16'][name][controller],maps['64'][name][controller]) for name in maps['16']})
    if summary is not None:
        require(summary['files']==files and summary['paired_duration']==paired,'report duration/raw inventory differs')
        require(all(summary.get(key)==value for key,value in budget.items() if key in
                    ('physical_rollouts','first_episodes','window_observations','new_training_transitions')),'report episode/window budget differs')
        require(set(summary['windows'])==set(WINDOWS) and set(summary['paired_duration'])==set(CONTROLLERS),'report window/controller inventory differs')
        require(all(summary['provenance'][key]==value for key,value in provenance.items()),'report provenance differs')
    if summary is not None:
        require(summary.get('evaluation_training_transitions')==0 and summary.get('reused_training_transitions')==98304000,
                'evaluation versus reused training accounting differs')
        require(all(summary['provenance'][key]==frozen[key] for key in ('training_frozen_sha256', 'trained_models_sha256',
                'historical_references_sha256', 'development_sha256', 'training_commands_sha256')), 'report training provenance differs')
    commands=command_ledger(root,directory,raw_files,devfiles)
    training_commands=training_ledger(root,directory,frozen)
    if summary is not None:
        require(summary['provenance']['ledger']==dict(scored_commands=28,unique_logs=28,verified=True),
                'summary scored ledger count differs')
    verifier_path=Path(__file__).resolve()
    test_path=ROOT/'tests/test_lr_raw_audit_v24.py'
    return dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),physical_rollouts=28,raw_files=28,
        raw_window_records=56,first_episodes=4900,window_observations=9800,dependent_window_observations=True,
        development_first_episodes=280,development_window_observations=560,new_training_transitions=132677632,
        successful_gpu_commands=commands+training_commands,successful_training_gpu_commands=training_commands,
        successful_evaluation_gpu_commands=commands,scored_physical_commands=28,windows=scores,paired_duration=paired,
        report_comparison=compare_report,summary_sha256=sha(directory/'summary.json') if summary is not None else None,
        experiment_freeze_sha256=sha(directory/'frozen.json'),
        post_experiment_verifier_sha256={str(verifier_path.relative_to(ROOT)):sha(verifier_path),str(test_path.relative_to(ROOT)):sha(test_path)},
        limitations='Independent stdlib arithmetic, JSON equality and byte-hash provenance. Snapshot passivity hashes attest recorded state equality; simulator state and neural tensors are not independently reconstructed. Contact/posture geometry, trained optimizer tensors, LR updates and TensorBoard scalars are not independently recomputed; the separate CPU training audit is required. The 39-evaluation-command check applies to the fixed successful attempt without extra infrastructure retries. Windows/maps/history are not independent training replicates; no causal or safety guarantee.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'artifacts/terrain_demo/lr_continuation_v24')
    parser.add_argument('--cache-root',type=Path,default=Path('/tmp/isaaclab/terrains'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--skip-summary',action='store_true')
    args=parser.parse_args()
    output=args.output or args.directory/'independent_raw_audit.json'
    require(not output.exists(),'refusing to overwrite independent audit')
    result=audit(directory=args.directory,cache_root=args.cache_root,compare_report=not args.skip_summary)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:result[key] for key in ('passed','raw_files','first_episodes','window_observations')},indent=2))


if __name__=='__main__':
    main()
