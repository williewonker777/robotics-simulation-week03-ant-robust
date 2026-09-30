"""Post-experiment independent v20 verification, using Python's standard library only.

This verifier is deliberately OUTSIDE the frozen experimental source inventory.
It does not import the experiment's scoring, study, schedule or telemetry helpers.
The output pins this verifier's own bytes and explicitly limits its claims.
"""
from __future__ import annotations

import argparse
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
CONTROLLERS = ('parent', 'immediate', 'ramped', 'history_parent', 'history_immediate', 'history_ramped')
MAPS = ((111, 71), (112, 72))
FAMILIES = ('rough', 'slope', 'stairs', 'waves', 'obstacles', 'stepping_stones', 'flat')
PAIRS = (('ramped', 'immediate'), ('history_ramped', 'history_immediate'),
         ('immediate', 'parent'), ('ramped', 'parent'),
         ('history_immediate', 'history_parent'), ('history_ramped', 'history_parent'))
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


def raw_rows(data, *, controller, geometry, reset, scenario):
    """Recompute first-episode counts; no saved aggregate is used as input."""
    finite(data)
    seconds, n = (16, 175) if scenario == 'mixed' else (64, 10)
    require((data['schema'], data['task'], data['phase'], data['controller'], data['geometry_seed'],
             data['reset_seed'], data['scenario'], data['num_envs']) ==
            ('week03_ant_contact_curriculum_v20_v1', 'Week03-Ant-Contact-v16-Eval-v0', 'holdout',
             controller, geometry, reset, scenario, n), 'raw evaluation identity differs')
    require((geometry, reset) in MAPS and controller in CONTROLLERS, 'undeclared map/controller')
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
        maximum = data['maximum_distance_m'][i]
        require(type(maximum) in (int, float) and maximum + 1.e-5 >= distance, 'maximum distance contradicts final')
        for key, threshold in (('first_hit_one_seconds', one), ('first_hit_six_seconds', six)):
            hit = data[key][i]
            require((hit is not None) == (maximum >= threshold), 'first hit contradicts maximum distance')
            if hit is not None:
                require(type(hit) in (int, float) and 0 < hit <= length / 60 + 1.e-7, 'first hit outside first episode')
        rows.append(dict(family=FAMILIES[family], level=level, one=int(success[0]), six=int(success[1]),
                         falls=int(fall), lane=int(lane), world=int(world), survival=int(flags[5]),
                         speed=distance / (length * c['dt'])))
    expected = ({(family, level): 5 for family in FAMILIES for level in range(5)}
                if scenario == 'mixed' else {('stepping_stones', 4): 10})
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


def compare_section(groups, section, primary, *, breakdown=True):
    require(set(section['groups']) == set(CONTROLLERS), 'summary controller inventory differs')
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
                require(target['comparisons'] == {}, 'flat level has no rough gate')
                for name, rows in subset.items():
                    compare_group(rows, target['groups'][name])
    require(set(section['family_level']) == expected_family_levels, 'family/level cells differ')
    for level in levels:
        subset = {name: [r for r in rows if r['level'] == level] for name, rows in groups.items()}
        compare_section(subset, section['level'][str(level)], primary, breakdown=False)


def schedule(record, arm):
    """Independently derive 8,000 coefficients and actual reward*dt accounting."""
    finite(record)
    require(record['mode'] == record['arm'] == arm and record['num_envs'] == 4096
            and record['ramp_steps'] == 4000 and record['policy_steps'] == 8000, 'training schedule shape differs')
    keys = ('common_step_counters', 'coefficients', 'raw_reward_mean', 'scaled_reward_mean',
            'valid_rows', 'step_dt', 'raw_reward_dt_sum', 'scaled_reward_dt_sum')
    require(all(isinstance(record[k], list) and len(record[k]) == 8000 for k in keys), 'incomplete reward-time sequence')
    for i in range(8000):
        step = record['common_step_counters'][i]
        require(type(step) is int and step == i + 1, 'global training clock reset/skipped')
        expected = 1. if arm == 'immediate' or i >= 3999 else i / 3999
        require(type(record['coefficients'][i]) in (int, float) and record['coefficients'][i] == expected,
                'actual coefficient trajectory differs')
        raw, scaled = record['raw_reward_mean'][i], record['scaled_reward_mean'][i]
        require(type(raw) in (int, float) and type(scaled) in (int, float)
                and -1 <= raw <= 0 and -1 <= scaled <= 0, 'reward outside declared bounds')
        close(scaled, raw * expected, 'coefficient not applied to same raw reward', tolerance=2.e-6, absolute=2.e-7)
        valid = record['valid_rows'][i]
        require(type(valid) is int and 4096 * .99 <= valid <= 4096, 'excess invalid reward rows')
        close(record['step_dt'][i], 1 / 60, 'reward dt differs')
        for name, mean in (('raw', raw), ('scaled', scaled)):
            close(record[name + '_reward_dt_sum'][i], mean * 4096 / 60,
                  'realized reward dt accounting differs', tolerance=2.e-6, absolute=2.e-7)
    mass = 8000 if arm == 'immediate' else 6000
    close(record['coefficient_sum'], mass, 'coefficient-step mass differs', tolerance=1.e-9)
    realized = -math.fsum(record['scaled_reward_dt_sum'])
    close(record['realized_penalty_sum'], realized, 'realized penalty sum differs', tolerance=1.e-9)
    proof = record['episode_randomization']
    require(proof['matches_raw_environment'] is True and digest(proof['assigned_sha256']) == digest(proof['actual_sha256'])
            and 0 < proof['nonzero'] <= 4096 and 0 <= proof['minimum'] < proof['maximum'],
            'random horizons did not reach actual environment')
    return dict(coefficient_step_mass=mass, realized_penalty_sum=realized)


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


def training(root, directory, cache_root):
    frozen, trained, validation = [read(directory / name) for name in
                                  ('training_frozen.json', 'trained_models.json', 'training_validation.json')]
    require((frozen['seed'], frozen['geometry'], frozen['envs'], frozen['iterations_per_arm'],
             frozen['steps_per_env'], frozen['ramp_steps'], frozen['transitions_per_arm']) ==
            (51, 110, 4096, 250, 32, 4000, TRANSITIONS), 'frozen training budget differs')
    hashes(root, frozen['source_sha256'])
    require(frozen['source_checkpoint_sha256'] == PARENT_SHA, 'training parent differs')
    hashes(root, {frozen['starting_checkpoint']: frozen['starting_checkpoint_sha256']})
    for key, name in (('preflight_sha256', 'preflight.json'), ('capacity_sha256', 'capacity.json'),
                      ('initial_shared_sha256', 'initial_shared.json'), ('training_cache_sha256', 'training_cache.json')):
        require(frozen[key] == sha(directory / name), 'training freeze proof changed')
    require(frozen['terrain_cache']['geometries'] == [110], 'training geometry cache differs')
    cache_hashes(frozen['terrain_cache'], cache_root)
    require(trained['total_training_transitions'] == 2 * TRANSITIONS
            and trained['training_freeze_sha256'] == sha(directory / 'training_frozen.json')
            and trained['training_validation_sha256'] == sha(directory / 'training_validation.json')
            and set(trained['models']) == set(validation['records']) == {'immediate', 'ramped'}
            and validation['paired_initialization'] is True and validation['actual_random_horizons_paired'] is True,
            'training manifests differ')
    first_initial = first_randomization = None
    result = {}
    for arm in ('immediate', 'ramped'):
        rec, model = validation['records'][arm], trained['models'][arm]
        initial, raw_schedule, rollout = [linked(root, rec[key]) for key in ('initial', 'schedule', 'rollout')]
        require(initial['arm'] == arm and initial['num_envs'] == 4096 and initial['iteration'] == 0
                and initial['common_step_counter'] == 0 and initial['optimizer_state_empty'] is True
                and initial['contact_slip_weight'] == 1. and initial['observation_dimensions'] == [4096, 91]
                and all(initial['initial_policy_parity'][k] is True for k in ('actor', 'critic', 'teacher')),
                'training start differs')
        initialization(initial, True)
        paired = {k: initial[k] for k in ('initial_state_sha256', 'initial_prefix_sha256', 'policy_state_sha256',
                                         'rng_sha256', 'num_envs', 'dt', 'normalized_parameters')}
        if first_initial is not None:
            require(paired == first_initial and raw_schedule['episode_randomization'] == first_randomization,
                    'training inputs/random horizons not paired')
        first_initial, first_randomization = paired, raw_schedule['episode_randomization']
        for name, expected in initial['saved_parameter_sha256'].items():
            require(sha(inside(root, initial['log_dir']) / 'params' / (name + '.yaml')) == expected,
                    'actual training YAML changed')
        result[arm] = schedule(raw_schedule, arm)
        for key, value in result[arm].items():
            close(rec[key], value, 'training validation total differs', tolerance=1.e-9)
        require(rollout['arm'] == arm and rollout['passive_post_action'] is True
                and rollout['rollout']['samples'] == 8000, 'passive contact rollout budget differs')
        require(model['iteration'] == 249 and model['transitions'] == TRANSITIONS
                and model['initial_audit_sha256'] == rec['initial']['sha256']
                and model['schedule_sha256'] == rec['schedule']['sha256'], 'final checkpoint provenance differs')
        hashes(root, {model['checkpoint']: model['sha256'], model['source_checkpoint']: model['sha256']})
        log = rec['learning_log']
        hashes(root, {log['text_log']: log['text_log_sha256'], **log['event_files_sha256']})
        text = inside(root, log['text_log']).read_text()
        require([(int(a), int(b)) for a, b in re.findall(r'Learning iteration (\d+)/(\d+)', text)] ==
                [(i, 250) for i in range(250)], 'actual learning iteration sequence differs')
        require([int(v) for v in re.findall(r'Total timesteps:\s*(\d+)', text)] ==
                [(i + 1) * 4096 * 32 for i in range(250)], 'actual learning transition sequence differs')
        require(log['iterations_logged'] == 250 and log['transitions_logged'] == TRANSITIONS
                and log['all_scalars_finite'] is True and all(v == 250 for v in log['scalar_counts'].values()),
                'training scalar manifest budget differs')
    return result


def ledger(root, directory, inputs, raw_paths):
    pin = read(directory / 'preholdout_ledger.json')
    path = inside(root, pin['path'])
    require(path == (directory / 'commands.jsonl').resolve(), 'wrong pinned command ledger')
    content = path.read_bytes()
    require(type(pin['bytes']) is int and type(pin['lines']) is int and pin['bytes'] > 0 and pin['lines'] > 0,
            'invalid ledger prefix size')
    prefix = content[:pin['bytes']]
    require(prefix.endswith(b'\n') and len(prefix) == pin['bytes'] and len(prefix.splitlines()) == pin['lines']
            and hashlib.sha256(prefix).hexdigest() == pin['sha256'], 'pre-scoring ledger prefix changed')
    expected = {p.resolve(): data for p, data in raw_paths.items()}
    matched, logs = Counter(), set()
    for line_number, line in enumerate(content.splitlines()):
        rec = json.loads(line, object_pairs_hook=unique_object)
        finite(rec)
        command = rec['command']
        if '--phase' not in command or command[command.index('--phase') + 1] != 'holdout':
            continue
        require(line_number >= pin['lines'], 'holdout command inside pre-scoring prefix')
        def value(flag):
            require(command.count(flag) == 1 and command.index(flag) + 1 < len(command), 'missing/duplicate command flag')
            return command[command.index(flag) + 1]
        output = (root / value('--output')).resolve()
        require(output in expected, 'undeclared scoring command output')
        data = expected[output]
        require(value('--controller') == data['controller'] and int(value('--geometry')) == data['geometry_seed']
                and int(value('--seed')) == data['reset_seed'] and value('--scenario') == data['scenario']
                and int(value('--num_envs')) == data['num_envs'] and int(value('--seconds')) == data['condition']['seconds']
                and Path(value('--checkpoint')).resolve() == inside(root, data['checkpoint'])
                and value('--device') == 'cuda:1', 'scoring command does not match raw evidence')
        matched[output] += 1
        log = inside(root, rec['log'])
        require(type(rec['returncode']) is int and rec['returncode'] == 0 and log not in logs
                and sha(log) == rec['log_sha256'] and rec['evaluation_inputs_sha256'] == sha(directory / 'evaluation_inputs.json')
                and rec['terrain_cache_manifest_sha256'] == inputs['terrain_cache_manifest_sha256'], 'scoring log/result differs')
        logs.add(log)
    require(matched == Counter({p: 1 for p in expected}), 'missing/duplicate scored command')
    return len(matched)


def audit(root=ROOT, directory=None, cache_root=Path('/tmp/isaaclab/terrains')):
    root = Path(root).resolve()
    directory = Path(directory or root / 'artifacts/terrain_demo/contact_curriculum_v20').resolve()
    # Missing or incomplete inputs fail before an output file can be opened.
    summary, frozen, inputs = [read(directory / name) for name in ('summary.json', 'frozen.json', 'evaluation_inputs.json')]
    require(frozen['controllers'] == list(CONTROLLERS) and frozen['holdouts'] == [list(pair) for pair in MAPS]
            and frozen['new_training_transitions'] == 2 * TRANSITIONS, 'evaluation freeze inventory differs')
    require(summary['new_training_transitions'] == 2 * TRANSITIONS and summary['evaluation_training_transitions'] == 0,
            'summary transition accounting differs')
    for entries in (frozen['source_sha256'], frozen['legacy']['sources'], frozen['legacy']['models']):
        hashes(root, entries)
    require(str(Path(__file__).resolve().relative_to(ROOT)) not in frozen['source_sha256'],
            'post-experiment verifier unexpectedly part of frozen definitions')
    for key, name in (('development_sha256', 'development.json'), ('training_freeze_sha256', 'training_frozen.json'),
                      ('training_validation_sha256', 'training_validation.json'), ('trained_models_sha256', 'trained_models.json')):
        require(frozen[key] == sha(directory / name) == summary['provenance'][key], 'frozen evidence link changed')
    development = read(directory / 'development.json')
    require(development['source_sha256'] == frozen['source_sha256']
            and development['models'] == frozen['models']
            and {r['controller'] for r in development['records']} == set(CONTROLLERS)
            and len(development['records']) == 6, 'development inventory/source/model linkage differs')
    development_raw = {record['controller']: linked(root, record) for record in development['records']}
    reference_config = {key: development_raw['parent'][key] for key in
                        ('gate_config', 'posture_config', 'contact_config', 'v5_sha256')}
    require(reference_config['v5_sha256'] == '889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e',
            'frozen v5 teacher identity differs')
    first = initialization(development_raw['parent'])
    for value in development_raw.values():
        require(initialization(value) == first and all(value[k] == v for k, v in reference_config.items()),
                'development initialization/configuration not paired')
    for record in development['parity']:
        legacy = linked(root, record)
        new = development_raw[record['controller']]
        require(all(legacy[key] == new[key] for key in record['equal_fields']), 'development legacy parity differs')
    models = frozen['models']
    require(set(models) == {'parent', 'immediate', 'ramped'} and models['parent']['sha256'] == PARENT_SHA,
            'evaluation model inventory differs')
    require({k: models[k] for k in ('immediate', 'ramped')} == read(directory / 'trained_models.json')['models'],
            'training/evaluation model manifests differ')
    for model in models.values():
        hashes(root, {model['checkpoint']: model['sha256']})
    training_result = training(root, directory, Path(cache_root))
    for arm, metrics in training_result.items():
        for key, value in metrics.items():
            close(summary['training'][arm][key], value, 'summary training evidence differs', tolerance=1.e-9)
    for key, name in (('experiment_freeze_sha256', 'frozen.json'), ('terrain_cache_manifest_sha256', 'terrain_cache.json'),
                      ('preholdout_ledger_sha256', 'preholdout_ledger.json')):
        require(inputs[key] == sha(directory / name), 'evaluation input link differs')
    cache = read(directory / 'terrain_cache.json')
    require(cache['cache']['geometries'] == [111, 112] and len(cache['preparation']) == 2, 'holdout cache inventory differs')
    cache_hashes(cache['cache'], Path(cache_root))
    for record in cache['preparation']:
        require(linked(root, record)['scored_episodes'] == 0, 'terrain preparation scored episodes')
    provenance = {key: sha(directory / name) for key, name in (
        ('experiment_freeze_sha256', 'frozen.json'), ('evaluation_inputs_sha256', 'evaluation_inputs.json'),
        ('terrain_cache_manifest_sha256', 'terrain_cache.json'))}
    require(all(summary['provenance'][key] == val for key, val in provenance.items()), 'summary raw provenance differs')
    raw_paths, inventory, paired_initial, flat_evidence = {}, [], {}, {}
    for scenario, folder in (('mixed', 'evaluations'), ('stones', 'horizon')):
        names = {f'{c}__geometry{g}_reset{r}.json' for c in CONTROLLERS for g, r in MAPS}
        require({p.name for p in (directory / folder).glob('*.json')} == names, 'raw result matrix incomplete/extra')
        groups, per_map = defaultdict(list), defaultdict(dict)
        for controller in CONTROLLERS:
            for geometry, reset in MAPS:
                path = directory / folder / f'{controller}__geometry{geometry}_reset{reset}.json'
                data = read(path)
                require(all(data[k] == v for k, v in provenance.items()), 'raw provenance differs')
                require(all(data[k] == v for k, v in reference_config.items()), 'raw gate/telemetry/teacher configuration differs')
                model = models[controller.removeprefix('history_')]
                require(data['checkpoint'] == model['checkpoint'] and data['checkpoint_sha256'] == model['sha256']
                        and data['evaluation_plan_sha256'] == sha(root / 'docs/experiment_plans/contact_curriculum_v20.md'),
                        'raw model/plan link differs')
                require(datetime.fromisoformat(inputs['created_utc']) < datetime.fromisoformat(data['started_utc'])
                        <= datetime.fromisoformat(data['finished_utc']), 'scoring chronology differs')
                rows = raw_rows(data, controller=controller, geometry=geometry, reset=reset, scenario=scenario)
                pair = (scenario, geometry, reset)
                initial = initialization(data)
                require(pair not in paired_initial or paired_initial[pair] == initial, 'evaluation initial states/RNG not paired')
                paired_initial[pair] = initial
                if scenario == 'mixed' and controller.startswith('history_'):
                    indices = [i for i, row in enumerate(rows) if row['family'] == 'flat']
                    flat = {key: [data[key][i] for i in indices] for key in FLAT_FIELDS}
                    flat['indices'] = indices
                    flat['levels'] = [data['level_indices'][i] for i in indices]
                    require(pair not in flat_evidence or flat_evidence[pair] == flat, 'history flat physical/routing arrays differ')
                    flat_evidence[pair] = flat
                groups[controller].extend(rows)
                per_map[f'geometry{geometry}_reset{reset}'][controller] = rows
                raw_paths[path] = data
                inventory.append(dict(path=str(path.relative_to(directory)), sha256=sha(path), episodes=len(rows)))
        compare_section(groups, summary[folder], scenario == 'mixed')
        require(set(summary[folder]['per_map']) == set(per_map), 'summary per-map inventory differs')
        for name, selected in per_map.items():
            compare_section(selected, summary[folder]['per_map'][name], scenario == 'mixed')
    require(len(inventory) == 24 and sum(row['episodes'] for row in inventory) == 2220, 'incomplete scored budget')
    require(summary['files'] == inventory, 'summary raw file inventory/digests differ')
    commands = ledger(root, directory, inputs, raw_paths)
    return dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), raw_files=24, first_episodes=2220,
                scored_commands=commands, training=training_result, summary_sha256=sha(directory / 'summary.json'),
                experiment_freeze_sha256=sha(directory / 'frozen.json'), verifier_sha256=sha(Path(__file__)),
                verifier_scope='Post-experiment verification code, not part of frozen experimental definitions.',
                verified='Independent strict counts, all six descriptive gates/paired outcomes at pooled/map/family/level, provenance, clocks and reward accounting.',
                limitations='Stdlib verifier hashes checkpoint/TensorBoard bytes and validates recorded tensor/scalar proofs; it does not deserialize tensors or independently decode TensorBoard scalars, nor independently recompute contact/posture geometry telemetry.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'artifacts/terrain_demo/contact_curriculum_v20')
    parser.add_argument('--cache-root', type=Path, default=Path('/tmp/isaaclab/terrains'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.directory / 'independent_raw_audit.json'
    require(not output.exists(), 'refusing to replace existing independent audit')
    result = audit(directory=args.directory, cache_root=args.cache_root)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
