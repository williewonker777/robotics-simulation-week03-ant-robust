"""Post-experiment independent v21 verification, using Python's standard library only.

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
CONTROLLERS = ('parent', 'no_cost', 'immediate', 'ramped', 'history_parent', 'history_no_cost', 'history_immediate', 'history_ramped')
MAPS = ((113, 73), (114, 74))
FAMILIES = ('rough', 'slope', 'stairs', 'waves', 'obstacles', 'stepping_stones', 'flat')
PAIRS = (('no_cost', 'parent'), ('immediate', 'no_cost'), ('ramped', 'no_cost'),
         ('ramped', 'immediate'), ('history_no_cost', 'history_parent'),
         ('history_immediate', 'history_no_cost'), ('history_ramped', 'history_no_cost'),
         ('history_ramped', 'history_immediate'), ('immediate', 'parent'), ('ramped', 'parent'),
         ('history_immediate', 'history_parent'), ('history_ramped', 'history_parent'))
INIT_SHA = '10d8485d6d18f1dd6c01724ff69fc6a2ced3bbcd3d7b0db1285b7097cb5c21bd'
DEVELOPMENT_TRANSITIONS = 2 * (256 + 4096) * 64
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
    seconds, n = (16, 175) if scenario == 'mixed' else (64, 10)
    if development:
        require((geometry, reset, scenario) == (51, 24, 'mixed'), 'development matrix differs')
        n = 35
    require((data['schema'], data['task'], data['phase'], data['controller'], data['geometry_seed'],
             data['reset_seed'], data['scenario'], data['num_envs']) ==
            ('week03_ant_contact_continuation_v21_v1', 'Week03-Ant-Contact-v16-Eval-v0', 'smoke' if development else 'holdout',
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
    expected = ({(family, level): 1 if development else 5 for family in FAMILIES for level in range(5)}
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


def computed_section(groups, primary, breakdown=True):
    result = dict(groups={name: counts(rows) for name, rows in groups.items()},
                  comparisons={f'{a}_vs_{b}': judgment(groups[a], groups[b], primary) for a, b in PAIRS})
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
    return result


def schedule(record, arm, *, envs=4096, steps=8000, ramp_steps=4000):
    """Independently derive 8,000 coefficients and actual reward*dt accounting."""
    finite(record)
    require(arm in ('no_cost', 'immediate', 'ramped'), 'undeclared schedule arm')
    require((envs, steps, ramp_steps) in ((256, 64, 32), (4096, 64, 4000), (4096, 8000, 4000)),
            'undeclared schedule budget')
    require(record['mode'] == record['arm'] == arm and record['num_envs'] == envs
            and record['ramp_steps'] == ramp_steps and record['policy_steps'] == steps, 'training schedule shape differs')
    keys = ('common_step_counters', 'coefficients', 'raw_reward_mean', 'scaled_reward_mean',
            'valid_rows', 'step_dt', 'raw_reward_dt_sum', 'scaled_reward_dt_sum')
    require(all(isinstance(record[k], list) and len(record[k]) == steps for k in keys), 'incomplete reward-time sequence')
    factors = []
    for i in range(steps):
        step = record['common_step_counters'][i]
        require(type(step) is int and step == i + 1, 'global training clock reset/skipped')
        expected = 0. if arm == 'no_cost' else 1. if arm == 'immediate' else min(i / (ramp_steps - 1), 1.)
        factors.append(expected)
        require(type(record['coefficients'][i]) in (int, float) and record['coefficients'][i] == expected,
                'actual coefficient trajectory differs')
        raw, scaled = record['raw_reward_mean'][i], record['scaled_reward_mean'][i]
        require(type(raw) in (int, float) and type(scaled) in (int, float)
                and -1 <= raw <= 0 and -1 <= scaled <= 0, 'reward outside declared bounds')
        close(scaled, raw * expected, 'coefficient not applied to same raw reward', tolerance=2.e-6, absolute=2.e-7)
        valid = record['valid_rows'][i]
        require(type(valid) is int and envs * .99 <= valid <= envs, 'excess invalid reward rows')
        close(record['step_dt'][i], 1 / 60, 'reward dt differs')
        for name, mean in (('raw', raw), ('scaled', scaled)):
            close(record[name + '_reward_dt_sum'][i], mean * envs / 60,
                  'realized reward dt accounting differs', tolerance=2.e-6, absolute=2.e-7)
        if arm == 'no_cost':
            require(scaled == record['scaled_reward_dt_sum'][i] == 0., 'no-cost penalty is not exactly zero')
    mass = math.fsum(factors)
    close(record['coefficient_sum'], mass, 'coefficient-step mass differs', tolerance=1.e-9)
    realized = -math.fsum(record['scaled_reward_dt_sum'])
    close(record['realized_penalty_sum'], realized, 'realized penalty sum differs', tolerance=1.e-9)
    if arm == 'no_cost':
        require(record['coefficient_sum'] == record['realized_penalty_sum'] == 0., 'no-cost totals not exactly zero')
    proof = record['episode_randomization']
    require(proof['matches_raw_environment'] is True and digest(proof['assigned_sha256']) == digest(proof['actual_sha256'])
            and type(proof['nonzero']) is int and 0 < proof['nonzero'] <= envs
            and type(proof['minimum']) is int and type(proof['maximum']) is int
            and 0 <= proof['minimum'] < proof['maximum'], 'random horizons did not reach actual environment')
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


def paired_training(initial, reference, schedule_raw, reference_schedule):
    initialization(initial, True)
    initialization(reference, True)
    for key in ('initial_state_sha256', 'initial_prefix_sha256', 'policy_state_sha256',
                'rng_sha256', 'num_envs', 'dt', 'normalized_parameters'):
        require(initial[key] == reference[key], 'historical training initialization differs: ' + key)
    require(schedule_raw['episode_randomization'] == reference_schedule['episode_randomization'],
            'historical real randomized horizons differ')


def training_record(root, record, arm, *, envs=4096, iterations=250, ramp_steps=4000):
    initial, schedule_raw, rollout = [linked(root, record[key]) for key in ('initial', 'schedule', 'rollout')]
    require(initial['arm'] == arm and initial['num_envs'] == envs and initial['iteration'] == 0
            and initial['common_step_counter'] == 0 and initial['optimizer_state_empty'] is True
            and initial['contact_slip_weight'] == 1. and initial['observation_dimensions'] == [envs, 91]
            and all(initial['initial_policy_parity'][k] is True for k in ('actor', 'critic', 'teacher')),
            'training start differs')
    initialization(initial, True)
    for name, expected in initial['saved_parameter_sha256'].items():
        require(sha(inside(root, initial['log_dir']) / 'params' / (name + '.yaml')) == expected,
                'actual training YAML changed')
    result = schedule(schedule_raw, arm, envs=envs, steps=iterations * 32, ramp_steps=ramp_steps)
    for key, value in result.items():
        close(record[key], value, 'training validation total differs', tolerance=1.e-9)
    require(rollout['arm'] == arm and rollout['passive_post_action'] is True
            and rollout['rollout']['samples'] == iterations * 32, 'passive contact rollout budget differs')
    log = record['learning_log']
    hashes(root, {log['text_log']: log['text_log_sha256'], **log['event_files_sha256']})
    text = inside(root, log['text_log']).read_text()
    require([(int(a), int(b)) for a, b in re.findall(r'Learning iteration (\d+)/(\d+)', text)] ==
            [(i, iterations) for i in range(iterations)], 'actual learning iteration sequence differs')
    require([int(v) for v in re.findall(r'Total timesteps:\s*(\d+)', text)] ==
            [(i + 1) * envs * 32 for i in range(iterations)], 'actual learning transition sequence differs')
    require(log['iterations_logged'] == iterations and log['transitions_logged'] == envs * iterations * 32
            and log['all_scalars_finite'] is True and log['scalar_counts']
            and all(v == iterations for v in log['scalar_counts'].values()), 'training scalar manifest budget differs')
    return result, initial, schedule_raw, rollout


def training(root, directory, cache_root, *, reused=False):
    frozen, trained, validation = [read(directory / name) for name in
                                  ('training_frozen.json', 'trained_models.json', 'training_validation.json')]
    arms = ('immediate', 'ramped') if reused else ('no_cost',)
    require((frozen['seed'], frozen['geometry'], frozen['envs'], frozen['iterations_per_arm'],
             frozen['steps_per_env'], frozen['ramp_steps'], frozen['transitions_per_arm']) ==
            (51, 110, 4096, 250, 32, 4000, TRANSITIONS) and frozen['arms'] == list(arms), 'frozen training budget differs')
    hashes(root, frozen['source_sha256'])
    for entries in frozen['legacy'].values():
        hashes(root, entries)
    require(frozen['source_checkpoint_sha256'] == PARENT_SHA, 'training parent differs')
    hashes(root, {frozen['starting_checkpoint']: frozen['starting_checkpoint_sha256']})
    require(frozen['starting_checkpoint_sha256'] == INIT_SHA, 'starting checkpoint is not byte-identical v20 init')
    for key, name in (('preflight_sha256', 'preflight.json'), ('capacity_sha256', 'capacity.json'),
                      ('initial_shared_sha256', 'initial_shared.json'), ('training_cache_sha256', 'training_cache.json')):
        require(frozen[key] == sha(directory / name), 'training freeze proof changed')
    require(frozen['terrain_cache']['geometries'] == [110], 'training geometry cache differs')
    cache_hashes(frozen['terrain_cache'], cache_root)
    require(trained['total_training_transitions'] == len(arms) * TRANSITIONS
            and trained['training_freeze_sha256'] == sha(directory / 'training_frozen.json')
            and trained['training_validation_sha256'] == sha(directory / 'training_validation.json')
            and set(trained['models']) == set(validation['records']) == set(arms)
            and validation['paired_initialization' if reused else 'historical_initial_pairing'] is True
            and validation['actual_random_horizons_paired'] is True, 'training manifests differ')
    result, previous = {}, None
    old_dir = root / 'artifacts/terrain_demo/contact_curriculum_v20'
    old_record = read(old_dir / 'training_validation.json')['records']['immediate']
    old_initial, old_schedule = [linked(root, old_record[key]) for key in ('initial', 'schedule')]
    if not reused:
        require(frozen['terrain_cache'] == read(old_dir / 'training_cache.json')['cache'],
                'new training cache differs from historical v20')
        references = historical_development(root, directory, frozen)
        require(validation['historical_references_sha256'] == sha(directory / 'historical_references.json'),
                'main historical reference link differs')
        require(all(references['stages']['main'][key] == old_record[key] for key in ('initial', 'schedule', 'rollout')),
                'historical main reference substituted')
    for arm in arms:
        rec, model = validation['records'][arm], trained['models'][arm]
        metrics, initial, schedule_raw, _ = training_record(root, rec, arm)
        paired_training(initial, old_initial, schedule_raw, old_schedule)
        if previous is not None:
            paired_training(initial, previous[0], schedule_raw, previous[1])
        previous = initial, schedule_raw
        require(model['iteration'] == 249 and model['transitions'] == TRANSITIONS
                and model['initial_audit_sha256'] == rec['initial']['sha256']
                and model['schedule_sha256'] == rec['schedule']['sha256'], 'final checkpoint provenance differs')
        hashes(root, {model['checkpoint']: model['sha256'], model['source_checkpoint']: model['sha256']})
        result[arm] = metrics
    return result


def historical_development(root, directory, frozen):
    references = read(directory / 'historical_references.json')
    require(frozen['historical_references_sha256'] == sha(directory / 'historical_references.json')
            and set(references['stages']) == {'preflight', 'capacity', 'main'}, 'historical pin linkage differs')
    hashes(root, references['sha256'])
    for stage in references['stages'].values():
        require(set(stage) == {'initial', 'schedule', 'rollout', 'checkpoint'}, 'historical stage inventory differs')
        for item in stage.values():
            require(references['sha256'].get(item['path']) == item['sha256'], 'historical reference not in pin')
    total = 0
    for phase, envs, ramp in (('preflight', 256, 32), ('capacity', 4096, 4000)):
        proof = read(directory / (phase + '.json'))
        require(proof['phase'] == phase and proof['passed'] is True and proof['development_only'] is True
                and proof['historical_initial_pairing'] is True and proof['actual_random_horizons_paired'] is True
                and proof['source_sha256'] == frozen['source_sha256'] and proof['terrain_cache'] == frozen['terrain_cache']
                and set(proof['records']) == {'immediate', 'no_cost'}
                and proof['historical_references_sha256'] == sha(directory / 'historical_references.json'),
                'positive-control development proof differs')
        reference = references['stages'][phase]
        old_initial, old_schedule, old_rollout = [linked(root, reference[key]) for key in ('initial', 'schedule', 'rollout')]
        old_manifest = read(root / 'artifacts/terrain_demo/contact_curriculum_v20' / (phase + '.json'))
        require(inside(root, reference['checkpoint']['path']) == inside(root, old_initial['log_dir']) / 'model_1.pt',
                'historical positive-control checkpoint was substituted')
        require(all(reference[key] == old_manifest['records']['immediate'][key] for key in ('initial', 'schedule', 'rollout')),
                'historical development reference substituted')
        for arm in ('immediate', 'no_cost'):
            _, initial, schedule_raw, rollout = training_record(root, proof['records'][arm], arm,
                                                               envs=envs, iterations=2, ramp_steps=ramp)
            paired_training(initial, old_initial, schedule_raw, old_schedule)
            require(datetime.fromisoformat(references['created_utc']) < datetime.fromisoformat(initial['created_utc']),
                    'historical references were not pinned before replay')
            if arm == 'immediate':
                require(schedule_raw == old_schedule and rollout['rollout'] == old_rollout['rollout'],
                        'positive-control exact reward/contact replay differs')
            total += envs * 64
        replay = proof['positive_control_exact_replay']
        require(replay['exact_model_optimizer_infos'] is True and replay['exact_reward_contact'] is True
                and replay['historical_checkpoint'] == reference['checkpoint'], 'positive-control checkpoint proof differs')
        new_initial = linked(root, proof['records']['immediate']['initial'])
        require(inside(root, replay['new_checkpoint']) == inside(root, new_initial['log_dir']) / 'model_1.pt',
                'positive-control new checkpoint was substituted')
        hashes(root, {replay['new_checkpoint']: replay['new_checkpoint_sha256']})
        tags = replay['scalars']
        new_tags = set(proof['records']['immediate']['learning_log']['scalar_counts'])
        old_tags = set(old_manifest['records']['immediate']['learning_log']['scalar_counts'])
        require(new_tags == old_tags and set(tags['exact_tags']) == new_tags - set(tags['timing_only_exclusions']),
                'positive-control scalar inventory omitted non-timing values')
        require(set(tags['exact_tags']) >= {'Loss/value_function', 'Loss/surrogate', 'Loss/prior_loss', 'Policy/mean_noise_std'}
                and set(tags['timing_only_exclusions']) == {'Perf/total_fps', 'Perf/collection time', 'Perf/learning_time',
                    'Train/mean_reward/time', 'Train/mean_episode_length/time'}, 'positive-control scalar proof differs')
    require(total == DEVELOPMENT_TRANSITIONS == 557056, 'new development transition budget differs')
    return references


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


def command_inventory(root, directory, raw_paths):
    """The completed attempt01 study has 49 successful, uniquely logged GPU jobs."""
    expected = {path.resolve(): None for path in raw_paths}
    development = read(directory / 'development.json')
    terrain = read(directory / 'terrain_cache.json')
    training_cache = read(directory / 'training_cache.json')
    preparation = {'path': training_cache['preparation'], 'sha256': training_cache['preparation_sha256']}
    for record in [preparation, development['preparation'], *development['records'], *terrain['preparation']]:
        path = inside(root, record['path'])
        require(sha(path) == record['sha256'] and path not in expected, 'duplicate/changed GPU output evidence')
        expected[path] = None
    for phase, envs, iterations, ramp in (('preflight', 256, 2, 32), ('capacity', 4096, 2, 4000),
                                         ('training_validation', 4096, 250, 4000)):
        records = read(directory / (phase + '.json'))['records']
        for arm, record in records.items():
            path = inside(root, record['initial']['path'])
            require(path not in expected, 'duplicate training GPU output evidence')
            expected[path] = (arm, envs, iterations, ramp, record)
    require(len(expected) == 49, 'declared GPU output inventory differs from attempt01 budget')
    seen, logs = set(), set()
    records = [json.loads(line, object_pairs_hook=unique_object) for line in (directory / 'commands.jsonl').read_text().splitlines()]
    require(len(records) == 49, 'unexpected extra/missing GPU commands or undeclared retry')
    for record in records:
        finite(record)
        command = record['command']
        def value(flag):
            require(command.count(flag) == 1 and command.index(flag) + 1 < len(command), 'missing/duplicate GPU command flag')
            return command[command.index(flag) + 1]
        training_job = '--audit-output' in command
        output = (root / value('--audit-output' if training_job else '--output')).resolve()
        require(output in expected and output not in seen, 'unexpected/repeated GPU output')
        log = inside(root, record['log'])
        require(log not in logs and type(record['returncode']) is int and record['returncode'] == 0
                and sha(log) == record['log_sha256'] and value('--device') == 'cuda:1', 'GPU command log/exit/device differs')
        specification = expected[output]
        require(training_job == (specification is not None), 'wrong command kind for GPU evidence')
        if specification is not None:
            arm, envs, iterations, ramp, run = specification
            require(value('--arm') == arm and int(value('--num_envs')) == envs
                    and int(value('--max_iterations')) == iterations and int(value('--ramp-steps')) == ramp
                    and int(value('--seed')) == 51 and value('--load_run') == 'v21_init'
                    and value('--checkpoint') == 'model_0.pt'
                    and (root / value('--schedule-output')).resolve() == inside(root, run['schedule']['path'])
                    and log == inside(root, run['learning_log']['text_log']), 'actual training GPU budget/provenance differs')
        seen.add(output)
        logs.add(log)
    return len(records)


def legacy_parity(record, legacy, current):
    require(record.get('equal_fields') == list(PARITY_FIELDS),
            'development parity field inventory differs')
    require(all(key in legacy and key in current for key in PARITY_FIELDS),
            'development parity evidence is missing a required field')
    require(all(legacy[key] == current[key] for key in PARITY_FIELDS),
            'development legacy parity differs')


def audit(root=ROOT, directory=None, cache_root=Path('/tmp/isaaclab/terrains'), *, compare_report=True):
    root = Path(root).resolve()
    directory = Path(directory or root / 'artifacts/terrain_demo/contact_continuation_v21').resolve()
    # Missing or incomplete inputs fail before an output file can be opened.
    summary = read(directory / 'summary.json') if compare_report else None
    frozen, inputs = [read(directory / name) for name in ('frozen.json', 'evaluation_inputs.json')]
    require(frozen['controllers'] == list(CONTROLLERS) and frozen['holdouts'] == [list(pair) for pair in MAPS]
            and frozen['new_training_transitions'] == TRANSITIONS and frozen['reused_training_transitions'] == 2 * TRANSITIONS, 'evaluation freeze inventory differs')
    if summary is not None:
        require(summary['new_training_transitions'] == TRANSITIONS and summary['reused_training_transitions'] == 2 * TRANSITIONS
                and summary['evaluation_training_transitions'] == 0, 'summary transition accounting differs')
    for entries in (frozen['source_sha256'], frozen['legacy']['sources'], frozen['legacy']['models']):
        hashes(root, entries)
    require(str(Path(__file__).resolve().relative_to(ROOT)) not in frozen['source_sha256'],
            'post-experiment verifier unexpectedly part of frozen definitions')
    for key, name in (('development_sha256', 'development.json'), ('training_freeze_sha256', 'training_frozen.json'),
                      ('training_validation_sha256', 'training_validation.json'), ('trained_models_sha256', 'trained_models.json')):
        require(frozen[key] == sha(directory / name), 'frozen evidence link changed')
        if summary is not None:
            require(summary['provenance'][key] == frozen[key], 'summary frozen evidence link changed')
    development = read(directory / 'development.json')
    require(development['source_sha256'] == frozen['source_sha256']
            and development['models'] == frozen['models']
            and {r['controller'] for r in development['records']} == set(CONTROLLERS)
            and len(development['records']) == 8, 'development inventory/source/model linkage differs')
    development_raw = {record['controller']: linked(root, record) for record in development['records']}
    reference_config = {key: development_raw['parent'][key] for key in
                        ('gate_config', 'posture_config', 'contact_config', 'v5_sha256')}
    require(reference_config['v5_sha256'] == '889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e',
            'frozen v5 teacher identity differs')
    require(development['terrain_cache']['geometries'] == [51], 'evaluation development cache geometry differs')
    cache_hashes(development['terrain_cache'], Path(cache_root))
    require(linked(root, development['preparation'])['scored_episodes'] == 0, 'evaluation preparation scored episodes')
    first = initialization(development_raw['parent'])
    development_rows, development_flat = {}, None
    for controller, value in development_raw.items():
        development_rows[controller] = raw_rows(value, controller=controller, geometry=51, reset=24, scenario='mixed', development=True)
        require(initialization(value) == first and all(value[k] == v for k, v in reference_config.items()),
                'development initialization/configuration not paired')
        if controller.startswith('history_'):
            indices = [i for i, row in enumerate(development_rows[controller]) if row['family'] == 'flat']
            flat = {key: [value[key][i] for i in indices] for key in FLAT_FIELDS}
            flat['indices'] = indices
            flat['levels'] = [value['level_indices'][i] for i in indices]
            require(development_flat is None or development_flat == flat, 'development history flat evidence differs')
            development_flat = flat
    require(sum(len(rows) for rows in development_rows.values()) == 280, 'evaluation development episode budget differs')
    require({record['controller'] for record in development['parity']} == set(CONTROLLERS) - {'no_cost', 'history_no_cost'}
            and len(development['parity']) == 6, 'six historical evaluation replays required')
    for record in development['parity']:
        legacy = linked(root, record)
        new = development_raw[record['controller']]
        legacy_parity(record, legacy, new)
    models = frozen['models']
    require(set(models) == {'parent', 'no_cost', 'immediate', 'ramped'} and models['parent']['sha256'] == PARENT_SHA,
            'evaluation model inventory differs')
    require({'no_cost': models['no_cost']} == read(directory / 'trained_models.json')['models'],
            'training/evaluation model manifests differ')
    for model in models.values():
        hashes(root, {model['checkpoint']: model['sha256']})
    old_dir = root / 'artifacts/terrain_demo/contact_curriculum_v20'
    require({k: models[k] for k in ('immediate', 'ramped')} == read(old_dir / 'trained_models.json')['models'],
            'historical cost-arm models were replaced')
    training_result = training(root, directory, Path(cache_root))
    training_result['reused_v20'] = training(root, old_dir, Path(cache_root), reused=True)
    if summary is not None:
        for arm, metrics in training_result.items():
            if arm == 'reused_v20':
                for old_arm, values in metrics.items():
                    for key, value in values.items():
                        close(summary['training'][arm][old_arm][key], value, 'summary reused training differs', tolerance=1.e-9)
            else:
                for key, value in metrics.items():
                    close(summary['training'][arm][key], value, 'summary new training differs', tolerance=1.e-9)
    for key, name in (('experiment_freeze_sha256', 'frozen.json'), ('terrain_cache_manifest_sha256', 'terrain_cache.json'),
                      ('preholdout_ledger_sha256', 'preholdout_ledger.json')):
        require(inputs[key] == sha(directory / name), 'evaluation input link differs')
    cache = read(directory / 'terrain_cache.json')
    require(cache['cache']['geometries'] == [113, 114] and len(cache['preparation']) == 2, 'holdout cache inventory differs')
    cache_hashes(cache['cache'], Path(cache_root))
    for record in cache['preparation']:
        require(linked(root, record)['scored_episodes'] == 0, 'terrain preparation scored episodes')
    provenance = {key: sha(directory / name) for key, name in (
        ('experiment_freeze_sha256', 'frozen.json'), ('evaluation_inputs_sha256', 'evaluation_inputs.json'),
        ('terrain_cache_manifest_sha256', 'terrain_cache.json'))}
    if summary is not None:
        require(all(summary['provenance'][key] == val for key, val in provenance.items()), 'summary raw provenance differs')
    raw_paths, inventory, paired_initial, flat_evidence, scores = {}, [], {}, {}, {}
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
                        and data['evaluation_plan_sha256'] == sha(root / 'docs/experiment_plans/contact_continuation_v21.md'),
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
        scores[folder] = computed_section(groups, scenario == 'mixed')
        scores[folder]['per_map'] = {name: computed_section(selected, scenario == 'mixed') for name, selected in per_map.items()}
        if summary is not None:
            compare_section(groups, summary[folder], scenario == 'mixed')
            require(set(summary[folder]['per_map']) == set(per_map), 'summary per-map inventory differs')
            for name, selected in per_map.items():
                compare_section(selected, summary[folder]['per_map'][name], scenario == 'mixed')
    require(len(inventory) == 32 and sum(row['episodes'] for row in inventory) == 2960, 'incomplete scored budget')
    if summary is not None:
        require(summary['files'] == inventory, 'summary raw file inventory/digests differ')
    commands = ledger(root, directory, inputs, raw_paths)
    gpu_commands = command_inventory(root, directory, raw_paths)
    return dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), raw_files=32, first_episodes=2960,
                new_main_training_transitions=TRANSITIONS, reused_main_training_transitions=2 * TRANSITIONS,
                new_development_transitions=DEVELOPMENT_TRANSITIONS,
                new_total_training_environment_transitions=TRANSITIONS + DEVELOPMENT_TRANSITIONS,
                evaluation_development_first_episodes=280, report_comparison=compare_report, scores=scores,
                scored_commands=commands, successful_gpu_commands=gpu_commands, training=training_result, summary_sha256=sha(directory / 'summary.json') if summary is not None else None,
                experiment_freeze_sha256=sha(directory / 'frozen.json'), verifier_sha256=sha(Path(__file__)),
                verifier_scope='Post-experiment verification code, not part of frozen experimental definitions.',
                verified='Independent strict counts, all twelve descriptive gates/paired outcomes at pooled/map/family/level, provenance, clocks and reward accounting.',
                limitations='Stdlib verifier hashes checkpoint/TensorBoard bytes but does not deserialize tensors or decode TensorBoard scalars. Positive-control tensor/scalar equality remains a hash-linked recorded proof; JSON reward/contact equality is independently checked. Contact/posture geometry telemetry is not recomputed. The command budget verifies the successful attempt01 study; an additional authorized attempt needs an explicit verifier update, not silent acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'artifacts/terrain_demo/contact_continuation_v21')
    parser.add_argument('--cache-root', type=Path, default=Path('/tmp/isaaclab/terrains'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-summary', action='store_true', help='Compute raw evidence without requiring/comparing summary.json')
    args = parser.parse_args()
    output = args.output or args.directory / 'independent_raw_audit.json'
    require(not output.exists(), 'refusing to replace existing independent audit')
    result = audit(directory=args.directory, cache_root=args.cache_root, compare_report=not args.skip_summary)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('passed', 'raw_files', 'first_episodes', 'report_comparison', 'verifier_sha256')}, indent=2))


if __name__ == '__main__':
    main()
