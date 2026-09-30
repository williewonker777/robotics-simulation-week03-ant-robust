"""Audit dependent 16s/64s windows without changing their physical condition."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
import struct
from pathlib import Path

from week03_ant import paired_horizon_study_v23 as study
from week03_ant.contact_telemetry import audit_contact
from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from week03_ant.posture_study import cache_snapshot
from summarize_contact_v16 import CONTACT_CONFIG, FAMILIES, HISTORY, POSTURE, _paired, _flat_identity
from summarize_seed_v22 import _section
from summarize_rebaseline_v19 import _finite, audit_ledger


def read(path):
    return json.loads(Path(path).read_text())


def audit(data):
    _finite(data)
    controller = data.get('controller')
    condition = data.get('condition', {})
    seconds = condition.get('seconds')
    if data.get('schema') != study.SCHEMA or data.get('task') != study.TASK or seconds not in study.WINDOWS:
        raise ValueError('invalid window identity')
    study.validate_request(controller, data.get('geometry_seed'), data.get('reset_seed'),
                           data.get('scenario'), 64, data.get('num_envs'), data.get('phase'),
                           instrumented=data.get('phase') != 'reference')
    if data['phase'] not in ('smoke', 'holdout', 'reference'):
        raise ValueError('preparation is not scored evidence')
    if (condition.get('snapshot_seconds') != (8 if seconds == 16 else 16)
            or any(type(condition.get(k)) is not int or condition[k] != v for k, v in
                   dict(observations=91, expert_observations=91, v5_observations=60, actions=8).items())
            or data.get('command_mode') != study.command_mode(controller)
            or type(data.get('command_schema_version')) is not int or data['command_schema_version'] != 1
            or data.get('mode') != study.policy_mode(controller)
            or data.get('contact_config') != CONTACT_CONFIG
            or data.get('gate_config') != asdict(HistoryGateConfig())
            or data.get('posture_config') != asdict(PostureConfig())
            or data.get('teacher_tensor_identity_verified') is not True
            or data.get('v5_sha256') != POSTURE.V5_SHA
            or data.get('default_config_parity') is not True
            or type(data.get('new_training_transitions')) is not int or data['new_training_transitions'] != 0
            or data.get('difficulties') != [.2, .4, .6, .8, 1.]):
        raise ValueError('changed frozen policy/physical contract')
    model = study.models()[study.model_key(controller)]
    seed = None if study.model_key(controller) == 'parent' else study.training_seed(study.model_key(controller))
    if (data.get('checkpoint') != model['checkpoint'] or data.get('checkpoint_sha256') != model['sha256']
            or data.get('training_seed') != seed
            or data.get('policy_training_transitions') != (0 if seed is None else model['transitions'])
            or data.get('evaluation_plan_sha256') != study.sha(study.PLAN)):
        raise ValueError('model/plan provenance differs')
    HISTORY._hashes(data.get('initial_state_sha256'), HISTORY.INITIAL_KEYS)
    HISTORY._hashes({'prefix': data.get('initial_prefix_sha256')})
    HISTORY._hashes(data.get('initial_rng_sha256'), ('cpu', 'cuda'))
    if data['phase'] == 'holdout':
        HISTORY._hashes({k: data.get(k) for k in ('experiment_freeze_sha256', 'evaluation_inputs_sha256', 'terrain_cache_manifest_sha256')})
    projected = deepcopy(data)
    projected['schema'] = HISTORY.BASE_SCHEMA
    projected['condition']['observations'] = 88
    rows = POSTURE.BASE.audit(projected)
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
            if event['step'] != data['episode_switch_steps'][i][j] or event['target_v10'] != data['episode_switch_to_v10'][i][j]:
                raise ValueError('switch evidence disagrees')
            previous = event['step']
    POSTURE._posture(data, rows)
    audit_contact(data, rows)
    if Counter((r['family'], r['level']) for r in rows) != {(f, l): data['num_envs'] // 35 for f in FAMILIES for l in range(5)}:
        raise ValueError('mixed population differs')
    return rows


def audit_bundle(bundle):
    _finite(bundle)
    if (bundle.get('schema') != study.BUNDLE_SCHEMA or type(bundle.get('instrumented')) is not bool
            or bundle.get('physical_condition') != dict(seconds=64, max_steps=3840, dt=1 / 60)):
        raise ValueError('invalid physical bundle')
    windows = bundle.get('windows', {})
    expected = {'16', '64'} if bundle['instrumented'] else {'64'}
    if set(windows) != expected:
        raise ValueError('missing/unexpected scoring windows')
    if set(bundle.get('snapshot_passivity', {})) != expected:
        raise ValueError('passivity window inventory differs')
    rows = {}
    for window, data in windows.items():
        if data.get('condition', {}).get('seconds') != int(window):
            raise ValueError('window condition mismatch')
        if not bundle['instrumented'] and (data.get('phase') != 'reference' or data.get('controller') not in ('parent', 'history_parent')):
            raise ValueError('uninstrumented mode is development-only')
        for key in ('controller', 'geometry_seed', 'reset_seed', 'phase', 'num_envs', 'scenario'):
            if key not in bundle or bundle[key] != data.get(key):
                raise ValueError('outer/window identity differs')
        rows[window] = audit(data)
        n = len(rows[window])
        cutoff = int(window) * 60
        captured = data.get('captured_step')
        if type(captured) is not int or not max(data['episode_lengths']) <= captured <= cutoff or data.get('window_seconds') != int(window):
            raise ValueError('invalid capture bounds')
        reason = data.get('capture_reason')
        if reason not in ('window_cutoff', 'all_first_episodes_finished') or (reason == 'window_cutoff' and captured != cutoff):
            raise ValueError('invalid capture reason')
        for key in ('episode_physical_done', 'episode_physical_timeout', 'episode_window_censored'):
            if not isinstance(data.get(key), list) or len(data[key]) != n or any(type(x) is not bool for x in data[key]):
                raise ValueError('missing physical/censor evidence')
        for i in range(n):
            done, timeout, censor, fall = (data[k][i] for k in ('episode_physical_done', 'episode_physical_timeout', 'episode_window_censored', 'episode_terminated'))
            if (window == '64' and not done) or (timeout and data['episode_lengths'][i] != 3840):
                raise ValueError('physical timeout/completion differs')
            if timeout != (done and not fall) or (fall and not done) or censor != (window == '16' and not done):
                raise ValueError('physical completion/censor mismatch')
            if not done and data['episode_lengths'][i] != cutoff:
                raise ValueError('unfinished first episode truncated')
            if reason == 'all_first_episodes_finished' and not done:
                raise ValueError('early capture fabricates surviving rows')
        proof = bundle.get('snapshot_passivity', {}).get(window, {})
        if proof.get('unchanged') is not True or not proof.get('before') or proof.get('before') != proof.get('after'):
            raise ValueError('missing/changed snapshot passivity proof')
        before = proof['before']
        hash_keys = ('root_state_sha256', 'joint_pos_sha256', 'joint_vel_sha256', 'episode_length_buf_sha256', 'observations_sha256', 'cpu_rng_sha256', 'cuda_rng_sha256', 'policy_state_sha256', 'gate_state_sha256', 'full_first_episode_active_sha256')
        if set(before) != set(hash_keys) | {'common_step_counter', 'policy_mode', 'policy_training'} or type(before['common_step_counter']) is not int or before['policy_mode'] != data['mode'] or before['policy_training'] is not False:
            raise ValueError('snapshot proof inventory/policy differs')
        HISTORY._hashes({k: before[k] for k in hash_keys})
    if (bundle.get('physical_steps_executed') != windows['64']['captured_step']
            or bundle.get('executed_first_episodes') != bundle['num_envs']
            or bundle.get('window_observations') != bundle['num_envs'] * len(expected)):
        raise ValueError('physical episode/window accounting differs')
    if bundle['instrumented']:
        short, long = windows['16'], windows['64']
        if short['captured_step'] > long['captured_step']:
            raise ValueError('capture order differs')
        identity = ('controller', 'geometry_seed', 'reset_seed', 'phase', 'num_envs', 'scenario', 'training_seed', 'mode',
                    'checkpoint', 'checkpoint_sha256', 'initial_state_sha256', 'initial_prefix_sha256', 'initial_rng_sha256',
                    'family_names', 'difficulties', 'family_indices', 'level_indices')
        if any(k not in short or short[k] != long.get(k) for k in identity):
            raise ValueError('window row/policy/map/initial identity differs')
        for i in range(len(rows['16'])):
            for key in ('episode_lengths', 'maximum_distance_m', 'episode_terminated', 'episode_out_of_lane', 'episode_world_exit'):
                if long[key][i] < short[key][i]:
                    raise ValueError('first-episode prefix regressed')
            for key in ('episode_switch_steps', 'episode_switch_to_v10', 'history_switch_events'):
                if long[key][i][:len(short[key][i])] != short[key][i]:
                    raise ValueError('routing event prefix changed')
            for key in ('first_hit_one_seconds', 'first_hit_six_seconds'):
                if short[key][i] is None and long[key][i] is not None and long[key][i] <= 16:
                    raise ValueError('missing first-window threshold hit')
                if short[key][i] is not None and short[key][i] != long[key][i]:
                    raise ValueError('first hit changed')
            if short['episode_physical_done'][i]:
                if rows['16'][i] != rows['64'][i]:
                    raise ValueError('completed first episode telemetry changed')
                for key, values in short.items():
                    if key.startswith('episode_') or key in ('forward_distance', 'maximum_distance_m', 'history_switch_events'):
                        if key in ('episode_window_censored', 'episode_full_horizon_survival'):
                            continue
                        if isinstance(values, list) and len(values) == len(rows['16']) and values[i] != long[key][i]:
                            raise ValueError('completed first episode changed')
            rows['16'][i]['first_six_hit'] = short['first_hit_six_seconds'][i]
            rows['64'][i]['first_six_hit'] = long['first_hit_six_seconds'][i]
    return rows


def duration_pairs(short, long):
    if len(short) != len(long):
        raise ValueError('unequal paired denominators')
    cells = dict(neither=0, late_gain=0, late_loss=0, both=0)
    reasons = dict(new_physical_termination=0, new_lane_exit=0, new_world_exit=0, final_distance_below_53_1=0)
    hits = 0
    for a, b in zip(short, long):
        if (a['family'], a['level']) != (b['family'], b['level']):
            raise ValueError('paired row identity differs')
        cell = ('both' if b['six'] else 'late_loss') if a['six'] else ('late_gain' if b['six'] else 'neither')
        cells[cell] += 1
        if cell == 'late_loss':
            for name, flag in dict(new_physical_termination=b['falls'] and not a['falls'], new_lane_exit=b['lane'] and not a['lane'],
                                   new_world_exit=b['world'] and not a['world'], final_distance_below_53_1=b['distance'] < struct.unpack('<f', struct.pack('<f', 53.1))[0]).items():
                reasons[name] += int(bool(flag))
        hits += int(b.get('first_six_hit') is not None and 16 < b['first_six_hit'] <= 64)
    return dict(n=len(short), strict_six=cells, late_loss_flags=reasons, first_six_hit_in_16_64=hits,
                flags_overlap=True, threshold_hit_is_not_strict_success=True,
                strict_six_float32_threshold=struct.unpack('<f', struct.pack('<f', 53.1))[0])


def duration_sections(short, long):
    selectors = {'all': lambda r: True, 'rough': lambda r: r['family'] != 'flat'}
    for family in FAMILIES:
        selectors[f'family/{family}'] = lambda r, f=family: r['family'] == f
        for level in range(5):
            selectors[f'family_level/{family}/{level}'] = lambda r, f=family, l=level: r['family'] == f and r['level'] == l
    for level in range(5):
        selectors[f'level/{level}'] = lambda r, l=level: r['level'] == l
    return {name: duration_pairs([r for r in short if select(r)], [r for r in long if select(r)]) for name, select in selectors.items()}


def summarize(directory=study.ART):
    directory = Path(directory)
    study.verify_frozen()
    inputs = study.evaluation_inputs()
    from run_paired_horizon_v23 import verify_development, verify_ledger
    verify_development()
    if study.sha(directory / 'frozen.json') != study.sha(study.ART / 'frozen.json'):
        raise ValueError('summary freeze differs')
    cache = read(directory / 'terrain_cache.json')
    if cache['cache'] != cache_snapshot(Path('/tmp/isaaclab/terrains'), [g for g, _ in study.HOLDOUTS]):
        raise ValueError('pinned cache changed')
    provenance = {k: study.sha(directory / f) for k, f in (('experiment_freeze_sha256', 'frozen.json'), ('evaluation_inputs_sha256', 'evaluation_inputs.json'), ('terrain_cache_manifest_sha256', 'terrain_cache.json'))}
    expected = {f'{c}__geometry{g}_reset{r}.json' for c in study.CONTROLLERS for g, r in study.HOLDOUTS}
    if {p.name for p in (directory / 'evaluations').glob('*.json')} != expected:
        raise ValueError('missing/unexpected physical bundle matrix')
    groups = {w: defaultdict(list) for w in ('16', '64')}
    maps = {w: defaultdict(dict) for w in groups}
    initial, flats, files = {}, {}, []
    for controller in study.CONTROLLERS:
        for geometry, reset in study.HOLDOUTS:
            path = directory / 'evaluations' / f'{controller}__geometry{geometry}_reset{reset}.json'
            data = read(path)
            if data.get('instrumented') is not True:
                raise ValueError('holdout requires paired windows')
            audited = audit_bundle(data)
            for window, rows in audited.items():
                raw = data['windows'][window]
                if (raw['controller'], raw['geometry_seed'], raw['reset_seed'], raw['phase']) != (controller, geometry, reset, 'holdout') or any(raw.get(k) != v for k, v in provenance.items()):
                    raise ValueError('raw input linkage differs')
                if not HISTORY._timestamp(inputs['created_utc']) < HISTORY._timestamp(raw['started_utc']) <= HISTORY._timestamp(raw['finished_utc']):
                    raise ValueError('scoring precedes input freeze')
                _paired(initial, (geometry, reset), raw)
                if controller.startswith('history_'):
                    _flat_identity(flats, (window, geometry, reset), raw, rows)
                groups[window][controller].extend(rows)
                maps[window][f'geometry{geometry}_reset{reset}'][controller] = rows
            files.append(dict(path=str(path.relative_to(directory)), sha256=study.sha(path), episodes=175, window_observations=350))
    result = dict(files=files, provenance=provenance, physical_rollouts=16, first_episodes=2800, window_observations=5600,
                  new_training_transitions=0, windows={}, paired_duration={})
    for window in groups:
        result['windows'][window] = _section(groups[window], True)
        result['windows'][window]['role'] = 'primary' if window == '16' else 'secondary descriptive'
        result['windows'][window]['per_map'] = {m: _section(g, True) for m, g in maps[window].items()}
    for controller in study.CONTROLLERS:
        result['paired_duration'][controller] = dict(pooled=duration_sections(groups['16'][controller], groups['64'][controller]),
            per_map={m: duration_sections(maps['16'][m][controller], maps['64'][m][controller]) for m in maps['16']})
    result['provenance']['ledger'] = audit_ledger(directory, files, inputs)
    verify_ledger(complete=True)
    result['interpretation'] = '2,800 first episodes, 5,600 dependent window observations; three continuation seeds on one parent and two maps. Parent counted once. No significance, causal safety, general robustness or automatic promotion; 64s cannot override 16s.'
    return result


def markdown(result):
    lines = ['# v23 paired 16s/64s first-episode windows', '', result['interpretation'], '']
    for window, section in result['windows'].items():
        lines += [f'## {window}s — {section["role"]}', '']
        for scope, group in [('pooled', section), *section['per_map'].items()]:
            lines += [f'### {scope}', '']
            for name, gate in group['comparisons'].items():
                failed = ', '.join(k for k, v in gate['checks'].items() if not v) or 'none'
                lines.append(f'{name}: {"PASS" if gate["passed"] else "FAIL"}; failed checks: {failed}; paired six: {gate["paired_six"]}.')
    lines += ['', 'All controller metrics, family/level/cell gates, per-seed ranges and matching-parent deltas, overlapping late-loss flags and interval first-hit counts are in summary.json.', '']
    return '\n'.join(lines)
