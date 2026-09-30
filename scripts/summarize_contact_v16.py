"""Audit v16 raw first-episode evidence and compare the predeclared paired arms."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

from week03_ant.contact_telemetry import aggregate_contact, audit_contact
from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from week03_ant.contact_study import (
    ARMS, CONTROLLERS, HOLDOUTS, ROOT, SCHEMA, SOURCE_FILES, TRAINING_SOURCES,
    START_SHA, V15_SHA, V5_SHA, TASK, evaluation_inputs, legacy_hashes,
)

_SPEC = importlib.util.spec_from_file_location('v16_frozen_posture_summary', ROOT / 'scripts/summarize_posture_v13.py')
POSTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(POSTURE)
HISTORY = POSTURE.HISTORY
FAMILIES = POSTURE.FAMILIES
improvement = POSTURE.improvement

PROJECTED_CONTROLLERS = {
    'v15_control': 'adaptive', 'control': 'control', 'slip': 'adaptive',
    'history_original': 'history_original', 'history_control': 'history_control',
    'history_slip': 'history_adaptive',
}
FLAT_IDENTITY_FIELDS = (
    'episode_return', 'forward_distance', 'maximum_distance_m', 'episode_lengths',
    'episode_active_steps', 'episode_terminated', 'episode_out_of_lane', 'episode_world_exit',
    'episode_strict_one_tile_success', 'episode_strict_all_tiles_success',
    'episode_full_horizon_survival', 'first_hit_one_seconds', 'first_hit_six_seconds',
    'distance_at_snapshot_m', 'episode_v10_target_steps', 'episode_v10_duty',
    'episode_alpha_sum', 'episode_switch_count', 'episode_switch_steps',
    'episode_switch_to_v10', 'history_switch_events', 'episode_uncertain_steps',
    'episode_fall_within_switch_window', 'episode_max_action_jump_rms',
)
CONTACT_CONFIG = {'force_threshold_n': 2.0, 'speed_scale_m_s': 1.0, 'contact_targets': 2}


def _inside(name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('provenance path must be repository-relative')
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('provenance path escapes repository')
    return path


def audit(data):
    """Validate v16 metadata before projecting a copy to frozen physical scoring."""
    controller = data.get('controller')
    if data.get('schema') != SCHEMA or data.get('task') != TASK or controller not in CONTROLLERS:
        raise ValueError('unexpected v16 task/schema/controller')
    expert = controller.removeprefix('history_')
    mode = None if expert == 'original' else 'conditioned'
    condition = data.get('condition', {})
    if (condition.get('observations') != 91 or type(condition.get('observations')) is not int
            or condition.get('expert_observations') != (91 if mode else 88)
            or type(condition.get('expert_observations')) is not int
            or condition.get('v5_observations') != 60 or condition.get('actions') != 8
            or data.get('command_mode') != mode or data.get('command_schema_version') != 1
            or type(data.get('command_schema_version')) is not int):
        raise ValueError('changed conditioned policy/observation contract')
    if data.get('contact_config') != CONTACT_CONFIG:
        raise ValueError('changed v16 contact metric configuration')
    HISTORY._hashes({'plan': data.get('evaluation_plan_sha256')})
    HISTORY._hashes(data.get('initial_state_sha256'), HISTORY.INITIAL_KEYS)
    HISTORY._hashes({'prefix': data.get('initial_prefix_sha256')})
    HISTORY._hashes(data.get('initial_rng_sha256'), ('cpu', 'cuda'))
    if data.get('phase') not in ('smoke', 'holdout') or data.get('scenario') not in ('mixed', 'stones'):
        raise ValueError('invalid v16 scoring phase/scenario')
    if data['phase'] == 'holdout':
        HISTORY._hashes({key: data.get(key) for key in (
            'experiment_freeze_sha256', 'evaluation_inputs_sha256', 'terrain_cache_manifest_sha256')})
        if ((data.get('geometry_seed'), data.get('reset_seed')) not in HOLDOUTS
                or type(data.get('num_envs')) is not int
                or data['num_envs'] != (175 if data['scenario'] == 'mixed' else 10)):
            raise ValueError('changed holdout geometry/reset/count')
    projected = deepcopy(data)
    projected.update(schema=POSTURE.SCHEMA, task=POSTURE.TASK,
                     controller=PROJECTED_CONTROLLERS[controller], phase='smoke')
    projected['condition']['observations'] = 88
    rows = POSTURE.audit(projected)
    audit_contact(data, rows)
    return rows


def aggregate(rows):
    result = POSTURE.aggregate(rows)
    result['contact'] = aggregate_contact(rows)
    return result


def _paired(initial, pair, data):
    state = {key: data[key] for key in ('initial_state_sha256', 'initial_prefix_sha256', 'initial_rng_sha256')}
    if pair in initial and initial[pair] != state:
        raise ValueError('paired controllers differ in full initial state/prefix/RNG')
    initial[pair] = deepcopy(state)


def _flat_identity(initial, pair, data, rows):
    indices = [i for i, row in enumerate(rows) if row['family'] == 'flat']
    evidence = {'indices': indices, 'levels': [data['level_indices'][i] for i in indices]}
    for key in FLAT_IDENTITY_FIELDS:
        values = data.get(key)
        if not isinstance(values, list) or len(values) != len(rows):
            raise ValueError(f'missing per-environment flat identity field: {key}')
        evidence[key] = [values[i] for i in indices]
    if pair in initial and initial[pair] != evidence:
        raise ValueError('fixed-v5 flat outcomes/routing differ across history hybrids')
    initial[pair] = deepcopy(evidence)


def _verify_sources(frozen):
    HISTORY._hashes(frozen.get('source_sha256'), SOURCE_FILES)
    HISTORY._hashes(frozen['legacy']['sources'], count=123)
    HISTORY._hashes(frozen['legacy']['models'], count=10)
    if frozen['legacy'] != legacy_hashes():
        raise ValueError('v16 prior frozen reference inventory differs')
    for name, digest in {**frozen['source_sha256'], **frozen['legacy']['sources'], **frozen['legacy']['models']}.items():
        if HISTORY.sha(_inside(name)) != digest:
            raise ValueError(f'frozen provenance bytes changed: {name}')


def _verify_training(directory, frozen):
    training_path = directory / 'training_frozen.json'
    if HISTORY.sha(training_path) != frozen['training_freeze_sha256']:
        raise ValueError('training freeze digest mismatch')
    training = json.loads(training_path.read_text())
    HISTORY._hashes(training['source_sha256'], SOURCE_FILES)
    if (training['legacy'] != frozen['legacy']
            or any(frozen['source_sha256'].get(k) != v for k, v in training['source_sha256'].items())
            or training['source_checkpoint_sha256'] != V15_SHA
            or training['arms'] != list(ARMS)):
        raise ValueError('training source/reference/arm contract changed')
    for key, value in dict(training_seed=48, geometry=98, iterations_per_arm=250,
                           envs=4096, steps_per_env=32, transitions_per_arm=32768000).items():
        if type(training.get(key)) is not int or training[key] != value:
            raise ValueError('training budget/seed changed')
    if HISTORY.sha(_inside(training['starting_checkpoint'])) != training['starting_checkpoint_sha256']:
        raise ValueError('shared initial checkpoint changed')
    for name in ('capacity.json', 'preflight.json'):
        key = name.replace('.json', '_sha256')
        if HISTORY.sha(directory / name) != training[key]:
            raise ValueError(f'{name} changed')
    trained_path = directory / 'trained_models.json'
    if HISTORY.sha(trained_path) != frozen['trained_models_sha256']:
        raise ValueError('trained model manifest changed')
    trained = json.loads(trained_path.read_text())
    if (trained.get('training_freeze_sha256') != frozen['training_freeze_sha256']
            or trained.get('total_training_transitions') != 65536000
            or trained.get('models') != {arm: frozen['models'][arm] for arm in ARMS}):
        raise ValueError('trained model/budget contract changed')
    if HISTORY.sha(directory / 'training_validation.json') != trained['training_validation_sha256']:
        raise ValueError('training validation evidence changed')
    validation = json.loads((directory / 'training_validation.json').read_text())
    if set(validation.get('records', {})) != set(ARMS):
        raise ValueError('missing training validation arms')
    for arm in ARMS:
        record = validation['records'][arm]
        if (record.get('iterations_logged') != 250 or record.get('transitions_logged') != 32768000
                or record.get('all_scalars_finite') is not True):
            raise ValueError('incomplete/nonfinite training log')
        if HISTORY.sha(_inside(record['text_log'])) != record['text_log_sha256']:
            raise ValueError('training text log changed')
        for path, digest in record['event_files_sha256'].items():
            if HISTORY.sha(_inside(path)) != digest:
                raise ValueError('training scalar file changed')
        initial = directory / 'training' / f'{arm}_initial.json'
        if HISTORY.sha(initial) != frozen['models'][arm]['initial_audit_sha256']:
            raise ValueError('training initial audit changed')
    return trained


def audit_development_proofs(directory, frozen, models):
    """Re-open every frozen smoke/parity byte; flags alone are not evidence."""
    directory = Path(directory)
    smokes_path, parity_path = directory / 'final_model_smokes.json', directory / 'evaluator_parity.json'
    if (HISTORY.sha(smokes_path) != frozen['final_model_smokes_sha256']
            or HISTORY.sha(parity_path) != frozen['evaluator_parity_sha256']):
        raise ValueError('frozen smoke/parity manifest digest changed')
    smokes, parity = json.loads(smokes_path.read_text()), json.loads(parity_path.read_text())
    records = smokes.get('records')
    if (smokes.get('passed') is not True or smokes.get('development_only') is not True
            or smokes.get('exact_initial_pairing') is not True
            or smokes.get('v15_reference_parity_sha256') != frozen['evaluator_parity_sha256']
            or not isinstance(records, list) or len(records) != len(CONTROLLERS)
            or {record.get('controller') for record in records} != set(CONTROLLERS)):
        raise ValueError('final smoke proof inventory/flags changed')
    if (parity.get('exact_reference_parity') is not True or parity.get('development_only') is not True
            or parity.get('two_live_collision_prims') != [
                '/World/ground/terrain/mesh', '/World/flatPlane/GroundPlane/CollisionPlane']):
        raise ValueError('v15 sensor evaluation parity proof changed')
    old_path, new_path, probe_path = (_inside(parity[key]) for key in
                                      ('reference_path', 'new_path', 'sensor_probe_path'))
    for path, key in ((old_path, 'reference_sha256'), (new_path, 'new_sha256'),
                      (probe_path, 'sensor_probe_sha256')):
        if HISTORY.sha(path) != parity[key]:
            raise ValueError(f'evaluator parity raw evidence changed: {path}')
    old, new, probe = (json.loads(path.read_text()) for path in (old_path, new_path, probe_path))
    fields = parity.get('equal_fields')
    if (not isinstance(fields, list) or len(fields) < 10 or len(fields) != len(set(fields))
            or any(field not in old or field not in new or old[field] != new[field] for field in fields)
            or new.get('checkpoint_sha256') != V15_SHA
            or new.get('controller') != 'v15_control'):
        raise ValueError('v15-control raw physical/parity fields differ')
    if (probe.get('configured_targets') != parity['two_live_collision_prims']
            or probe.get('collision_prims') != parity['two_live_collision_prims']
            or any(not any(measure['per_family'][family] > 0 for measure in probe['measures'])
                   for family in FAMILIES)):
        raise ValueError('development sensor probe did not cover both collision targets/families')
    initial = {}
    for record in records:
        if set(record) != {'path', 'sha256', 'controller', 'checkpoint_sha256'}:
            raise ValueError('final smoke record shape changed')
        path = _inside(record['path'])
        if HISTORY.sha(path) != record['sha256']:
            raise ValueError('final smoke raw result changed')
        data = json.loads(path.read_text())
        controller = record['controller']
        model = models[controller.removeprefix('history_')]
        if (data.get('controller') != controller
                or record['checkpoint_sha256'] != model['sha256']
                or data.get('checkpoint') != model['checkpoint']
                or data.get('checkpoint_sha256') != model['sha256']
                or data.get('evaluation_plan_sha256') != frozen['source_sha256']['docs/experiment_plans/contact_slip_v16.md']
                or (data.get('phase'), data.get('scenario'), data.get('geometry_seed'),
                    data.get('reset_seed'), data.get('num_envs'), data.get('condition', {}).get('seconds'))
                   != ('smoke', 'mixed', 51, 24, 35, 16)):
            raise ValueError('final smoke model/condition/plan differs')
        audit(data)
        _paired(initial, (51, 24), data)
    if initial[(51, 24)] != {key: new[key] for key in
                              ('initial_state_sha256', 'initial_prefix_sha256', 'initial_rng_sha256')}:
        raise ValueError('final smoke initial state differs from frozen v15 parity')
    if not HISTORY._timestamp(smokes['created_utc']) <= HISTORY._timestamp(frozen['frozen_at']):
        raise ValueError('smoke proof postdates evaluation freeze')
    return {'smokes_sha256': frozen['final_model_smokes_sha256'],
            'parity_sha256': frozen['evaluator_parity_sha256'],
            'six_smokes_verified': True, 'v15_physical_parity_verified': True}


def summarize(directory):
    directory = Path(directory)
    frozen_path = directory / 'frozen.json'
    frozen = json.loads(frozen_path.read_text())
    if frozen['holdouts'] != [list(pair) for pair in HOLDOUTS] or frozen['controllers'] != list(CONTROLLERS):
        raise ValueError('predeclared holdout/controller inventory changed')
    if frozen['gate_config'] != asdict(HistoryGateConfig()) or frozen['posture_config'] != asdict(PostureConfig()):
        raise ValueError('frozen history/posture config changed')
    if frozen.get('contact_config') != CONTACT_CONFIG:
        raise ValueError('frozen contact config changed')
    _verify_sources(frozen)
    trained = _verify_training(directory, frozen)
    models = frozen['models']
    if set(models) != {'original', 'v15_control', *ARMS} or models['original']['sha256'] != START_SHA or models['v15_control']['sha256'] != V15_SHA:
        raise ValueError('frozen checkpoint inventory changed')
    for name, item in models.items():
        if HISTORY.sha(_inside(item['checkpoint'])) != item['sha256']:
            raise ValueError(f'frozen model changed: {name}')
        if name in ARMS and (item['iteration'] != 249 or item['transitions'] != 32768000):
            raise ValueError('trained model not final249')
    development = audit_development_proofs(directory, frozen, models)
    for label, expected in (('primary', dict(files=12, episodes=2100, seconds=16, envs=175)),
                            ('secondary', dict(files=12, episodes=120, seconds=64, envs=10))):
        if frozen.get(label) != expected:
            raise ValueError('predeclared evaluation inventory changed')
    frozen_sha = HISTORY.sha(frozen_path)
    inputs = evaluation_inputs(directory)
    input_sha = HISTORY.sha(directory / 'evaluation_inputs.json')
    cache_sha = inputs['terrain_cache_manifest_sha256']
    if inputs['experiment_freeze_sha256'] != frozen_sha:
        raise ValueError('evaluation freeze linkage changed')
    # The frozen prefix covers all development runs, final smokes and cache preparation.
    from summarize_direction_v15 import audit_preholdout_ledger
    ledger = audit_preholdout_ledger(directory, inputs)
    result = {'files': [], 'provenance': {
        'experiment_freeze_sha256': frozen_sha,
        'training_freeze_sha256': frozen['training_freeze_sha256'],
        'trained_models_sha256': frozen['trained_models_sha256'],
        'training_validation_sha256': trained['training_validation_sha256'],
        'evaluation_inputs_sha256': input_sha,
        'terrain_cache_manifest_sha256': cache_sha,
        'preholdout_ledger': ledger,
        'development_proofs': development,
    }}
    for scenario, folder, envs, seconds in (('mixed', 'evaluations', 175, 16), ('stones', 'horizon', 10, 64)):
        groups, initial, flats = defaultdict(list), {}, {}
        expected_paths = {f'{c}__geometry{g}_reset{r}.json' for c in CONTROLLERS for g, r in HOLDOUTS}
        if {path.name for path in (directory / folder).glob('*.json')} != expected_paths:
            raise ValueError('missing or unexpected evaluation file')
        for controller in CONTROLLERS:
            model = models[controller.removeprefix('history_')]
            for geometry, reset in HOLDOUTS:
                path = directory / folder / f'{controller}__geometry{geometry}_reset{reset}.json'
                data = json.loads(path.read_text())
                actual = (data.get('controller'), data.get('geometry_seed'), data.get('reset_seed'),
                          data.get('scenario'), data.get('num_envs'), data.get('condition', {}).get('seconds'), data.get('phase'))
                if actual != (controller, geometry, reset, scenario, envs, seconds, 'holdout'):
                    raise ValueError('evaluation condition differs from freeze')
                if data.get('checkpoint') != model['checkpoint'] or data.get('checkpoint_sha256') != model['sha256']:
                    raise ValueError('evaluation checkpoint differs from freeze')
                if (data.get('evaluation_plan_sha256') != frozen['source_sha256']['docs/experiment_plans/contact_slip_v16.md']
                        or data.get('experiment_freeze_sha256') != frozen_sha
                        or data.get('evaluation_inputs_sha256') != input_sha
                        or data.get('terrain_cache_manifest_sha256') != cache_sha):
                    raise ValueError('evaluation source/input/cache linkage differs')
                if HISTORY._timestamp(data['started_utc']) <= HISTORY._timestamp(inputs['created_utc']):
                    raise ValueError('evaluation predates pinned inputs')
                rows = audit(data)
                expected_counts = ({(family, level): 5 for family in FAMILIES for level in range(5)}
                                   if scenario == 'mixed' else {('stepping_stones', 4): 10})
                if Counter((row['family'], row['level']) for row in rows) != expected_counts:
                    raise ValueError('family/difficulty inventory changed')
                _paired(initial, (geometry, reset), data)
                if scenario == 'mixed' and controller.startswith('history_'):
                    _flat_identity(flats, (geometry, reset), data, rows)
                groups[controller].extend(rows)
                result['files'].append({'path': str(path.relative_to(directory)), 'sha256': HISTORY.sha(path), 'episodes': len(rows)})
        section = {
            'groups': {name: aggregate(rows) for name, rows in groups.items()},
            'family': {name: {family: aggregate([row for row in rows if row['family'] == family])
                               for family in sorted({row['family'] for row in rows})} for name, rows in groups.items()},
            'family_level': {name: {f'{family}/level{level}': aggregate([row for row in rows if (row['family'], row['level']) == (family, level)])
                                     for family, level in sorted({(row['family'], row['level']) for row in rows})}
                             for name, rows in groups.items()},
        }
        section['actor_slip_vs_control'] = improvement(section['groups']['slip'], section['groups']['control'], primary=scenario == 'mixed')
        section['hybrid_slip_vs_control'] = improvement(section['groups']['history_slip'], section['groups']['history_control'],
                                                         hybrid=True, primary=scenario == 'mixed')
        if scenario == 'mixed':
            identity = {'verified': len(flats) == len(HOLDOUTS), 'maps': len(flats),
                        'flat_episodes_per_controller': sum(len(value['indices']) for value in flats.values()),
                        'fields': list(FLAT_IDENTITY_FIELDS)}
            section['hybrid_flat_raw_identity'] = identity
            section['hybrid_slip_vs_control']['checks']['fixed_v5_per_environment_flat_raw_identity'] = identity['verified']
            section['hybrid_slip_vs_control']['passed'] = all(section['hybrid_slip_vs_control']['checks'].values())
        section['contact_mechanism'] = {}
        for label, candidate, control in (('actor', 'slip', 'control'), ('hybrid', 'history_slip', 'history_control')):
            new, old = section['groups'][candidate]['contact'], section['groups'][control]['contact']
            section['contact_mechanism'][label] = {name: {
                'slip': new['means'][name], 'control': old['means'][name],
                'delta': (new['means'][name] - old['means'][name]) if new['means'][name] is not None and old['means'][name] is not None else None,
                'slip_valid_steps': new['sums']['valid_steps'], 'control_valid_steps': old['sums']['valid_steps'],
            } for name in ('bounded_cost', 'contact_fraction', 'contacted_tip_speed')}
        result[folder] = section
    if len(result['files']) != 24 or sum(item['episodes'] for item in result['files']) != 2220:
        raise ValueError('incomplete 24-file/2220-episode inventory')
    result['interpretation'] = ('One training seed and two fresh maps: exploratory, not universal or statistically significant. '
        'The only paired treatment is the contact-conditioned foot-tip-speed proxy, with both arms sensor-enabled. '
        'V15 control and original history hybrid are descriptive references, not substitute comparators. '
        '16s primary and 64s secondary are separate; secondary cannot override primary failure. '
        'Contact/posture metrics are conditional on visited valid states, not matched-state causality. '
        'Filtered foot contact is not whole-body airborne, and speed is not exact contact-point slip. '
        'Ideal ray depth and simulation do not establish real-camera or real-robot safety.')
    return result


def markdown(result):
    lines = ['# v16 contact-slip: matched reward comparison', '', result['interpretation'], '']
    for folder, title in (('evaluations', 'Primary: 16s mixed terrain'), ('horizon', 'Secondary: 64s hardest stones')):
        section = result[folder]
        lines += [f'## {title}', '', '| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for controller in CONTROLLERS:
            group = section['groups'][controller]
            speed = '—' if group['flat_mean_episode_speed'] is None else f"{group['flat_mean_episode_speed']:.4f}"
            lines.append(f"| {controller} | {group['one']}/{group['n']} | {group['six']}/{group['n']} | {group['falls']}/{group['n']} | {group['lane']}/{group['n']} | {group['world'] + group['flat_world']} | {group['flat_falls']}/{group['flat_n']} | {speed} |")
        for key in ('actor_slip_vs_control', 'hybrid_slip_vs_control'):
            judgment = section[key]
            lines += ['', f"{key}: **{'PASS' if judgment['passed'] else 'FAIL'}**"]
            lines += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in judgment['checks'].items()]
        lines += ['', 'Contact mechanism (slip minus control; visited-state only, not causal):']
        for label, fields in section['contact_mechanism'].items():
            parts = [f"{key}: {field['delta']:.4f}" if field['delta'] is not None else f'{key}: unavailable'
                     for key, field in fields.items()]
            lines.append(f"- {label}: " + ', '.join(parts))
        lines.append('')
    lines += ['Flat speed is distance/(first-episode steps × dt), retaining falls. '
              'See JSON for per-family/level scores and contact coverage, no-detected-foot-contact fraction, '
              'speed-cap saturation, posture metrics, provenance and exact flat-branch identity.']
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
