"""Audit true 91D v15 evidence; project only a copy for frozen scoring primitives."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import hashlib
import json
from pathlib import Path

from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from week03_ant.direction_math import DirectionConfig
from week03_ant.direction_telemetry import audit_direction, aggregate_direction
from week03_ant.direction_study import (
    ARMS, CONTROLLERS, HOLDOUTS, ROOT, SCHEMA, SOURCE_FILES, TRAINING_SOURCES,
    START_SHA, V14_SHA, V5_SHA, TASK, evaluation_inputs, legacy_hashes,
)

_SPEC = importlib.util.spec_from_file_location('v15_frozen_posture_summary', ROOT / 'scripts/summarize_posture_v13.py')
POSTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(POSTURE)
HISTORY = POSTURE.HISTORY
FAMILIES = POSTURE.FAMILIES

improvement = POSTURE.improvement
PROJECTED_CONTROLLERS = {
    'v14': 'adaptive', 'control': 'control', 'stable': 'adaptive',
    'history_original': 'history_original', 'history_control': 'history_control',
    'history_stable': 'history_adaptive',
}


def aggregate(rows):
    result = POSTURE.aggregate(rows)
    result['direction'] = aggregate_direction(rows)
    return result


def audit(data):
    """Validate v15 first; never mutate or relabel the underlying 91D evidence."""
    controller = data.get('controller')
    if data.get('schema') != SCHEMA or data.get('task') != TASK or controller not in CONTROLLERS:
        raise ValueError('unexpected v15 task/schema/controller')
    arm = controller.removeprefix('history_')
    expected_mode = None if arm == 'original' else 'conditioned'
    condition = data.get('condition', {})
    if (type(condition.get('observations')) is not int or condition['observations'] != 91
            or type(condition.get('expert_observations')) is not int
            or condition['expert_observations'] != (91 if expected_mode else 88)
            or type(condition.get('v5_observations')) is not int or condition['v5_observations'] != 60
            or type(condition.get('actions')) is not int or condition['actions'] != 8
            or 'command_mode' not in data or data['command_mode'] != expected_mode
            or type(data.get('command_schema_version')) is not int or data['command_schema_version'] != 1):
        raise ValueError('changed 91D command observation/mode/schema contract')
    if data.get('direction_config') != asdict(DirectionConfig()):
        raise ValueError('changed directional reward configuration')
    HISTORY._hashes({'plan': data.get('evaluation_plan_sha256')})
    HISTORY._hashes(data.get('initial_state_sha256'), HISTORY.INITIAL_KEYS)
    HISTORY._hashes({'prefix': data.get('initial_prefix_sha256')})
    HISTORY._hashes(data.get('initial_rng_sha256'), ('cpu', 'cuda'))
    if data.get('phase') not in ('smoke', 'holdout') or data.get('scenario') not in ('mixed', 'stones'):
        raise ValueError('invalid scoring phase/scenario')
    if data['phase'] == 'holdout':
        HISTORY._hashes({key: data.get(key) for key in (
            'experiment_freeze_sha256', 'evaluation_inputs_sha256', 'terrain_cache_manifest_sha256')})
        if ((data.get('geometry_seed'), data.get('reset_seed')) not in HOLDOUTS
                or type(data.get('num_envs')) is not int
                or data['num_envs'] != (175 if data['scenario'] == 'mixed' else 10)):
            raise ValueError('changed holdout geometry/reset/shape')
    # The old audit owns unchanged physical scoring, routing and posture identities.
    # Its legacy holdout inventory is deliberately bypassed ONLY after our checks.
    projected = deepcopy(data)
    projected.update(schema=POSTURE.SCHEMA, task=POSTURE.TASK,
                     controller=PROJECTED_CONTROLLERS[controller], phase='smoke')
    projected['condition']['observations'] = 88
    rows = POSTURE.audit(projected)
    audit_direction(data, rows)
    return rows


# Counterfactual unused-expert disagreement is intentionally excluded: it can
# change while the fixed-v5 action and every physical outcome remain identical.
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


def _flat_identity(initial, pair, data, rows):
    indices = [i for i, row in enumerate(rows) if row['family'] == 'flat']
    evidence = {'indices': indices, 'levels': [data['level_indices'][i] for i in indices]}
    for key in FLAT_IDENTITY_FIELDS:
        values = data.get(key)
        if not isinstance(values, list) or len(values) != len(rows):
            raise ValueError(f'missing per-environment flat identity field: {key}')
        evidence[key] = [values[i] for i in indices]
    for value in evidence['episode_return']:
        POSTURE._finite(value)
    if pair in initial and initial[pair] != evidence:
        raise ValueError('hybrid per-environment flat outcomes/routing are not exactly identical')
    initial[pair] = deepcopy(evidence)


def _paired(initial, pair, data):
    state = {key: data[key] for key in (
        'initial_state_sha256', 'initial_prefix_sha256', 'initial_rng_sha256')}
    if pair in initial and initial[pair] != state:
        raise ValueError('paired controllers have different initial states/prefix/RNG')
    initial[pair] = deepcopy(state)


def _inside(name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('provenance paths must be repository-relative')
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('provenance path escapes repository')
    return path


def audit_preholdout_ledger(directory, inputs):
    path = directory / 'preholdout_ledger.json'
    HISTORY._hashes({'preholdout_ledger': inputs.get('preholdout_ledger_sha256')})
    if HISTORY.sha(path) != inputs['preholdout_ledger_sha256']:
        raise ValueError('preholdout ledger manifest digest mismatch')
    manifest = json.loads(path.read_text())
    if set(manifest) != {'created_utc', 'path', 'lines', 'bytes', 'sha256'}:
        raise ValueError('invalid preholdout ledger manifest')
    HISTORY._hashes({'prefix': manifest['sha256']})
    ledger = _inside(manifest['path'])
    if ledger != (directory / 'commands.jsonl').resolve():
        raise ValueError('preholdout ledger path mismatch')
    if any(type(manifest[key]) is not int or manifest[key] <= 0 for key in ('lines', 'bytes')):
        raise ValueError('invalid preholdout ledger prefix size')
    prefix = ledger.read_bytes()[:manifest['bytes']]
    if (len(prefix) != manifest['bytes'] or not prefix.endswith(b'\n')
            or len(prefix.splitlines()) != manifest['lines']
            or hashlib.sha256(prefix).hexdigest() != manifest['sha256']):
        raise ValueError('preholdout command ledger prefix changed')
    pinned = HISTORY._timestamp(manifest['created_utc'])
    if pinned > HISTORY._timestamp(inputs['created_utc']):
        raise ValueError('preholdout ledger was pinned after evaluation inputs')
    for line in prefix.splitlines():
        record = json.loads(line)
        start, finish = HISTORY._timestamp(record['started_utc']), HISTORY._timestamp(record['finished_utc'])
        if not start <= finish <= pinned:
            raise ValueError('preholdout command chronology mismatch')
        HISTORY._hashes({'log': record.get('log_sha256')})
        if HISTORY.sha(_inside(record['log'])) != record['log_sha256']:
            raise ValueError('preholdout command log changed')
    return dict(manifest_sha256=inputs['preholdout_ledger_sha256'], **manifest)


def audit_development(directory, training):
    capacity = json.loads((directory / 'capacity.json').read_text())
    preflight = json.loads((directory / 'preflight.json').read_text())
    for key in ('default_config_parity', 'paired_initial_and_saved_configs', 'development_only'):
        if capacity.get(key) is not True:
            raise ValueError('capacity proof flags missing')
    if (preflight.get('passed') is not True or preflight.get('compileall') is not True
            or preflight.get('diff_check') is not True or preflight.get('legacy_source_count') != 106
            or preflight.get('legacy_model_count') != 8
            or preflight.get('capacity_sha256') != training['capacity_sha256']):
        raise ValueError('preflight proof flags/inventory mismatch')
    if not (HISTORY._timestamp(capacity['created_utc']) <= HISTORY._timestamp(preflight['created_utc'])
            <= HISTORY._timestamp(training['frozen_at'])):
        raise ValueError('development evidence chronology mismatch')
    for name, key in (('evaluator_parity.json', 'evaluator_parity_sha256'),
                      ('implementation_review.json', 'review_sha256')):
        HISTORY._hashes({key: preflight.get(key)})
        if HISTORY.sha(directory / name) != preflight[key]:
            raise ValueError('preflight linked evidence changed')
    HISTORY._hashes({'cpu_log': preflight.get('cpu_log_sha256')})
    if HISTORY.sha(_inside(preflight['cpu_log'])) != preflight['cpu_log_sha256']:
        raise ValueError('preflight CPU evidence changed')
    records = capacity.get('records')
    if not isinstance(records, dict) or set(records) != set(ARMS) or capacity.get('terrain_cache') != training.get('terrain_cache'):
        raise ValueError('capacity arms/cache changed')
    paired = None
    for arm, record in records.items():
        HISTORY._hashes({'initial': record.get('sha256')})
        path = _inside(record['initial'])
        if HISTORY.sha(path) != record['sha256']:
            raise ValueError('capacity initial evidence changed')
        data = json.loads(path.read_text())
        if (data.get('arm') != arm or data.get('num_envs') != 4096
                or data.get('observation_dimensions') != [4096, 91]
                or data.get('reward_weight') != 1. or data.get('direction_reward_weight') != float(arm == 'stable')
                or data.get('default_config_parity') is not True or data.get('optimizer_state_empty') is not True
                or data.get('iteration') != 0 or data.get('saved_parameter_sha256') != record.get('saved_config_sha256')):
            raise ValueError('capacity actual initial contract mismatch')
        HISTORY._hashes(data.get('initial_state_sha256'), HISTORY.INITIAL_KEYS)
        HISTORY._hashes({'prefix': data.get('initial_prefix_sha256')})
        HISTORY._hashes(data.get('rng_sha256'), ('cpu', 'cuda'))
        HISTORY._hashes(data.get('policy_state_sha256'))
        HISTORY._hashes(data.get('saved_parameter_sha256'))
        for name, digest in data['saved_parameter_sha256'].items():
            if HISTORY.sha(_inside(data['log_dir']) / 'params' / f'{name}.yaml') != digest:
                raise ValueError('capacity saved configuration bytes changed')
        current = {key: data[key] for key in ('initial_state_sha256', 'initial_prefix_sha256', 'rng_sha256',
                                             'policy_state_sha256', 'normalized_parameters', 'dt')}
        if paired is not None and paired != current:
            raise ValueError('capacity actual initial pairing mismatch')
        paired = current


def audit_final_smokes(directory, frozen, models):
    path = directory / 'final_model_smokes.json'
    HISTORY._hashes({'final_smokes': frozen.get('final_model_smokes_sha256')})
    if HISTORY.sha(path) != frozen['final_model_smokes_sha256']:
        raise ValueError('final model smoke manifest changed')
    manifest = json.loads(path.read_text())
    if any(manifest.get(key) is not True for key in ('passed', 'development_only', 'exact_initial_pairing')):
        raise ValueError('final model smokes lack development pairing proof')
    finished = HISTORY._timestamp(manifest['created_utc'])
    if finished > HISTORY._timestamp(frozen['frozen_at']):
        raise ValueError('final model smokes postdate evaluation freeze')
    records = manifest.get('records')
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError('invalid final smoke inventory')
    initial, controllers = {}, set()
    for record in records:
        if set(record) != {'path', 'sha256'}:
            raise ValueError('invalid final smoke file record')
        HISTORY._hashes({'smoke': record['sha256']})
        path = _inside(record['path'])
        if HISTORY.sha(path) != record['sha256']:
            raise ValueError('final smoke evidence changed')
        data = json.loads(path.read_text())
        controller = data.get('controller')
        if controller not in ('control', 'stable', 'history_stable') or controller in controllers:
            raise ValueError('final smoke controller inventory changed')
        controllers.add(controller)
        model = models[controller.removeprefix('history_')]
        if (data.get('phase'), data.get('scenario'), data.get('geometry_seed'), data.get('reset_seed'),
                data.get('num_envs'), data.get('condition', {}).get('seconds')) != ('smoke', 'mixed', 51, 24, 35, 16):
            raise ValueError('final smoke development contract changed')
        if data.get('checkpoint') != model['checkpoint'] or data.get('checkpoint_sha256') != model['sha256']:
            raise ValueError('final smoke model mismatch')
        if not HISTORY._timestamp(data['started_utc']) <= HISTORY._timestamp(data['finished_utc']) <= finished:
            raise ValueError('final smoke chronology changed')
        if data.get('evaluation_plan_sha256') != frozen['source_sha256']['docs/experiment_plans/directional_stability_v15.md']:
            raise ValueError('final smoke plan linkage changed')
        audit(data)
        _paired(initial, (51, 24), data)


def summarize(directory):
    directory = Path(directory)
    frozen_path = directory / 'frozen.json'
    frozen = json.loads(frozen_path.read_text())
    if frozen['holdouts'] != [list(pair) for pair in HOLDOUTS] or frozen['controllers'] != list(CONTROLLERS):
        raise ValueError('changed predeclared holdout/controller inventory')
    if frozen['gate_config'] != asdict(HistoryGateConfig()) or frozen['posture_config'] != asdict(PostureConfig()):
        raise ValueError('changed frozen configuration')
    if frozen.get('direction_config') != asdict(DirectionConfig()):
        raise ValueError('changed frozen direction configuration')
    HISTORY._hashes(frozen['source_sha256'], SOURCE_FILES)
    HISTORY._hashes(frozen['legacy']['sources'], count=106)
    HISTORY._hashes(frozen['legacy']['models'], count=8)
    if frozen['legacy'] != legacy_hashes():
        raise ValueError('legacy source/model inventory differs from canonical freeze')
    for name, digest in {**frozen['source_sha256'], **frozen['legacy']['sources'], **frozen['legacy']['models']}.items():
        if HISTORY.sha(_inside(name)) != digest:
            raise ValueError(f'frozen provenance bytes changed: {name}')
    HISTORY._hashes({'training_freeze': frozen['training_freeze_sha256']})
    if HISTORY.sha(directory / 'training_frozen.json') != frozen['training_freeze_sha256']:
        raise ValueError('training freeze digest mismatch')
    training = json.loads((directory / 'training_frozen.json').read_text())
    HISTORY._hashes(training.get('source_sha256'), TRAINING_SOURCES)
    if (training.get('legacy') != frozen['legacy']
            or any(frozen['source_sha256'].get(key) != value for key, value in training['source_sha256'].items())
            or training.get('starting_checkpoint_sha256') != V14_SHA
            or training.get('arms') != list(ARMS)
            or training.get('posture_config') != asdict(PostureConfig())
            or training.get('direction_config') != asdict(DirectionConfig())
            or training.get('history_config') != asdict(HistoryGateConfig())):
        raise ValueError('changed training source/configuration contract')
    for key, value in dict(training_seed=47, geometry=95, iterations_per_arm=250,
                           envs=4096, steps_per_env=32, transitions_per_arm=32768000).items():
        if type(training.get(key)) is not int or training[key] != value:
            raise ValueError('changed training budget/seed contract')
    HISTORY._hashes({key: training.get(key) for key in ('capacity_sha256', 'preflight_sha256')})
    for key, filename in (('capacity_sha256', 'capacity.json'), ('preflight_sha256', 'preflight.json')):
        if HISTORY.sha(directory / filename) != training[key]:
            raise ValueError('training preflight/capacity bytes changed')
    audit_development(directory, training)
    checkpoints = training.get('initial_checkpoints')
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(ARMS):
        raise ValueError('changed initial checkpoint inventory')
    if checkpoints['control'] != checkpoints['stable']:
        raise ValueError('paired arms must share one initial checkpoint')
    for item in checkpoints.values():
        if set(item) != {'checkpoint', 'sha256'}:
            raise ValueError('invalid initial checkpoint declaration')
        HISTORY._hashes({'checkpoint': item['sha256']})
        if HISTORY.sha(_inside(item['checkpoint'])) != item['sha256']:
            raise ValueError('initial checkpoint bytes changed')
    for name, expected in (
        ('primary', dict(files=12, episodes=2100, seconds=16, envs=175)),
        ('secondary', dict(files=12, episodes=120, seconds=64, envs=10)),
    ):
        actual = frozen.get(name)
        if actual != expected or any(type(v) is not int for v in actual.values()):
            raise ValueError('changed predeclared horizon inventory')
    models = frozen['models']
    if set(models) != {'original', 'v14', 'control', 'stable'} or models['original']['sha256'] != START_SHA or models['v14']['sha256'] != V14_SHA:
        raise ValueError('invalid frozen model inventory')
    trained_path = directory / 'trained_models.json'
    HISTORY._hashes({'trained_models': frozen['trained_models_sha256']})
    if HISTORY.sha(trained_path) != frozen['trained_models_sha256']:
        raise ValueError('trained model manifest digest mismatch')
    trained = json.loads(trained_path.read_text())
    if (trained.get('training_freeze_sha256') != frozen['training_freeze_sha256']
            or type(trained.get('total_training_transitions')) is not int
            or trained['total_training_transitions'] != 65536000
            or trained.get('models') != {arm: models[arm] for arm in ARMS}):
        raise ValueError('trained model manifest differs from declared paired training')
    HISTORY._hashes({'training_validation': trained.get('training_validation_sha256')})
    validation_path = directory / 'training_validation.json'
    if HISTORY.sha(validation_path) != trained['training_validation_sha256']:
        raise ValueError('training validation digest mismatch')
    validation = json.loads(validation_path.read_text())
    records = validation.get('records')
    if not isinstance(records, dict) or set(records) != set(ARMS):
        raise ValueError('training validation arm inventory mismatch')
    required_tags = {'Episode_Reward/adaptive_posture', 'Loss/value_function',
                     'Loss/surrogate', 'Loss/prior_loss', 'Train/mean_reward', 'Policy/mean_noise_std'}
    for arm, record in records.items():
        if arm == "stable" and "Episode_Reward/directional_stability" not in record.get("scalar_counts", {}):
            raise ValueError("stable training lacks directional reward tag")
        counts = record.get('scalar_counts')
        if (type(record.get('iterations_logged')) is not int or record['iterations_logged'] != 250
                or type(record.get('transitions_logged')) is not int or record['transitions_logged'] != 32768000
                or record.get('all_scalars_finite') is not True
                or not isinstance(counts, dict) or not required_tags.issubset(counts)
                or any(not isinstance(key, str) or type(value) is not int or value != 250
                       for key, value in counts.items())):
            raise ValueError('incomplete/nonfinite training log validation')
        HISTORY._hashes({'text_log': record.get('text_log_sha256')})
        HISTORY._hashes(record.get('event_files_sha256'))
        if not record['event_files_sha256']:
            raise ValueError('training validation lacks event files')
        for name, digest in {record['text_log']: record['text_log_sha256'],
                             **record['event_files_sha256']}.items():
            if HISTORY.sha(_inside(name)) != digest:
                raise ValueError('validated training log/event bytes changed')
    if not (HISTORY._timestamp(training['frozen_at']) < HISTORY._timestamp(trained['completed_utc'])
            <= HISTORY._timestamp(frozen['frozen_at'])):
        raise ValueError('training completion/freeze chronology mismatch')
    training_initial = None
    for name, model in models.items():
        expected_keys = {'checkpoint', 'sha256'}
        if name in ARMS:
            expected_keys |= {'source_checkpoint', 'iteration', 'transitions', 'initial_audit_sha256'}
        if set(model) != expected_keys:
            raise ValueError('invalid checkpoint declaration')
        HISTORY._hashes({'checkpoint': model['sha256']})
        if HISTORY.sha(_inside(model['checkpoint'])) != model['sha256']:
            raise ValueError('frozen checkpoint bytes changed')
        if name in ARMS:
            if (type(model['iteration']) is not int or model['iteration'] != 249
                    or type(model['transitions']) is not int or model['transitions'] != 32768000):
                raise ValueError('trained model must be final249 with declared transition budget')
            source = _inside(model['source_checkpoint'])
            if source.name != 'model_249.pt' or HISTORY.sha(source) != model['sha256']:
                raise ValueError('final training source checkpoint mismatch')
            HISTORY._hashes({'initial_audit': model['initial_audit_sha256']})
            audit_path = directory / 'training' / f'{name}_initial.json'
            if HISTORY.sha(audit_path) != model['initial_audit_sha256']:
                raise ValueError('training initial audit digest mismatch')
            initial_audit = json.loads(audit_path.read_text())
            if (initial_audit.get('arm') != name
                    or initial_audit.get('default_config_parity') is not True
                    or initial_audit.get('optimizer_state_empty') is not True
                    or type(initial_audit.get('iteration')) is not int or initial_audit['iteration'] != 0
                    or initial_audit.get('num_envs') != 4096
                    or initial_audit.get('reward_weight') != 1.0
                    or initial_audit.get('direction_reward_weight') != float(name == 'stable')
                    or initial_audit.get('observation_dimensions') != [4096, 91]):
                raise ValueError('training initial audit arm/state contract mismatch')
            HISTORY._hashes(initial_audit.get('initial_state_sha256'), HISTORY.INITIAL_KEYS)
            HISTORY._hashes({'prefix': initial_audit.get('initial_prefix_sha256')})
            HISTORY._hashes(initial_audit.get('rng_sha256'), ('cpu', 'cuda'))
            HISTORY._hashes(initial_audit.get('policy_state_sha256'))
            paired = {key: initial_audit[key] for key in (
                'initial_state_sha256', 'initial_prefix_sha256', 'rng_sha256',
                'policy_state_sha256', 'normalized_parameters', 'dt')}
            if training_initial is not None and training_initial != paired:
                raise ValueError('training arms differ in actual initial state/configuration')
            training_initial = paired
    audit_final_smokes(directory, frozen, models)
    frozen_sha = HISTORY.sha(frozen_path)
    inputs = evaluation_inputs(directory)
    input_sha = HISTORY.sha(directory / 'evaluation_inputs.json')
    cache_sha = inputs['terrain_cache_manifest_sha256']
    ledger = audit_preholdout_ledger(directory, inputs)
    if not HISTORY._timestamp(frozen['frozen_at']) <= HISTORY._timestamp(inputs['created_utc']):
        raise ValueError('evaluation inputs precede freeze')
    HISTORY._hashes({'evaluation_inputs': input_sha, 'terrain_cache_manifest': cache_sha})
    result = {'files': [], 'provenance': {
        'experiment_freeze_sha256': frozen_sha,
        'training_freeze_sha256': frozen['training_freeze_sha256'],
        'trained_models_sha256': frozen['trained_models_sha256'],
        'training_validation_sha256': trained['training_validation_sha256'],
        'evaluation_inputs_sha256': input_sha,
        'terrain_cache_manifest_sha256': cache_sha,
        'preholdout_ledger': ledger,
    }}
    for scenario, folder, envs, seconds in (('mixed', 'evaluations', 175, 16), ('stones', 'horizon', 10, 64)):
        groups, initial, flat_initial = defaultdict(list), {}, {}
        expected_paths = {f'{c}__geometry{g}_reset{r}.json' for c in CONTROLLERS for g, r in HOLDOUTS}
        if {path.name for path in (directory / folder).glob('*.json')} != expected_paths:
            raise ValueError('missing or unexpected evaluation file')
        for controller in CONTROLLERS:
            model = models[controller.removeprefix('history_')]
            for geometry, reset in HOLDOUTS:
                path = directory / folder / f'{controller}__geometry{geometry}_reset{reset}.json'
                data = json.loads(path.read_text())
                expected = (controller, geometry, reset, scenario, envs, seconds, 'holdout')
                actual = (data.get('controller'), data.get('geometry_seed'), data.get('reset_seed'), data.get('scenario'),
                          data.get('num_envs'), data.get('condition', {}).get('seconds'), data.get('phase'))
                if actual != expected or data.get('checkpoint') != model['checkpoint'] or data.get('checkpoint_sha256') != model['sha256']:
                    raise ValueError('evaluation/model contract differs from freeze')
                if data.get('evaluation_plan_sha256') != frozen['source_sha256']['docs/experiment_plans/directional_stability_v15.md']:
                    raise ValueError('evaluation plan linkage mismatch')
                if data.get('experiment_freeze_sha256') != frozen_sha:
                    raise ValueError('evaluation source/model freeze linkage mismatch')
                if (data.get('evaluation_inputs_sha256') != input_sha
                        or data.get('terrain_cache_manifest_sha256') != cache_sha):
                    raise ValueError('evaluation input/cache manifest linkage mismatch')
                if HISTORY._timestamp(data['started_utc']) <= HISTORY._timestamp(inputs['created_utc']):
                    raise ValueError('holdout evaluation precedes pinned inputs')
                rows = audit(data)
                expected_counts = ({(f, level): 5 for f in FAMILIES for level in range(5)}
                                   if scenario == 'mixed' else {('stepping_stones', 4): 10})
                if Counter((row['family'], row['level']) for row in rows) != expected_counts:
                    raise ValueError('family/difficulty inventory mismatch')
                _paired(initial, (geometry, reset), data)
                if scenario == 'mixed' and controller.startswith('history_'):
                    _flat_identity(flat_initial, (geometry, reset), data, rows)
                groups[controller].extend(rows)
                result['files'].append({'path': str(path.relative_to(directory)), 'sha256': HISTORY.sha(path), 'episodes': len(rows)})
        section = {
            'groups': {c: aggregate(rows) for c, rows in groups.items()},
            'family': {c: {f: aggregate([r for r in rows if r['family'] == f]) for f in sorted({r['family'] for r in rows})}
                       for c, rows in groups.items()},
            'family_level': {c: {f'{f}/level{level}': aggregate([r for r in rows if (r['family'], r['level']) == (f, level)])
                                 for f, level in sorted({(r['family'], r['level']) for r in rows})} for c, rows in groups.items()},
        }
        section['actor_stable_vs_control'] = improvement(section['groups']['stable'], section['groups']['control'], primary=scenario == 'mixed')
        section['hybrid_stable_vs_control'] = improvement(section['groups']['history_stable'], section['groups']['history_control'], hybrid=True, primary=scenario == 'mixed')
        if scenario == 'mixed':
            identity = {'verified': len(flat_initial) == len(HOLDOUTS),
                        'controllers': [c for c in CONTROLLERS if c.startswith('history_')],
                        'maps': len(flat_initial),
                        'flat_episodes_per_controller': sum(len(v['indices']) for v in flat_initial.values()),
                        'fields': list(FLAT_IDENTITY_FIELDS)}
            section['hybrid_flat_raw_identity'] = identity
            judgment = section['hybrid_stable_vs_control']
            judgment['checks']['fixed_v5_per_environment_flat_raw_identity'] = identity['verified']
            judgment['passed'] = all(judgment['checks'].values())
        section['posture_mechanism'] = {}
        for label, candidate, control in (('actor', 'stable', 'control'),
                                           ('hybrid', 'history_stable', 'history_control')):
            diagnostics = {}
            for name, group, field in (('rough_target_error', 'rough', 'absolute_body_error'),
                                        ('clear_body_clearance', 'clear', 'body_clearance')):
                new = section['groups'][candidate]['posture'][group]
                old = section['groups'][control]['posture'][group]
                a, b = new['means'][field], old['means'][field]
                diagnostics[name] = {'stable': a, 'control': b, 'delta': a - b if a is not None and b is not None else None,
                                     'stable_samples': new['sums']['steps'], 'control_samples': old['sums']['steps']}
            if scenario == 'mixed':
                a = section['family'][candidate]['flat']['posture']['all_valid']['means']['body_clearance']
                b = section['family'][control]['flat']['posture']['all_valid']['means']['body_clearance']
                diagnostics['flat_family_body_clearance'] = {'stable': a, 'control': b,
                    'delta': a - b if a is not None and b is not None else None}
            section['posture_mechanism'][label] = diagnostics
        section['direction_mechanism'] = {}
        for label, candidate, control in (('actor', 'stable', 'control'),
                                          ('hybrid', 'history_stable', 'history_control')):
            new, old = section['groups'][candidate]['direction'], section['groups'][control]['direction']
            section['direction_mechanism'][label] = {
                field: {'stable': new['means'][field], 'control': old['means'][field],
                        'delta': new['means'][field] - old['means'][field]
                                 if new['means'][field] is not None and old['means'][field] is not None else None,
                        'stable_valid_steps': new['sums']['valid_steps'], 'control_valid_steps': old['sums']['valid_steps'],
                        'stable_valid_coverage': new['valid_coverage'], 'control_valid_coverage': old['valid_coverage']}
                for field in ('absolute_lateral_velocity', 'absolute_heading_error', 'absolute_yaw_error', 'reward')}
        result[folder] = section
    if len(result['files']) != 24 or sum(item['episodes'] for item in result['files']) != 2220:
        raise ValueError('incomplete declared first-episode inventory')
    result['interpretation'] = ('One starting actor/fine-tuning seed and two maps: exploratory, not universal or statistically significant. '
        'Stable is compared with equal-budget control training; frozen v14 and original hybrid are baselines. '
        'Only the directional training reward differs; actor and critic retain identical conditioned 91D access. '
        '16s primary and 64s secondary remain separate; secondary cannot override primary failure. '
        'Direction and posture means are conditional on visited valid states, not matched-state causal effects. '
        'Invalid direction samples are excluded, never counted as good tracking. Reward is unweighted and not dt-integrated. '
        'World-Z angular velocity is not Euler yaw derivative under tilt. Ideal ray depth and a kinematic swing proxy do not establish real-camera or robot safety.')
    return result


def markdown(result):
    lines = ['# v15 directional stability: matched reward comparison', '', result['interpretation'], '']
    for folder, title in (('evaluations', 'Primary: 16s mixed terrain'), ('horizon', 'Secondary: 64s hardest stones')):
        section = result[folder]
        lines += [f'## {title}', '', '| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for controller in CONTROLLERS:
            d = section['groups'][controller]
            speed = '—' if d['flat_mean_episode_speed'] is None else f"{d['flat_mean_episode_speed']:.4f}"
            lines.append(f"| {controller} | {d['one']}/{d['n']} | {d['six']}/{d['n']} | {d['falls']}/{d['n']} | {d['lane']}/{d['n']} | {d['world'] + d['flat_world']} | {d['flat_falls']}/{d['flat_n']} | {speed} |")
        for key in ('actor_stable_vs_control', 'hybrid_stable_vs_control'):
            judgment = section[key]
            lines += ['', f"{key}: **{'PASS' if judgment['passed'] else 'FAIL'}**"]
            lines += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in judgment['checks'].items()]
        lines += ['', 'Visited-state posture mechanism (stable minus control; not causal):']
        for label, diagnostics in section['posture_mechanism'].items():
            values = ', '.join(f"{key}: {value['delta']:.4f}m" if value['delta'] is not None else f"{key}: unavailable"
                               for key, value in diagnostics.items())
            lines.append(f'- {label}: {values}')
        lines += ['', 'Visited-state directional diagnostics (stable minus control; not causal):']
        for label, diagnostics in section['direction_mechanism'].items():
            values = ', '.join(f"{key}: {value['delta']:.4f}" if value['delta'] is not None else f'{key}: unavailable'
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
