"""Audit true 91D v14 evidence; project only a copy for frozen scoring primitives."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from week03_ant.command_study import (
    ARMS, CONTROLLERS, HOLDOUTS, ROOT, SCHEMA, SOURCE_FILES, TRAINING_SOURCES,
    START_SHA, V13_SHA, V5_SHA, TASK, evaluation_inputs, legacy_hashes,
)

_SPEC = importlib.util.spec_from_file_location('v14_frozen_posture_summary', ROOT / 'scripts/summarize_posture_v13.py')
POSTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(POSTURE)
HISTORY = POSTURE.HISTORY
FAMILIES = POSTURE.FAMILIES
aggregate = POSTURE.aggregate
improvement = POSTURE.improvement
PROJECTED_CONTROLLERS = {
    'v13': 'adaptive', 'masked': 'control', 'conditioned': 'adaptive',
    'history_original': 'history_original', 'history_masked': 'history_control',
    'history_conditioned': 'history_adaptive',
}


def audit(data):
    """Validate v14 first; never mutate or relabel the underlying 91D evidence."""
    controller = data.get('controller')
    if data.get('schema') != SCHEMA or data.get('task') != TASK or controller not in CONTROLLERS:
        raise ValueError('unexpected v14 task/schema/controller')
    arm = controller.removeprefix('history_')
    expected_mode = arm if arm in ARMS else None
    condition = data.get('condition', {})
    if (type(condition.get('observations')) is not int or condition['observations'] != 91
            or type(condition.get('expert_observations')) is not int
            or condition['expert_observations'] != (91 if expected_mode else 88)
            or type(condition.get('v5_observations')) is not int or condition['v5_observations'] != 60
            or type(condition.get('actions')) is not int or condition['actions'] != 8
            or 'command_mode' not in data or data['command_mode'] != expected_mode
            or type(data.get('command_schema_version')) is not int or data['command_schema_version'] != 1):
        raise ValueError('changed 91D command observation/mode/schema contract')
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
    return POSTURE.audit(projected)


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


def summarize(directory):
    directory = Path(directory)
    frozen_path = directory / 'frozen.json'
    frozen = json.loads(frozen_path.read_text())
    if frozen['holdouts'] != [list(pair) for pair in HOLDOUTS] or frozen['controllers'] != list(CONTROLLERS):
        raise ValueError('changed predeclared holdout/controller inventory')
    if frozen['gate_config'] != asdict(HistoryGateConfig()) or frozen['posture_config'] != asdict(PostureConfig()):
        raise ValueError('changed frozen configuration')
    HISTORY._hashes(frozen['source_sha256'], SOURCE_FILES)
    HISTORY._hashes(frozen['legacy']['sources'], count=89)
    HISTORY._hashes(frozen['legacy']['models'], count=6)
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
            or training.get('starting_checkpoint_sha256') != START_SHA
            or training.get('arms') != list(ARMS)
            or training.get('posture_config') != asdict(PostureConfig())
            or training.get('history_config') != asdict(HistoryGateConfig())):
        raise ValueError('changed training source/configuration contract')
    for key, value in dict(training_seed=46, geometry=85, iterations_per_arm=250,
                           envs=4096, steps_per_env=32, transitions_per_arm=32768000).items():
        if type(training.get(key)) is not int or training[key] != value:
            raise ValueError('changed training budget/seed contract')
    HISTORY._hashes({key: training.get(key) for key in ('capacity_sha256', 'preflight_sha256')})
    for key, filename in (('capacity_sha256', 'capacity.json'), ('preflight_sha256', 'preflight.json')):
        if HISTORY.sha(directory / filename) != training[key]:
            raise ValueError('training preflight/capacity bytes changed')
    checkpoints = training.get('initial_checkpoints')
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(ARMS):
        raise ValueError('changed initial checkpoint inventory')
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
    if set(models) != {'original', 'v13', 'masked', 'conditioned'} or models['original']['sha256'] != START_SHA or models['v13']['sha256'] != V13_SHA:
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
    for record in records.values():
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
                    or initial_audit.get('reward_weight') != 1
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
    frozen_sha = HISTORY.sha(frozen_path)
    inputs = evaluation_inputs(directory)
    input_sha = HISTORY.sha(directory / 'evaluation_inputs.json')
    cache_sha = inputs['terrain_cache_manifest_sha256']
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
        section['actor_conditioned_vs_masked'] = improvement(section['groups']['conditioned'], section['groups']['masked'], primary=scenario == 'mixed')
        section['hybrid_conditioned_vs_masked'] = improvement(section['groups']['history_conditioned'], section['groups']['history_masked'], hybrid=True, primary=scenario == 'mixed')
        if scenario == 'mixed':
            identity = {'verified': len(flat_initial) == len(HOLDOUTS),
                        'controllers': [c for c in CONTROLLERS if c.startswith('history_')],
                        'maps': len(flat_initial),
                        'flat_episodes_per_controller': sum(len(v['indices']) for v in flat_initial.values()),
                        'fields': list(FLAT_IDENTITY_FIELDS)}
            section['hybrid_flat_raw_identity'] = identity
            judgment = section['hybrid_conditioned_vs_masked']
            judgment['checks']['fixed_v5_per_environment_flat_raw_identity'] = identity['verified']
            judgment['passed'] = all(judgment['checks'].values())
        section['posture_mechanism'] = {}
        for label, candidate, control in (('actor', 'conditioned', 'masked'),
                                           ('hybrid', 'history_conditioned', 'history_masked')):
            diagnostics = {}
            for name, group, field in (('rough_target_error', 'rough', 'absolute_body_error'),
                                        ('clear_body_clearance', 'clear', 'body_clearance')):
                new = section['groups'][candidate]['posture'][group]
                old = section['groups'][control]['posture'][group]
                a, b = new['means'][field], old['means'][field]
                diagnostics[name] = {'conditioned': a, 'masked': b, 'delta': a - b if a is not None and b is not None else None,
                                     'conditioned_samples': new['sums']['steps'], 'masked_samples': old['sums']['steps']}
            if scenario == 'mixed':
                a = section['family'][candidate]['flat']['posture']['all_valid']['means']['body_clearance']
                b = section['family'][control]['flat']['posture']['all_valid']['means']['body_clearance']
                diagnostics['flat_family_body_clearance'] = {'conditioned': a, 'masked': b,
                    'delta': a - b if a is not None and b is not None else None}
            section['posture_mechanism'][label] = diagnostics
        result[folder] = section
    if len(result['files']) != 24 or sum(item['episodes'] for item in result['files']) != 2220:
        raise ValueError('incomplete declared first-episode inventory')
    result['interpretation'] = ('One starting actor/fine-tuning seed and two maps: exploratory, not universal or statistically significant. '
        'Conditioned is compared with equal-budget masked training; v13 and the original hybrid are baselines. The contrast changes actor AND critic command-plus-feedback access, not actor-only or goal-only causality. Action sensitivity does not establish arbitrary command tracking. '
        '16s primary and 64s secondary remain separate; secondary cannot override primary failure. '
        'Posture means are conditional on visited valid states, not matched-state causal effects. '
        'Ideal ray depth and a kinematic swing proxy do not establish real-camera or robot safety.')
    return result


def markdown(result):
    lines = ['# v14 command conditioning: matched masked comparison', '', result['interpretation'], '']
    for folder, title in (('evaluations', 'Primary: 16s mixed terrain'), ('horizon', 'Secondary: 64s hardest stones')):
        section = result[folder]
        lines += [f'## {title}', '', '| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for controller in CONTROLLERS:
            d = section['groups'][controller]
            speed = '—' if d['flat_mean_episode_speed'] is None else f"{d['flat_mean_episode_speed']:.4f}"
            lines.append(f"| {controller} | {d['one']}/{d['n']} | {d['six']}/{d['n']} | {d['falls']}/{d['n']} | {d['lane']}/{d['n']} | {d['world'] + d['flat_world']} | {d['flat_falls']}/{d['flat_n']} | {speed} |")
        for key in ('actor_conditioned_vs_masked', 'hybrid_conditioned_vs_masked'):
            judgment = section[key]
            lines += ['', f"{key}: **{'PASS' if judgment['passed'] else 'FAIL'}**"]
            lines += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in judgment['checks'].items()]
        lines += ['', 'Visited-state posture mechanism (conditioned minus masked; not causal):']
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
