"""Independent CPU tensor/Adam/LR/scalar checks, outside v24 study definitions.

This post-experiment verifier never imports experiment validators or policies.
It checks serialized tensors and decoded logs, not unobserved simulator behavior.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/terrain_demo/lr_continuation_v24'
INIT = ROOT / 'logs/rsl_rl/week03_ant_contact_continuation_v21/v21_init/model_0.pt'
INIT_SHA = '10d8485d6d18f1dd6c01724ff69fc6a2ced3bbcd3d7b0db1285b7097cb5c21bd'
V5 = ROOT / 'artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt'
V5_SHA = '889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e'
RATES = {'high': 1.e-4, 'low': 1.e-5}
ACTOR = ('actor.0.weight', 'actor.0.bias', 'actor.0.command_weight',
         'actor.2.weight', 'actor.2.bias', 'actor.4.weight', 'actor.4.bias',
         'actor.6.weight', 'actor.6.bias')
CRITIC = tuple(name.replace('actor.', 'critic.') for name in ACTOR)
TEACHER = tuple(name.replace('actor.', 'teacher.') for name in ACTOR
                if not name.endswith('command_weight'))
TRAINABLE = ('std', *ACTOR, *CRITIC)
PARAMETERS = (*TRAINABLE, *TEACHER)
TIMING = {'Perf/total_fps', 'Perf/collection time', 'Perf/learning_time',
          'Train/mean_reward/time', 'Train/mean_episode_length/time'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    result = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    finite(result)
    return result


def finite(value):
    if torch.is_tensor(value):
        require(bool(torch.isfinite(value).all()), 'nonfinite tensor')
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            finite(child)
    elif isinstance(value, float):
        require(math.isfinite(value), 'nonfinite number')


def inside(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'nonrelative evidence path')
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'evidence path escapes repository')
    return path


def linked(root, item):
    path = inside(root, item['path'])
    require(sha(path) == item['sha256'], 'linked evidence changed: ' + item['path'])
    return read(path)


def exact(left, right, context='root'):
    if torch.is_tensor(left):
        require(torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left, right), 'tensor differs: ' + context)
    else:
        require(type(left) is type(right), 'type differs: ' + context)
        if isinstance(left, dict):
            require(left.keys() == right.keys(), 'keys differ: ' + context)
            for key in left:
                exact(left[key], right[key], context + '/' + str(key))
        elif isinstance(left, (tuple, list)):
            require(len(left) == len(right), 'length differs: ' + context)
            for index, (a, b) in enumerate(zip(left, right)):
                exact(a, b, context + '/' + str(index))
        else:
            require(left == right, 'value differs: ' + context)


def json_form(value):
    return json.loads(json.dumps(value))


def initializer_pair(original, high, low):
    exact(original, high)
    require(high['iter'] == 0 and high['optimizer_state_dict']['state'] == {}, 'nonempty initializer')
    require(torch.equal(high['model_state_dict']['std'], torch.full((8,), .2)), 'initial std differs')
    projected = copy.deepcopy(low)
    groups = projected['optimizer_state_dict']['param_groups']
    require(len(groups) == 1 and groups[0]['lr'] == 1.e-5, 'low initial LR differs')
    groups[0]['lr'] = 1.e-4
    exact(high, projected)


def checkpoint(state, original, teacher, rate, updates):
    require(rate in RATES.values() and updates in (2, 250), 'undeclared checkpoint budget/LR')
    finite(state)
    require(state['iter'] == updates - 1, 'final checkpoint iteration differs')
    policy, initial = state['model_state_dict'], original['model_state_dict']
    require(list(policy) == list(initial) and len(policy) == 33, 'policy tensor inventory differs')
    require(sum(value.numel() for value in policy.values()) == 400631, 'policy value count differs')
    for name, value in policy.items():
        require(value.shape == initial[name].shape and value.dtype == initial[name].dtype,
                'policy shape/dtype differs: ' + name)
        if name not in TRAINABLE:
            exact(value, initial[name], name)
        if name in TEACHER:
            exact(value, teacher[name.replace('teacher.', 'actor.')], 'original v5/' + name)
    require(bool((policy['std'] > 0).all()), 'nonpositive final std')
    optimizer = state['optimizer_state_dict']
    expected_groups = copy.deepcopy(original['optimizer_state_dict']['param_groups'])
    require(len(expected_groups) == 1 and expected_groups[0]['params'] == list(range(27)),
            'historical optimizer coverage differs')
    expected_groups[0]['lr'] = rate
    exact(expected_groups, optimizer['param_groups'], 'Adam options/order/LR')
    require(set(optimizer['state']) == set(range(19)), 'Adam actor/critic/std state coverage differs')
    for index, name in enumerate(TRAINABLE):
        values = optimizer['state'][index]
        require(set(values) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Adam state fields differ')
        require(values['step'].numel() == 1 and values['step'].item() == updates * 20,
                'Adam actual step count differs')
        for key in ('exp_avg', 'exp_avg_sq'):
            require(values[key].shape == policy[name].shape and values[key].dtype == policy[name].dtype,
                    'Adam moments shape/dtype differs')
        require(bool((values['exp_avg_sq'] >= 0).all()), 'negative Adam second moment')
    return {'policy_tensors': 33, 'policy_values': 400631, 'teacher_tensors_exact_v5': 8,
            'trainable_optimizer_states': 19, 'optimizer_parameter_entries': 27,
            'adam_steps': updates * 20, 'iteration': updates - 1, 'learning_rate': rate,
            'std_min': policy['std'].min().item(), 'std_max': policy['std'].max().item()}


def lr_trace(data, rate, updates, initial_groups, final_groups):
    finite(data)
    condition = next(key for key, value in RATES.items() if value == rate)
    require((data['lr_condition'], data['learning_rate'], data['expected_updates'],
             data['update_count'], data['adam_step_count'], data['teacher_unchanged'], data['passed'])
            == (condition, rate, updates, updates, updates * 20, True, True), 'LR trace budget/result differs')
    observation = dict(saved_learning_rate=rate, algorithm_learning_rate=rate,
                       group_learning_rates=[rate], schedule='fixed', desired_kl=None,
                       parameter_names=list(PARAMETERS), trainable_parameter_names=list(TRAINABLE))
    require(data['initial'] == observation and data['final'] == observation, 'LR endpoint/coverage differs')
    require(data['initial_optimizer_groups'] == json_form(initial_groups)
            and data['final_optimizer_groups'] == json_form(final_groups), 'LR optimizer metadata differs')
    require(len(data['updates']) == updates and len(data['adam_steps']) == updates * 20,
            'LR evidence array count differs')
    for index, row in enumerate(data['updates']):
        require(row == dict(index=index, before=observation, after=observation,
                            step_start=index * 20, step_end=(index + 1) * 20), 'PPO update LR sequence differs')
    for index, row in enumerate(data['adam_steps']):
        require(row == dict(index=index, update_index=index // 20, before=observation, after=observation),
                'Adam step LR sequence differs')
    return {'ppo_updates': updates, 'adam_steps': updates * 20,
            'checked_observations': 2 + 2 * updates + 40 * updates,
            'actor_critic_std_global_coverage': True, 'teacher_boundary_invariance_recorded': True}


def scalar_series(logdir, updates, rate):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(logdir), size_guidance={'scalars': 0}).Reload()
    tags = set(accumulator.Tags()['scalars'])
    require(len(tags) == 38 and TIMING <= tags, 'scalar inventory differs')
    result = {}
    for tag in sorted(tags):
        rows = accumulator.Scalars(tag)
        require(len(rows) == updates, 'scalar count differs: ' + tag)
        require(all(math.isfinite(row.value) for row in rows), 'nonfinite scalar: ' + tag)
        steps = [row.step for row in rows]
        if tag in {'Train/mean_reward/time', 'Train/mean_episode_length/time'}:
            require(steps == sorted(steps) and min(steps) >= 0, 'scalar time sequence differs: ' + tag)
        else:
            require(steps == list(range(updates)), 'scalar iteration sequence differs: ' + tag)
        result[tag] = [(row.step, row.value) for row in rows]
    # TensorBoard encodes scalar values as float32; do not require Python-float equality.
    encoded_rate = struct.unpack('f', struct.pack('f', rate))[0]
    require(all(value == encoded_rate for _, value in result['Loss/learning_rate']), 'logged LR differs')
    return result


def initial_pair(left, right, *, project_lr=False, project_iterations=False):
    """Only explicitly allowed pre-learning projections; never final trajectories."""
    projected = copy.deepcopy(left)
    if project_lr:
        require(float(projected['normalized_parameters']['agent']['algorithm']['learning_rate']) == 1.e-5
                and float(right['normalized_parameters']['agent']['algorithm']['learning_rate']) == 1.e-4,
                'same-seed LR projection operands differ')
        projected['normalized_parameters']['agent']['algorithm']['learning_rate'] = \
            right['normalized_parameters']['agent']['algorithm']['learning_rate']
    if project_iterations:
        require(projected['normalized_parameters']['agent']['max_iterations'] == '2'
                and right['normalized_parameters']['agent']['max_iterations'] == '250',
                'same-arm capacity projection operands differ')
        projected['normalized_parameters']['agent']['max_iterations'] = '250'
    fields = ('initial_state_sha256', 'initial_prefix_sha256', 'policy_state_sha256', 'rng_sha256',
              'num_envs', 'dt', 'normalized_parameters', 'initial_policy_parity', 'optimizer_state_empty',
              'iteration', 'common_step_counter', 'contact_slip_weight', 'actual_seed', 'observation_dimensions')
    for name in fields:
        require(projected[name] == right[name], 'pre-learning pairing differs: ' + name)


def normalized_configs(env, agent, logdir, condition, ramp_steps):
    """Independently decode the complete config with only enumerated projections."""
    env, agent = copy.deepcopy(env), copy.deepcopy(agent)
    require(float(env['rewards']['adaptive_posture']['weight']) == 1., 'saved posture reward differs')
    env['rewards']['adaptive_posture']['weight'] = '<paired-treatment>'
    for field in ('log_dir', 'io_descriptors_output_dir'):
        require(Path(env.pop(field)).resolve() == logdir.resolve(), 'saved output directory differs')
    require(isinstance(agent.pop('run_name'), str), 'saved run name missing')
    contact = env['rewards']['contact_slip']
    require(contact['func'] == 'week03_ant.tasks.contact_continuation_v21_cfg:ContinuationContactSlipReward'
            and float(contact['weight']) == 1.
            and contact['params'] == {'mode': 'no_cost', 'ramp_steps': str(ramp_steps)},
            'saved no-cost reward differs')
    require(env['scene']['terrain']['terrain_generator']['seed'] == '110', 'saved training geometry differs')
    require(agent['experiment_name'] == 'week03_ant_lr_continuation_v24'
            and agent['load_run'] == f'v24_{condition}_init' and agent['load_checkpoint'] == 'model_0.pt',
            'saved experiment/initialization identity differs')
    contact['func'] = 'week03_ant.tasks.contact_curriculum_v20_cfg:CurriculumContactSlipReward'
    contact['params']['mode'] = '<paired-coefficient-trajectory>'
    agent['experiment_name'], agent['load_run'] = 'week03_ant_contact_curriculum_v20', 'v20_init'
    return {'env': env, 'agent': agent}


def audit_record(root, record, condition, seed, envs, updates, original, teacher, initializer):
    import yaml  # type: ignore[import-untyped]
    data = {name: linked(root, record[name]) for name in ('initial', 'schedule', 'rollout', 'lr')}
    initial = data['initial']
    require(initial['training_seed'] == seed and initial['lr_condition'] == condition
            and initial['learning_rate'] == RATES[condition], 'training arm/seed/LR differs')
    require(initial['actual_seed'] == dict.fromkeys(('requested', 'saved_agent', 'saved_environment', 'live_environment'), seed)
            and all(type(value) is int for value in initial['actual_seed'].values()), 'actual training seed differs')
    require(initial['optimizer_state_empty'] is True and initial['iteration'] == 0
            and initial['common_step_counter'] == 0 and initial['num_envs'] == envs
            and initial['observation_dimensions'] == [envs, 91], 'pre-learning state differs')
    expected_hashes = {name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
                       for name, value in original['model_state_dict'].items()}
    require(initial['policy_state_sha256'] == expected_hashes, 'initial policy tensor hashes differ')
    parameters = initial['normalized_parameters']
    require(parameters['agent']['seed'] == parameters['env']['seed'] == str(seed)
            and parameters['agent']['max_iterations'] == str(updates), 'normalized seed/budget differs')
    logdir = inside(root, initial['log_dir'])
    for kind in ('env', 'agent'):
        path = logdir / 'params' / f'{kind}.yaml'
        require(sha(path) == initial['saved_parameter_sha256'][kind], 'saved YAML bytes differ')
    agent = yaml.load((logdir / 'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    env = yaml.load((logdir / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    require(normalized_configs(env, agent, logdir, condition, 32 if envs == 256 else 4000) == parameters,
            'complete saved/normalized configuration differs')
    require(agent['seed'] == env['seed'] == str(seed) and agent['device'] == 'cuda:1'
            and agent['max_iterations'] == str(updates) and agent['num_steps_per_env'] == '32'
            and agent['experiment_name'] == 'week03_ant_lr_continuation_v24'
            and agent['load_run'] == f'v24_{condition}_init' and agent['load_checkpoint'] == 'model_0.pt',
            'saved agent/initializer configuration differs')
    require(agent['algorithm'] == parameters['agent']['algorithm'], 'saved/normalized algorithm differs')
    algorithm = agent['algorithm']
    require(float(algorithm['learning_rate']) == RATES[condition] and algorithm['schedule'] == 'fixed'
            and algorithm['desired_kl'] == 'null' and algorithm['num_learning_epochs'] == '5'
            and algorithm['num_mini_batches'] == '4', 'saved algorithm schedule/LR/update budget differs')
    state_path = logdir / f'model_{updates - 1}.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    tensor_result = checkpoint(state, original, teacher, RATES[condition], updates)
    lr_result = lr_trace(data['lr'], RATES[condition], updates,
                         initializer['optimizer_state_dict']['param_groups'], state['optimizer_state_dict']['param_groups'])
    learning = record['learning_log']
    text_path = inside(root, learning['text_log'])
    require(sha(text_path) == learning['text_log_sha256'], 'training console bytes differ')
    text = text_path.read_text()
    pairs = [(int(a), int(b)) for a, b in re.findall(r'Learning iteration (\d+)/(\d+)', text)]
    require(pairs == [(index, updates) for index in range(updates)], 'console update sequence differs')
    totals = [int(value) for value in re.findall(r'Total timesteps:\s*(\d+)', text)]
    require(totals == [(index + 1) * envs * 32 for index in range(updates)], 'console transition budget differs')
    actual_event_files = {str(path.relative_to(root)): sha(path) for path in logdir.glob('events.out.tfevents.*')}
    require(actual_event_files == learning['event_files_sha256'], 'TensorBoard event inventory differs')
    scalars = scalar_series(logdir, updates, RATES[condition])
    schedule = data['schedule']
    require(schedule['episode_randomization']['matches_raw_environment'] is True
            and schedule['episode_randomization']['assigned_sha256'] == schedule['episode_randomization']['actual_sha256'],
            'actual randomized initial horizons differ')
    output = dict(checkpoint=str(state_path.relative_to(root)), checkpoint_sha256=sha(state_path),
                  tensors=tensor_result, learning_rate_trace=lr_result, scalar_tags=38,
                  finite_scalar_values=38 * updates, transitions=updates * 32 * envs,
                  raw_evidence={key: record[key] for key in ('initial', 'schedule', 'rollout', 'lr')})
    return output, data, state, scalars


def replay(root, data, state, scalars, reference, updates):
    old = {name: linked(root, reference[name]) for name in ('initial', 'schedule', 'rollout')}
    initial_pair(data['initial'], old['initial'])
    require(data['schedule'] == old['schedule'], 'historical reward-time replay differs')
    require(data['rollout']['rollout'] == old['rollout']['rollout'], 'historical contact replay differs')
    path = inside(root, reference['checkpoint']['path'])
    require(sha(path) == reference['checkpoint']['sha256'], 'historical checkpoint bytes changed')
    exact(state, torch.load(path, map_location='cpu', weights_only=False), 'full replay model/Adam/metadata')
    previous = scalar_series(inside(root, old['initial']['log_dir']), updates, 1.e-4)
    require(set(scalars) == set(previous), 'historical scalar tag inventory differs')
    for tag in set(scalars) - TIMING:
        require(scalars[tag] == previous[tag], 'historical non-time scalar replay differs: ' + tag)
    return {'whole_checkpoint_model_adam_metadata_exact': True, 'reward_contact_exact': True,
            'exact_non_time_scalar_tags': sorted(set(scalars) - TIMING), 'time_only_exclusions': sorted(TIMING)}


def audit(root=ROOT, *, development_only=False):
    torch.set_num_threads(1)
    directory = root / 'artifacts/terrain_demo/lr_continuation_v24'
    original_path = root / INIT.relative_to(ROOT)
    teacher_path = root / V5.relative_to(ROOT)
    require(sha(original_path) == INIT_SHA and sha(teacher_path) == V5_SHA, 'original init/teacher changed')
    original = torch.load(original_path, map_location='cpu', weights_only=False)
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)['model_state_dict']
    manifest = read(directory / 'initial_shared.json')
    require(set(manifest['initializers']) == {'high', 'low'}, 'initializer inventory differs')
    initializers = {}
    for condition, item in manifest['initializers'].items():
        path = inside(root, item['checkpoint'])
        require(sha(path) == item['sha256'], 'initializer manifest SHA differs')
        initializers[condition] = torch.load(path, map_location='cpu', weights_only=False)
    require(manifest['initializers']['high']['sha256'] == INIT_SHA, 'high byte-copy differs')
    initializer_pair(original, initializers['high'], initializers['low'])
    references = read(directory / 'historical_references.json')
    for name, expected in references['sha256'].items():
        require(sha(inside(root, name)) == expected, 'historical reference bytes changed')
    results, capacities = {}, {}
    proof_paths = ['initial_shared.json', 'historical_references.json', 'preflight.json', 'capacity.json']
    development_transitions = 0
    for phase, envs, seeds in (('preflight', 256, (51,)), ('capacity', 4096, (51, 52, 53))):
        proof = read(directory / (phase + '.json'))
        require(proof['passed'] is True and set(proof['records']) ==
                {f'{condition}{seed}' for seed in seeds for condition in ('high', 'low')}, 'development inventory differs')
        require(len(proof['source_sha256']) == 12 and len(proof['legacy']['sources']) == 237
                and len(proof['legacy']['models']) == 22, 'preserved source/model inventory differs')
        for inventory in (proof['source_sha256'], proof['legacy']['sources'], proof['legacy']['models']):
            for name, expected in inventory.items():
                require(sha(inside(root, name)) == expected, 'frozen source/model bytes changed')
        for seed in seeds:
            pair = {}
            for condition in ('high', 'low'):
                name = f'{condition}{seed}'
                output, data, state, scalars = audit_record(root, proof['records'][name], condition, seed, envs,
                                                           2, original, teacher, initializers[condition])
                pair[condition] = data
                if condition == 'high':
                    output['historical_replay'] = replay(root, data, state, scalars,
                                                         references['stages'][phase][f'seed{seed}'], 2)
                results[f'{phase}/{name}'] = output
                development_transitions += output['transitions']
                if phase == 'capacity':
                    capacities[name] = data
            initial_pair(pair['low']['initial'], pair['high']['initial'], project_lr=True)
            require(pair['low']['schedule']['episode_randomization'] == pair['high']['schedule']['episode_randomization'],
                    'same-seed low/high initial randomization differs')
    require(development_transitions == 1605632, 'new development transitions differ')
    if not development_only:
        proof_paths += ['training_frozen.json', 'full_replay.json', 'training_validation.json', 'trained_models.json']
        frozen = read(directory / 'training_frozen.json')
        full = read(directory / 'full_replay.json')
        trained = read(directory / 'trained_models.json')
        validation = read(directory / 'training_validation.json')
        require(full['training_freeze_sha256'] == trained['training_freeze_sha256'] == sha(directory / 'training_frozen.json')
                and trained['training_validation_sha256'] == sha(directory / 'training_validation.json')
                and validation['full_replay_sha256'] == sha(directory / 'full_replay.json'), 'training proof links differ')
        require(set(trained['models']) == set(validation['records']) == {'low51', 'low52', 'low53'},
                'low trained seed inventory differs')
        records = {'high51': full['record'], **validation['records']}
        for name, record in records.items():
            condition, seed = ('high' if name == 'high51' else 'low'), int(name[-2:])
            output, data, state, scalars = audit_record(root, record, condition, seed, 4096, 250,
                                                       original, teacher, initializers[condition])
            require(datetime.fromisoformat(frozen['frozen_at']) < datetime.fromisoformat(data['initial']['created_utc']),
                    'main training preceded training freeze')
            initial_pair(capacities[name]['initial'], data['initial'], project_iterations=True)
            require(capacities[name]['schedule']['episode_randomization'] == data['schedule']['episode_randomization'],
                    'same-arm capacity/main actual randomization differs')
            if condition == 'high':
                output['historical_replay'] = replay(root, data, state, scalars, references['stages']['main']['seed51'], 250)
            else:
                item = trained['models'][name]
                expected_path = f'artifacts/terrain_demo/lr_continuation_v24/runs/{name}/model_249.pt'
                require(item['checkpoint'] == expected_path and item['source_checkpoint'] == output['checkpoint']
                        and item['sha256'] == output['checkpoint_sha256'] == sha(inside(root, expected_path))
                        and item['iteration'] == 249 and item['training_seed'] == seed
                        and item['transitions'] == 32768000 and item['learning_rate'] == 1.e-5,
                        'fresh low final249 model binding differs')
            results['main/' + name] = output
        require(len({trained['models'][key]['source_checkpoint'] for key in trained['models']}) == 3,
                'fresh low model aliases')
    total = sum(item['transitions'] for item in results.values())
    require(total == (1605632 if development_only else 132677632), 'new training budget differs')
    sources = ['scripts/audit_lr_v24_training.py', 'tests/test_lr_tensor_audit_v24.py']
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                phase='development' if development_only else 'complete_training', cpu_only=True,
                production_validator_imports=False, source_sha256={name: sha(root / name) for name in sources},
                evidence_sha256={str((directory / name).relative_to(root)): sha(directory / name) for name in proof_paths},
                initializers={'high_byte_identical': True, 'low_deserialized_change_only_empty_Adam_LR': True},
                runs=results, total_new_transitions=total,
                limitation='CPU deserialization/log verification; not an independent replay of simulator physics or evidence generation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--development-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(development_only=args.development_only)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'phase': result['phase'], 'runs': len(result['runs']),
                      'transitions': result['total_new_transitions'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
