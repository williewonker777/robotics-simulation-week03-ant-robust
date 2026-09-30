"""CPU regression tests for exact LR-only initialization and actual Adam guards."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from week03_ant import lr_study_v24 as study

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import lr_continuation_v24 as launcher
from run_lr_training_v24 import validate_lr


def args(condition='low', seed=51, envs=4096, iterations=250):
    return ['--device', 'cuda:1', '--seed', str(seed), '--num_envs', str(envs),
            '--max_iterations', str(iterations), '--resume', '--load_run', f'v24_{condition}_init', '--checkpoint', 'model_0.pt']


@pytest.mark.parametrize('condition', ['high', 'low'])
@pytest.mark.parametrize('seed', [51, 52, 53])
@pytest.mark.parametrize('envs,iterations,ramp', [(256, 2, 32), (4096, 2, 4000), (4096, 250, 4000)])
def test_launch_fixed_lr(condition, seed, envs, iterations, ramp):
    values = launcher.training_arguments(args(condition, seed, envs, iterations), training_seed=seed, ramp_steps=ramp, condition=condition)
    if (envs == 256 and seed != 51) or (iterations == 250 and condition == 'high' and seed != 51):
        with pytest.raises(ValueError): launcher.validate_budget(values, ramp, seed, condition)
    else:
        launcher.validate_budget(values, ramp, seed, condition)
    assert f'agent.algorithm.learning_rate={study.learning_rate(condition)}' in values


@pytest.mark.parametrize('extra', ['agent.algorithm.learning_rate=0.1', 'agent.algorithm.schedule=adaptive',
                                    'agent.algorithm.desired_kl=0.01', 'env.seed=52', 'agent.device=cuda:0'])
def test_forbidden_override(extra):
    with pytest.raises(ValueError):
        launcher.training_arguments(args() + [extra], training_seed=51)


def original():
    return torch.load(study.INIT_SOURCE, map_location='cpu', weights_only=False)


def test_lr_only_initializer_exact_and_no_input_mutation():
    high = original()
    low = deepcopy(high)
    low['optimizer_state_dict']['param_groups'][0]['lr'] = 1.e-5
    before = deepcopy(low)
    assert study.validate_init_difference(high, low)
    assert low['optimizer_state_dict'] == before['optimizer_state_dict']
    assert high['optimizer_state_dict']['param_groups'][0]['lr'] == 1.e-4


@pytest.mark.parametrize('mutation', ['std', 'infos', 'iteration', 'beta', 'state', 'lr'])
def test_initializer_rejects_other_changes(mutation):
    high, low = original(), original()
    low['optimizer_state_dict']['param_groups'][0]['lr'] = 1.e-5
    if mutation == 'std':
        low['model_state_dict']['std'][0] += .01
    elif mutation == 'infos':
        low['infos'] = {'different': True}
    elif mutation == 'iteration':
        low['iter'] = 1
    elif mutation == 'beta':
        low['optimizer_state_dict']['param_groups'][0]['betas'] = (.8, .999)
    elif mutation == 'state':
        low['optimizer_state_dict']['state'][0] = {'step': 1}
    else:
        low['optimizer_state_dict']['param_groups'][0]['lr'] = 1.e-4
    with pytest.raises(ValueError):
        study.validate_init_difference(high, low)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.std = torch.nn.Parameter(torch.ones(2))
        self.actor, self.critic = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
        self.teacher = torch.nn.Linear(2, 2)
        self.teacher.requires_grad_(False)


def algorithm():
    policy = TinyPolicy()
    alg = SimpleNamespace(policy=policy, learning_rate=1.e-5, schedule='fixed', desired_kl=None)
    alg.optimizer = torch.optim.Adam(policy.parameters(), lr=1.e-5)
    def update():
        for _ in range(20):
            alg.optimizer.zero_grad()
            (policy.actor.weight.sum() + policy.critic.weight.sum() + policy.std.sum()).backward()
            alg.optimizer.step()
    alg.update = update
    return alg


def guard(alg):
    return study.LearningRateAudit(alg, dict(learning_rate='1e-05', schedule='fixed', desired_kl='null'), 'low')


def test_all_updates_and_actual_adam_steps_recorded_without_new_logs():
    alg = algorithm()
    audit = guard(alg)
    audit.install()
    alg.update()
    alg.update()
    proof = audit.finish(2)
    validate_lr(proof, 'low', 2)
    assert proof['update_count'] == 2 and proof['adam_step_count'] == 40
    assert proof['adam_steps'][20]['update_index'] == 1
    assert int(next(iter(alg.optimizer.state.values()))['step']) == 40


@pytest.mark.parametrize('target', ['algorithm_lr', 'optimizer_lr', 'schedule', 'kl', 'coverage', 'options'])
def test_wrong_loaded_runtime_rejected_before_update(target):
    alg = algorithm()
    if target == 'algorithm_lr': alg.learning_rate = 1.e-4
    elif target == 'optimizer_lr': alg.optimizer.param_groups[0]['lr'] = 1.e-4
    elif target == 'schedule': alg.schedule = 'adaptive'
    elif target == 'kl': alg.desired_kl = .01
    elif target == 'coverage': alg.optimizer.param_groups[0]['params'].pop()
    else: alg.optimizer.param_groups[0]['betas'] = (.8, .9)
    if target == 'options':
        audit = guard(alg)
        alg.optimizer.param_groups[0]['betas'] = (.9, .999)
        with pytest.raises(ValueError): audit.observe()
    else:
        with pytest.raises(ValueError): guard(alg)
    assert not alg.optimizer.state


def test_drift_during_actual_adam_step_rejected_after_step():
    alg = algorithm()
    original_step = alg.optimizer.step
    def changed(*a, **kw):
        value = original_step(*a, **kw)
        alg.optimizer.param_groups[0]['lr'] = 1.e-4
        return value
    alg.optimizer.step = changed
    audit = guard(alg)
    audit.install()
    with pytest.raises(ValueError, match='LR'):
        alg.update()
    assert len(audit.steps) == 1 and 'after' not in audit.steps[0]


def test_teacher_drift_rejected_at_update_boundary():
    alg = algorithm()
    audit = guard(alg)
    audit.install()
    with torch.no_grad(): alg.policy.teacher.weight.add_(1)
    with pytest.raises(ValueError, match='teacher'): alg.update()
    assert not audit.steps


def test_lr_raw_tampering_rejected():
    alg = algorithm()
    audit = guard(alg); audit.install(); alg.update()
    proof = audit.finish(1)
    proof['adam_steps'][0]['after']['group_learning_rates'] = [1.e-4]
    with pytest.raises(ValueError): validate_lr(proof, 'low', 1)


def test_failed_lr_guard_retains_partial_trace_for_exclusive_failure_evidence(tmp_path):
    alg = algorithm()
    audit = guard(alg)
    audit.install()
    alg.optimizer.param_groups[0]['lr'] = 1.e-4
    with pytest.raises(ValueError):
        alg.update()
    proof = audit.finish(1)
    path = tmp_path / 'failure.json'
    study.save_json(path, proof)
    saved = study.read(path)
    assert not saved['passed'] and saved['failure_reason']
    assert saved['adam_step_count'] == 0
    with pytest.raises(FileExistsError): study.save_json(path, proof)


def test_real_checkpoint_high_bytecopy_low_lr_only_and_exclusive(tmp_path, monkeypatch):
    source = tmp_path / 'historical.pt'
    source.write_bytes(study.INIT_SOURCE.read_bytes())
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'INIT_SOURCE', source)
    high_path, low_path = tmp_path / 'high/model_0.pt', tmp_path / 'low/model_0.pt'
    high = study.prepare_checkpoint(high_path, 'high')
    low = study.prepare_checkpoint(low_path, 'low')
    assert high_path.read_bytes() == source.read_bytes()
    assert high['sha256'] == study.INIT_SHA and low['learning_rate'] == 1.e-5
    assert study.validate_init_difference(torch.load(high_path, weights_only=False), torch.load(low_path, weights_only=False))
    with pytest.raises(FileExistsError): study.prepare_checkpoint(low_path, 'low')


def test_loaded_checkpoint_lr_is_rejected_not_coerced():
    state = original()
    parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in state['optimizer_state_dict']['param_groups'][0]['params']]
    optimizer = torch.optim.Adam(parameters, lr=1.e-5)
    optimizer.load_state_dict(state['optimizer_state_dict'])
    with pytest.raises(ValueError, match='never coerce'):
        study.validate_training_start(state['model_state_dict'], optimizer, 51, 'low')
    assert optimizer.param_groups[0]['lr'] == 1.e-4
    study.validate_training_start(state['model_state_dict'], optimizer, 51, 'high')
