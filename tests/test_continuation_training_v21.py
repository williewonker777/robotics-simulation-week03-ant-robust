"""Strict launch and real-environment episode randomization regression proofs."""
import importlib.util
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('curriculum_launcher', ROOT / 'scripts/contact_continuation_v21.py')
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def base(envs=4096, iterations=250):
    return ['--device', 'cuda:1', '--num_envs', str(envs), '--max_iterations', str(iterations), '--seed', '51',
            '--resume', '--load_run', 'v21_init', '--checkpoint', 'model_0.pt']


def test_arms_differ_only_by_coefficient_mode():
    immediate, ramped = [launcher.training_arguments(base(), arm) for arm in ('immediate', 'no_cost')]
    assert [(a, b) for a, b in zip(immediate, ramped) if a != b] == [
        ('env.rewards.contact_slip.params.mode=immediate', 'env.rewards.contact_slip.params.mode=no_cost')]
    assert 'env.rewards.contact_slip.weight=1.0' in immediate
    launcher.validate_budget(ramped, 4000)


@pytest.mark.parametrize('args', [
    ['env.rewards.contact_slip.weight=0.0'], ['env.rewards.contact_slip.func=other'],
    ['env.scene.robot=other'], ['+env.rewards.contact_slip.params.mode=no_cost'],
    ['agent.algorithm.teacher_coef=0.0'], ['--task=other'], ['--task', 'other'],
    ['agent.policy.command_mode=masked'], ['env.rewards.contact_slip.params.ramp_steps=32'],
    ['--num_envs', '4096', '--num_envs', '256'],
])
def test_rejects_undeclared_substitutions(args):
    with pytest.raises(ValueError):
        launcher.training_arguments(args, 'immediate')


@pytest.mark.parametrize('envs,iterations,ramp,valid', [(256, 2, 32, True), (4096, 2, 4000, True),
    (4096, 250, 4000, True), (256, 2, 4000, False), (4096, 250, 32, False), (4096, 251, 4000, False)])
def test_strict_stage_budget(envs, iterations, ramp, valid):
    args = launcher.training_arguments(base(envs, iterations), 'no_cost', ramp_steps=ramp)
    if valid:
        launcher.validate_budget(args, ramp)
    else:
        with pytest.raises(ValueError):
            launcher.validate_budget(args, ramp)


def test_episode_length_assignment_reaches_actual_environment():
    raw = SimpleNamespace(episode_length_buf=torch.zeros(4, dtype=torch.long))
    class Vector:
        @property
        def episode_length_buf(self):
            return raw.episode_length_buf
        @episode_length_buf.setter
        def episode_length_buf(self, value):
            raw.episode_length_buf = value
    wrapper = launcher.ContinuationSamplingEnv(Vector(), raw, wrap_count_getter=lambda: torch.zeros(4))
    random_horizons = torch.tensor([3, 11, 0, 8])
    wrapper.episode_length_buf = random_horizons
    assert raw.episode_length_buf is random_horizons
    assert wrapper.randomization_proof['matches_raw_environment']
    assert wrapper.randomization_proof['nonzero'] == 3
    assert 'episode_length_buf' not in wrapper.__dict__


def test_shadowed_vector_assignment_is_rejected():
    raw = SimpleNamespace(episode_length_buf=torch.zeros(4, dtype=torch.long))
    vector = SimpleNamespace(episode_length_buf=raw.episode_length_buf)
    wrapper = launcher.ContinuationSamplingEnv(vector, raw, wrap_count_getter=lambda: torch.zeros(4))
    with pytest.raises(ValueError, match='shadowed'):
        wrapper.episode_length_buf = torch.ones(4, dtype=torch.long)


@pytest.mark.parametrize('device,agent', [('cuda:0', 'cuda:1'), ('cuda:1', 'cuda:0'), ('cpu', 'cpu')])
def test_agent_simulator_device_mismatch_rejected(device, agent):
    args = base()
    args[args.index('--device') + 1] = device
    with pytest.raises(ValueError):
        launcher.training_arguments(args + ['agent.device=' + agent], 'immediate')


def test_agent_device_explicitly_matches_simulator():
    args = launcher.training_arguments(base() + ['agent.device=cuda:1'], 'immediate')
    assert args.count('agent.device=cuda:1') == 1


def test_raw_reward_evidence_is_saved_in_finally_before_validation():
    tree = ast.parse((ROOT / 'scripts/contact_continuation_v21.py').read_text())
    learn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'learn')
    block = next(n for n in learn.body if isinstance(n, ast.Try))
    saves = [n for n in ast.walk(ast.Module(body=block.finalbody, type_ignores=[]))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'save_json']
    assert len(saves) == 2
    audit = next(n for n in ast.walk(learn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'audit_schedule')
    assert all(save.lineno < audit.lineno for save in saves)


@pytest.mark.parametrize('envs,iterations,ramp,valid', [(256, 2, 32, True), (4096, 2, 4000, True),
                                                      (4096, 250, 4000, False)])
def test_immediate_is_development_only(envs, iterations, ramp, valid):
    args = launcher.training_arguments(base(envs, iterations), 'immediate', ramp_steps=ramp)
    if valid:
        launcher.validate_budget(args, ramp)
    else:
        with pytest.raises(ValueError, match='positive-control'):
            launcher.validate_budget(args, ramp)


def test_ramped_treatment_is_not_a_v21_arm():
    with pytest.raises(ValueError):
        launcher.training_arguments(base(), 'ramped')
