"""Seed reachability, strict launch boundaries and inherited no-cost task invariants."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('seed_launcher_v22', ROOT / 'scripts/seed_continuation_v22.py')
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def arguments(seed=51, envs=4096, iterations=250):
    return ['--device', 'cuda:1', '--seed', str(seed), '--num_envs', str(envs),
            '--max_iterations', str(iterations), '--resume', '--load_run', 'v22_init', '--checkpoint', 'model_0.pt']


@pytest.mark.parametrize('seed', [51, 52, 53])
@pytest.mark.parametrize('envs,iterations,ramp', [(256, 2, 32), (4096, 2, 4000), (4096, 250, 4000)])
def test_all_declared_seeds_and_budgets(seed, envs, iterations, ramp):
    result = launcher.training_arguments(arguments(seed, envs, iterations), training_seed=seed, ramp_steps=ramp)
    launcher.validate_budget(result, ramp, seed)
    assert 'agent.device=cuda:1' in result
    assert 'env.rewards.contact_slip.params.mode=no_cost' in result
    assert 'env.scene.terrain.terrain_generator.seed=110' in result
    assert 'env.rewards.contact_slip.weight=1.0' in result


@pytest.mark.parametrize('seed', [50, 54, True, '51'])
def test_undeclared_or_coerced_seed_rejected(seed):
    with pytest.raises(ValueError):
        launcher.training_arguments(arguments(), training_seed=seed)


@pytest.mark.parametrize('extra', [['--seed', '52'], ['agent.seed=52'], ['env.seed=52'],
    ['agent.device=cuda:0'], ['env.rewards.contact_slip.weight=0.0'], ['env.rewards.contact_slip.func=other'],
    ['env.scene.terrain.terrain_generator.seed=111'], ['agent.algorithm.teacher_coef=0.0'], ['--task=other']])
def test_config_or_seed_substitution_rejected(extra):
    with pytest.raises(ValueError):
        launcher.training_arguments(arguments() + extra, training_seed=51)


@pytest.mark.parametrize('arm', ['immediate', 'ramped'])
def test_only_no_cost_arm_allowed(arm):
    with pytest.raises(ValueError):
        launcher.training_arguments(arguments(), arm, training_seed=51)


def test_forwarded_and_requested_seed_must_match():
    with pytest.raises(ValueError, match='forwarded seed'):
        launcher.training_arguments(arguments(52), training_seed=51)


@pytest.mark.parametrize('envs,iterations,ramp', [(256, 2, 4000), (4096, 250, 32), (4096, 251, 4000)])
def test_nonstandard_budget_rejected(envs, iterations, ramp):
    values = launcher.training_arguments(arguments(52, envs, iterations), training_seed=52, ramp_steps=ramp)
    with pytest.raises(ValueError):
        launcher.validate_budget(values, ramp, 52)


@pytest.mark.parametrize('seed', [51, 52, 53])
def test_seed_proof_preserves_actual_saved_seeds(seed):
    saved = {'normalized': {'agent': {'seed': str(seed)}, 'env': {'seed': str(seed)}}}
    before = deepcopy(saved)
    result = launcher.seed_configuration(saved, seed, seed)
    assert result == dict(requested=seed, saved_agent=seed, saved_environment=seed, live_environment=seed)
    assert saved == before
    with pytest.raises(ValueError):
        launcher.seed_configuration(saved, seed + 1, seed)
    saved['normalized']['agent']['seed'] = '51' if seed != 51 else '52'
    with pytest.raises(ValueError):
        launcher.seed_configuration(saved, seed, seed)


def test_episode_horizon_assignment_forwards_to_real_environment():
    raw = SimpleNamespace(episode_length_buf=torch.zeros(4, dtype=torch.long))
    class Vector:
        @property
        def episode_length_buf(self):
            return raw.episode_length_buf
        @episode_length_buf.setter
        def episode_length_buf(self, value):
            raw.episode_length_buf = value
    wrapper = launcher.ContinuationSamplingEnv(Vector(), raw, wrap_count_getter=lambda: torch.zeros(4))
    assigned = torch.tensor([1, 2, 0, 4])
    wrapper.episode_length_buf = assigned
    assert raw.episode_length_buf is assigned
    assert wrapper.randomization_proof['matches_raw_environment']
    assert 'episode_length_buf' not in wrapper.__dict__


def test_no_new_reward_implementation_and_environment_inherits_unchanged():
    tree = ast.parse((ROOT / 'src/week03_ant/tasks/seed_continuation_v22_cfg.py').read_text())
    env = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SeedContinuationTrainAntEnvCfg')
    assert len(env.body) == 1 and isinstance(env.body[0], ast.Pass)
    assert [ast.unparse(base) for base in env.bases] == ['ContinuationContactTrainAntEnvCfg']
    assert not any(isinstance(node, ast.FunctionDef) and node.name in ('__call__', 'geometry') for node in ast.walk(tree))


def test_raw_failure_evidence_saved_before_guards_and_on_learning_failure():
    tree = ast.parse((ROOT / 'scripts/seed_continuation_v22.py').read_text())
    learn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'learn')
    block = next(node for node in learn.body if isinstance(node, ast.Try))
    saves = [node for node in ast.walk(ast.Module(body=block.finalbody, type_ignores=[]))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'save_json']
    audit = next(node for node in ast.walk(learn) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'audit_schedule')
    assert len(saves) == 2 and all(node.lineno < audit.lineno for node in saves)
