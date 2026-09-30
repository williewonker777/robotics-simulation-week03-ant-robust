"""Mocked import contract and real frozen geometry; GPU parity is a separate gate."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_posture_adapter import adapter, environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def command_adapter(adapter, monkeypatch):
    managers = sys.modules['isaaclab.managers']
    monkeypatch.setattr(managers, 'ObservationTermCfg', SimpleNamespace, raising=False)
    foothold = sys.modules['week03_ant.tasks.foothold_v9_cfg']
    class Observation:
        class PolicyCfg:
            original_prefix = 'unchanged'
    for name in ('FootholdObservationCfg', 'FootholdTrainObservationCfg'):
        monkeypatch.setattr(foothold, name, Observation, raising=False)
    monkeypatch.setattr(foothold, 'FootholdEvalAntEnvCfg', type('Eval', (), {}), raising=False)
    monkeypatch.setitem(sys.modules, 'week03_ant.tasks.posture_v13_cfg', adapter)
    prior = ModuleType('week03_ant.tasks.agents.prior_v10_cfg')
    prior.PriorPolicyCfg = SimpleNamespace
    prior.PriorAntPPORunnerCfg = type('Runner', (), {'algorithm': object()})
    monkeypatch.setitem(sys.modules, prior.__name__, prior)
    spec = importlib.util.spec_from_file_location(
        'week03_ant.tasks._command_cpu_test', ROOT / 'src/week03_ant/tasks/command_v14_cfg.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_geometry_after_partial_reset_and_portal(command_adapter):
    env, data, sensor, calls = environment()
    for name, value in vars(data).items():
        if isinstance(value, torch.Tensor):
            setattr(data, name, value.repeat(2, *([1] * (value.ndim - 1))))
    sensor.data.ray_hits_w = sensor.data.ray_hits_w.repeat(2, 1, 1)
    env.num_envs = 2
    env.lane.target_direction = lambda pos: torch.tensor([[1., 0.], [1., 0.]])
    term = command_adapter.CommandObservation(None, env)
    initial = term(env)
    torch.testing.assert_close(initial, torch.tensor([[0., 0., 1.], [0., 0., 1.]]))
    data.root_pos_w[0, 2] = .58  # partial reset: second environment must not change
    reset = term(env)
    torch.testing.assert_close(reset[0], torch.tensor([0., 1., 1.]))
    assert torch.equal(reset[1], initial[1])
    # World translation models portal relocation with fresh root/link/ray states.
    delta = torch.tensor([17., -8., 2.])
    data.root_pos_w[0] += delta
    data.body_link_pos_w[0] += delta
    sensor.data.ray_hits_w[0] += delta
    torch.testing.assert_close(term(env), reset, atol=1e-5, rtol=1e-5)
    sensor.data.ray_hits_w[0, :, 2] = float('inf')
    assert term(env)[0].eq(0).all()
    data.root_link_lin_vel_w[1, 0] = float('nan')
    assert term(env).eq(0).all()
    assert calls == [(0., True)] * 5


def test_train_eval_inherit_prefix_reward_and_runner(command_adapter, adapter):
    module = command_adapter
    assert issubclass(module.CommandTrainAntEnvCfg, adapter.AdaptivePostureTrainEnvCfg)
    assert module.CommandTrainAntEnvCfg.rewards.adaptive_posture.weight == 1.
    cfg = module.CommandTrainAntEnvCfg()
    cfg.__post_init__()
    assert cfg.overrides_applied
    for cls in (module.CommandObservationCfg, module.CommandTrainObservationCfg):
        assert cls.policy.original_prefix == 'unchanged'
        assert cls.policy.posture_command.func is module.CommandObservation
    assert module.CommandAntPPORunnerCfg.policy.class_name == 'CommandPriorActorCritic'
    assert module.CommandAntPPORunnerCfg.policy.command_mode == 'conditioned'
    assert module.CommandAntPPORunnerCfg.policy.require_config_match
    assert module.CommandAntPPORunnerCfg.algorithm is sys.modules[
        'week03_ant.tasks.agents.prior_v10_cfg'].PriorAntPPORunnerCfg.algorithm


def test_opt_in_registration_is_idempotent(monkeypatch):
    gym = ModuleType('gymnasium')
    gym.registry = {}
    calls = []
    def register(**kwargs):
        calls.append(kwargs)
        gym.registry[kwargs['id']] = kwargs
    gym.register = register
    monkeypatch.setitem(sys.modules, 'gymnasium', gym)
    spec = importlib.util.spec_from_file_location('_command_registry_test', ROOT / 'src/week03_ant/tasks/command_v14.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spec.loader.exec_module(module)
    assert len(calls) == 2
    for item, stage in zip(calls, ('Train', 'Eval')):
        assert item['id'] == f'Week03-Ant-Command-v14-{stage}-v0'
        assert item['kwargs']['env_cfg_entry_point'].endswith(f':Command{stage}AntEnvCfg')
        assert item['kwargs']['rsl_rl_cfg_entry_point'].endswith(':CommandAntPPORunnerCfg')
