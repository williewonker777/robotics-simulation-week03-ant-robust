"""Mocked simulator adapter/config contracts; real GPU parity is a separate gate."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_posture_adapter import adapter, environment
from test_command_adapter import command_adapter

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def direction_adapter(command_adapter, adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, 'week03_ant.tasks.command_v14_cfg', command_adapter)
    monkeypatch.setitem(sys.modules, 'week03_ant.tasks.posture_v13_cfg', adapter)
    spec = importlib.util.spec_from_file_location(
        'week03_ant.tasks._direction_cpu_test', ROOT / 'src/week03_ant/tasks/direction_v15_cfg.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_com_accessors_fresh_partial_reset_and_portal(direction_adapter):
    env, data, sensor, calls = environment()
    env.num_envs = 2
    data.root_pos_w = data.root_pos_w.repeat(2, 1)
    data.root_quat_w = data.root_quat_w.repeat(2, 1)
    data.root_com_lin_vel_w = torch.tensor([[2., 1., 9.], [2., 0., -8.]])
    data.root_com_ang_vel_w = torch.zeros(2, 3)
    # Stale link aliases must not be read; COM is the explicit contract.
    data.root_link_lin_vel_w = torch.full((2, 3), float('nan'))
    data.root_link_ang_vel_w = torch.full((2, 3), float('nan'))
    env.lane.target_direction = lambda pos: torch.stack((torch.ones_like(pos[:, 0]), -pos[:, 1]), -1)
    term = direction_adapter.DirectionReward(None, env)
    rng = torch.get_rng_state().clone()
    initial = term.geometry(env)
    assert initial['valid'].all()
    assert initial['lateral_velocity'].tolist() == [1., 0.]
    assert initial['reward'][0] < 0 and initial['reward'][1] == 0
    data.root_com_lin_vel_w[0, 1] = 0  # partial reset, same term instance
    reset = term.geometry(env)
    assert reset['reward'].eq(0).all()
    data.root_pos_w[0] += torch.tensor([17., -8., 2.])
    portal = term.geometry(env)
    assert portal['heading_error'][0] > 1
    assert portal['target_yaw_rate'][0] == 1
    for key in initial:
        assert torch.equal(portal[key][1], initial[key][1])
    data.root_com_ang_vel_w[0, 2] = 1
    assert term.geometry(env)['yaw_error'][0] == 0
    assert torch.equal(term(env), term.geometry(env)['reward'])
    assert torch.equal(rng, torch.get_rng_state())
    assert calls == []  # no sensor query, simulation step, or cache update
    assert set(vars(term)) == {'robot'}


def test_only_one_reward_added_all_other_config_inherited(direction_adapter, command_adapter, adapter):
    module = direction_adapter
    assert module.DirectionalRewardsCfg.__bases__ == (adapter.AdaptivePostureRewardsCfg,)
    assert module.DirectionTrainAntEnvCfg.__bases__ == (command_adapter.CommandTrainAntEnvCfg,)
    assert module.DirectionAntPPORunnerCfg.__bases__ == (command_adapter.CommandAntPPORunnerCfg,)
    public = lambda cls: {key for key in vars(cls) if not key.startswith('_')}
    assert public(module.DirectionalRewardsCfg) == {'directional_stability'}
    assert public(module.DirectionTrainAntEnvCfg) == {'rewards'}
    assert public(module.DirectionAntPPORunnerCfg) == {'experiment_name'}
    cfg = module.DirectionTrainAntEnvCfg()
    cfg.__post_init__()
    assert cfg.overrides_applied
    assert cfg.observations is command_adapter.CommandTrainAntEnvCfg.observations
    assert cfg.rewards.adaptive_posture is adapter.AdaptivePostureRewardsCfg.adaptive_posture
    assert cfg.rewards.directional_stability.func is module.DirectionReward
    assert cfg.rewards.directional_stability.weight == 1.
    assert module.DirectionAntPPORunnerCfg.policy is command_adapter.CommandAntPPORunnerCfg.policy
    assert module.DirectionAntPPORunnerCfg.algorithm is command_adapter.CommandAntPPORunnerCfg.algorithm
    assert module.DirectionAntPPORunnerCfg.experiment_name == 'week03_ant_direction_v15'


def test_train_only_idempotent_registration(monkeypatch):
    gym = ModuleType('gymnasium')
    gym.registry = {}
    calls = []
    def register(**kwargs):
        calls.append(kwargs)
        gym.registry[kwargs['id']] = kwargs
    gym.register = register
    monkeypatch.setitem(sys.modules, 'gymnasium', gym)
    spec = importlib.util.spec_from_file_location('_direction_registry_test', ROOT / 'src/week03_ant/tasks/direction_v15.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spec.loader.exec_module(module)
    assert len(calls) == 1
    item = calls[0]
    assert item['id'] == 'Week03-Ant-Direction-v15-Train-v0'
    assert item['kwargs']['env_cfg_entry_point'].endswith(':DirectionTrainAntEnvCfg')
    assert item['kwargs']['rsl_rl_cfg_entry_point'].endswith(':DirectionAntPPORunnerCfg')
