"""CPU adapter contract with mocked simulator imports, not a physical smoke test."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from week03_ant.foothold_math import GRID_RAYS
from week03_ant.footmap_math import FOOT_NAMES

ROOT = Path(__file__).resolve().parents[1]


def quat_apply(q, v):
    xyz = q[..., 1:]
    cross = torch.cross(xyz, v, dim=-1)
    return v + 2 * (q[..., :1] * cross + torch.cross(xyz, cross, dim=-1))


@pytest.fixture
def adapter(monkeypatch):
    class Base:
        def __init__(self, cfg, env):
            pass

    class Train:
        def __post_init__(self):
            self.overrides_applied = True

    modules = {
        'isaaclab.managers': dict(ManagerTermBase=Base, RewardTermCfg=SimpleNamespace),
        'isaaclab.utils': dict(configclass=lambda cls: cls),
        'isaaclab.utils.math': dict(quat_apply=quat_apply),
        'week03_ant.tasks.foothold_v9_cfg': dict(FootholdRewardsCfg=type('Rewards', (), {}),
                                               FootholdTrainAntEnvCfg=Train),
        'week03_ant.tasks.rough_v5_cfg': dict(FLAT_PLANE_HEIGHT=-50.),
        'week03_ant.tasks.lanes': dict(get_lane_state=lambda env: env.lane),
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        'week03_ant.tasks._posture_cpu_test', ROOT / 'src/week03_ant/tasks/posture_v13_cfg.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment():
    identity = torch.tensor([[1., 0., 0., 0.]])
    data = SimpleNamespace(
        root_pos_w=torch.tensor([[0., 0., .44]]), root_quat_w=identity.clone(),
        root_link_lin_vel_w=torch.tensor([[1., 0., 0.]]),
        body_link_quat_w=identity[:, None].repeat(1, 4, 1),
        body_link_pos_w=torch.tensor([[[.2, .2, .08], [-.2, .2, .08],
                                       [-.2, -.2, .08], [.2, -.2, .08]]]),
        body_link_lin_vel_w=torch.tensor([[[1., 0., 0.]]]).repeat(1, 4, 1),
        body_link_ang_vel_w=torch.zeros(1, 4, 3),
    )
    robot = SimpleNamespace(data=data, find_bodies=lambda names, preserve_order: (list(range(4)), list(FOOT_NAMES)))
    calls = []
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(offset=SimpleNamespace(pos=(.4, 0., 2.)), max_distance=4.),
        data=SimpleNamespace(ray_hits_w=torch.zeros(1, GRID_RAYS, 3)),
        update=lambda dt, force_recompute: calls.append((dt, force_recompute)),
    )
    env = SimpleNamespace(scene={'robot': robot, 'height_scanner': sensor}, device='cpu', num_envs=1,
                          lane=SimpleNamespace(target_direction=lambda pos: torch.tensor([[1., 0.]])))
    return env, data, sensor, calls


def test_adapter_geometry_capsule_and_reset_freshness(adapter):
    env, data, sensor, calls = environment()
    term = adapter.PostureReward(None, env)
    out = term.geometry(env)
    torch.testing.assert_close(out['body_clearance'], torch.tensor([.44]))
    torch.testing.assert_close(out['foot_clearance'], torch.zeros(1, 4), atol=1e-7, rtol=0)
    assert out['foot_valid'].all()
    before = term(env)
    data.root_pos_w[:, 2] = .58  # same term, new root and scan after reset
    after = term(env)
    assert not torch.equal(before, after)
    sensor.data.ray_hits_w[..., 2] = float('inf')
    assert term(env).eq(0).all()
    assert calls == [(0., True)] * 4


def test_link_origin_angular_velocity_and_world_lane_projection(adapter, monkeypatch):
    env, data, _, _ = environment()
    env.lane.target_direction = lambda pos: torch.tensor([[0., 1.]])
    # Root heading points -x, whereas lane points +y: neither norm nor body x.
    data.root_quat_w[:] = torch.tensor([[0., 0., 0., 1.]])
    data.root_link_lin_vel_w[:] = torch.tensor([[8., -2., 0.]])
    data.body_link_lin_vel_w[:] = torch.tensor([8., -2., 0.])
    data.body_link_ang_vel_w[..., 2] = 5.
    # COM velocities must not be used.
    data.body_lin_vel_w = torch.full((1, 4, 3), 999.)
    data.root_lin_vel_w = torch.full((1, 3), 999.)
    captured = {}
    original = adapter.terrain_posture_terms

    def capture(*args):
        captured['args'] = args
        return original(*args)

    monkeypatch.setattr(adapter, 'terrain_posture_terms', capture)
    out = adapter.PostureReward(None, env).geometry(env)
    torch.testing.assert_close(out['forward_speed'], torch.tensor([-2.]))
    torch.testing.assert_close(captured['args'][3], torch.tensor([[2., -2., -2., 2.]]))
    torch.testing.assert_close(captured['args'][2][0, 0], torch.tensor([-.6, -.6, -.36]))
    assert out['flat_speed_bonus'].eq(0).all() and out['foot_cost'].eq(0).all()


def test_world_translation_invariance(adapter):
    env, data, sensor, _ = environment()
    term = adapter.PostureReward(None, env)
    before = term.geometry(env)
    translation = torch.tensor([10., -7., 2.])
    data.root_pos_w += translation
    data.body_link_pos_w += translation
    sensor.data.ray_hits_w += translation
    after = term.geometry(env)
    for key in before:
        torch.testing.assert_close(after[key], before[key], atol=1e-5, rtol=1e-5)


def test_scan_clipping_shape_and_body_name_guards(adapter):
    env, _, sensor, _ = environment()
    term = adapter.PostureReward(None, env)
    sensor.data.ray_hits_w[..., 2] = -2.  # finite but clipped depth: abstain
    assert not term.geometry(env)['valid'].any()
    sensor.data.ray_hits_w = torch.zeros(1, 221, 3)
    with pytest.raises(ValueError, match='825'):
        term.geometry(env)
    env.scene['robot'].find_bodies = lambda names, preserve_order: ([0], ['wrong'])
    with pytest.raises(ValueError, match='unexpected feet'):
        adapter.PostureReward(None, env)


def test_inheritance_preserves_post_init_and_additive_weight(adapter):
    cfg = adapter.AdaptivePostureTrainEnvCfg()
    cfg.__post_init__()
    assert cfg.overrides_applied
    assert adapter.AdaptivePostureRewardsCfg.adaptive_posture.weight == 1.
    assert adapter.AdaptivePostureRewardsCfg.adaptive_posture.func is adapter.PostureReward
    # Source-level contract accompanies mocked inheritance; actual Isaac config
    # parity must additionally be verified in the simulator smoke run.
    text = (ROOT / 'src/week03_ant/tasks/posture_v13_cfg.py').read_text()
    assert 'class AdaptivePostureTrainEnvCfg(FootholdTrainAntEnvCfg)' in text
    assert 'class AdaptivePostureRewardsCfg(FootholdRewardsCfg)' in text
    assert 'def __post_init__' not in text


def test_registration_reuses_frozen_runner(monkeypatch):
    calls = []
    gym = ModuleType('gymnasium')
    gym.registry = {}

    def register(**kwargs):
        calls.append(kwargs)
        gym.registry[kwargs['id']] = kwargs

    gym.register = register
    monkeypatch.setitem(sys.modules, 'gymnasium', gym)
    spec = importlib.util.spec_from_file_location(
        '_posture_registry_cpu_test', ROOT / 'src/week03_ant/tasks/posture_v13.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spec.loader.exec_module(module)
    assert len(calls) == 1
    assert calls[0]['id'] == 'Week03-Ant-Adaptive-Posture-Train-v13'
    assert calls[0]['kwargs'] == {
        'env_cfg_entry_point': 'week03_ant.tasks.posture_v13_cfg:AdaptivePostureTrainEnvCfg',
        'rsl_rl_cfg_entry_point': 'week03_ant.tasks.agents.prior_v10_cfg:PriorAntPPORunnerCfg',
    }
