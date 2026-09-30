"""Mocked contact sensor wiring, kinematics and fail-closed guards."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_posture_adapter import adapter, environment
from test_command_adapter import command_adapter
from week03_ant.footmap_math import FOOT_NAMES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def contact_adapter(command_adapter, adapter, monkeypatch):
    class Config(SimpleNamespace):
        def replace(self, **kwargs):
            new = copy.deepcopy(self)
            vars(new).update(kwargs)
            return new
    class Scene(Config):
        # Real configclass materializes inherited asset fields on instances.
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if not hasattr(type(self), 'robot'):
                self.robot = Config(spawn=Config(activate_contact_sensors=False))
    monkeypatch.setitem(sys.modules, 'week03_ant.tasks.command_v14_cfg', command_adapter)
    monkeypatch.setattr(sys.modules['week03_ant.tasks.foothold_v9_cfg'], 'FootholdSceneCfg', Scene, raising=False)
    sensors = ModuleType('isaaclab.sensors'); sensors.ContactSensorCfg = Config
    monkeypatch.setitem(sys.modules, sensors.__name__, sensors)
    spec = importlib.util.spec_from_file_location('week03_ant.tasks._contact_cpu_test', ROOT / 'src/week03_ant/tasks/contact_v16_cfg.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def contact_environment(module, monkeypatch):
    env, data, _, calls = environment()
    monkeypatch.setattr(module, 'validate_terrain_contact_target', lambda: None)
    for foot, name in zip(FOOT_NAMES, module.CONTACT_SENSOR_NAMES):
        force = torch.zeros(1, 1, 2, 3)
        force[..., 0, 2] = 4.
        env.scene[name] = SimpleNamespace(
            body_names=[foot], cfg=module.terrain_contact_sensor(foot),
            data=SimpleNamespace(force_matrix_w=force),
            update=lambda dt, force_recompute: calls.append((dt, force_recompute)))
    return env, data, calls


def test_fresh_kinematics_reset_portal_and_no_randomness(contact_adapter, monkeypatch):
    m = contact_adapter; env, data, calls = contact_environment(m, monkeypatch)
    term = m.ContactSlipReward(None, env); rng = torch.get_rng_state().clone()
    assert term(env).item() == -1
    data.body_link_lin_vel_w.zero_()
    assert term(env).item() == 0
    data.body_link_ang_vel_w[..., 2] = 1.
    result = term.geometry(env)
    torch.testing.assert_close(result['reward'], torch.tensor([-.32]))
    data.root_pos_w += 100; data.body_link_pos_w += 100
    for key, value in result.items():
        torch.testing.assert_close(term.geometry(env)[key], value)
    for name in m.CONTACT_SENSOR_NAMES:
        env.scene[name].data.force_matrix_w.zero_()
    out = term.geometry(env)
    assert out['no_contact'].item() and out['reward'].item() == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert all(call == (0., True) for call in calls)


def test_flat_target_contact_and_shape_missing_filter_guards(contact_adapter, monkeypatch):
    m = contact_adapter; env, _, _ = contact_environment(m, monkeypatch)
    first = env.scene[m.CONTACT_SENSOR_NAMES[0]]
    for name in m.CONTACT_SENSOR_NAMES:
        f = env.scene[name].data.force_matrix_w
        f.zero_(); f[..., 1, 2] = 4.
    term = m.ContactSlipReward(None, env)
    assert term(env).item() == -1
    first.data.force_matrix_w = torch.zeros(1, 1, 1, 3)
    with pytest.raises(ValueError, match='force_matrix'):
        term(env)
    first.cfg.filter_prim_paths_expr = ['/World/ground']
    with pytest.raises(ValueError, match='filter exactly'):
        m.ContactSlipReward(None, env)
    first.cfg = m.terrain_contact_sensor(FOOT_NAMES[0]); first.body_names = ['wrong']
    with pytest.raises(ValueError, match='one contact sensor body'):
        m.ContactSlipReward(None, env)
    del env.scene[m.CONTACT_SENSOR_NAMES[0]]
    with pytest.raises(KeyError):
        m.ContactSlipReward(None, env)


def test_collision_prim_guard_for_both_targets(contact_adapter, monkeypatch):
    m = contact_adapter; queried = []; invalid = [None]
    assert m.FLAT_CONTACT_PATH == '/World/flatPlane/GroundPlane/CollisionPlane'
    class Stage:
        def GetPrimAtPath(self, path):
            queried.append(path)
            return SimpleNamespace(IsValid=lambda: path != invalid[0], HasAPI=lambda api: path != invalid[0])
    usd = ModuleType('omni.usd'); usd.get_context = lambda: SimpleNamespace(get_stage=lambda: Stage())
    omni = ModuleType('omni'); omni.usd = usd
    pxr = ModuleType('pxr'); pxr.UsdPhysics = SimpleNamespace(CollisionAPI=object())
    for name, mod in [('omni', omni), ('omni.usd', usd), ('pxr', pxr)]:
        monkeypatch.setitem(sys.modules, name, mod)
    m.validate_terrain_contact_target()
    assert queried == list(m.CONTACT_TARGET_PATHS)
    for path in m.CONTACT_TARGET_PATHS:
        invalid[0] = path
        with pytest.raises(ValueError, match='live collision prim'):
            m.validate_terrain_contact_target()


def test_config_inheritance_and_registration(contact_adapter, command_adapter, adapter, monkeypatch):
    m = contact_adapter
    assert m.ContactTrainAntEnvCfg.__bases__ == (command_adapter.CommandTrainAntEnvCfg,)
    assert m.ContactEvalAntEnvCfg.__bases__ == (command_adapter.CommandEvalAntEnvCfg,)
    assert m.ContactTrainAntEnvCfg.observations is command_adapter.CommandTrainAntEnvCfg.observations
    assert m.ContactEvalAntEnvCfg.observations is command_adapter.CommandEvalAntEnvCfg.observations
    assert m.ContactRewardsCfg.__bases__ == (adapter.AdaptivePostureRewardsCfg,)
    assert m.ContactRewardsCfg.contact_slip.func is m.ContactSlipReward
    assert m.ContactRewardsCfg.contact_slip.weight == 1.
    assert m.ContactSceneCfg.robot.spawn.activate_contact_sensors
    assert not m.FootholdSceneCfg().robot.spawn.activate_contact_sensors
    for foot, name in zip(FOOT_NAMES, m.CONTACT_SENSOR_NAMES):
        sensor = getattr(m.ContactSceneCfg, name)
        assert sensor.prim_path == '{ENV_REGEX_NS}/Robot/' + foot
        assert sensor.filter_prim_paths_expr == list(m.CONTACT_TARGET_PATHS)
        assert sensor.update_period == 0 and not sensor.track_air_time
    assert m.ContactAntPPORunnerCfg.policy is command_adapter.CommandAntPPORunnerCfg.policy
    gym = ModuleType('gymnasium'); gym.registry = {}; calls = []
    def register(**kwargs):
        calls.append(kwargs); gym.registry[kwargs['id']] = kwargs
    gym.register = register; monkeypatch.setitem(sys.modules, 'gymnasium', gym)
    spec = importlib.util.spec_from_file_location('_contact_register', ROOT / 'src/week03_ant/tasks/contact_v16.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); spec.loader.exec_module(module)
    assert len(calls) == 2
    assert [item['id'] for item in calls] == ['Week03-Ant-Contact-v16-Train-v0', 'Week03-Ant-Contact-v16-Eval-v0']


def test_two_env_partial_reset_invalid_rows_and_body_order(contact_adapter, monkeypatch):
    m = contact_adapter; env, data, _ = contact_environment(m, monkeypatch)
    env.num_envs = 2
    for name, value in vars(data).items():
        if isinstance(value, torch.Tensor):
            setattr(data, name, value.repeat(2, *([1] * (value.ndim - 1))))
    for name in m.CONTACT_SENSOR_NAMES:
        sensor = env.scene[name]
        sensor.data.force_matrix_w = sensor.data.force_matrix_w.repeat(2, 1, 1, 1)
    term = m.ContactSlipReward(None, env)
    before = term.geometry(env)
    data.body_link_lin_vel_w[0] = 0
    after = term.geometry(env)
    assert after['reward'].tolist() == [0., -1.]
    for key in before:
        torch.testing.assert_close(before[key][1], after[key][1])
    env.scene[m.CONTACT_SENSOR_NAMES[0]].data.force_matrix_w[0, 0, 0, 0] = float('nan')
    out = term.geometry(env)
    assert out['valid'].tolist() == [False, True]
    assert not out['no_contact'][0]
    env.scene['robot'].find_bodies = lambda names, preserve_order: ([0, 1, 2, 3], list(reversed(FOOT_NAMES)))
    with pytest.raises(ValueError, match='unexpected feet'):
        m.ContactSlipReward(None, env)


def test_full_config_parity_rejects_undeclared_change(contact_adapter, monkeypatch):
    m = contact_adapter
    baseline = {'scene': {'robot': {'spawn': {'activate_contact_sensors': None}}},
                'rewards': {'original': 'same'}, 'observations': {'policy': 91}, 'sim': {'dt': 1/120}}
    for added, original in ((m.ContactTrainAntEnvCfg, m.CommandTrainAntEnvCfg),
                            (m.ContactEvalAntEnvCfg, m.CommandEvalAntEnvCfg)):
        value = copy.deepcopy(baseline)
        value['scene']['robot']['spawn']['activate_contact_sensors'] = True
        value['scene'].update({name: 'sensor' for name in m.CONTACT_SENSOR_NAMES})
        if added is m.ContactTrainAntEnvCfg:
            value['rewards']['contact_slip'] = 'new'
        monkeypatch.setattr(original, 'to_dict', lambda self: copy.deepcopy(baseline), raising=False)
        monkeypatch.setattr(added, 'to_dict', lambda self, value=value: copy.deepcopy(value), raising=False)
    assert m.validate_contact_config_parity()
    original_method = m.ContactEvalAntEnvCfg.to_dict
    def changed(self):
        value = original_method(self); value['sim']['dt'] = .1; return value
    monkeypatch.setattr(m.ContactEvalAntEnvCfg, 'to_dict', changed)
    with pytest.raises(ValueError, match='undeclared environment'):
        m.validate_contact_config_parity()
