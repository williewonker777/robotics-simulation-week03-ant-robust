"""Pure schedule and reward-adapter regressions without Isaac/Omniverse startup."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from week03_ant.contact_continuation_v21 import coefficient, audit_schedule

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode,total', [('immediate', 8000), ('no_cost', 0)])
def test_full_schedule_endpoints_mass_and_monotonicity(mode, total):
    factors = [coefficient(mode, t) for t in range(1, 8001)]
    assert factors[0] == (1 if mode == 'immediate' else 0)
    assert factors[3999] == factors[-1] == (1 if mode == 'immediate' else 0)
    assert factors == sorted(factors)
    assert sum(factors) == pytest.approx(total)


@pytest.mark.parametrize('args', [('bad', 1, 4000), ('no_cost', 0, 4000), ('no_cost', True, 4000),
                                  ('no_cost', 1, 1), ('no_cost', 1, 2.5)])
def test_schedule_invalid_inputs(args):
    with pytest.raises(ValueError):
        coefficient(*args)


def reward_class():
    source = ast.parse((ROOT / 'src/week03_ant/tasks/contact_continuation_v21_cfg.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ContinuationContactSlipReward')
    class Base:
        def __init__(self, cfg, env):
            self.calls = 0
        def geometry(self, env):
            self.calls += 1
            return {'reward': torch.tensor([-.5, -1.]), 'valid': torch.tensor([True, True])}
    namespace = dict(ContactSlipReward=Base, torch=torch, coefficient=coefficient)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<reward-adapter>', 'exec'), namespace)
    return namespace['ContinuationContactSlipReward']


def test_same_geometry_even_at_zero_and_episode_resets_do_not_reset_clock():
    cls = reward_class()
    env = SimpleNamespace(common_step_counter=0, step_dt=1 / 60, episode_length_buf=torch.zeros(2))
    reward = cls(None, env)
    for step in range(1, 65):
        env.common_step_counter = step
        env.episode_length_buf.zero_()
        result = reward(env, 'no_cost', 32)
        assert torch.equal(result, torch.tensor([-.5, -1.]) * coefficient('no_cost', step, 32))
    assert reward.calls == 64
    before = copy.deepcopy(reward.history)
    reward.geometry(env)
    assert reward.history == before
    record = reward.schedule('no_cost', 32)
    record['num_envs'] = 2
    assert audit_schedule(record, mode='no_cost', policy_steps=64, ramp_steps=32)
    for key, value in [('coefficients', [1.] * 64), ('common_step_counters', [1] * 64),
                       ('realized_penalty_sum', 999), ('scaled_reward_mean', [-1.] * 64)]:
        corrupt = dict(record, **{key: value})
        with pytest.raises(ValueError):
            audit_schedule(corrupt, mode='no_cost', policy_steps=64, ramp_steps=32)


def test_factor_one_is_exact_original_reward():
    cls = reward_class()
    env = SimpleNamespace(common_step_counter=4000, step_dt=1 / 60)
    immediate, no_cost = cls(None, env), cls(None, env)
    assert torch.equal(immediate(env, 'immediate'), torch.tensor([-.5, -1.]))
    assert torch.equal(no_cost(env, 'no_cost'), torch.zeros(2))


def test_immediate_reward_and_history_exactly_replay_frozen_v20():
    from week03_ant.contact_curriculum_v20 import coefficient as old_coefficient

    source = ast.parse((ROOT / 'src/week03_ant/tasks/contact_curriculum_v20_cfg.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'CurriculumContactSlipReward')
    new_type = reward_class()
    namespace = dict(ContactSlipReward=new_type.__bases__[0], torch=torch, coefficient=old_coefficient)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen-reward-adapter>', 'exec'), namespace)
    env = SimpleNamespace(common_step_counter=0, step_dt=1 / 60)
    old, new = namespace['CurriculumContactSlipReward'](None, env), new_type(None, env)
    before = torch.get_rng_state().clone()
    for step in range(1, 65):
        env.common_step_counter = step
        assert torch.equal(old(env, 'immediate', 32), new(env, 'immediate', 32))
    assert old.schedule('immediate', 32) == new.schedule('immediate', 32)
    assert torch.equal(before, torch.get_rng_state())
    assert old.calls == new.calls == 64


def test_full_config_guard_normalizes_only_reward_function_and_mode():
    source = ast.parse((ROOT / 'src/week03_ant/tasks/contact_continuation_v21_cfg.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_continuation_config_parity')
    reference = dict(scene={'seed': 110, 'observations': 91}, rewards={'contact_slip': dict(
        func='old', weight=1., params={'mode': 'immediate', 'ramp_steps': 4000})})
    candidate = copy.deepcopy(reference)
    candidate['rewards']['contact_slip'].update(func='new', params={'mode': 'no_cost', 'ramp_steps': 4000})
    class New:
        def to_dict(self):
            return copy.deepcopy(candidate)
    class Old:
        def to_dict(self):
            return copy.deepcopy(reference)
    namespace = dict(ContinuationContactTrainAntEnvCfg=New, CurriculumContactTrainAntEnvCfg=Old)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<config-parity>', 'exec'), namespace)
    validate = namespace['validate_continuation_config_parity']
    assert validate()
    candidate['scene']['observations'] = 88
    with pytest.raises(ValueError, match='undeclared'):
        validate()
    candidate['scene']['observations'] = 91
    candidate['rewards']['contact_slip']['weight'] = 0.
    with pytest.raises(ValueError, match='coefficient'):
        validate()


@pytest.mark.parametrize('key', ['scaled_reward_mean', 'scaled_reward_dt_sum', 'realized_penalty_sum'])
def test_no_cost_rejects_even_tiny_nonzero_penalty(key):
    env = SimpleNamespace(common_step_counter=1, step_dt=1 / 60)
    reward = reward_class()(None, env)
    reward(env, 'no_cost', 32)
    record = reward.schedule('no_cost', 32)
    record['num_envs'] = 2
    if key == 'realized_penalty_sum':
        record[key] = 1.e-12
    else:
        record[key][0] = -1.e-12
    with pytest.raises(ValueError, match='exactly zero'):
        audit_schedule(record, mode='no_cost', policy_steps=1, ramp_steps=32)
