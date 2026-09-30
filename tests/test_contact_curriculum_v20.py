"""Pure schedule and reward-adapter regressions without Isaac/Omniverse startup."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from week03_ant.contact_curriculum_v20 import coefficient, audit_schedule

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode,total', [('immediate', 8000), ('ramped', 6000)])
def test_full_schedule_endpoints_mass_and_monotonicity(mode, total):
    factors = [coefficient(mode, t) for t in range(1, 8001)]
    assert factors[0] == (1 if mode == 'immediate' else 0)
    assert factors[3999] == factors[-1] == 1
    assert factors == sorted(factors)
    assert sum(factors) == pytest.approx(total)


@pytest.mark.parametrize('args', [('bad', 1, 4000), ('ramped', 0, 4000), ('ramped', True, 4000),
                                  ('ramped', 1, 1), ('ramped', 1, 2.5)])
def test_schedule_invalid_inputs(args):
    with pytest.raises(ValueError):
        coefficient(*args)


def reward_class():
    source = ast.parse((ROOT / 'src/week03_ant/tasks/contact_curriculum_v20_cfg.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'CurriculumContactSlipReward')
    class Base:
        def __init__(self, cfg, env):
            self.calls = 0
        def geometry(self, env):
            self.calls += 1
            return {'reward': torch.tensor([-.5, -1.]), 'valid': torch.tensor([True, True])}
    namespace = dict(ContactSlipReward=Base, torch=torch, coefficient=coefficient)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<reward-adapter>', 'exec'), namespace)
    return namespace['CurriculumContactSlipReward']


def test_same_geometry_even_at_zero_and_episode_resets_do_not_reset_clock():
    cls = reward_class()
    env = SimpleNamespace(common_step_counter=0, step_dt=1 / 60, episode_length_buf=torch.zeros(2))
    reward = cls(None, env)
    for step in range(1, 65):
        env.common_step_counter = step
        env.episode_length_buf.zero_()
        result = reward(env, 'ramped', 32)
        assert torch.equal(result, torch.tensor([-.5, -1.]) * coefficient('ramped', step, 32))
    assert reward.calls == 64
    before = copy.deepcopy(reward.history)
    reward.geometry(env)
    assert reward.history == before
    record = reward.schedule('ramped', 32)
    record['num_envs'] = 2
    assert audit_schedule(record, mode='ramped', policy_steps=64, ramp_steps=32)
    for key, value in [('coefficients', [1.] * 64), ('common_step_counters', [1] * 64),
                       ('realized_penalty_sum', 999), ('scaled_reward_mean', [-1.] * 64)]:
        corrupt = dict(record, **{key: value})
        with pytest.raises(ValueError):
            audit_schedule(corrupt, mode='ramped', policy_steps=64, ramp_steps=32)


def test_factor_one_is_exact_original_reward():
    cls = reward_class()
    env = SimpleNamespace(common_step_counter=4000, step_dt=1 / 60)
    immediate, ramped = cls(None, env), cls(None, env)
    assert torch.equal(immediate(env, 'immediate'), ramped(env, 'ramped'))
