from pathlib import Path

import pytest
import torch

from week03_ant.command_policy import (
    COMMAND_MODES, COMMAND_OBSERVATIONS, CommandPriorActorCritic,
    register_command_components, warmstart_command_from_prior,
)
from week03_ant.prior_policy import PriorActorCritic

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/terrain_demo/prior_v10/runs/v10_anchored_seed42/model_749.pt'
GROUPS = {'policy': ['policy'], 'critic': ['policy']}


def obs(n=7):
    return {'policy': torch.randn(n, COMMAND_OBSERVATIONS)}


def policy(mode='conditioned', **kwargs):
    return CommandPriorActorCritic(obs(), GROUPS, 8, command_mode=mode, **kwargs)


@pytest.mark.parametrize('mode', COMMAND_MODES)
def test_real_source_exact_prefix_actor_critic_teacher(mode):
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)['model_state_dict']
    inputs = obs(13)
    prefix = {'policy': inputs['policy'][:, :88].contiguous()}
    old = PriorActorCritic(prefix, GROUPS, 8, prior_mode='anchored')
    old.load_state_dict(source)
    new = policy(mode)
    info = warmstart_command_from_prior(new, source)
    assert info['teacher_preserved']
    for fn in ('act_inference', 'evaluate', 'teacher_mean'):
        assert torch.equal(getattr(old, fn)(prefix), getattr(new, fn)(inputs)), fn
    for key, value in source.items():
        if key != 'std':
            assert torch.equal(value, new.state_dict()[key]), key
    for name in ('actor', 'critic'):
        assert getattr(new, name)[0].command_weight.eq(0).all()
        assert getattr(new, name)[0].weight[:, 60:].ne(0).any()
    assert new.std.eq(.2).all()
    assert not new.teacher.training
    assert not any(p.requires_grad for p in new.teacher.parameters())


@pytest.mark.parametrize('mode', COMMAND_MODES)
def test_command_branch_gradient_and_masked_invariance(mode):
    model = policy(mode)
    inputs = obs()
    (model.act_inference(inputs).square().sum() + model.evaluate(inputs).square().sum()).backward()
    for name in ('actor', 'critic'):
        grad = getattr(model, name)[0].command_weight.grad
        assert torch.isfinite(grad).all()
        assert bool(grad.ne(0).any()) == (mode == 'conditioned')
    if mode == 'masked':
        with torch.no_grad():
            model.actor[0].command_weight.fill_(1.)
            model.critic[0].command_weight.fill_(1.)
        changed = {'policy': inputs['policy'].clone()}
        changed['policy'][:, 88:] = float('nan')
        for fn in ('act_inference', 'evaluate', 'teacher_mean'):
            assert torch.equal(getattr(model, fn)(inputs), getattr(model, fn)(changed))


def test_pair_only_differs_in_mode_and_fresh_optimizer():
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)['model_state_dict']
    models = [policy(mode) for mode in COMMAND_MODES]
    for model in models:
        warmstart_command_from_prior(model, source)
        assert torch.optim.Adam(model.parameters(), lr=1e-4).state_dict()['state'] == {}
    for key, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[key]) == (key != 'command_mode_code')


@pytest.mark.parametrize('key,bad', [
    ('command_mode_code', None), ('command_schema_code', torch.tensor(2)),
    ('command_mode_code', torch.tensor([0])), ('command_mode_code', torch.tensor(0.)),
    ('command_mode_code', torch.tensor(3)), ('prior_mode_code', torch.tensor(0)),
    ('input_mode_code', torch.tensor(0)),
])
def test_strict_state_guards(key, bad):
    model = policy()
    state = model.state_dict()
    if bad is None:
        del state[key]
    else:
        state[key] = bad
    with pytest.raises((ValueError, RuntimeError)):
        model.load_state_dict(state)


def test_mode_restore_config_and_shape_guards():
    state = policy('masked').state_dict()
    model = policy()
    model.load_state_dict(state)
    assert model.command_mode == 'masked'
    with pytest.raises(ValueError, match='training config'):
        policy(require_config_match=True).load_state_dict(state)
    with pytest.raises(ValueError):
        model.load_state_dict(state, strict=False)
    for kwargs in ({'prior_mode': 'free'}, {'noise_std_type': 'log'},
                   {'actor_obs_normalization': True}, {'input_mode': 'feet'}):
        with pytest.raises(ValueError):
            policy(**kwargs)
    with pytest.raises(ValueError):
        CommandPriorActorCritic({'policy': torch.zeros(2, 88)}, GROUPS, 8)
    with pytest.raises(ValueError):
        model.act_inference({'policy': torch.zeros(2, 88)})
    with pytest.raises(ValueError):
        warmstart_command_from_prior(model, {})
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)['model_state_dict']
    source['actor.0.weight'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        warmstart_command_from_prior(model, source)


def test_registration():
    import rsl_rl.runners.on_policy_runner as module
    from week03_ant.prior_ppo import PriorPPO
    register_command_components()
    assert module.CommandPriorActorCritic is CommandPriorActorCritic
    assert module.PriorPPO is PriorPPO
