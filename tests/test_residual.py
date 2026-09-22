import copy
from pathlib import Path

import pytest
import torch
from rsl_rl.modules import ActorCritic

from week03_ant.footmap_math import FOOTMAP_OBSERVATIONS, SCAN_RAYS
from week03_ant.residual_policy import ResidualActorCritic, warmstart_residual_from_60d


def observations(batch=8):
    raw = torch.randn(batch, FOOTMAP_OBSERVATIONS)
    raw[:, 60 + SCAN_RAYS:60 + 2 * SCAN_RAYS] = 1
    raw[:, -12:] *= 0.5
    return {"policy": raw}


def policy(mode="footmap", **kwargs):
    torch.set_num_threads(2)
    return ResidualActorCritic(observations(), {"policy": ["policy"], "critic": ["policy"]},
                               8, input_mode=mode, init_noise_std=0.2, **kwargs)


def source_state():
    path = Path(__file__).resolve().parents[1] / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
    return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]


@pytest.mark.parametrize("mode", ["blind", "footmap"])
def test_real_parent_identity_and_frozen_base_after_learning(mode):
    obs = observations()
    old = ActorCritic({"policy": obs["policy"][:, :60]}, {"policy": ["policy"], "critic": ["policy"]},
                      8, actor_hidden_dims=[400, 200, 100], critic_hidden_dims=[400, 200, 100], activation="elu")
    source = source_state()
    old.load_state_dict(source)
    p = policy(mode)
    warmstart_residual_from_60d(p, source)
    torch.testing.assert_close(p.act_inference(obs), old.act_inference({"policy": obs["policy"][:, :60]}),
                               rtol=0, atol=0)
    frozen = copy.deepcopy(p.actor.base.state_dict())
    optimizer = torch.optim.Adam(p.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        loss = p.act_inference(obs).square().mean() + p.evaluate(obs).square().mean()
        loss.backward()
        optimizer.step()
    for name, value in p.actor.base.state_dict().items():
        assert torch.equal(value, frozen[name])
    assert all(not v.requires_grad and v.grad is None for v in p.actor.base.parameters())
    assert p.actor.residual[-1].weight.grad.abs().sum() > 0
    if mode == "footmap":
        assert p.actor_encoder[0].weight.grad.abs().sum() > 0
    base = p.actor.base(obs["policy"][:, :60])
    assert (p.act_inference(obs) - base).abs().max() <= 0.500001


def test_large_residual_saturates_and_zero_limit_restores_base():
    p = policy()
    with torch.no_grad():
        p.actor.residual[-1].bias.fill_(1e5)
    obs = observations()
    base = p.actor.base(obs["policy"][:, :60])
    torch.testing.assert_close(p.act_inference(obs) - base, torch.full_like(base, .5))
    p.actor.residual_limit.zero_()
    torch.testing.assert_close(p.act_inference(obs), base, rtol=0, atol=0)


def test_ppo_distribution_is_over_total_actions():
    p = policy()
    obs = observations()
    action = p.act(obs)
    torch.testing.assert_close(p.action_mean, p.act_inference(obs))
    expected = torch.distributions.Normal(p.action_mean, p.std).log_prob(action).sum(-1)
    torch.testing.assert_close(p.get_actions_log_prob(action), expected)
    assert action.shape == (8, 8) and torch.isfinite(action).all()


def test_blind_is_independent_of_nonfinite_spatial_input_after_learning():
    p = policy("blind")
    with torch.no_grad():
        p.actor.residual[-1].weight.normal_()
    a = observations()
    b = {"policy": a["policy"].clone()}
    b["policy"][:, 60:] = float("nan")
    torch.testing.assert_close(p.act_inference(a), p.act_inference(b), rtol=0, atol=0)
    torch.testing.assert_close(p.evaluate(a), p.evaluate(b), rtol=0, atol=0)


def test_state_restores_mode_limit_and_frozen_base():
    p = policy("blind", residual_limit=.25)
    q = policy()
    assert q.load_state_dict(p.state_dict()) is True
    assert q.input_mode == "blind" and float(q.actor.residual_limit) == .25
    obs = observations()
    torch.testing.assert_close(p.act_inference(obs), q.act_inference(obs), rtol=0, atol=0)
    assert all(not v.requires_grad for v in q.actor.base.parameters())


def test_paired_initialization_differs_only_by_mode_and_matches_values():
    source = source_state()
    torch.manual_seed(42)
    a = policy("blind")
    warmstart_residual_from_60d(a, source)
    torch.manual_seed(42)
    b = policy("footmap")
    warmstart_residual_from_60d(b, source)
    for key, value in a.state_dict().items():
        if key != "input_mode_code":
            assert torch.equal(value, b.state_dict()[key]), key
    obs = observations()
    torch.testing.assert_close(a.act_inference(obs), b.act_inference(obs), rtol=0, atol=0)
    torch.testing.assert_close(a.evaluate(obs), b.evaluate(obs), rtol=0, atol=0)


def test_zero_head_learns_before_encoder_and_roundtrip_keeps_parent():
    p = policy()
    source = source_state()
    warmstart_residual_from_60d(p, source)
    obs = observations()
    loss = p.act_inference(obs).square().mean()
    loss.backward()
    assert p.actor.residual[-1].weight.grad.abs().sum() > 0
    assert p.actor_encoder[0].weight.grad.eq(0).all()
    optimizer = torch.optim.Adam(p.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()
    p.act_inference(obs).square().mean().backward()
    assert p.actor_encoder[0].weight.grad.abs().sum() > 0
    q = policy()
    q.load_state_dict(p.state_dict())
    for key, value in q.actor.base.state_dict().items():
        assert torch.equal(value, source[f"actor.{key}"])
    assert all(not v.requires_grad for v in q.actor.base.parameters())


@pytest.mark.parametrize("value", [torch.tensor([.5]), torch.tensor(.5, dtype=torch.float64)])
def test_checkpoint_rejects_wrong_limit_shape_dtype(value):
    p = policy()
    state = p.state_dict()
    state["actor.residual_limit"] = value
    with pytest.raises(ValueError):
        p.load_state_dict(state)


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_invalid_limits_rejected(bad):
    with pytest.raises(ValueError):
        policy(residual_limit=bad)
    p = policy()
    state = p.state_dict()
    state["actor.residual_limit"] = torch.tensor(bad, dtype=torch.float32)
    with pytest.raises(ValueError):
        p.load_state_dict(state)


def test_malformed_or_mismatched_training_checkpoint_rejected():
    p = policy(require_config_match=True)
    for state in (policy("blind").state_dict(), policy(residual_limit=.25).state_dict()):
        with pytest.raises(ValueError):
            p.load_state_dict(state)
    state = p.state_dict()
    del state["actor.residual_limit"]
    with pytest.raises(ValueError):
        p.load_state_dict(state)
    with pytest.raises(ValueError):
        p.load_state_dict(p.state_dict(), strict=False)
