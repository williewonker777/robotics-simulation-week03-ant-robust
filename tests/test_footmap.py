from pathlib import Path
import math

import pytest
import torch

from week03_ant.footmap_math import (
    FOOTMAP_OBSERVATIONS, SCAN_RAYS, scan_grid, feet_in_yaw_frame, foot_position_maps,
)
from week03_ant.footmap_policy import FootMapActorCritic, warmstart_from_60d


def observations(batch=4):
    x = torch.randn(batch, FOOTMAP_OBSERVATIONS)
    x[:, 60 + SCAN_RAYS:60 + 2 * SCAN_RAYS] = 1
    x[:, -12:] = torch.tensor([0.8, 0.8, -0.4, 0.8, -0.8, -0.4,
                               -0.8, 0.8, -0.4, -0.8, -0.8, -0.4])
    return {"policy": x}


def policy(mode="footmap"):
    return FootMapActorCritic(observations(), {"policy": ["policy"], "critic": ["policy"]}, 8,
                             input_mode=mode, init_noise_std=0.2)


def test_grid_is_y_rows_x_columns_with_scanner_offset():
    grid = scan_grid()
    assert grid.shape == (13, 17, 2)
    torch.testing.assert_close(grid[0, 0], torch.tensor([-1.2, -1.2]))
    torch.testing.assert_close(grid[-1, -1], torch.tensor([2.0, 1.2]))
    torch.testing.assert_close(grid[0, 1] - grid[0, 0], torch.tensor([0.2, 0.0]))


def test_four_maps_peak_at_corresponding_xy_cells():
    grid = scan_grid()
    locations = [(2, 3), (8, 3), (2, 12), (8, 12)]
    feet = torch.stack([grid[y, x] for y, x in locations])[None]
    maps = foot_position_maps(feet)
    assert maps.shape == (1, 4, 13, 17)
    assert maps.flatten(2).argmax(-1).tolist() == [[y * 17 + x for y, x in locations]]
    assert maps.flatten(2).max(-1).values.tolist() == [[1] * 4]


def test_offgrid_invalid_feet_are_not_clamped_to_edges():
    feet = torch.tensor([[[3., 0.], [0., 4.], [float("nan"), 0.], [0., float("inf")]]])
    assert torch.equal(foot_position_maps(feet), torch.zeros(1, 4, 13, 17))


@pytest.mark.parametrize("sigma", [0, -1, float("inf"), float("nan")])
def test_invalid_sigma_rejected(sigma):
    with pytest.raises(ValueError):
        foot_position_maps(torch.zeros(1, 4, 2), sigma)


def test_yaw_transform_and_translation_invariance():
    local = torch.tensor([[[1., 0., -.3], [0., 1., -.4], [-1., 0., -.5], [0., -1., -.6]]])
    root = torch.tensor([[7., 11., 3.]])
    world = torch.stack((-local[..., 1], local[..., 0], local[..., 2]), -1) + root[:, None]
    quat = torch.tensor([[math.sqrt(.5), 0, 0, math.sqrt(.5)]])
    torch.testing.assert_close(feet_in_yaw_frame(root, quat, world), local, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(feet_in_yaw_frame(root + 20, quat, world + 20), local, atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["blind", "height", "footmap"])
def test_policy_finite_gradient_checkpoint_contract(mode):
    torch.set_num_threads(2)
    p = policy(mode)
    obs = observations()
    action = p.act(obs)
    assert action.shape == (4, 8)
    loss = p.action_mean.square().mean() + p.evaluate(obs).square().mean() - .01 * p.entropy.mean()
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in p.parameters() if v.grad is not None)
    if mode != "blind":
        assert p.actor_encoder[0].weight.grad.abs().sum() > 0
        assert p.critic_encoder[0].weight.grad.abs().sum() > 0
    clone = policy("footmap")
    assert clone.load_state_dict(p.state_dict()) is True
    assert clone.input_mode == mode
    torch.testing.assert_close(clone.act_inference(obs), p.act_inference(obs), rtol=0, atol=0)


def test_blind_ignores_all_extra_inputs_even_nonfinite():
    p = policy("blind")
    a = observations()
    b = {"policy": a["policy"].clone()}
    b["policy"][:, 60:] = float("nan")
    torch.testing.assert_close(p.act_inference(a), p.act_inference(b), atol=0, rtol=0)
    torch.testing.assert_close(p.evaluate(a), p.evaluate(b), atol=0, rtol=0)


def test_height_ignores_foot_positions():
    p = policy("height")
    a = observations()
    b = {"policy": a["policy"].clone()}
    b["policy"][:, -12:] = float("nan")
    torch.testing.assert_close(p.act_inference(a), p.act_inference(b), atol=0, rtol=0)
    assert torch.equal(p.spatial_channels(a["policy"])[:, 2:], torch.zeros(4, 4, 13, 17))


def test_invalid_scan_returns_finite_unknown_not_safe_flat():
    p = policy()
    obs = observations()
    obs["policy"][:, 60:60 + SCAN_RAYS] = float("nan")
    channels = p.spatial_channels(obs["policy"])
    assert channels[:, 0].eq(1).all() and channels[:, 1].eq(0).all()
    assert torch.isfinite(p.act_inference(obs)).all()


def test_strict_checkpoint_and_dimensions_required():
    p = policy()
    state = p.state_dict()
    del state["input_mode_code"]
    with pytest.raises(ValueError):
        p.load_state_dict(state)
    with pytest.raises(ValueError):
        p.load_state_dict(p.state_dict(), strict=False)
    with pytest.raises(ValueError):
        FootMapActorCritic({"policy": torch.zeros(2, 60)}, {"policy": ["policy"], "critic": ["policy"]}, 8)


def test_real_parent_warmstart_preserves_actions_and_learns_new_columns():
    from rsl_rl.modules import ActorCritic
    root = Path(__file__).resolve().parents[1]
    source = root / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
    state = torch.load(source, map_location="cpu", weights_only=False)["model_state_dict"]
    obs = observations(12)
    old = ActorCritic({"policy": obs["policy"][:, :60]}, {"policy": ["policy"], "critic": ["policy"]}, 8,
                      actor_hidden_dims=[400, 200, 100], critic_hidden_dims=[400, 200, 100], activation="elu")
    old.load_state_dict(state)
    p = policy()
    warmstart_from_60d(p, state)
    expected = old.act_inference({"policy": obs["policy"][:, :60]})
    torch.testing.assert_close(p.act_inference(obs), expected, atol=1e-5, rtol=1e-6)
    assert p.actor[0].weight[:, 60:].eq(0).all()
    optimizer = torch.optim.Adam(p.parameters(), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad()
        (p.act_inference(obs).square().mean() + p.evaluate(obs).square().mean()).backward()
        optimizer.step()
    assert p.actor[0].weight[:, 60:].abs().sum() > 0
    assert p.actor_encoder[0].weight.grad.abs().sum() > 0
