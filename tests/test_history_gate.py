"""Regression tests for depth-history routing, independent of Isaac Sim."""
import pytest
import torch

from week03_ant.foothold_math import GRID_RAYS, foothold_grid
from week03_ant.history_gate import HistoryDepthPolicyGate, HistoryGateConfig
from week03_ant.hybrid_gate import PRESETS


def setup(n=1, dt=1 / 60, **overrides):
    gate = HistoryDepthPolicyGate(n, "cpu", dt, HistoryGateConfig(**overrides))
    xy = foothold_grid(device="cpu").reshape(-1, 2)
    rough = (xy[:, 0] > .5).float().expand(n, -1).clone() * .3
    flat = torch.zeros(n, GRID_RAYS)
    valid = torch.ones_like(flat, dtype=torch.bool)
    a, b = torch.zeros(n, 8), torch.ones(n, 8)
    return gate, rough, flat, valid, a, b


def advance(g, h, valid, a, b, count=1, mode="hybrid"):
    for _ in range(count):
        result = g.step(h, valid, a, b, mode=mode)
    return result


def test_single_spikes_and_alternating_rough_never_enter():
    g, rough, flat, valid, a, b = setup()
    for i in range(90):
        out = g.step(rough if i % 2 else flat, valid, a, b)
        assert not out["target_v10"].item()
    assert g.history_features.shape == (1, 60, 8)
    assert int(out["history_samples"]) == 60


def test_persistent_rough_needs_point_two_seconds_and_snapshots_are_stable():
    g, rough, _, valid, a, b = setup()
    before = advance(g, rough, valid, a, b, 11)
    assert not before["target_v10"].item()
    out = g.step(rough, valid, a, b)
    assert out["switched"].item() and out["target_v10"].item()
    assert out["rough_evidence_seconds"].item() == pytest.approx(.2)
    assert out["rough_fraction"].item() == 1
    g.reset()
    assert out["target_v10"].item()
    assert not before["target_v10"].item()
    assert int(g.history_recorded.sum()) == 0


def test_short_flat_gap_retains_v10_then_sustained_clear_exits():
    g, rough, flat, valid, a, b = setup()
    advance(g, rough, valid, a, b, 70)
    out = advance(g, flat, valid, a, b, 20)
    assert out["target_v10"].item()
    out = advance(g, flat, valid, a, b, 30)
    assert not out["target_v10"].item()
    assert out["clear_fraction"].item() == 1


def test_unknown_counts_in_denominator_and_interrupts_confirmation():
    g, rough, _, valid, a, b = setup()
    advance(g, rough, valid, a, b, 10)
    out = g.step(rough, ~valid, a, b)
    assert out["uncertain"].item()
    assert out["rough_fraction"].item() == pytest.approx(10 / 11)
    assert out["consecutive_rough_seconds"].item() == 0
    out = advance(g, rough, valid, a, b, 2)
    assert not out["target_v10"].item()
    out = advance(g, rough, valid, a, b, 4)
    assert out["target_v10"].item()


def test_long_unknown_expires_history_without_changing_target():
    g, rough, flat, valid, a, b = setup()
    advance(g, rough, valid, a, b, 70)
    out = advance(g, flat, ~valid, a, b, 15)
    assert out["target_v10"].item() and out["alpha"].item() == 1
    assert out["history_samples"].item() == 0
    assert torch.isnan(g.history_features).all()
    out = g.step(flat, valid, a, b)
    assert out["target_v10"].item()
    assert out["clear_evidence_seconds"].item() == pytest.approx(1 / 60)
    out = advance(g, flat, valid, a, b, 35)
    assert not out["target_v10"].item()


def test_per_env_reset_independence_and_no_portal_implicit_reset():
    g, rough, _, valid, a, b = setup(2)
    advance(g, rough, valid, a, b, 30)
    saved = g.history_features[1].clone()
    g.reset(torch.tensor([True, False]))
    assert not g.target_v10[0] and g.target_v10[1]
    torch.testing.assert_close(g.history_features[1], saved, equal_nan=True)
    out = g.step(rough, valid, a, b)
    assert out["history_samples"].tolist() == [1, 31]
    assert out["target_v10"].tolist() == [False, True]


def test_fixed_modes_exact_and_no_evidence_leakage():
    g, rough, flat, valid, a, b = setup()
    advance(g, rough, valid, a, b, 10)
    for mode, expected in (("v10", b), ("v5", a)):
        out = g.step(rough, valid, a, b, mode=mode)
        assert torch.equal(out["actions"], expected)
        assert out["history_samples"].item() == 0
    out = advance(g, rough, valid, a, b, 11)
    assert not out["target_v10"].item()
    assert g.step(rough, valid, a, b)["target_v10"].item()


def test_crossfade_endpoints_and_mixed_inference_context_reset():
    g, rough, flat, valid, a, b = setup()
    a.fill_(1.e20)
    b.fill_(-1.e20)
    assert torch.equal(g.step(flat, valid, a, b)["actions"], a)
    with torch.inference_mode():
        out = advance(g, rough, valid, a, b, 12)
    assert 0 < out["alpha"].item() < 1
    out = advance(g, rough, valid, a, b, 10)
    assert torch.equal(out["actions"], b)
    g.reset()
    assert torch.equal(g.step(flat, valid, a, b)["actions"], a)


@pytest.mark.parametrize("dt", [1 / 30, 1 / 60, 1 / 120, .07])
def test_enter_timing_dt_invariance(dt):
    g, rough, _, valid, a, b = setup(dt=dt)
    for step in range(1, 100):
        if g.step(rough, valid, a, b)["switched"].item():
            assert .2 - 1.e-8 <= step * dt < .2 + dt + 1.e-8
            break
    else:
        pytest.fail("persistent rough did not enter")


def test_rear_footprint_hazard_prevents_exit():
    g, rough, flat, valid, a, b = setup()
    advance(g, rough, valid, a, b, 70)
    rear = (foothold_grid(device="cpu").reshape(-1, 2)[:, 0] < -.2).float()[None] * .3
    out = advance(g, rear, valid, a, b, 70)
    assert out["enter_span"].item() == 0
    assert out["retain_span"].item() > .12
    assert out["target_v10"].item()
    assert out["clear_fraction"].item() == 0


@pytest.mark.parametrize("kwargs", [
    {"history_seconds": 0}, {"enter_fraction": 1.01}, {"exit_fraction": float("nan")},
    {"enter_min_seconds": .4}, {"exit_confirm_seconds": .9},
    {"exit_window_seconds": 1.1}, {"blend_seconds": True}, {"spatial": None},
    {"unknown_reset_seconds": float("inf")},
])
def test_invalid_configs(kwargs):
    with pytest.raises(ValueError):
        HistoryGateConfig(**kwargs)


def test_invalid_gate_inputs():
    with pytest.raises(ValueError):
        HistoryDepthPolicyGate(1, "cpu", 1 / 60, PRESETS["cautious"])
    with pytest.raises(ValueError):
        setup(dt=0)
    g, rough, flat, valid, a, b = setup()
    with pytest.raises(ValueError):
        g.step(rough, valid, a, b, mode="guess")
    with pytest.raises(ValueError):
        g.reset(torch.tensor([1]))
    with pytest.raises(ValueError):
        g.step(rough[:, :-1], valid, a, b)


def test_full_enter_window_requires_fifteen_of_eighteen_rough_votes():
    g, rough, flat, valid, a, b = setup()
    advance(g, flat, valid, a, b, 60)
    out = advance(g, rough, valid, a, b, 14)
    assert out["rough_votes"].item() == 14
    assert out["enter_history_samples"].item() == 18
    assert not out["target_v10"].item()
    out = g.step(rough, valid, a, b)
    assert out["rough_votes"].item() == 15
    assert out["target_v10"].item()


def test_full_exit_window_requires_forty_four_of_forty_eight_clear_votes():
    g, rough, flat, valid, a, b = setup()
    advance(g, rough, valid, a, b, 70)
    out = advance(g, flat, valid, a, b, 43)
    assert out["clear_votes"].item() == 43
    assert out["exit_history_samples"].item() == 48
    assert out["target_v10"].item()
    out = g.step(flat, valid, a, b)
    assert out["clear_votes"].item() == 44
    assert out["switched"].item() and not out["target_v10"].item()


def test_uncertainty_uses_retention_roi_while_v10_target():
    g, rough, flat, valid, a, b = setup()
    advance(g, rough, valid, a, b, 70)
    front_only = valid & (foothold_grid(device="cpu").reshape(-1, 2)[:, 0] >= -1.e-6)[None]
    out = g.step(flat, front_only, a, b)
    assert out["enter_coverage"].item() == 1
    assert out["retain_coverage"].item() < .9
    assert out["uncertain"].item()
    assert torch.equal(out["coverage"], out["retain_coverage"])
    assert out["target_v10"].item()


def test_exit_requires_eighteen_consecutive_clear_even_when_votes_suffice():
    g, rough, flat, valid, a, b = setup()
    advance(g, rough, valid, a, b, 70)
    advance(g, flat, valid, a, b, 43)
    g.step(rough, valid, a, b)
    out = advance(g, flat, valid, a, b, 17)
    assert out["clear_fraction"].item() >= .9
    assert out["target_v10"].item()
    out = g.step(flat, valid, a, b)
    assert out["consecutive_clear_seconds"].item() == pytest.approx(.3)
    assert out["switched"].item() and not out["target_v10"].item()


def test_isolated_one_frame_spike_is_forgotten():
    g, rough, flat, valid, a, b = setup()
    advance(g, flat, valid, a, b, 20)
    out = g.step(rough, valid, a, b)
    assert not out["target_v10"].item()
    out = advance(g, flat, valid, a, b, 60)
    assert out["rough_votes"].item() == 0
    assert not out["target_v10"].item()
