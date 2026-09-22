import pytest
import torch

from week03_ant.lane_math import (
    allocate_by_weight,
    body_height_deficit,
    cap_terrain_progress,
    lane_boundary_masks,
    lane_family_level,
    lane_odometer,
    lane_target_direction,
    lane_wrap_mask,
    lane_wrap_shift,
    swing_clearance_cost,
    weighted_lane_layout,
)


def test_swing_clearance_does_not_penalize_stance_or_backward_support():
    clearance = torch.tensor([[-0.3, -0.3, 0.2, 0.1]])
    speed = torch.tensor([[0.0, -3.0, 2.0, 2.0]])
    torch.testing.assert_close(swing_clearance_cost(clearance, speed, 0.1), torch.zeros(1))


def test_swing_clearance_rewards_lifting_instead_of_forward_dragging():
    clearance = torch.tensor([[-0.3, -0.3], [0.0, 0.0], [0.05, 0.05], [0.1, 0.1]])
    speed = torch.full_like(clearance, 2.0)
    torch.testing.assert_close(swing_clearance_cost(clearance, speed, 0.1), torch.tensor([4.0, 1.0, 0.25, 0.0]))
    assert torch.equal(swing_clearance_cost(clearance, speed * 10, 0.1), swing_clearance_cost(clearance, speed, 0.1))


def test_body_height_penalty_has_no_high_jump_bonus_and_is_bounded():
    heights = torch.tensor([-1.0, 0.3, 0.4, 0.5, 5.0])
    torch.testing.assert_close(body_height_deficit(heights, 0.5, 0.2), torch.tensor([4.0, 1.0, 0.25, 0.0, 0.0]))


def test_gap_trap_cost_cannot_be_evaded_by_stopping_or_swinging_backward():
    clearance = torch.tensor([[-0.3, -0.1], [-0.3, -0.1]])
    speed = torch.tensor([[0.0, 0.0], [-2.0, -2.0]])
    torch.testing.assert_close(swing_clearance_cost(clearance, speed, 0.1, trap_weight=1.0), torch.tensor([2.5, 2.5]))
    # The historical curriculum remains reproducible with the default zero weight.
    torch.testing.assert_close(swing_clearance_cost(clearance, speed, 0.1), torch.zeros(2))


def test_gap_trap_cost_preserves_nominal_top_contact_and_does_not_double_charge_swing():
    contact = torch.tensor([[0.0, 0.0, 0.1, 0.2]])
    torch.testing.assert_close(swing_clearance_cost(contact, torch.zeros_like(contact), 0.1, 1.0), torch.zeros(1))
    low = torch.tensor([[-0.3, -0.3, -0.3, -0.3]])
    torch.testing.assert_close(swing_clearance_cost(low, torch.full_like(low, 2.0), 0.1, 1.0), torch.tensor([4.0]))


def test_gap_trap_cost_rejects_invalid_weight():
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="trap_weight"):
            swing_clearance_cost(torch.zeros(1, 4), torch.zeros(1, 4), 0.1, bad)


def test_recovery_costs_reject_invalid_scales():
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            swing_clearance_cost(torch.zeros(1, 4), torch.zeros(1, 4), bad)
        with pytest.raises(ValueError):
            body_height_deficit(torch.zeros(1), 0.5, bad)
        with pytest.raises(ValueError):
            body_height_deficit(torch.zeros(1), bad, 0.2)


def test_target_direction_points_forward_on_centre_line():
    direction = lane_target_direction(torch.tensor([[3.0, 2.0]]), torch.tensor([2.0]), lookahead=12.0)
    torch.testing.assert_close(direction, torch.tensor([[1.0, 0.0]]))


def test_target_direction_steers_back_to_lane_centre():
    pos = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    direction = lane_target_direction(pos, torch.tensor([0.0, 0.0]), lookahead=12.0)
    torch.testing.assert_close(torch.linalg.vector_norm(direction, dim=-1), torch.ones(2))
    assert direction[0, 1] < 0.0 < direction[1, 1]
    assert torch.all(direction[:, 0] > 0.99)


def test_target_direction_rejects_non_positive_lookahead():
    with pytest.raises(ValueError):
        lane_target_direction(torch.zeros(1, 2), torch.zeros(1), lookahead=0.0)


def test_wrap_lands_on_entrance_portal_and_preserves_odometer():
    x_entrance, x_exit = torch.tensor([-28.0]), torch.tensor([28.0])
    shift = lane_wrap_shift(x_entrance, x_exit)
    x = torch.tensor([28.1])
    assert bool(lane_wrap_mask(x, x_exit)[0])
    wrapped = x - shift
    # 0.1 m of overshoot past the exit portal centre is kept past the entrance centre.
    torch.testing.assert_close(wrapped, x_entrance + 0.1)
    before = lane_odometer(x, torch.tensor([0]), shift)
    after = lane_odometer(wrapped, torch.tensor([1]), shift)
    torch.testing.assert_close(before, after)


def test_wrap_mask_ignores_torsos_before_exit_centre():
    assert not bool(lane_wrap_mask(torch.tensor([27.9]), torch.tensor([28.0]))[0])


def test_lane_family_level_split():
    family, level = lane_family_level(torch.tensor([0, 4, 5, 34]), num_levels=5)
    assert family.tolist() == [0, 0, 1, 6]
    assert level.tolist() == [0, 4, 0, 4]


def test_allocate_by_weight_sums_to_total():
    counts = allocate_by_weight(10, torch.tensor([2.0, 1.0, 1.0]))
    assert counts.tolist() == [5, 3, 2] or counts.tolist() == [5, 2, 3]
    assert int(counts.sum()) == 10
    assert allocate_by_weight(4096, torch.tensor([0.25] + [0.125] * 6)).tolist() == [1024] + [512] * 6


def test_allocate_by_weight_rejects_bad_weights():
    with pytest.raises(ValueError):
        allocate_by_weight(3, torch.tensor([0.0, 0.0]))


def test_weighted_lane_layout_cycles_levels_within_family():
    lanes = weighted_lane_layout(12, torch.tensor([2.0, 1.0]), num_levels=3)
    family, level = lane_family_level(lanes, 3)
    assert family.tolist() == [0] * 8 + [1] * 4
    assert level.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0]


def test_terrain_speed_cap_preserves_flat_and_negative_progress():
    progress = torch.tensor([10.0, 10.0, 2.0, -2.0])
    on_plane = torch.tensor([True, False, False, False])
    assert torch.equal(cap_terrain_progress(progress, on_plane, 3.0), torch.tensor([10.0, 3.0, 2.0, -2.0]))


def test_terrain_speed_cap_rejects_invalid_limits():
    for limit in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            cap_terrain_progress(torch.zeros(1), torch.zeros(1, dtype=torch.bool), limit)


def test_lane_boundaries_exclude_unbounded_plane_and_use_assigned_center():
    positions = torch.tensor([[0.0, 104.0], [0.0, 104.1], [35.1, 100.0], [0.0, 124.0], [1000.0, 1000.0]])
    out_of_lane, world_exit = lane_boundary_masks(
        positions, torch.full((5,), 100.0), 4.0, (35.0, 123.0), torch.tensor([True] * 4 + [False]),
    )
    assert out_of_lane.tolist() == [False, True, False, True, False]
    assert world_exit.tolist() == [False, False, True, True, False]


def test_lane_boundaries_treat_entrance_and_exit_portals_as_inside_world():
    positions = torch.tensor([[-28.0, 0.0], [28.2, 0.0], [-35.1, 0.0]])
    out_of_lane, world_exit = lane_boundary_masks(
        positions, torch.zeros(3), 4.0, (35.0, 123.0), torch.ones(3, dtype=torch.bool),
    )
    assert not out_of_lane.any()
    assert world_exit.tolist() == [False, False, True]


def test_conservative_footprint_cannot_borrow_an_adjacent_lane():
    positions = torch.tensor([[0.0, 2.89], [0.0, 2.91], [0.0, -2.91], [28.2, 0.0]])
    departed, world_exit = lane_boundary_masks(
        positions, torch.zeros(4), 4.0 - 1.1, (35.0 - 1.1, 123.0 - 1.1),
        torch.ones(4, dtype=torch.bool),
    )
    assert departed.tolist() == [False, True, True, False]
    assert not world_exit.any()
