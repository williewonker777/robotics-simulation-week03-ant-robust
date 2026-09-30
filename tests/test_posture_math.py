from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from week03_ant.foothold_math import GRID_RAYS, foothold_grid
from week03_ant.posture_math import PostureConfig, terrain_posture_terms


def inputs(n=1):
    feet = torch.tensor([[[.6, .6, -.36], [-.6, .6, -.36],
                          [-.6, -.6, -.36], [.6, -.6, -.36]]]).repeat(n, 1, 1)
    return dict(surface_z=torch.full((n, GRID_RAYS), -.44),
                valid=torch.ones(n, GRID_RAYS, dtype=torch.bool), feet=feet,
                forward_swing_speed=torch.zeros(n, 4), forward_speed=torch.full((n,), 6.),
                upright=torch.ones(n))


def test_frozen_config_and_flat_target_bonus():
    with pytest.raises(FrozenInstanceError):
        PostureConfig().clear_height = .3
    out = terrain_posture_terms(**inputs())
    assert out['valid'].all() and out['foot_valid'].all()
    torch.testing.assert_close(out['body_clearance'], torch.tensor([.44]))
    torch.testing.assert_close(out['foot_clearance'], torch.zeros(1, 4), atol=1e-7, rtol=0)
    assert out['severity'].eq(0).all() and out['body_cost'].eq(0).all()
    assert out['reward'].eq(.5).all()


@pytest.mark.parametrize('step', [.15, -.15, -.4])
def test_uniform_front_upstep_downstep_and_gap_are_rough(step):
    args = inputs()
    grid = foothold_grid().reshape(-1, 2)
    args['surface_z'][:, grid[:, 0] >= .4] += step
    out = terrain_posture_terms(**args)
    torch.testing.assert_close(out['severity'], torch.ones(1))
    torch.testing.assert_close(out['target_height'], torch.tensor([.58]))
    assert out['flat_speed_bonus'].eq(0).all()


def test_uniform_shelf_reference_and_slopes():
    args = inputs()
    baseline = terrain_posture_terms(**args)
    args['surface_z'] += .2
    args['feet'][..., 2] += .2
    shelf = terrain_posture_terms(**args)
    torch.testing.assert_close(shelf['severity'], baseline['severity'])
    torch.testing.assert_close(shelf['foot_clearance'], baseline['foot_clearance'])
    torch.testing.assert_close(shelf['body_clearance'], baseline['body_clearance'] - .2)
    args['surface_z'] += .1 * foothold_grid().reshape(-1, 2)[:, 0]
    assert terrain_posture_terms(**args)['severity'].item() > .5


def test_severity_thresholds():
    args = inputs(3)
    grid = foothold_grid().reshape(-1, 2)
    args['surface_z'][:, grid[:, 0] >= .4] += torch.tensor([.03, .09, .15])[:, None]
    out = terrain_posture_terms(**args)
    torch.testing.assert_close(out['severity'], torch.tensor([0., .5, 1.]), atol=1e-6, rtol=0)
    assert (out['target_height'] > .31).all()


def test_coverage_boundary_and_invalid_masking():
    args = inputs(2)
    grid = foothold_grid().reshape(-1, 2)
    local = (grid.abs() <= .300001).all(-1).nonzero().flatten()
    assert len(local) == 49
    args['valid'][0, local[:4]] = False  # 45/49 accepted
    args['valid'][1, local[:5]] = False  # 44/49 rejected
    args['surface_z'][~args['valid']] = float('nan')
    out = terrain_posture_terms(**args)
    assert out['valid'].tolist() == [True, False]
    assert out['reward'][1] == 0
    args['surface_z'].fill_(float('inf'))
    args['feet'].fill_(float('nan'))
    args['forward_swing_speed'].fill_(float('inf'))
    args['forward_speed'].fill_(float('nan'))
    args['upright'].fill_(float('nan'))
    out = terrain_posture_terms(**args)
    assert not out['valid'].any() and not out['foot_valid'].any()
    for value in out.values():
        assert torch.isfinite(value).all()
    assert out['reward'].eq(0).all()


def test_front_coverage_is_independent():
    args = inputs()
    grid = foothold_grid().reshape(-1, 2)
    args['valid'][:, grid[:, 0] > .9] = False
    assert not terrain_posture_terms(**args)['valid'].any()


def test_full_foot_patch_unknown_edges_and_conservative_support():
    args = inputs()
    args['feet'][0, 0, 0] = 1.9  # center in grid, full patch out
    grid = foothold_grid().reshape(-1, 2)
    missing = ((grid - torch.tensor([-.7, .5])).abs() < 1e-5).all(-1)
    args['valid'][:, missing] = False
    high = ((grid - torch.tensor([-.5, -.5])).abs() < 1e-5).all(-1)
    args['surface_z'][:, high] += .02
    args['feet'][0, 3] = float('nan')
    out = terrain_posture_terms(**args)
    assert out['foot_valid'].tolist() == [[False, False, True, False]]
    torch.testing.assert_close(out['foot_clearance'][0, 2], torch.tensor(-.02), atol=1e-7, rtol=0)


def test_patch_grid_boundary_is_included():
    args = inputs()
    args['feet'][..., :2] = torch.tensor([1.85, 1.05])
    assert terrain_posture_terms(**args)['foot_valid'].all()
    args['feet'][..., 0] += .001
    assert not terrain_posture_terms(**args)['foot_valid'].any()


def test_swing_proxy_gates_and_four_foot_divisor():
    args = inputs(4)
    args['forward_swing_speed'][:] = torch.tensor([2., 0., -2., 1.])
    args['forward_speed'][:] = torch.tensor([1., 0., -1., 1.])
    args['feet'][3, 0, 0] = 3.
    out = terrain_posture_terms(**args)
    # Supported stance velocity relative to forward-moving root is nonpositive.
    expected = ((.04 / .12) ** 2) * torch.tensor([1.5 / 4, 0., 0., .5 / 4])
    torch.testing.assert_close(out['foot_cost'], expected)
    args['feet'][..., 2] += .2
    assert terrain_posture_terms(**args)['foot_cost'].eq(0).all()


def test_bonus_direction_upright_threshold_and_gaussian():
    args = inputs(6)
    args['forward_speed'][:] = torch.tensor([0., -3., 6., 6., 6., 6.])
    args['upright'][:] = torch.tensor([1., 1., -1., math.cos(1.2) - .001, math.cos(1.2), 1.])
    args['surface_z'][5] -= .24
    out = terrain_posture_terms(**args)
    assert out['flat_speed_bonus'][:4].eq(0).all()
    torch.testing.assert_close(out['flat_speed_bonus'][4], torch.tensor(math.cos(1.2)))
    torch.testing.assert_close(out['flat_speed_bonus'][5], torch.tensor(math.exp(-4.)))
    assert out['body_cost'][5] == 1


def test_bounds_and_shapes():
    args = inputs(128)
    generator = torch.Generator().manual_seed(13)
    args['surface_z'] += torch.rand(args['surface_z'].shape, generator=generator) * .5
    args['feet'][..., 2] -= 3
    args['forward_swing_speed'].fill_(100)
    out = terrain_posture_terms(**args)
    assert ((out['combined_cost'] >= 0) & (out['combined_cost'] <= 1.25)).all()
    assert ((out['reward'] >= -1.25) & (out['reward'] <= .5)).all()
    with pytest.raises(ValueError):
        terrain_posture_terms(**(args | {'surface_z': torch.zeros(128, 824)}))


@pytest.mark.parametrize('override', [
    {'body_error_scale': 0}, {'foot_error_scale': -1}, {'severity_span': 0},
    {'speed_scale': 0}, {'swing_speed_scale': 0}, {'clear_height': .31},
    {'clear_height': float('nan')}, {'rough_height_delta': float('inf')},
    {'rough_height_delta': -.1}, {'foot_cost_weight': -.1}, {'flat_bonus_weight': -.1},
    {'min_coverage': 0}, {'min_coverage': 1.01}, {'min_upright': -1},
    {'local_half_width': 1.3}, {'front_min_x': 1.6}, {'front_max_x': 2.1},
    {'front_min_x': .31, 'front_max_x': .39}, {'front_half_width': 0},
    {'foot_patch_half_width': 0}, {'foot_patch_half_width': 1.3},
    {'min_foot_samples': 0}, {'min_foot_samples': 100}, {'min_foot_samples': 9.5},
    {'min_foot_samples': True}, {'foot_patch_half_width': .01},
])
def test_invalid_config_fails_closed(override):
    with pytest.raises(ValueError):
        PostureConfig(**override)


def test_roi_coverage_diagnostics():
    args = inputs()
    out = terrain_posture_terms(**args)
    assert out['local_coverage'].eq(1).all() and out['front_coverage'].eq(1).all()
    args['valid'].zero_()
    out = terrain_posture_terms(**args)
    assert out['local_coverage'].eq(0).all() and out['front_coverage'].eq(0).all()
