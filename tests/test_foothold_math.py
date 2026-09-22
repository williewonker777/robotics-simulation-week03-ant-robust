import torch

from week03_ant.foothold_math import (
    GRID_RAYS, foothold_grid, plateau_mask, select_footholds,
    foothold_features, endpoint_support_cost,
)


def feet():
    return torch.tensor([[[.6, .6, -.42], [-.6, .6, -.42], [-.6, -.6, -.42], [.6, -.6, -.42]]])


def flat():
    return torch.full((1, GRID_RAYS), -.5), torch.ones(1, GRID_RAYS, dtype=torch.bool)


def test_grid_order_and_extent():
    grid = foothold_grid()
    assert grid.shape == (25, 33, 2)
    torch.testing.assert_close(grid[0, 0], torch.tensor([-1.2, -1.2]))
    torch.testing.assert_close(grid[-1, -1], torch.tensor([2., 1.2]))
    torch.testing.assert_close(grid[0, 1] - grid[0, 0], torch.tensor([.1, 0.]), atol=1e-6, rtol=0)


def test_flat_forward_target_and_stance_cost_zero():
    heights, valid = flat()
    proposal, support, available, plateau = select_footholds(heights, valid, feet())
    assert available.all()
    torch.testing.assert_close(proposal[..., 0], feet()[..., 0] + .2, atol=1e-6, rtol=0)
    torch.testing.assert_close(support, feet(), atol=1e-6, rtol=0)
    assert endpoint_support_cost(feet(), support, available).eq(0).all()
    assert plateau.sum() == 23 * 31
    assert foothold_features(feet(), proposal, available).shape == (1, 16)


def test_edges_holes_and_single_high_rays_are_not_plateaus():
    height, valid = flat()
    h = height.reshape(1, 25, 33)
    h[:, 10:13, 10:13] = -.8
    h[:, 5, 5] = -.2
    mask = plateau_mask(height, valid).reshape(1, 25, 33)
    assert not mask[:, 0].any() and not mask[:, -1].any()
    assert not mask[:, :, 0].any() and not mask[:, :, -1].any()
    assert not mask[:, 4:7, 4:7].any()
    assert not mask[:, 9:11, 9:14].any()


def test_invalid_neighbor_and_no_targets_produce_zero_features():
    height, valid = flat()
    height[:, 123] = float('nan')
    assert not plateau_mask(height, valid).flatten()[123]
    valid.zero_()
    proposal, support, available, _ = select_footholds(height, valid, feet())
    assert not available.any() and proposal.eq(0).all() and support.eq(0).all()
    assert foothold_features(feet(), proposal, available).eq(0).all()
    assert endpoint_support_cost(feet(), support, available).eq(0).all()


def test_vertical_translation_invariance_and_foot_displacement_limit():
    height, valid = flat()
    a, support, mask, _ = select_footholds(height, valid, feet())
    shifted = feet().clone()
    shifted[..., 2] += 7
    b, _, bm, _ = select_footholds(height + 7, valid, shifted)
    torch.testing.assert_close(foothold_features(feet(), a, mask), foothold_features(shifted, b, bm), atol=1e-6, rtol=0)
    assert (a[..., :2] - feet()[..., :2]).norm(dim=-1).max() <= .45


def test_stationary_buried_feet_penalized_high_swings_free_bounded():
    foot = feet()
    anchor = foot.clone()
    valid = torch.ones(1, 4, dtype=torch.bool)
    foot[..., 2] -= .3
    assert endpoint_support_cost(foot, anchor, valid).item() == 1
    foot[..., :2] += 2
    assert endpoint_support_cost(foot, anchor, valid).item() == 2
    foot[..., 2] = anchor[..., 2] + .3
    assert endpoint_support_cost(foot, anchor, valid).eq(0).all()


def test_no_boundary_clamping_and_nonfinite_foot_safe():
    height, valid = flat()
    foot = feet()
    foot[:, 0] = torch.tensor([5., 0., -.4])
    foot[:, 1] = float('nan')
    p, s, ok, _ = select_footholds(height, valid, foot)
    assert not ok[:, :2].any()
    assert torch.isfinite(foothold_features(foot, p, ok)).all()
    assert torch.isfinite(endpoint_support_cost(foot, s, ok)).all()


def test_body_quadrants_preserved():
    h, v = flat()
    p, _, ok, _ = select_footholds(h, v, feet())
    assert ok.all()
    assert (p[0, :, 0] * torch.tensor([1, -1, -1, 1]) >= .1).all()
    assert (p[0, :, 1] * torch.tensor([1, 1, -1, -1]) >= .1).all()


def test_descending_tread_is_retained_but_deep_gap_is_rejected():
    h, v = flat()
    x = foothold_grid().reshape(-1, 2)[:, 0]
    foot = feet()
    foot[:, 0] = torch.tensor([.8, .6, -.57])
    h[:, x >= .7] = -.65  # a .15m descending tread, not a stone pit
    proposal, _, ok, _ = select_footholds(h, v, foot)
    assert ok[:, 0].all() and proposal[0, 0, 0] > .8
    torch.testing.assert_close(proposal[0, 0, 2], torch.tensor(-.57))
    h[:, x >= .7] = -.8
    proposal, _, ok, _ = select_footholds(h, v, foot)
    assert ok[:, 0].all() and proposal[0, 0, 0] < .7
    torch.testing.assert_close(proposal[0, 0, 2], torch.tensor(-.42))


def test_unreachable_stone_tops_do_not_allow_pit_fallback():
    h, v = flat()
    x = foothold_grid().reshape(-1, 2)[:, 0]
    h[:, x >= .7] = -.8
    foot = feet()
    foot[:, 0] = torch.tensor([.8, .6, -1.05])
    proposal, support, ok, _ = select_footholds(h, v, foot)
    assert not ok[0, 0]
    assert proposal[0, 0].eq(0).all() and support[0, 0].eq(0).all()


def test_patch_maximum_not_center_ray_sets_support_height():
    h, v = flat()
    x = foothold_grid().reshape(-1, 2)[:, 0]
    h += .02 * x
    foot = feet()
    _, support, ok, _ = select_footholds(h, v, foot)
    assert ok.all()
    expected = -.5 + .02 * (support[..., 0] + .1) + .08
    torch.testing.assert_close(support[..., 2], expected, atol=1e-6, rtol=0)


def test_vertical_search_is_bounded_even_on_flat_plane():
    h, v = flat()
    foot = feet()
    foot[..., 2] += .41
    _, _, ok, _ = select_footholds(h, v, foot)
    assert not ok.any()
