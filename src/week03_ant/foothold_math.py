"""Causal distal-endpoint hints, not certified whole-capsule foothold planning."""

import torch
from torch.nn import functional as F

GRID_WIDTH, GRID_HEIGHT = 33, 25
GRID_RAYS = GRID_WIDTH * GRID_HEIGHT
OBSERVATIONS = 88
FOOT_RADIUS = 0.08
SEARCH_RADIUS = 0.45


def foothold_grid(*, device=None, dtype=torch.float32):
    x = torch.linspace(-1.2, 2.0, GRID_WIDTH, device=device, dtype=dtype)
    y = torch.linspace(-1.2, 1.2, GRID_HEIGHT, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    return torch.stack((xx, yy), dim=-1)


def plateau_mask(height, valid):
    """Require an observed flat 3x3 neighborhood; never pad unknown edges as safe."""
    if height.ndim != 2 or height.shape[-1] != GRID_RAYS or valid.shape != height.shape:
        raise ValueError("expected [N,825] heights and validity")
    finite = torch.isfinite(height) & valid.bool()
    grid = torch.where(finite, height, torch.zeros_like(height)).reshape(-1, 1, GRID_HEIGHT, GRID_WIDTH)
    missing = (~finite).to(height.dtype).reshape_as(grid)
    complete = F.max_pool2d(missing, 3, stride=1, padding=1).eq(0)
    span = F.max_pool2d(grid, 3, stride=1, padding=1) + F.max_pool2d(-grid, 3, stride=1, padding=1)
    interior = torch.zeros_like(complete)
    interior[:, :, 1:-1, 1:-1] = True
    return (complete & interior & (span <= .06)).flatten(1)


def select_footholds(surface_z, valid, feet):
    """Return forward proposal and nearest current support center in the yaw frame.

    Only a local displacement/body-quadrant heuristic is enforced; this is NOT
    exact 2DOF inverse kinematics or a whole-foot stability certificate.
    Heights and feet share the torso-relative, yaw-only coordinate frame.
    """
    if feet.shape != (surface_z.shape[0], 4, 3):
        raise ValueError("expected four distal centers[N,4,3]")
    plateau = plateau_mask(surface_z, valid)
    grid = foothold_grid(device=surface_z.device, dtype=surface_z.dtype).reshape(-1, 2)
    foot_valid = torch.isfinite(feet).all(-1)
    safe_feet = torch.where(foot_valid[..., None], feet, torch.zeros_like(feet))
    finite = torch.isfinite(surface_z) & valid.bool()
    height = torch.where(finite, surface_z, torch.zeros_like(surface_z))
    # A tilted but locally acceptable patch supports the endpoint at its maximum
    # observed height, not just the center ray (conservative, still not a proof).
    support_z = F.max_pool2d(height.reshape(-1, 1, GRID_HEIGHT, GRID_WIDTH), 3,
                            stride=1, padding=1).flatten(1)
    distance2 = (grid[None, None] - safe_feet[:, :, None, :2]).square().sum(-1)
    near = distance2 <= SEARCH_RADIUS ** 2
    # FOOT_NAMES order follows the Ant asset, whose names do not encode our
    # x-forward/y-left convention. Rest XY quadrants are ++, -+, --, +-.
    # These envelopes prevent cross-body hints but are not joint-limit proofs.
    x_sign = surface_z.new_tensor([1, -1, -1, 1])
    y_sign = surface_z.new_tensor([1, 1, -1, -1])
    quadrant = ((grid[None, :, 0] * x_sign[:, None] >= .1)
                & (grid[None, :, 1] * y_sign[:, None] >= .1))
    vertical_reach = (support_z[:, None] + FOOT_RADIUS - safe_feet[..., 2, None]).abs() <= .4
    observed = near & quadrant[None] & finite[:, None] & foot_valid[..., None]
    top = height[:, None].expand(-1, 4, -1).masked_fill(~observed, -torch.inf).amax(-1)
    # Preserve ordinary .15m descending treads, reject .25m-or-deeper stone gaps.
    eligible = observed & vertical_reach & plateau[:, None] & (support_z[:, None] >= top[..., None] - .18)
    available = eligible.any(-1)
    nominal = safe_feet[:, :, :2].clone()
    nominal[:, :, 0] += .2
    target_cost = (grid[None, None] - nominal[:, :, None]).square().sum(-1)
    proposal_index = target_cost.masked_fill(~eligible, torch.inf).argmin(-1)
    support_index = distance2.masked_fill(~eligible, torch.inf).argmin(-1)

    def centers(index):
        z = support_z.gather(1, index) + FOOT_RADIUS
        center = torch.cat((grid[index], z[..., None]), dim=-1)
        return torch.where(available[..., None], center, torch.zeros_like(center))

    return centers(proposal_index), centers(support_index), available, plateau


def foothold_features(feet, proposal, available):
    valid = available & torch.isfinite(feet).all(-1) & torch.isfinite(proposal).all(-1)
    delta = torch.where(valid[..., None], proposal - feet, torch.zeros_like(feet))
    return torch.cat((delta, valid[..., None].to(feet.dtype)), dim=-1).flatten(1)


def endpoint_support_cost(feet, support_center, available):
    """Bounded stationary intrusion/unsupported-low-endpoint cost, no speed gate.

    Nearest CURRENT support is used, not a moving forward target. A supported
    stance foot is not forced to chase the next step. High swing feet are free.
    """
    valid = available & torch.isfinite(feet).all(-1) & torch.isfinite(support_center).all(-1)
    foot = torch.where(valid[..., None], feet, torch.zeros_like(feet))
    target = torch.where(valid[..., None], support_center, torch.zeros_like(support_center))
    horizontal = ((foot[..., :2] - target[..., :2]).norm(dim=-1) - .10).clamp(0, .2) / .2
    intrusion = (target[..., 2] - foot[..., 2]).clamp(0, .2) / .2
    low = ((target[..., 2] + .15 - foot[..., 2]) / .15).clamp(0, 1)
    cost = intrusion.square() + low * horizontal.square()
    return (cost * valid).mean(-1)
