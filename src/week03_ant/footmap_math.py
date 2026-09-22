"""Geometry shared by the v7 sensor and policy; no simulator dependency."""

import math

import torch

SCAN_HEIGHT = 13  # y rows: Isaac Lab GridPattern ordering='xy'
SCAN_WIDTH = 17  # x columns
SCAN_RAYS = SCAN_HEIGHT * SCAN_WIDTH
FOOTMAP_OBSERVATIONS = 60 + 2 * SCAN_RAYS + 12
FOOT_NAMES = ("front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot")
TIP_OFFSETS = ((0.4, 0.4, 0.0), (-0.4, 0.4, 0.0), (-0.4, -0.4, 0.0), (0.4, -0.4, 0.0))


def scan_grid(*, device=None, dtype=torch.float32):
    """Return [y,x,xy] torso-yaw coordinates, including the scanner offset."""
    x = torch.linspace(-1.2, 2.0, SCAN_WIDTH, device=device, dtype=dtype)
    y = torch.linspace(-1.2, 1.2, SCAN_HEIGHT, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    return torch.stack((xx, yy), dim=-1)


def feet_in_yaw_frame(root_pos, root_quat, tips_world):
    """Distal capsule centers relative to torso in its yaw-only frame (wxyz)."""
    if root_pos.shape != (tips_world.shape[0], 3) or tips_world.shape[1:] != (4, 3):
        raise ValueError("expected root[N,3] and four feet[N,4,3]")
    if root_quat.shape != (tips_world.shape[0], 4):
        raise ValueError("expected wxyz root quaternion[N,4]")
    w, x, y, z = root_quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    relative = tips_world - root_pos[:, None, :]
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    return torch.stack((c * relative[..., 0] + s * relative[..., 1],
                        -s * relative[..., 0] + c * relative[..., 1], relative[..., 2]), dim=-1)


def foot_position_maps(feet_xy, sigma=0.15):
    """Four Gaussian maps. Invalid/off-grid centers are absent, never clamped."""
    if feet_xy.ndim != 3 or feet_xy.shape[1:] != (4, 2):
        raise ValueError("expected four foot positions[N,4,2]")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    valid = (torch.isfinite(feet_xy).all(-1) & (feet_xy[..., 0] >= -1.2)
             & (feet_xy[..., 0] <= 2.0) & (feet_xy[..., 1].abs() <= 1.2))
    safe = torch.where(valid[..., None], feet_xy, torch.zeros_like(feet_xy))
    grid = scan_grid(device=feet_xy.device, dtype=feet_xy.dtype)
    distance2 = (grid[None, None] - safe[:, :, None, None]).square().sum(-1)
    return torch.exp(-distance2 / (2 * sigma ** 2)) * valid[:, :, None, None]
