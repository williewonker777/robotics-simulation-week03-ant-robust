"""Current-scan posture shaping; no simulator state, labels, or temporal cache."""

from dataclasses import dataclass, fields
import math

import torch

from .foothold_math import FOOT_RADIUS, GRID_HEIGHT, GRID_RAYS, GRID_WIDTH, foothold_grid


@dataclass(frozen=True)
class PostureConfig:
    local_half_width: float = 0.3
    front_min_x: float = 0.3
    front_max_x: float = 1.5
    front_half_width: float = 0.6
    min_coverage: float = 0.9
    severity_start: float = 0.03
    severity_span: float = 0.12
    clear_height: float = 0.44
    rough_height_delta: float = 0.14
    body_error_scale: float = 0.12
    foot_patch_half_width: float = 0.15
    min_foot_samples: int = 9
    clear_foot_clearance: float = 0.04
    rough_foot_clearance_delta: float = 0.08
    foot_error_scale: float = 0.12
    swing_speed_scale: float = 2.0
    foot_cost_weight: float = 0.25
    speed_scale: float = 6.0
    min_upright: float = math.cos(1.2)
    flat_bonus_weight: float = 0.5

    def __post_init__(self):
        if not all(math.isfinite(getattr(self, field.name)) for field in fields(self)):
            raise ValueError("posture constants must be finite")
        positive = ("severity_span", "body_error_scale", "foot_error_scale",
                    "swing_speed_scale", "speed_scale")
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("posture scales must be positive")
        nonnegative = ("severity_start", "rough_height_delta", "clear_foot_clearance",
                       "rough_foot_clearance_delta", "foot_cost_weight", "flat_bonus_weight")
        if any(getattr(self, name) < 0 for name in nonnegative):
            raise ValueError("posture increments, clearances and weights must be nonnegative")
        if self.clear_height <= .31:
            raise ValueError("clear_height must exceed the unchanged 0.31m fall bound")
        if not 0 < self.min_coverage <= 1 or not 0 <= self.min_upright <= 1:
            raise ValueError("coverage must be in (0,1] and upright threshold in [0,1]")
        if not (0 < self.local_half_width <= 1.2
                and -1.2 <= self.front_min_x < self.front_max_x <= 2.
                and 0 < self.front_half_width <= 1.2):
            raise ValueError("posture ROIs must be nonempty and inside the scan grid")
        # Require at least one sampled front column, not merely a continuous ROI.
        if math.ceil((self.front_min_x + 1.2) / .1 - 1e-6) > math.floor(
                (self.front_max_x + 1.2) / .1 + 1e-6):
            raise ValueError("front ROI must contain scan samples")
        if not 0 < self.foot_patch_half_width <= 1.2:
            raise ValueError("foot patch must fit inside the scan grid")
        max_side_samples = math.floor(2 * self.foot_patch_half_width / .1 + 1e-6) + 1
        max_samples = min(GRID_WIDTH, max_side_samples) * min(GRID_HEIGHT, max_side_samples)
        if (not isinstance(self.min_foot_samples, int) or isinstance(self.min_foot_samples, bool)
                or not 1 <= self.min_foot_samples <= max_samples):
            raise ValueError("min_foot_samples must be a positive achievable integer")


def terrain_posture_terms(surface_z, valid, feet, forward_swing_speed,
                          forward_speed, upright, config=PostureConfig()):
    """Return bounded costs/bonus and diagnostics in a torso-relative yaw frame.

    Invalid body coverage abstains from the entire new reward. Invalid foot
    patches abstain individually (the divisor remains four). The forward-swing
    proxy is kinematic, not a contact detector or proof of stance exclusion.
    """
    if surface_z.ndim != 2 or surface_z.shape[1] != GRID_RAYS or valid.shape != surface_z.shape:
        raise ValueError("expected surface_z[N,825] and matching validity")
    n = surface_z.shape[0]
    if feet.shape != (n, 4, 3) or forward_swing_speed.shape != (n, 4):
        raise ValueError("expected feet[N,4,3] and forward_swing_speed[N,4]")
    if forward_speed.shape != (n,) or upright.shape != (n,):
        raise ValueError("expected forward_speed[N] and upright[N]")
    c = config
    grid = foothold_grid(device=surface_z.device, dtype=surface_z.dtype).reshape(-1, 2)
    # Tolerance admits mathematical grid boundaries despite linspace rounding.
    eps = 1e-6
    local = (grid.abs() <= c.local_half_width + eps).all(-1)
    front = ((grid[:, 0] >= c.front_min_x - eps)
             & (grid[:, 0] <= c.front_max_x + eps)
             & (grid[:, 1].abs() <= c.front_half_width + eps))
    observed = valid.bool() & torch.isfinite(valid) & torch.isfinite(surface_z)
    height = torch.where(observed, surface_z, torch.zeros_like(surface_z))
    local_valid = observed & local
    front_valid = observed & front
    local_coverage = local_valid.sum(-1).to(surface_z.dtype) / local.sum()
    front_coverage = front_valid.sum(-1).to(surface_z.dtype) / front.sum()
    coverage = ((local_valid.sum(-1) >= c.min_coverage * local.sum())
                & (front_valid.sum(-1) >= c.min_coverage * front.sum()))
    finite_motion = torch.isfinite(forward_speed) & torch.isfinite(upright)
    body_valid = coverage & finite_motion
    ground = height.masked_fill(~local_valid, -torch.inf).amax(-1)
    high = height.masked_fill(~front_valid, -torch.inf).amax(-1)
    low = height.masked_fill(~front_valid, torch.inf).amin(-1)
    # Replace missing reductions BEFORE subtraction, not by multiplying NaNs by 0.
    ground = torch.where(body_valid, ground, torch.zeros_like(ground))
    high = torch.where(body_valid, high, torch.zeros_like(high))
    low = torch.where(body_valid, low, torch.zeros_like(low))
    measure = torch.maximum(high, ground) - torch.minimum(low, ground)
    severity = ((measure - c.severity_start) / c.severity_span).clamp(0, 1)
    target_height = c.clear_height + c.rough_height_delta * severity
    body_clearance = -ground
    body_error_squared = ((body_clearance - target_height) / c.body_error_scale).square()
    body_cost = torch.where(body_valid, body_error_squared.clamp(max=1), 0.)

    finite_feet = torch.isfinite(feet).all(-1)
    safe_feet = torch.where(finite_feet[..., None], feet, torch.zeros_like(feet))
    radius = c.foot_patch_half_width
    patch = ((grid[None, None] - safe_feet[:, :, None, :2]).abs() <= radius + eps).all(-1)
    in_bounds = ((safe_feet[..., :2] - radius >= grid.amin(0) - eps)
                 & (safe_feet[..., :2] + radius <= grid.amax(0) + eps)).all(-1)
    foot_valid = (finite_feet & in_bounds & (patch.sum(-1) >= c.min_foot_samples)
                  & (~patch | observed[:, None]).all(-1) & body_valid[:, None]
                  & torch.isfinite(forward_swing_speed))
    support = height[:, None].expand(-1, 4, -1).masked_fill(~patch, -torch.inf).amax(-1)
    support = torch.where(foot_valid, support, torch.zeros_like(support))
    foot_z = torch.where(foot_valid, safe_feet[..., 2], torch.zeros_like(support))
    foot_clearance = torch.where(foot_valid, foot_z - FOOT_RADIUS - support, 0.)
    target_foot_clearance = c.clear_foot_clearance + c.rough_foot_clearance_delta * severity
    foot_deficit = ((target_foot_clearance[:, None] - foot_clearance)
                    / c.foot_error_scale).clamp(0, 1).square()
    foot_deficit = torch.where(foot_valid, foot_deficit, 0.)
    speed = torch.where(finite_motion, forward_speed, torch.zeros_like(forward_speed))
    safe_upright = torch.where(finite_motion, upright, torch.zeros_like(upright))
    swing = torch.where(foot_valid & (speed[:, None] > 0), forward_swing_speed, 0.)
    swing_factor = swing.clamp(0, c.swing_speed_scale) / c.swing_speed_scale
    foot_cost = (foot_deficit * swing_factor).mean(-1)
    combined_cost = body_cost + c.foot_cost_weight * foot_cost
    flat_speed_bonus = ((1 - severity) * torch.exp(-body_error_squared)
                        * speed.clamp(0, c.speed_scale) / c.speed_scale
                        * safe_upright.clamp(0, 1))
    flat_speed_bonus = torch.where(body_valid & (safe_upright >= c.min_upright), flat_speed_bonus, 0.)
    return {
        "body_clearance": body_clearance, "target_height": target_height,
        "motion_valid": finite_motion,
        "severity": severity, "valid": body_valid, "body_cost": body_cost,
        "local_coverage": local_coverage, "front_coverage": front_coverage,
        "foot_cost": foot_cost, "combined_cost": combined_cost,
        "flat_speed_bonus": flat_speed_bonus,
        "reward": -combined_cost + c.flat_bonus_weight * flat_speed_bonus,
        "forward_speed": speed, "foot_clearance": foot_clearance,
        "foot_valid": foot_valid, "foot_deficit": foot_deficit,
        "target_foot_clearance": target_foot_clearance,
    }
