"""Stateless, planar directional regularization; no simulator or terrain labels."""

from dataclasses import dataclass, fields
import math

import torch


@dataclass(frozen=True)
class DirectionConfig:
    lateral_weight: float = .75
    yaw_weight: float = .25
    lateral_scale: float = 3.
    yaw_scale: float = 1.
    heading_gain: float = 1.
    max_yaw_rate: float = 1.
    min_forward_xy: float = .1

    def __post_init__(self):
        if not all(math.isfinite(getattr(self, field.name)) for field in fields(self)):
            raise ValueError("direction constants must be finite")
        if self.lateral_weight < 0 or self.yaw_weight < 0 or not math.isclose(
                self.lateral_weight + self.yaw_weight, 1., abs_tol=1e-12):
            raise ValueError("direction weights must be nonnegative and sum to one")
        if any(getattr(self, name) <= 0 for name in (
                'lateral_scale', 'yaw_scale', 'heading_gain', 'max_yaw_rate', 'min_forward_xy')):
            raise ValueError("direction scales and thresholds must be positive")


def direction_metrics(target_direction, body_forward, linear_velocity, angular_velocity,
                      config=DirectionConfig()):
    """Return per-environment unweighted reward and signed planar diagnostics.

    Velocities are world-frame COM quantities. World-Z angular velocity is a
    planar approximation, not the Euler yaw derivative on tilted bodies. Invalid
    rows abstain entirely; vertical velocity is checked for validity but unpenalized.
    RewardManager, not this function, applies the training weight and timestep.
    """
    if target_direction.ndim != 2 or target_direction.shape[1] != 2:
        raise ValueError("expected target_direction[N,2]")
    tensors = (target_direction, body_forward, linear_velocity, angular_velocity)
    expected = (target_direction.shape[0], 3)
    if any(value.shape != expected for value in tensors[1:]):
        raise ValueError("expected body_forward and velocities[N,3]")
    if any(not value.is_floating_point() for value in tensors):
        raise ValueError("direction inputs must be floating point")
    if any(value.device != target_direction.device or value.dtype != target_direction.dtype
           for value in tensors[1:]):
        raise ValueError("direction inputs must share dtype and device")
    valid = torch.stack([torch.isfinite(value).all(-1) for value in tensors]).all(0)
    # Mask before arithmetic so NaN/Inf cannot contaminate even invalid diagnostics.
    d, forward, velocity, angular = [torch.where(valid[:, None], value, 0.) for value in tensors]
    # Scale first to normalize finite directions without overflow/underflow.
    magnitude = d.abs().amax(-1)
    d = d / torch.where(magnitude > 0, magnitude, 1.)[:, None]
    norm = torch.linalg.vector_norm(d, dim=-1)
    d = d / torch.where(norm > 0, norm, 1.)[:, None]
    valid = valid & (magnitude > 0) & (torch.linalg.vector_norm(forward[:, :2], dim=-1)
                                             >= config.min_forward_xy)
    lateral = -d[:, 1] * velocity[:, 0] + d[:, 0] * velocity[:, 1]
    heading = torch.atan2(d[:, 1], d[:, 0]) - torch.atan2(forward[:, 1], forward[:, 0])
    heading = torch.atan2(torch.sin(heading), torch.cos(heading))
    target_yaw = (config.heading_gain * heading).clamp(-config.max_yaw_rate, config.max_yaw_rate)
    yaw_error = angular[:, 2] - target_yaw
    reward = (-config.lateral_weight * (1. - torch.exp(-config.lateral_scale * lateral.square()))
              - config.yaw_weight * (1. - torch.exp(-config.yaw_scale * yaw_error.square())))
    metrics = dict(reward=reward, lateral_velocity=lateral, heading_error=heading,
                   yaw_error=yaw_error, target_yaw_rate=target_yaw)
    valid = valid & torch.stack([torch.isfinite(value) for value in metrics.values()]).all(0)
    return {**{name: torch.where(valid, value, 0.) for name, value in metrics.items()}, 'valid': valid}
