"""Three stateless posture inputs, using the frozen reward's geometry and scales."""

import torch
from .posture_math import PostureConfig


def command_features(geometry):
    config = PostureConfig()
    target, clearance = geometry["target_height"], geometry["body_clearance"]
    valid, motion = geometry["valid"], geometry["motion_valid"]
    if (target.ndim != 1 or clearance.shape != target.shape
            or valid.shape != target.shape or motion.shape != target.shape):
        raise ValueError("command geometry requires matching one-dimensional tensors")
    available = (valid.bool() & motion.bool() & torch.isfinite(valid)
                 & torch.isfinite(motion) & torch.isfinite(target) & torch.isfinite(clearance))
    values = torch.stack(((target - config.clear_height) / config.rough_height_delta,
                          ((clearance - config.clear_height) / config.rough_height_delta).clamp(-8, 8),
                          available.to(target.dtype)), dim=-1)
    return torch.where(available[:, None], values, torch.zeros_like(values))
