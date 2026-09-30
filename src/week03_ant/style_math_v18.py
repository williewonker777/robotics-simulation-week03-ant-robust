"""Training-only binary teacher mask derived from the existing depth scan."""

from __future__ import annotations

import torch

from .foothold_math import GRID_RAYS, foothold_grid


def teacher_mask_from_scan(surface_z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return 1 for flat/unknown and 0 for observable non-flat terrain.

    Heights may be translated vertically without changing the answer. Missing
    rays are never treated as zero-height terrain. The output is a separate
    train-only observation and is not a terrain-family label or actor input.
    """
    if (surface_z.ndim != 2 or surface_z.shape[1] != GRID_RAYS
            or valid.shape != surface_z.shape or valid.dtype != torch.bool
            or not surface_z.is_floating_point() or surface_z.device != valid.device):
        raise ValueError("expected height float[N,825] and valid bool[N,825] on one device")
    xy = foothold_grid(device=surface_z.device, dtype=surface_z.dtype).reshape(-1, 2)
    roi = (xy[:, 0] >= 0.) & (xy[:, 0] <= 1.5) & (xy[:, 1].abs() <= .8)
    observed = valid & torch.isfinite(surface_z) & roi[None, :]
    count = observed.sum(dim=1)
    coverage = count.to(surface_z.dtype) / roi.sum()
    safe = torch.where(observed, surface_z, torch.zeros_like(surface_z))
    mean = safe.sum(dim=1) / count.clamp_min(1)
    variance = torch.where(observed, (surface_z - mean[:, None]).square(), 0.).sum(dim=1)
    std = (variance / count.clamp_min(1)).sqrt()
    known = coverage >= .20
    rough = known & ((std > .035) | (coverage < .90))
    return (~rough).to(surface_z.dtype).unsqueeze(1)
