"""Finite, frame-invariant encoding for the v6 vertical terrain-depth scan."""

from __future__ import annotations

import math

import torch


def encode_height_scan(
    root_z: torch.Tensor,
    mesh_hit_z: torch.Tensor,
    *,
    ray_offset_z: float = 2.0,
    max_distance: float = 4.0,
    plane_height: float = -50.0,
    height_offset: float = 0.5,
    height_clip: float = 1.0,
) -> torch.Tensor:
    """Return relative heights followed by validity bits, without lane labels.

    Downward rays see the nearest of the terrain mesh and the existing infinite
    ground plane. The analytic plane is needed because Isaac Lab's RayCaster
    currently accepts one mesh, whereas the unchanged v5 scene has two surfaces.
    Misses encode as +1 (unknown/deep) with a zero validity bit, never as safe flat.
    """
    if root_z.ndim != 1 or mesh_hit_z.ndim != 2 or root_z.shape[0] != mesh_hit_z.shape[0]:
        raise ValueError("expected root_z[N] and mesh_hit_z[N, rays]")
    if not all(math.isfinite(value) for value in (ray_offset_z, max_distance, plane_height, height_offset, height_clip)):
        raise ValueError("scan constants must be finite")
    if ray_offset_z < 0.0 or max_distance <= 0.0 or height_clip <= 0.0:
        raise ValueError("ray offset must be nonnegative; distance and clip must be positive")
    origins = root_z[:, None] + ray_offset_z
    mesh_distance = origins - mesh_hit_z
    mesh_valid = torch.isfinite(mesh_hit_z) & (mesh_distance >= 0.0) & (mesh_distance <= max_distance)
    plane_distance = origins - plane_height
    plane_valid = torch.isfinite(plane_distance) & (plane_distance >= 0.0) & (plane_distance <= max_distance)
    mesh = torch.where(mesh_valid, mesh_hit_z, -torch.inf)
    plane = torch.where(plane_valid, torch.full_like(origins, plane_height), -torch.inf)
    hit_z = torch.maximum(mesh, plane)
    valid = torch.isfinite(hit_z) & torch.isfinite(root_z[:, None])
    height = torch.where(valid, root_z[:, None] - hit_z - height_offset, height_clip)
    return torch.cat((height.clamp(-height_clip, height_clip) / height_clip, valid.to(height.dtype)), dim=-1)
