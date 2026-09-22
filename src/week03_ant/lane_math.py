"""Pure tensor helpers for the v5 treadmill terrain lanes.

The v5 rough-terrain task lays every terrain family out as a lane along +x that
starts and ends with a flat portal tile. An Ant that reaches the centre of the
exit portal is moved back to the centre of the entrance portal. These helpers
contain the geometry only, so they can be unit-tested without Isaac Sim.
"""

from __future__ import annotations

import torch


def swing_clearance_cost(
    foot_clearance: torch.Tensor, forward_swing_speed: torch.Tensor, target_clearance: float,
    trap_weight: float = 0.0,
) -> torch.Tensor:
    """Bounded per-foot deficit, charged only while that foot swings forward.

    Clearance is the distal capsule bottom above the stone-top reference, not
    ankle-frame height or height above the bottom of a gap. A stationary stance
    foot and a backward-moving support foot incur no swing cost. With a positive
    trap_weight, a foot below nominal stone tops is also charged while stopped;
    otherwise reducing swing speed could hide a foot trapped in a gap. Zero
    preserves the original swing-only curriculum for reproducibility.
    """
    if not 0.0 < target_clearance < float("inf"):
        raise ValueError("target_clearance must be finite and positive")
    if not 0.0 <= trap_weight < float("inf"):
        raise ValueError("trap_weight must be finite and non-negative")
    deficit = ((target_clearance - foot_clearance) / target_clearance).clamp(0.0, 2.0)
    swing = forward_swing_speed.clamp(0.0, 2.0) / 2.0
    trapped = (-foot_clearance / target_clearance).clamp(0.0, 2.0).square() * trap_weight
    return torch.maximum(deficit.square() * swing, trapped).mean(dim=-1)


def body_height_deficit(height: torch.Tensor, target_height: float, tolerance: float) -> torch.Tensor:
    """Bounded low-body penalty; no incentive to jump above the target height."""
    if not 0.0 < tolerance < float("inf") or not 0.0 < target_height < float("inf"):
        raise ValueError("target_height and tolerance must be finite and positive")
    return ((target_height - height) / tolerance).clamp(0.0, 2.0).square()


def cap_terrain_progress(progress: torch.Tensor, on_plane: torch.Tensor, speed_limit: float) -> torch.Tensor:
    """Cap positive terrain speed reward without changing flat-ground or reverse progress."""
    if not 0.0 < speed_limit < float("inf"):
        raise ValueError("speed_limit must be finite and positive")
    return torch.where(on_plane, progress, progress.clamp(max=speed_limit))


def lane_boundary_masks(
    pos_xy: torch.Tensor, center_y: torch.Tensor, half_width: float,
    world_half_size: tuple[float, float], on_mesh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torso departures from the assigned lane or finite mesh; the plane is unbounded."""
    out_of_lane = on_mesh & ((pos_xy[:, 1] - center_y).abs() > half_width)
    world_exit = on_mesh & (
        (pos_xy[:, 0].abs() > world_half_size[0]) | (pos_xy[:, 1].abs() > world_half_size[1])
    )
    return out_of_lane, world_exit


def lane_target_direction(pos_xy: torch.Tensor, lane_center_y: torch.Tensor, lookahead: float) -> torch.Tensor:
    """Unit xy vector towards a point ``lookahead`` metres ahead on the lane centre line.

    The course task walks towards a far target at ``(1000, 0)``. Lanes sit at many
    lateral offsets, so each environment instead follows its own lane: the target
    stays ahead in +x and pulls the Ant back to the centre line when it drifts.
    """
    if lookahead <= 0.0:
        raise ValueError(f"lookahead must be positive, got {lookahead}")
    dx = torch.full_like(pos_xy[:, 0], lookahead)
    dy = lane_center_y - pos_xy[:, 1]
    direction = torch.stack((dx, dy), dim=-1)
    return direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)


def lane_wrap_shift(x_entrance: torch.Tensor, x_exit: torch.Tensor) -> torch.Tensor:
    """Distance an Ant is moved back when it reaches the exit portal centre."""
    return x_exit - x_entrance


def lane_wrap_mask(x: torch.Tensor, x_exit: torch.Tensor) -> torch.Tensor:
    """Environments whose torso passed the centre of their lane's exit portal."""
    return x > x_exit


def lane_odometer(x: torch.Tensor, wrap_count: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Unwrapped forward coordinate: world x plus the distance removed by wraps."""
    return x + wrap_count.to(x.dtype) * shift


def lane_family_level(lane: torch.Tensor, num_levels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a lane index into its (terrain family, difficulty level) pair."""
    if num_levels <= 0:
        raise ValueError(f"num_levels must be positive, got {num_levels}")
    return torch.div(lane, num_levels, rounding_mode="floor"), torch.remainder(lane, num_levels)


def allocate_by_weight(total: int, weights: torch.Tensor) -> torch.Tensor:
    """Split ``total`` items into integer counts proportional to ``weights``.

    Uses the largest-remainder method, so the counts always sum to ``total`` and the
    allocation is deterministic.
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    weights = weights.to(torch.float64)
    if weights.numel() == 0 or bool((weights < 0).any()) or float(weights.sum()) <= 0.0:
        raise ValueError("weights must be non-negative with a positive sum")
    exact = weights / weights.sum() * total
    counts = torch.floor(exact).to(torch.long)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = torch.argsort(exact - counts.to(torch.float64), descending=True, stable=True)
        counts[order[:remainder]] += 1
    return counts


def weighted_lane_layout(num_envs: int, family_weights: torch.Tensor, num_levels: int) -> torch.Tensor:
    """Lane index per environment: families by weight, levels cycled within each family."""
    counts = allocate_by_weight(num_envs, family_weights)
    families = torch.repeat_interleave(torch.arange(len(counts)), counts)
    offsets = torch.cumsum(counts, 0) - counts
    within_family = torch.arange(num_envs) - torch.repeat_interleave(offsets, counts)
    return families * num_levels + torch.remainder(within_family, num_levels)
