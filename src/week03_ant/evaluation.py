"""Pure accumulation utilities shared by the evaluation script and tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LaneTraversalGeometry:
    """Longitudinal geometry used to prove traversal of a rough lane.

    Distances are measured from the entrance-portal centre.  Clearing a tile
    means that the Ant's torso has moved one full foot reach beyond the far
    edge, rather than merely touching or entering the tile.
    """

    tile_length: float
    terrain_tiles: int
    foot_reach: float

    def __post_init__(self) -> None:
        if self.tile_length <= 0.0:
            raise ValueError("tile_length must be positive")
        if self.terrain_tiles <= 0:
            raise ValueError("terrain_tiles must be positive")
        if self.foot_reach < 0.0:
            raise ValueError("foot_reach must be non-negative")

    @property
    def entrance_half_length(self) -> float:
        return 0.5 * self.tile_length

    @property
    def one_tile_clearance(self) -> float:
        return self.entrance_half_length + self.tile_length + self.foot_reach

    @property
    def all_tiles_clearance(self) -> float:
        return self.entrance_half_length + self.terrain_tiles * self.tile_length + self.foot_reach

    @property
    def lane_period(self) -> float:
        # Portal-centre to portal-centre: half an entrance portal, all terrain
        # tiles, then half an exit portal.
        return (self.terrain_tiles + 1) * self.tile_length


@dataclass(frozen=True)
class LaneTraversalClassification:
    """Per-episode distance evidence and termination-aware terrain successes."""

    applicable: torch.Tensor
    distance_cleared_one_tile: torch.Tensor
    distance_cleared_all_tiles: torch.Tensor
    distance_complete_lane_loops: torch.Tensor
    cleared_one_tile: torch.Tensor
    cleared_all_tiles: torch.Tensor
    complete_lane_loops: torch.Tensor


def classify_lane_traversal(
    forward_distance: torch.Tensor,
    terminated: torch.Tensor,
    plane_family: torch.Tensor,
    geometry: LaneTraversalGeometry,
    out_of_lane: torch.Tensor | None = None,
    world_exit: torch.Tensor | None = None,
) -> LaneTraversalClassification:
    """Classify terrain traversal without treating survival as traversal.

    A successful crossing requires both geometric clearance and an episode
    without a terminal condition.  Raw distance-only evidence is retained so
    a run that crossed terrain and subsequently fell remains diagnosable, but
    is not counted as successful.  Plane-family episodes are explicitly
    inapplicable because travel on an unbounded plane is not terrain traversal.
    """
    distance = forward_distance.reshape(-1)
    terminated = terminated.reshape(-1).to(device=distance.device, dtype=torch.bool)
    plane_family = plane_family.reshape(-1).to(device=distance.device, dtype=torch.bool)
    if distance.shape != terminated.shape or distance.shape != plane_family.shape:
        raise ValueError("forward_distance, terminated, and plane_family must have equal lengths")
    if out_of_lane is None:
        out_of_lane = torch.zeros_like(terminated)
    else:
        out_of_lane = out_of_lane.reshape(-1).to(device=distance.device, dtype=torch.bool)
    if world_exit is None:
        world_exit = torch.zeros_like(terminated)
    else:
        world_exit = world_exit.reshape(-1).to(device=distance.device, dtype=torch.bool)
    if distance.shape != out_of_lane.shape or distance.shape != world_exit.shape:
        raise ValueError("boundary flags must have the same length as forward_distance")

    finite_distance = torch.isfinite(distance)
    finite_nonnegative_distance = torch.where(
        finite_distance, torch.clamp_min(distance, 0.0), torch.zeros_like(distance)
    )
    applicable = ~plane_family
    distance_cleared_one = applicable & finite_distance & (distance >= geometry.one_tile_clearance)
    distance_cleared_all = applicable & finite_distance & (distance >= geometry.all_tiles_clearance)
    distance_loops = torch.floor(finite_nonnegative_distance / geometry.lane_period).to(torch.long)
    distance_loops = torch.where(applicable, distance_loops, torch.zeros_like(distance_loops))
    successful = applicable & ~terminated & ~out_of_lane & ~world_exit
    return LaneTraversalClassification(
        applicable=applicable,
        distance_cleared_one_tile=distance_cleared_one,
        distance_cleared_all_tiles=distance_cleared_all,
        distance_complete_lane_loops=distance_loops,
        cleared_one_tile=distance_cleared_one & successful,
        cleared_all_tiles=distance_cleared_all & successful,
        complete_lane_loops=torch.where(successful, distance_loops, torch.zeros_like(distance_loops)),
    )


@dataclass
class EvaluationSummary:
    returns: torch.Tensor
    lengths: torch.Tensor

    def to_dict(self) -> dict[str, object]:
        returns = self.returns.detach().cpu().to(torch.float64)
        lengths = self.lengths.detach().cpu().to(torch.float64)
        return {
            "completed_episodes": int(returns.numel()),
            "episode_return_mean": float(returns.mean()),
            "episode_return_std": float(returns.std(unbiased=False)),
            "episode_return_min": float(returns.min()),
            "episode_return_max": float(returns.max()),
            "episode_length_mean": float(lengths.mean()),
            "episode_length_std": float(lengths.std(unbiased=False)),
            "episode_returns": returns.tolist(),
            "episode_lengths": lengths.tolist(),
        }


class FirstEpisodeAccumulator:
    """Accumulate exactly one episode per vectorized environment."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.returns = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.finished = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        rewards = rewards.reshape(-1)
        dones = dones.reshape(-1).to(torch.bool)
        active = ~self.finished
        self.returns[active] += rewards[active]
        self.lengths[active] += 1
        self.finished |= dones & active

    @property
    def complete(self) -> bool:
        return bool(torch.all(self.finished).item())

    def summary(self) -> EvaluationSummary:
        if not self.complete:
            missing = int((~self.finished).sum().item())
            raise RuntimeError(f"evaluation incomplete: {missing} environments have no finished episode")
        return EvaluationSummary(self.returns.clone(), self.lengths.clone())
