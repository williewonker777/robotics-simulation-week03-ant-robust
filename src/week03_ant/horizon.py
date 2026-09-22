"""Pure first-episode tracking for the v10 long-horizon diagnostic."""

from __future__ import annotations

import math

import torch


class HorizonEpisodeTracker:
    """Track one long episode without admitting post-auto-reset state.

    ``current_distance`` and the current boundary flags describe environments
    that remain active after a vectorized step.  For newly completed episodes,
    callers must provide the pre-reset evidence saved by ``LaneState`` through
    the corresponding ``last_*`` arguments.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        dt: float,
        max_steps: int,
        snapshot_seconds: float = 16.0,
        one_tile_threshold: float = 13.1,
        all_tiles_threshold: float = 53.1,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        snapshot_step = snapshot_seconds / dt
        if not math.isclose(snapshot_step, round(snapshot_step), abs_tol=1.0e-6):
            raise ValueError("snapshot_seconds must fall on a control step")
        if not 0 < round(snapshot_step) < max_steps:
            raise ValueError("snapshot must be strictly inside the episode horizon")
        if not 0.0 < one_tile_threshold < all_tiles_threshold:
            raise ValueError("distance thresholds must be positive and ordered")

        self.num_envs = num_envs
        self.dt = float(dt)
        self.max_steps = max_steps
        self.snapshot_step = int(round(snapshot_step))
        self.one_tile_threshold = float(one_tile_threshold)
        self.all_tiles_threshold = float(all_tiles_threshold)
        self.finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.final_distance = torch.full((num_envs,), float("nan"), device=device)
        self.maximum_distance = torch.full((num_envs,), float("-inf"), device=device)
        self.distance_at_snapshot = torch.full((num_envs,), float("nan"), device=device)
        self.first_hit_one_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.first_hit_all_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.posture_terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.out_of_lane = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.world_exit = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _vector(self, value: torch.Tensor, *, dtype=None) -> torch.Tensor:
        result = value.reshape(-1).to(device=self.finished.device, dtype=dtype)
        if result.shape != self.finished.shape:
            raise ValueError(f"expected {self.num_envs} values, got {result.numel()}")
        return result

    def update(
        self,
        *,
        step: int,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        reset_terminated: torch.Tensor,
        current_distance: torch.Tensor,
        last_distance: torch.Tensor,
        current_out_of_lane: torch.Tensor,
        last_out_of_lane: torch.Tensor,
        current_world_exit: torch.Tensor,
        last_world_exit: torch.Tensor,
    ) -> None:
        """Consume one post-step sample from a vectorized auto-reset environment."""
        if not 1 <= step <= self.max_steps:
            raise ValueError("step is outside the configured horizon")

        rewards = self._vector(rewards, dtype=torch.float32)
        dones = self._vector(dones, dtype=torch.bool)
        reset_terminated = self._vector(reset_terminated, dtype=torch.bool)
        current_distance = self._vector(current_distance)
        last_distance = self._vector(last_distance)
        current_out_of_lane = self._vector(current_out_of_lane, dtype=torch.bool)
        last_out_of_lane = self._vector(last_out_of_lane, dtype=torch.bool)
        current_world_exit = self._vector(current_world_exit, dtype=torch.bool)
        last_world_exit = self._vector(last_world_exit, dtype=torch.bool)

        active = ~self.finished
        newly_done = active & dones
        still_active = active & ~dones
        # Auto-reset has already replaced the root state for newly-done
        # environments.  Select only LaneState's saved pre-reset evidence.
        sampled_distance = torch.where(newly_done, last_distance, current_distance)
        sampled_lane = torch.where(newly_done, last_out_of_lane, current_out_of_lane)
        sampled_world = torch.where(newly_done, last_world_exit, current_world_exit)

        self.returns[active] += rewards[active]
        self.steps[active] += 1
        self.maximum_distance[active] = torch.maximum(
            self.maximum_distance[active], sampled_distance[active]
        )
        self.out_of_lane[active] |= sampled_lane[active]
        self.world_exit[active] |= sampled_world[active]

        hit_one = active & (self.first_hit_one_step == 0) & (sampled_distance >= self.one_tile_threshold)
        hit_all = active & (self.first_hit_all_step == 0) & (sampled_distance >= self.all_tiles_threshold)
        self.first_hit_one_step[hit_one] = step
        self.first_hit_all_step[hit_all] = step

        if step == self.snapshot_step:
            # A terminal sample at the snapshot instant is censored: the first
            # episode no longer survives through the requested 16 s snapshot.
            self.distance_at_snapshot[still_active] = current_distance[still_active]

        self.final_distance[newly_done] = last_distance[newly_done]
        self.posture_terminated[newly_done] = reset_terminated[newly_done]
        self.finished |= newly_done

    @property
    def complete(self) -> bool:
        return bool(torch.all(self.finished).item())

    @staticmethod
    def _optional_floats(values: torch.Tensor) -> list[float | None]:
        return [float(value) if math.isfinite(float(value)) else None for value in values.detach().cpu()]

    @staticmethod
    def _optional_times(steps: torch.Tensor, dt: float) -> list[float | None]:
        return [float(step) * dt if int(step) > 0 else None for step in steps.detach().cpu()]

    def to_dict(self) -> dict[str, object]:
        """Return raw arrays plus aggregates after all first episodes finish."""
        if not self.complete:
            missing = int((~self.finished).sum().item())
            raise RuntimeError(f"horizon diagnostic incomplete: {missing} first episodes unfinished")
        finite = torch.isfinite(self.final_distance) & torch.isfinite(self.maximum_distance)
        if not bool(torch.all(finite).item()):
            raise RuntimeError("horizon diagnostic contains non-finite final distance evidence")

        full_survival = (self.steps >= self.max_steps) & ~self.posture_terminated
        safe = ~self.posture_terminated & ~self.out_of_lane & ~self.world_exit
        strict_one = safe & (self.final_distance >= self.one_tile_threshold)
        strict_all = safe & (self.final_distance >= self.all_tiles_threshold)
        hit_one = self.first_hit_one_step > 0
        hit_all = self.first_hit_all_step > 0
        count = self.num_envs

        forward_distance = self.final_distance.detach().cpu().tolist()
        maximum_distance = self.maximum_distance.detach().cpu().tolist()
        distance_at_16s = self._optional_floats(self.distance_at_snapshot)
        first_hit_one = self._optional_times(self.first_hit_one_step, self.dt)
        first_hit_six = self._optional_times(self.first_hit_all_step, self.dt)
        episode_terminated = self.posture_terminated.detach().cpu().tolist()
        return {
            # Primary-benchmark-compatible names make the 14 diagnostic files
            # straightforward to audit alongside the short-horizon JSON.
            "forward_distance": forward_distance,
            "episode_terminated": episode_terminated,
            "episode_lengths": self.steps.detach().cpu().tolist(),
            "episode_out_of_lane": self.out_of_lane.detach().cpu().tolist(),
            "episode_world_exit": self.world_exit.detach().cpu().tolist(),
            # Long-horizon-only evidence.
            "first_hit_one_seconds": first_hit_one,
            "first_hit_six_seconds": first_hit_six,
            "distance_at_16s": distance_at_16s,
            "maximum_distance_m": maximum_distance,
            "episode_return": self.returns.detach().cpu().tolist(),
            "episode_steps": self.steps.detach().cpu().tolist(),
            "episode_full_64s_survival": full_survival.detach().cpu().tolist(),
            "episode_strict_one_tile_success": strict_one.detach().cpu().tolist(),
            "episode_strict_all_tiles_success": strict_all.detach().cpu().tolist(),
            "aggregates": {
                "episodes": count,
                "posture_terminations": int(self.posture_terminated.sum().item()),
                "out_of_lane_exits": int(self.out_of_lane.sum().item()),
                "world_exits": int(self.world_exit.sum().item()),
                "full_64s_survivals": int(full_survival.sum().item()),
                "distance_only_one_tile_hits": int(hit_one.sum().item()),
                "distance_only_all_tiles_hits": int(hit_all.sum().item()),
                "strict_one_tile_successes": int(strict_one.sum().item()),
                "strict_all_tiles_successes": int(strict_all.sum().item()),
                "strict_one_tile_success_rate": float(strict_one.float().mean().item()),
                "strict_all_tiles_success_rate": float(strict_all.float().mean().item()),
                "final_forward_distance_mean_m": float(self.final_distance.mean().item()),
                "maximum_forward_distance_mean_m": float(self.maximum_distance.mean().item()),
            },
        }
