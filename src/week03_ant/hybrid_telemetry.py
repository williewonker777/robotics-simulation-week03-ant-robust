"""First-episode telemetry for frozen-policy switching (no simulator imports)."""
from __future__ import annotations

import math
import numbers

import torch


class HybridTelemetry:
    """Record only the first episode of each vectorized environment.

    Steps are 1-based. A switch happens before its action at ``(step-1)*dt``;
    termination happens after the action at ``step*dt``. The inclusive 0.5 s
    fall window is a temporal association, not a causal attribution. Action
    statistics concern normalized policy commands, not physical joint torque.
    """

    def __init__(self, num_envs: int, device: torch.device | str, dt: float):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if isinstance(dt, bool) or not isinstance(dt, numbers.Real) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.num_envs, self.device, self.dt = num_envs, torch.device(device), float(dt)
        self.finished = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.alpha_sum = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        self.target_steps = torch.zeros_like(self.steps)
        self.uncertain_steps = torch.zeros_like(self.steps)
        self.disagreement_sum = torch.zeros_like(self.alpha_sum)
        self.max_jump = torch.zeros_like(self.alpha_sum)
        self.previous_actions = torch.zeros((num_envs, 8), dtype=torch.float64, device=self.device)
        self.min_coverage = torch.ones_like(self.alpha_sum)
        self.feature_max = {key: torch.full_like(self.alpha_sum, float("nan")) for key in (
            "enter_span", "enter_edge", "retain_span", "retain_edge")}
        self.last_switch = torch.zeros_like(self.steps)
        self.fall_near_switch = torch.zeros_like(self.finished)
        self.switch_steps: list[list[int]] = [[] for _ in range(num_envs)]
        self.switch_targets: list[list[bool]] = [[] for _ in range(num_envs)]
        self._next_step = 1
        self._pending: torch.Tensor | None = None

    def _tensor(self, value, name: str, *, boolean: bool = False, actions: bool = False):
        shape = (self.num_envs, 8) if actions else (self.num_envs,)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must be a tensor with shape {shape}")
        if boolean:
            if value.dtype != torch.bool:
                raise ValueError(f"{name} must have boolean dtype")
            return value.to(self.device)
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite floating point")
        return value.detach().to(device=self.device, dtype=torch.float64)

    def _step(self, step):
        if isinstance(step, bool) or not isinstance(step, int) or step != self._next_step:
            raise ValueError(f"expected sequential step {self._next_step}")

    @torch.no_grad()
    def before_step(self, step, active, gate_output, v5_actions, v10_actions):
        """Consume pre-step commands and selector evidence; pair with after_step."""
        self._step(step)
        if self._pending is not None:
            raise RuntimeError("after_step must finish the preceding before_step")
        active = self._tensor(active, "active", boolean=True)
        if not torch.equal(active, ~self.finished):
            raise ValueError("active must match unfinished first episodes")
        commands = self._tensor(gate_output["actions"], "actions", actions=True)
        old = self._tensor(v5_actions, "v5_actions", actions=True)
        new = self._tensor(v10_actions, "v10_actions", actions=True)
        fields = {key: self._tensor(gate_output[key], key) for key in (
            "alpha", "coverage")}
        for key in self.feature_max:
            value = gate_output[key]
            if not isinstance(value, torch.Tensor) or value.shape != self.steps.shape or not value.is_floating_point():
                raise ValueError(f"{key} must be a floating point vector")
            if bool(torch.isinf(value).any()):
                raise ValueError(f"{key} must not contain infinity")
            fields[key] = value.detach().to(device=self.device, dtype=torch.float64)
        flags = {key: self._tensor(gate_output[key], key, boolean=True) for key in (
            "switched", "target_v10", "uncertain")}
        for key in ("alpha", "coverage"):
            if bool(((fields[key] < 0) | (fields[key] > 1)).any()):
                raise ValueError(f"{key} must lie in [0,1]")
        if any(bool((fields[key] < 0).any()) for key in self.feature_max):
            raise ValueError("terrain span and edge features must be nonnegative")
        disagreement = ((old - new).square().mean(dim=1)).sqrt()
        jump = ((commands - self.previous_actions).square().mean(dim=1)).sqrt()
        if not bool(torch.isfinite(disagreement).all() & torch.isfinite(jump).all()):
            raise ValueError("action statistics overflow")
        previous = active & (self.steps > 0)
        self.max_jump[previous] = torch.maximum(self.max_jump[previous], jump[previous])
        self.previous_actions[active] = commands[active]
        self.steps[active] += 1
        self.alpha_sum[active] += fields["alpha"][active]
        self.target_steps[active] += flags["target_v10"][active].long()
        self.uncertain_steps[active] += flags["uncertain"][active].long()
        self.disagreement_sum[active] += disagreement[active]
        self.min_coverage[active] = torch.minimum(self.min_coverage[active], fields["coverage"][active])
        for key, maximum in self.feature_max.items():
            maximum[active] = torch.fmax(maximum[active], fields[key][active])
        switched = active & flags["switched"]
        self.last_switch[switched] = step
        for index in switched.nonzero(as_tuple=False).flatten().cpu().tolist():
            self.switch_steps[index].append(step)
            self.switch_targets[index].append(bool(flags["target_v10"][index]))
        self._pending = active.clone()

    @torch.no_grad()
    def after_step(self, step, active, dones, reset_terminated):
        """Use the saved pre-step active mask, never an updated tracker mask."""
        self._step(step)
        if self._pending is None:
            raise RuntimeError("before_step must precede after_step")
        active = self._tensor(active, "active", boolean=True)
        dones = self._tensor(dones, "dones", boolean=True)
        terminated = self._tensor(reset_terminated, "reset_terminated", boolean=True)
        if not torch.equal(active, self._pending):
            raise ValueError("after_step active must equal saved pre-step mask")
        if bool((active & terminated & ~dones).any()):
            raise ValueError("active posture termination must also be done")
        newly_done = active & dones
        elapsed = (step - self.last_switch + 1).double() * self.dt
        self.fall_near_switch |= newly_done & terminated & (self.last_switch > 0) & (elapsed <= 0.5 + 1e-12)
        self.finished |= newly_done
        self._pending = None
        self._next_step += 1

    def to_dict(self, episode_lengths):
        """Return JSON-safe per-environment arrays, checking exact episode lengths.

        Extra fields report active steps, alpha sum, switch seconds/first/last,
        minimum scan coverage and maximum observed enter/retain span/edge. Missing
        feature samples are ignored; a never-observed maximum is null. Absent switch
        times are null. A zero-sample row has zero moments and coverage zero.
        """
        if self._pending is not None:
            raise RuntimeError("cannot serialize an unfinished step")
        lengths = torch.as_tensor(episode_lengths, device=self.device)
        if lengths.dtype == torch.bool or lengths.is_floating_point() or lengths.is_complex():
            raise ValueError("episode_lengths must contain integers")
        if lengths.shape != self.steps.shape or not torch.equal(lengths, self.steps):
            raise ValueError("telemetry counts differ from first-episode lengths")
        denominator = self.steps.clamp_min(1)
        def values(tensor):
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise RuntimeError("non-finite accumulated telemetry")
            return tensor.cpu().tolist()
        switch_seconds = [[(step - 1) * self.dt for step in events] for events in self.switch_steps]
        result = {
            "episode_active_steps": values(self.steps),
            "episode_alpha_sum": values(self.alpha_sum),
            "episode_v10_duty": values(self.alpha_sum / denominator),
            "episode_v10_target_steps": values(self.target_steps),
            "episode_uncertain_steps": values(self.uncertain_steps),
            "episode_switch_count": [len(events) for events in self.switch_steps],
            "episode_switch_steps": [events.copy() for events in self.switch_steps],
            "episode_switch_to_v10": [events.copy() for events in self.switch_targets],
            "episode_switch_seconds": switch_seconds,
            "episode_first_switch_seconds": [events[0] if events else None for events in switch_seconds],
            "episode_last_switch_seconds": [events[-1] if events else None for events in switch_seconds],
            "episode_fall_within_switch_window": values(self.fall_near_switch),
            "episode_mean_action_disagreement_rms": values(self.disagreement_sum / denominator),
            "episode_max_action_jump_rms": values(self.max_jump),
            "episode_min_coverage": values(torch.where(self.steps > 0, self.min_coverage, 0.0)),
        }
        result.update({f"episode_max_{key}": [float(x) if math.isfinite(float(x)) else None
                       for x in value.cpu()] for key, value in self.feature_max.items()})
        return result
