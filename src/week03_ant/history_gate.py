"""Temporal depth-selector history, not recurrent policy training or a world map."""
from dataclasses import dataclass, field
import math

import torch

from week03_ant.hybrid_gate import DepthPolicyGate as SpatialDepthPolicyGate
from week03_ant.hybrid_gate import GateConfig, PRESETS


@dataclass(frozen=True)
class HistoryGateConfig:
    spatial: GateConfig = field(default_factory=lambda: PRESETS["cautious"])
    history_seconds: float = 1.0
    enter_window_seconds: float = .3
    enter_min_seconds: float = .2
    enter_fraction: float = .8
    enter_confirm_seconds: float = .1
    exit_window_seconds: float = .8
    exit_min_seconds: float = .6
    exit_fraction: float = .9
    exit_confirm_seconds: float = .3
    min_dwell_seconds: float = .5
    blend_seconds: float = .15
    unknown_reset_seconds: float = .25

    def __post_init__(self):
        if not isinstance(self.spatial, GateConfig):
            raise ValueError("spatial must be GateConfig")
        for name, value in vars(self).items():
            if name == "spatial":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for direction in ("enter", "exit"):
            window = getattr(self, f"{direction}_window_seconds")
            if (getattr(self, f"{direction}_min_seconds") > window
                    or getattr(self, f"{direction}_confirm_seconds") > window
                    or window > self.history_seconds):
                raise ValueError("minimum/confirmation must be <= window <= history")
            if getattr(self, f"{direction}_fraction") > 1:
                raise ValueError("evidence fractions must be <=1")


class HistoryDepthPolicyGate:
    """Bounded per-environment depth evidence with sustained enter/exit decisions.

    Only depth-derived features are stored: no family labels, pose history, map,
    rewards, or action-performance evidence. Unknown recorded samples count in
    vote denominators. Long relevant-ROI unknown intervals erase stale history
    but retain the target and allow an existing crossfade to finish. Episode
    reset clears selected environments; callers must not reset on portal wraps.
    Feature history is a ring in FEATURE_NAMES order; history_recorded marks
    occupied slots. Diagnostics returned by step() are independent snapshots.
    """

    FEATURE_NAMES = ("enter_span", "enter_edge", "enter_coverage", "enter_edge_coverage",
                     "retain_span", "retain_edge", "retain_coverage", "retain_edge_coverage")

    def __init__(self, num_envs, device, dt, config):
        if not isinstance(config, HistoryGateConfig):
            raise ValueError("config must be HistoryGateConfig")
        self._spatial = SpatialDepthPolicyGate(num_envs, device, dt, config.spatial)
        self.num_envs, self.device, self.dt, self.config = num_envs, torch.device(device), dt, config
        self._counts = {name: max(1, math.ceil(value / dt - 1.e-9))
                        for name, value in vars(config).items() if name.endswith("seconds")}
        size = self._counts["history_seconds"]
        self.history_features = torch.full((num_envs, size, 8), torch.nan, device=device)
        self.history_recorded = torch.zeros((num_envs, size), dtype=torch.bool, device=device)
        self._rough_history = torch.zeros_like(self.history_recorded)
        self._clear_history = torch.zeros_like(self.history_recorded)
        self._cursor = 0
        self.target_v10 = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.alpha = torch.zeros(num_envs, device=device)
        self._age = torch.full((num_envs,), self._counts["min_dwell_seconds"], dtype=torch.long, device=device)
        self._rough_run = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._clear_run = torch.zeros_like(self._rough_run)
        self._unknown_run = torch.zeros_like(self._rough_run)
        self._offsets = {direction: torch.arange(self._counts[f"{direction}_window_seconds"], device=device)
                         for direction in ("enter", "exit")}

    @torch.inference_mode()
    def _clear_history_for(self, mask):
        self.history_features[mask] = torch.nan
        self.history_recorded[mask] = False
        self._rough_history[mask] = False
        self._clear_history[mask] = False
        self._rough_run[mask] = 0
        self._clear_run[mask] = 0

    @torch.inference_mode()
    def reset(self, done_mask=None):
        if done_mask is None:
            done_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Reuse the original reset mask validation without changing its source.
        self._spatial.reset(done_mask)
        self._clear_history_for(done_mask)
        self.target_v10[done_mask] = False
        self.alpha[done_mask] = 0
        self._age[done_mask] = self._counts["min_dwell_seconds"]
        self._unknown_run[done_mask] = 0

    def _votes(self, direction, evidence):
        indices = (self._cursor - self._offsets[direction]) % self.history_recorded.shape[1]
        recorded = self.history_recorded[:, indices]
        samples = recorded.sum(1)
        votes = (evidence[:, indices] & recorded).sum(1)
        fraction = votes.float() / samples.clamp_min(1)
        return votes, fraction, samples

    @torch.inference_mode()
    def step(self, height, valid, v5_actions, v10_actions, mode="hybrid"):
        if mode not in ("hybrid", "v5", "v10"):
            raise ValueError("mode must be hybrid, v5, or v10")
        result = self._spatial.step(height, valid, v5_actions, v10_actions, mode="v5")
        cfg = self.config
        spatial = cfg.spatial
        known = {}
        for roi in ("enter", "retain"):
            known[roi] = ((result[f"{roi}_coverage"] >= spatial.min_coverage)
                          & (result[f"{roi}_edge_coverage"] >= spatial.min_coverage)
                          & torch.isfinite(result[f"{roi}_span"])
                          & torch.isfinite(result[f"{roi}_edge"]))
        old_target = self.target_v10.clone()
        uncertain = ~torch.where(old_target, known["retain"], known["enter"])
        coverage = torch.where(old_target, result["retain_coverage"], result["enter_coverage"])
        rough = known["enter"] & ((result["enter_span"] >= spatial.span_enter)
                                    | (result["enter_edge"] >= spatial.edge_enter))
        clear = known["retain"] & (result["retain_span"] <= spatial.exit_ratio * spatial.span_enter)
        clear &= result["retain_edge"] <= spatial.exit_ratio * spatial.edge_enter
        switched = torch.zeros_like(old_target)
        expired = torch.zeros_like(old_target)
        if mode == "hybrid":
            self.history_features[:, self._cursor] = torch.stack([result[name] for name in self.FEATURE_NAMES], dim=1)
            self.history_recorded[:, self._cursor] = True
            self._rough_history[:, self._cursor] = rough
            self._clear_history[:, self._cursor] = clear
            self._rough_run = torch.where(rough, self._rough_run + 1, 0)
            self._clear_run = torch.where(clear, self._clear_run + 1, 0)
            self._unknown_run = torch.where(uncertain, self._unknown_run + 1, 0)
            expired = self._unknown_run >= self._counts["unknown_reset_seconds"]
            self._clear_history_for(expired)
            self._age = (self._age + 1).clamp_max(self._counts["min_dwell_seconds"])
        else:
            self.reset()
            self.target_v10.fill_(mode == "v10")
            self.alpha.fill_(float(mode == "v10"))
        rough_votes, rough_fraction, enter_samples = self._votes("enter", self._rough_history)
        clear_votes, clear_fraction, exit_samples = self._votes("exit", self._clear_history)
        # Capture confirmation lengths before clearing switch-related counters.
        rough_seconds = self._rough_run.float() * self.dt
        clear_seconds = self._clear_run.float() * self.dt
        if mode == "hybrid":
            dwell = self._age >= self._counts["min_dwell_seconds"]
            enter = (~old_target & rough & (rough_votes >= self._counts["enter_min_seconds"])
                     & (rough_fraction >= cfg.enter_fraction)
                     & (self._rough_run >= self._counts["enter_confirm_seconds"]) & dwell)
            leave = (old_target & clear & (clear_votes >= self._counts["exit_min_seconds"])
                     & (clear_fraction >= cfg.exit_fraction)
                     & (self._clear_run >= self._counts["exit_confirm_seconds"]) & dwell)
            self.target_v10 = (old_target | enter) & ~leave
            switched = self.target_v10 != old_target
            self._age[switched] = 0
            self._rough_run[switched] = 0
            self._clear_run[switched] = 0
            delta = self.dt / cfg.blend_seconds
            self.alpha = (self.alpha + torch.where(self.target_v10, delta, -delta)).clamp(0, 1)
            alpha = self.alpha.to(v5_actions.dtype)[:, None]
            blended = (1 - alpha) * v5_actions + alpha * v10_actions
            actions = torch.where(alpha == 0, v5_actions, torch.where(alpha == 1, v10_actions, blended))
            self._cursor = (self._cursor + 1) % self.history_recorded.shape[1]
        else:
            actions = v5_actions if mode == "v5" else v10_actions
        result.update(actions=actions, alpha=self.alpha.clone(), target_v10=self.target_v10.clone(),
                      switched=switched, uncertain=uncertain, coverage=coverage,
                      rough_fraction=rough_fraction, clear_fraction=clear_fraction,
                      rough_votes=rough_votes, clear_votes=clear_votes,
                      rough_evidence_seconds=rough_votes.float() * self.dt,
                      clear_evidence_seconds=clear_votes.float() * self.dt,
                      history_samples=self.history_recorded.sum(1),
                      enter_history_samples=enter_samples, exit_history_samples=exit_samples,
                      consecutive_rough_seconds=rough_seconds, consecutive_clear_seconds=clear_seconds,
                      unknown_seconds=self._unknown_run.float() * self.dt, history_expired=expired)
        return result
