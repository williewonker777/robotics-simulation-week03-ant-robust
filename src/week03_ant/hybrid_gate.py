"""Depth-only routing of frozen policies; crossfade is not a stability guarantee."""

from dataclasses import dataclass
import math

import torch

from week03_ant.foothold_math import GRID_HEIGHT, GRID_RAYS, GRID_WIDTH, foothold_grid


@dataclass(frozen=True)
class GateConfig:
    span_enter: float
    edge_enter: float
    exit_ratio: float = .6
    min_coverage: float = .9
    enter_steps: int = 2
    min_dwell_seconds: float = .5
    clear_seconds: float = .3
    blend_seconds: float = .15

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.enter_steps, bool) or not isinstance(self.enter_steps, int):
            raise ValueError("enter_steps must be a positive integer")
        if self.exit_ratio >= 1 or self.min_coverage > 1:
            raise ValueError("exit_ratio must be <1 and min_coverage <=1")


PRESETS = {
    "cautious": GateConfig(.12, .08),
    "balanced": GateConfig(.18, .12),
    "selective": GateConfig(.27, .18),
}


class DepthPolicyGate:
    """Per-environment hysteresis, continuous clearance, dwell, and action blending.

    Heights use the existing xy-indexed 25x33 vertical-ray grid. Caller must mark
    clipped rays invalid. Invalid/nonfinite heights never contribute to features.
    Unknown coverage holds the target (an already-started fade still completes).
    reset() is for episode resets, not terrain-portal wraps. step() advances dt.
    Both state-mutating methods own their inference-mode boundary, so callers
    may freely mix inference-wrapped policy steps and ordinary reset calls.
    """

    def __init__(self, num_envs, device, dt, config):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(config, GateConfig):
            raise ValueError("config must be GateConfig")
        self.num_envs, self.device, self.dt, self.config = num_envs, torch.device(device), dt, config
        self.target_v10 = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.alpha = torch.zeros(num_envs, device=device)
        self._age = torch.full((num_envs,), torch.inf, device=device)
        self._clear = torch.zeros(num_envs, device=device)
        self._enter_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        xy = foothold_grid(device=device)
        x, y = xy.unbind(-1)
        tolerance = 1e-6
        self._enter_roi = (x >= -tolerance) & (x <= 2 + tolerance) & (y.abs() <= 1 + tolerance)
        self._retain_roi = (x >= -1.2 - tolerance) & (x <= 2 + tolerance) & (y.abs() <= 1.1 + tolerance)

    @torch.inference_mode()
    def reset(self, done_mask=None):
        if done_mask is None:
            done_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if done_mask.shape != (self.num_envs,) or done_mask.dtype != torch.bool or done_mask.device != self.alpha.device:
            raise ValueError("done_mask must be bool[N] on gate device")
        self.target_v10[done_mask] = False
        self.alpha[done_mask] = 0
        self._age[done_mask] = torch.inf
        self._clear[done_mask] = 0
        self._enter_count[done_mask] = 0

    @staticmethod
    def _features(height, valid, roi):
        mask = valid & roi
        coverage = mask.sum((1, 2)).to(height.dtype) / roi.sum()
        samples = height.flatten(1).masked_fill(~mask.flatten(1), torch.nan)
        quantiles = torch.nanquantile(samples, height.new_tensor([.05, .95]), dim=1)
        span = quantiles[1] - quantiles[0]
        # Both adjacent endpoints must be observed and inside the same ROI.
        dx = (height[:, :, 1:] - height[:, :, :-1]).abs()
        dy = (height[:, 1:, :] - height[:, :-1, :]).abs()
        mx = mask[:, :, 1:] & mask[:, :, :-1]
        my = mask[:, 1:, :] & mask[:, :-1, :]
        possible_edges = (roi[:, 1:] & roi[:, :-1]).sum() + (roi[1:, :] & roi[:-1, :]).sum()
        edge_coverage = (mx.sum((1, 2)) + my.sum((1, 2))).to(height.dtype) / possible_edges
        dx = dx.masked_fill(~mx, torch.nan)
        dy = dy.masked_fill(~my, torch.nan)
        edge = torch.nanquantile(torch.cat((dx.flatten(1), dy.flatten(1)), dim=1), .95, dim=1)
        # NaN deliberately retained when no observations/neighbors exist.
        return span, edge, coverage, edge_coverage

    @torch.inference_mode()
    def step(self, height, valid, v5_actions, v10_actions, mode="hybrid"):
        if mode not in ("hybrid", "v5", "v10"):
            raise ValueError("mode must be hybrid, v5, or v10")
        if height.shape != (self.num_envs, GRID_RAYS) or height.dtype not in (torch.float32, torch.float64):
            raise ValueError("height must be float32/float64[N,825]")
        if valid.shape != height.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be bool[N,825]")
        tensors = (height, valid, v5_actions, v10_actions)
        if any(t.device != self.alpha.device for t in tensors):
            raise ValueError("all inputs must be on gate device")
        if any(a.shape != (self.num_envs, 8) or not a.is_floating_point() for a in (v5_actions, v10_actions)):
            raise ValueError("actions must be floating[N,8]")
        if v5_actions.dtype != v10_actions.dtype or not all(torch.isfinite(a).all().item() for a in (v5_actions, v10_actions)):
            raise ValueError("action dtypes must match and actions must be finite")
        observed = (valid & torch.isfinite(height)).reshape(-1, GRID_HEIGHT, GRID_WIDTH)
        grid = height.reshape_as(observed)
        es, ee, ec, eec = self._features(grid, observed, self._enter_roi)
        rs, re, rc, rec = self._features(grid, observed, self._retain_roi)
        cfg = self.config
        enter_known = (ec >= cfg.min_coverage) & (eec >= cfg.min_coverage) & torch.isfinite(es) & torch.isfinite(ee)
        retain_known = (rc >= cfg.min_coverage) & (rec >= cfg.min_coverage) & torch.isfinite(rs) & torch.isfinite(re)
        old_target = self.target_v10.clone()
        coverage = torch.where(old_target, rc, ec)
        uncertain = ~torch.where(old_target, retain_known, enter_known)
        switched = torch.zeros_like(old_target)
        if mode == "hybrid":
            self._age += self.dt
            clear = retain_known & (rs <= cfg.exit_ratio * cfg.span_enter) & (re <= cfg.exit_ratio * cfg.edge_enter)
            self._clear = torch.where(old_target & clear, self._clear + self.dt, 0.)
            dwell_ready = self._age + 1e-7 >= cfg.min_dwell_seconds
            enter_candidate = ~old_target & enter_known & ((es >= cfg.span_enter) | (ee >= cfg.edge_enter))
            self._enter_count = torch.where(enter_candidate, self._enter_count + 1, 0)
            enter = (self._enter_count >= cfg.enter_steps) & dwell_ready
            leave = old_target & (self._clear + 1e-7 >= cfg.clear_seconds) & dwell_ready
            self.target_v10 = (old_target | enter) & ~leave
            switched = self.target_v10 != old_target
            self._age[switched] = 0
            self._clear[switched] = 0
            self._enter_count[switched] = 0
            delta = self.dt / cfg.blend_seconds
            self.alpha = (self.alpha + torch.where(self.target_v10, delta, -delta)).clamp(0, 1)
            alpha = self.alpha.to(v5_actions.dtype)[:, None]
            blended = (1 - alpha) * v5_actions + alpha * v10_actions
            actions = torch.where(alpha == 0, v5_actions, torch.where(alpha == 1, v10_actions, blended))
        else:
            self.target_v10.fill_(mode == "v10")
            self.alpha.fill_(float(mode == "v10"))
            self._age.fill_(torch.inf)
            self._clear.zero_()
            self._enter_count.zero_()
            actions = v5_actions if mode == "v5" else v10_actions
        return {"actions": actions, "alpha": self.alpha.clone(), "target_v10": self.target_v10.clone(),
                "switched": switched, "uncertain": uncertain, "coverage": coverage,
                "enter_span": es, "enter_edge": ee, "retain_span": rs, "retain_edge": re,
                "enter_coverage": ec, "retain_coverage": rc,
                "enter_edge_coverage": eec, "retain_edge_coverage": rec}
