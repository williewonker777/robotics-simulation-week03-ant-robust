"""Passive, pre-action first-episode posture sums; no policy feedback."""

import torch

GROUPS = ("all_valid", "clear", "rough", "intermediate")
SUM_FIELDS = ("body_clearance", "target_height", "absolute_body_error", "forward_speed",
              "body_cost", "foot_cost", "flat_speed_bonus", "reward", "local_coverage", "front_coverage")


class PostureTelemetry:
    def __init__(self, num_envs, device):
        self.active_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.motion_steps = torch.zeros_like(self.active_steps)
        self.low_speed_steps = torch.zeros_like(self.active_steps)
        self.groups = {}
        for group in GROUPS:
            values = {key + "_sum": torch.zeros(num_envs, dtype=torch.float64, device=device)
                      for key in (*SUM_FIELDS, "foot_clearance", "foot_deficit")}
            values.update({key: torch.zeros_like(self.active_steps)
                           for key in ("steps", "low_speed_steps", "foot_samples")})
            self.groups[group] = values

    @torch.inference_mode()
    def before_step(self, active, terms):
        if active.shape != self.active_steps.shape or active.dtype != torch.bool:
            raise ValueError("expected a boolean first-episode active mask")
        self.active_steps += active.long()
        self.motion_steps += (active & terms["motion_valid"]).long()
        self.low_speed_steps += (active & terms["motion_valid"] & (terms["forward_speed"] < 1.)).long()
        valid = active & terms["valid"]
        clear, rough = terms["severity"] <= .1, terms["severity"] >= .9
        selectors = {"all_valid": valid, "clear": valid & clear, "rough": valid & rough,
                     "intermediate": valid & ~clear & ~rough}
        values = {key: terms[key] for key in SUM_FIELDS if key != "absolute_body_error"}
        values["absolute_body_error"] = (terms["body_clearance"] - terms["target_height"]).abs()
        for group, mask in selectors.items():
            state = self.groups[group]
            state["steps"] += mask.long()
            state["low_speed_steps"] += (mask & (terms["forward_speed"] < 1.)).long()
            for key, value in values.items():
                selected = torch.where(mask, value, 0.)
                if not torch.isfinite(selected).all():
                    raise ValueError(f"nonfinite posture telemetry: {key}")
                state[key + "_sum"] += selected.double()
            feet = mask[:, None] & terms["foot_valid"]
            state["foot_samples"] += feet.sum(-1)
            for key in ("foot_clearance", "foot_deficit"):
                selected = torch.where(feet, terms[key], 0.)
                if not torch.isfinite(selected).all():
                    raise ValueError(f"nonfinite posture telemetry: {key}")
                state[key + "_sum"] += selected.double().sum(-1)

    def to_dict(self, episode_lengths):
        if not torch.equal(self.active_steps, episode_lengths):
            raise ValueError("posture telemetry duration differs from first episodes")
        return {"pre_action_first_episode_only": True,
                "active_steps": self.active_steps.cpu().tolist(),
                "motion_steps": self.motion_steps.cpu().tolist(),
                "low_speed_steps": self.low_speed_steps.cpu().tolist(),
                "groups": {name: {key: value.cpu().tolist() for key, value in values.items()}
                           for name, values in self.groups.items()}}
