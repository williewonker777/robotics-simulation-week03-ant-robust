"""Binary depth-gated v5 teacher target; frozen PriorPPO mathematics unchanged."""

from __future__ import annotations

import torch

from .command_policy import CommandPriorActorCritic, register_command_components
from .prior_ppo import PriorPPO


class StyleCommandPriorActorCritic(CommandPriorActorCritic):
    """No persistent new tensors: v18 remains a strict 91D v16 checkpoint."""

    def __init__(self, *args, style_mode="always", **kwargs):
        if style_mode not in ("always", "gated"):
            raise ValueError("style_mode must be always or gated")
        super().__init__(*args, **kwargs)
        self.style_mode = style_mode
        self._style_loss_active = False
        self._style_gate_sum = 0
        self._style_gate_count = 0

    def teacher_mean(self, obs):
        teacher = super().teacher_mean(obs)
        if not self._style_loss_active:
            return teacher
        if "style_gate" not in obs:
            raise ValueError("v18 update requires separate style_gate observation")
        gate = obs["style_gate"]
        if (gate.shape != (teacher.shape[0], 1) or gate.device != teacher.device
                or not gate.is_floating_point() or not torch.isfinite(gate).all()
                or not torch.all((gate == 0.) | (gate == 1.))):
            raise ValueError("v18 style_gate must be finite binary float[N,1]")
        self._style_gate_sum += int(gate.detach().sum().item())
        self._style_gate_count += gate.numel()
        if self.style_mode == "always":
            return teacher
        student = self.action_mean
        if student is None or student.shape != teacher.shape or not torch.isfinite(student).all():
            raise ValueError("v18 current student mean missing or nonfinite")
        return torch.where(gate.bool(), teacher, student.detach())


class StylePriorPPO(PriorPPO):
    """Activate the teacher-target override only around the inherited PPO update."""

    def __init__(self, policy, **kwargs):
        if not isinstance(policy, StyleCommandPriorActorCritic):
            raise TypeError("StylePriorPPO requires StyleCommandPriorActorCritic")
        super().__init__(policy, **kwargs)

    def update(self):
        if self.policy._style_loss_active:
            raise RuntimeError("nested v18 PPO update")
        self.policy._style_gate_sum = 0
        self.policy._style_gate_count = 0
        self.policy._style_loss_active = True
        try:
            metrics = super().update()
        finally:
            self.policy._style_loss_active = False
        if self.policy._style_gate_count == 0:
            raise ValueError("v18 update observed no style_gate samples")
        metrics["style_teacher_mask_fraction"] = (
            self.policy._style_gate_sum / self.policy._style_gate_count)
        return metrics


def register_style_components():
    register_command_components()
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.StyleCommandPriorActorCritic = StyleCommandPriorActorCritic
    runner_module.StylePriorPPO = StylePriorPPO
