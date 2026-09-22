"""Frozen v5 actor plus bounded mean correction; not a stability guarantee."""

import copy
import math

import torch
from torch import nn

from .footmap_policy import FootMapActorCritic, INPUT_MODES


class BoundedResidualActor(nn.Module):
    def __init__(self, residual, limit):
        super().__init__()
        self.base = copy.deepcopy(residual)
        self.base[0] = nn.Linear(60, self.base[0].out_features)
        self.base.requires_grad_(False)
        self.residual = residual
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer("residual_limit", torch.tensor(limit, dtype=torch.float32))

    def forward(self, obs):
        with torch.no_grad():
            base = self.base(obs[:, :60])
        return base + self.residual_limit * torch.tanh(self.residual(obs))


class ResidualActorCritic(FootMapActorCritic):
    """Reuse v7 input encoding and PPO distribution; only the actor is replaced.

    PPO samples a Gaussian around the SUM, and stores/log-probs those same total
    actions. The 0.5 default bounds the deterministic mean correction, not the
    exploration samples or the deviation between two closed-loop trajectories.
    """

    def __init__(self, obs, obs_groups, num_actions, residual_limit=0.5,
                 require_config_match=False, **kwargs):
        if not math.isfinite(residual_limit) or residual_limit < 0:
            raise ValueError("residual limit must be finite and nonnegative")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.actor = BoundedResidualActor(self.actor, residual_limit)
        self.require_config_match = require_config_match
        self.configured_mode = self.input_mode
        self.configured_limit = float(self.actor.residual_limit)

    def load_state_dict(self, state_dict, strict=True):
        limit = state_dict.get("actor.residual_limit")
        if (limit is None or limit.shape != torch.Size([]) or limit.dtype != torch.float32
                or not torch.isfinite(limit) or float(limit) < 0):
            raise ValueError("missing or invalid checkpoint residual limit")
        if self.require_config_match:
            code = state_dict.get("input_mode_code")
            if (code is None or code.shape != torch.Size([]) or code.dtype != torch.int64
                    or int(code) != INPUT_MODES.index(self.configured_mode)
                    or float(limit) != self.configured_limit):
                raise ValueError("v8 training config differs from loaded mode/limit")
        result = super().load_state_dict(state_dict, strict=strict)
        self.actor.base.requires_grad_(False)
        return result


def register_residual_policy():
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.ResidualActorCritic = ResidualActorCritic


def warmstart_residual_from_60d(policy, source_state, std=0.2):
    """Preserve base exactly, zero residual output, warm-start expanded critic."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be positive and finite")
    state = policy.state_dict()
    for key in state:
        if key.startswith("actor.base."):
            source_key = key.replace("actor.base.", "actor.", 1)
            if state[key].shape != source_state[source_key].shape:
                raise ValueError(f"incompatible source layer {source_key}")
            state[key].copy_(source_state[source_key])
        elif key == "critic.0.weight":
            if source_state[key].shape != (state[key].shape[0], 60):
                raise ValueError("warmstart requires a 60D critic")
            state[key].zero_()
            state[key][:, :60].copy_(source_state[key])
        elif key.startswith("critic."):
            if state[key].shape != source_state[key].shape:
                raise ValueError(f"incompatible source layer {key}")
            state[key].copy_(source_state[key])
        elif key == "std":
            state[key].fill_(std)
    policy.load_state_dict(state)
    nn.init.zeros_(policy.actor.residual[-1].weight)
    nn.init.zeros_(policy.actor.residual[-1].bias)
