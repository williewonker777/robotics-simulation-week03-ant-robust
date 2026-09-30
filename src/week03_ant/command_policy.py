"""Opt-in 91D command access with exact preservation of the learned 88D affine."""

import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.modules import ActorCritic

from .prior_policy import PriorActorCritic, register_prior_components

COMMAND_OBSERVATIONS = 91
COMMAND_MODES = ("masked", "conditioned")
COMMAND_SCHEMA = 1
BASE_OBSERVATIONS = 88


class CommandAffine(nn.Module):
    """An augmented affine, keeping the original contiguous-prefix GEMM."""

    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.in_features = COMMAND_OBSERVATIONS
        self.out_features = original.out_features
        self.command_weight = nn.Parameter(self.weight.new_zeros(self.out_features, 3))

    def forward(self, value):
        return (F.linear(value[:, :BASE_OBSERVATIONS].contiguous(), self.weight, self.bias)
                + F.linear(value[:, BASE_OBSERVATIONS:], self.command_weight))


class CommandPriorActorCritic(PriorActorCritic):
    def __init__(self, obs, obs_groups, num_actions, command_mode="conditioned",
                 prior_mode="anchored", **kwargs):
        if command_mode not in COMMAND_MODES:
            raise ValueError("invalid command mode")
        if prior_mode != "anchored":
            raise ValueError("v14 requires anchored prior")
        if kwargs.get("noise_std_type", "scalar") != "scalar":
            raise ValueError("v14 requires scalar std")
        keys = set()
        for group in ("policy", "critic"):
            if len(obs_groups[group]) != 1:
                raise ValueError("v14 requires one flat observation group per network")
            key = obs_groups[group][0]
            if obs[key].ndim != 2 or obs[key].shape[-1] != COMMAND_OBSERVATIONS:
                raise ValueError("v14 requires 91D observations")
            keys.add(key)
        prefix = {key: obs[key][:, :BASE_OBSERVATIONS].contiguous() for key in keys}
        super().__init__(prefix, obs_groups, num_actions, prior_mode=prior_mode, **kwargs)
        self.actor[0] = CommandAffine(self.actor[0])
        self.critic[0] = CommandAffine(self.critic[0])
        self.register_buffer("command_mode_code", torch.tensor(COMMAND_MODES.index(command_mode), dtype=torch.int64))
        self.register_buffer("command_schema_code", torch.tensor(COMMAND_SCHEMA, dtype=torch.int64))
        self.command_mode = self.configured_command_mode = command_mode

    def _mode_obs(self, raw):
        if raw.ndim != 2 or raw.shape[-1] != COMMAND_OBSERVATIONS:
            raise ValueError("unexpected v14 observation shape; expected 91D")
        if self.command_mode == "conditioned":
            return raw
        masked = raw.clone()
        masked[:, BASE_OBSERVATIONS:] = 0
        return masked

    def teacher_only_obs(self, obs):
        raw = ActorCritic.get_actor_obs(self, obs)
        if raw.ndim != 2 or raw.shape[-1] != COMMAND_OBSERVATIONS:
            raise ValueError("unexpected v14 teacher observation shape")
        return raw[:, :60]

    def load_state_dict(self, state_dict, strict=True):
        if not strict:
            raise ValueError("v14 requires strict checkpoint loading")
        for key, allowed in (("command_mode_code", range(len(COMMAND_MODES))),
                             ("command_schema_code", (COMMAND_SCHEMA,))):
            value = state_dict.get(key)
            if (value is None or value.shape != torch.Size([])
                    or value.dtype != torch.int64 or int(value) not in allowed):
                raise ValueError(f"invalid checkpoint {key}")
        mode = COMMAND_MODES[int(state_dict["command_mode_code"])]
        if self.require_config_match and mode != self.configured_command_mode:
            raise ValueError("v14 training config differs from loaded command mode")
        # v14 never permits the other legacy input/prior variants, even at inference.
        if int(state_dict.get("input_mode_code", torch.tensor(-1))) != 1:
            raise ValueError("v14 requires targets checkpoint")
        if int(state_dict.get("prior_mode_code", torch.tensor(-1))) != 1:
            raise ValueError("v14 requires anchored checkpoint")
        result = super().load_state_dict(state_dict, strict=True)
        self.command_mode = mode
        return result


def warmstart_command_from_prior(policy, source_state):
    """Preserve every learned 88D tensor; append zeros and reset exploration only.

    Optimizer state is deliberately outside this helper: callers create fresh Adam.
    Source teacher identity is checked against its copied tensors; the study caller
    additionally verifies those tensors against the pinned 60D v5 artifact.
    """
    if not isinstance(policy, CommandPriorActorCritic):
        raise TypeError("expected CommandPriorActorCritic")
    state = {key: value.clone() for key, value in policy.state_dict().items()}
    added = {"command_mode_code", "command_schema_code", "actor.0.command_weight", "critic.0.command_weight"}
    if set(source_state) != set(state) - added:
        raise ValueError("warmstart requires exact legacy 88D prior state keys")
    for key, value in source_state.items():
        if value.shape != state[key].shape or value.dtype != state[key].dtype:
            raise ValueError(f"incompatible source tensor {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite source tensor {key}")
        state[key] = value.to(device=state[key].device).clone()
    state["std"].fill_(.2)
    for key in ("actor.0.command_weight", "critic.0.command_weight"):
        state[key].zero_()
    policy.load_state_dict(state)
    for key, value in policy.state_dict().items():
        if key not in added | {"std"} and not torch.equal(value.cpu(), source_state[key].cpu()):
            raise RuntimeError(f"source preservation failed: {key}")
    return {"source_observations": 88, "command_observations": 91,
            "command_mode": policy.command_mode, "std": .2, "teacher_preserved": True}


def register_command_components():
    register_prior_components()
    import rsl_rl.runners.on_policy_runner as runner_module
    runner_module.CommandPriorActorCritic = CommandPriorActorCritic
