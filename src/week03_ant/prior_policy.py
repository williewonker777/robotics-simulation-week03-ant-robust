"""Training-only frozen-v5 mean prior for the v10 controlled experiment."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .foothold_math import OBSERVATIONS
from .foothold_policy import BASE_OBSERVATIONS, FootholdActorCritic


PRIOR_MODES = ("free", "anchored")
PRIOR_COEFFICIENTS = {"free": 0.0, "anchored": 0.02}


class PriorActorCritic(FootholdActorCritic):
    """An 88D learnable actor-critic with a frozen 60D training teacher.

    The teacher is checkpoint state, but is deliberately absent from inference:
    :meth:`act_inference` remains the inherited student-only implementation.
    """

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        prior_mode="free",
        require_config_match=False,
        input_mode="targets",
        **kwargs,
    ):
        if input_mode != "targets":
            raise ValueError("v10 requires the targets input mode")
        if prior_mode not in PRIOR_MODES:
            raise ValueError(f"unknown prior mode: {prior_mode}")
        required_dims = [400, 200, 100]
        if kwargs.get("actor_hidden_dims", required_dims) != required_dims:
            raise ValueError("v10 requires actor hidden dimensions 400/200/100")
        if kwargs.get("critic_hidden_dims", required_dims) != required_dims:
            raise ValueError("v10 requires critic hidden dimensions 400/200/100")
        if kwargs.get("activation", "elu").lower() != "elu":
            raise ValueError("v10 requires ELU activation")
        kwargs["actor_hidden_dims"] = required_dims
        kwargs["critic_hidden_dims"] = required_dims
        kwargs["activation"] = "elu"
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            input_mode="targets",
            require_config_match=require_config_match,
            **kwargs,
        )

        # Reuse the exact actor layout while changing only its input width.  The
        # warm-start helper overwrites every tensor with the original v5 actor.
        self.teacher = copy.deepcopy(self.actor)
        self.teacher[0] = nn.Linear(BASE_OBSERVATIONS, self.teacher[0].out_features)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.register_buffer(
            "prior_mode_code", torch.tensor(PRIOR_MODES.index(prior_mode), dtype=torch.int64)
        )
        self.register_buffer(
            "prior_teacher_coef", torch.tensor(PRIOR_COEFFICIENTS[prior_mode], dtype=torch.float32)
        )
        self.register_buffer("prior_reference_std", torch.tensor(0.2, dtype=torch.float32))
        self.prior_mode = prior_mode
        self.configured_prior_mode = prior_mode

    def train(self, mode: bool = True):
        result = super().train(mode)
        self.teacher.eval()
        return result

    def teacher_only_obs(self, obs):
        """Return exactly the unchanged 60D prefix used by the v5 actor."""
        raw = super().get_actor_obs(obs)
        if raw.ndim != 2 or raw.shape[-1] != OBSERVATIONS:
            raise ValueError("unexpected v10 observation shape")
        return raw[:, :BASE_OBSERVATIONS]

    def teacher_mean(self, obs):
        """Compute a detached teacher mean for a training mini-batch."""
        with torch.no_grad():
            return self.teacher(self.teacher_only_obs(obs)).detach()

    def load_state_dict(self, state_dict, strict=True):
        if not strict or "prior_mode_code" not in state_dict:
            raise ValueError("v10 requires a strict, self-describing v10 checkpoint")
        code = state_dict["prior_mode_code"]
        if (
            code.shape != torch.Size([])
            or code.dtype != torch.int64
            or int(code) not in range(len(PRIOR_MODES))
        ):
            raise ValueError("invalid checkpoint prior mode")
        checkpoint_mode = PRIOR_MODES[int(code)]
        coefficient = state_dict.get("prior_teacher_coef")
        reference_std = state_dict.get("prior_reference_std")
        if (
            coefficient is None
            or coefficient.shape != torch.Size([])
            or coefficient.dtype != torch.float32
            or not torch.equal(
                coefficient.cpu(), torch.tensor(PRIOR_COEFFICIENTS[checkpoint_mode], dtype=torch.float32)
            )
        ):
            raise ValueError("invalid checkpoint teacher coefficient")
        if (
            reference_std is None
            or reference_std.shape != torch.Size([])
            or reference_std.dtype != torch.float32
            or not torch.equal(reference_std.cpu(), torch.tensor(0.2, dtype=torch.float32))
        ):
            raise ValueError("invalid checkpoint reference std")
        if self.require_config_match and checkpoint_mode != self.configured_prior_mode:
            raise ValueError("v10 training config differs from loaded prior mode")
        result = super().load_state_dict(state_dict, strict=True)
        self.prior_mode = checkpoint_mode
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        return result


def warmstart_prior_from_60d(policy, source_state, std=0.2):
    """Warm-start student/value networks and copy the v5 actor into the teacher."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be positive and finite")
    state = policy.state_dict()
    for key, destination in state.items():
        if key in (
            "input_mode_code",
            "prior_mode_code",
            "prior_teacher_coef",
            "prior_reference_std",
        ):
            continue
        if key == "std":
            destination.fill_(std)
        elif key in ("actor.0.weight", "critic.0.weight"):
            if source_state[key].shape != (destination.shape[0], BASE_OBSERVATIONS):
                raise ValueError("warmstart requires the 60D source network")
            destination.zero_()
            destination[:, :BASE_OBSERVATIONS].copy_(source_state[key])
        elif key.startswith("teacher."):
            source_key = key.replace("teacher.", "actor.", 1)
            if source_key not in source_state or destination.shape != source_state[source_key].shape:
                raise ValueError(f"incompatible source layer {source_key}")
            destination.copy_(source_state[source_key])
        else:
            if key not in source_state or destination.shape != source_state[key].shape:
                raise ValueError(f"incompatible source layer {key}")
            destination.copy_(source_state[key])
    policy.load_state_dict(state)
    for key, value in policy.teacher.state_dict().items():
        if not torch.equal(value, source_state[f"actor.{key}"]):
            raise RuntimeError(f"teacher differs from source actor tensor {key}")


def register_prior_components():
    """Expose both v10 classes where RSL-RL 3.0.1 resolves class names."""
    import rsl_rl.runners.on_policy_runner as runner_module

    from .prior_ppo import PriorPPO

    runner_module.PriorActorCritic = PriorActorCritic
    runner_module.PriorPPO = PriorPPO
