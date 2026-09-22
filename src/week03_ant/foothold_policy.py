"""Flat foothold-feature policy for the v9 controlled experiment."""

import math

import torch

from rsl_rl.modules import ActorCritic

from .foothold_math import OBSERVATIONS


INPUT_MODES = ("feet", "targets", "guided")
BASE_OBSERVATIONS = 60
FOOT_OBSERVATIONS = 12
TARGET_OBSERVATIONS = 16


class FootholdActorCritic(ActorCritic):
    """An 88D MLP whose observation ablation is stored in its checkpoint.

    ``feet`` masks the 16D target proposal while retaining the four 3D distal
    centers. ``targets`` and ``guided`` both consume all inputs; their
    distinction belongs to the training contract (the guided reward), not to a
    different policy architecture.
    """

    def __init__(self, obs, obs_groups, num_actions, input_mode="guided",
                 require_config_match=False, **kwargs):
        if input_mode not in INPUT_MODES:
            raise ValueError(f"unknown input mode: {input_mode}")
        if num_actions != 8:
            raise ValueError("v9 is specific to the eight-effort Ant")
        for group in ("policy", "critic"):
            if any(obs[key].ndim != 2 for key in obs_groups[group]):
                raise ValueError("v9 observations must be flat")
            if sum(obs[key].shape[-1] for key in obs_groups[group]) != OBSERVATIONS:
                raise ValueError(f"v9 requires {OBSERVATIONS}D {group} observations")
        if kwargs.get("actor_obs_normalization") or kwargs.get("critic_obs_normalization"):
            raise ValueError("v9 checkpoint contract disables observation normalization")
        kwargs.setdefault("actor_hidden_dims", [400, 200, 100])
        kwargs.setdefault("critic_hidden_dims", [400, 200, 100])
        kwargs.setdefault("activation", "elu")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.register_buffer("input_mode_code", torch.tensor(INPUT_MODES.index(input_mode), dtype=torch.int64))
        self.input_mode = input_mode
        self.configured_mode = input_mode
        self.require_config_match = require_config_match

    def _mode_obs(self, raw):
        if raw.ndim != 2 or raw.shape[-1] != OBSERVATIONS:
            raise ValueError("unexpected v9 observation shape")
        if self.input_mode != "feet":
            return raw
        # Assignment masks NaN target hints as well as finite values.
        masked = raw.clone()
        masked[:, BASE_OBSERVATIONS + FOOT_OBSERVATIONS:] = 0
        return masked

    def get_actor_obs(self, obs):
        return self._mode_obs(super().get_actor_obs(obs))

    def get_critic_obs(self, obs):
        return self._mode_obs(super().get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        if not strict or "input_mode_code" not in state_dict:
            raise ValueError("v9 requires a strict, self-describing v9 checkpoint")
        code = state_dict["input_mode_code"]
        if (code.shape != torch.Size([]) or code.dtype != torch.int64
                or int(code) not in range(len(INPUT_MODES))):
            raise ValueError("invalid checkpoint input mode")
        checkpoint_mode = INPUT_MODES[int(code)]
        if self.require_config_match and checkpoint_mode != self.configured_mode:
            raise ValueError("v9 training config differs from loaded input mode")
        result = super().load_state_dict(state_dict, strict=True)
        self.input_mode = checkpoint_mode
        return result


def register_foothold_policy():
    """Expose the class where RSL-RL 3.0.1 resolves ``class_name``."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.FootholdActorCritic = FootholdActorCritic


def warmstart_from_60d(policy, source_state, std=0.2):
    """Copy the original MLP, zeroing only the 28 newly added columns."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be positive and finite")
    state = policy.state_dict()
    for key in state:
        if key == "input_mode_code":
            continue
        if key == "std":
            state[key].fill_(std)
        elif key in ("actor.0.weight", "critic.0.weight"):
            if source_state[key].shape != (state[key].shape[0], BASE_OBSERVATIONS):
                raise ValueError("warmstart requires the 60D source network")
            state[key].zero_()
            state[key][:, :BASE_OBSERVATIONS].copy_(source_state[key])
        else:
            if key not in source_state or state[key].shape != source_state[key].shape:
                raise ValueError(f"incompatible source layer {key}")
            state[key].copy_(source_state[key])
    policy.load_state_dict(state)
