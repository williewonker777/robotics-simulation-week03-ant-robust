"""Small spatial policy for RSL-RL 3.0.1; old task policies remain unchanged."""

import torch
from torch import nn
from rsl_rl.modules import ActorCritic

from .footmap_math import SCAN_HEIGHT, SCAN_WIDTH, SCAN_RAYS, FOOTMAP_OBSERVATIONS, foot_position_maps

INPUT_MODES = ("blind", "height", "footmap")
EMBEDDING_SIZE = 32


def terrain_encoder():
    return nn.Sequential(
        nn.Conv2d(6, 8, 3, stride=2, padding=1), nn.ELU(),
        nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ELU(), nn.Flatten(),
        nn.Linear(16 * 4 * 5, EMBEDDING_SIZE), nn.ELU(),
    )


class FootMapActorCritic(ActorCritic):
    """Same trainable architecture across three input ablations.

    The mode is checkpoint state, not an easily forgotten evaluation flag.
    Actor/critic encoders are separate. No history, residual action, or teacher
    routing is introduced in this first controlled spatial-encoding experiment.
    """

    def __init__(self, obs, obs_groups, num_actions, input_mode="footmap", **kwargs):
        if input_mode not in INPUT_MODES:
            raise ValueError(f"unknown input mode: {input_mode}")
        if num_actions != 8:
            raise ValueError("v7 is specific to the eight-effort Ant")
        for group in ("policy", "critic"):
            if any(obs[key].ndim != 2 for key in obs_groups[group]):
                raise ValueError("v7 observations must be flat")
            if sum(obs[key].shape[-1] for key in obs_groups[group]) != FOOTMAP_OBSERVATIONS:
                raise ValueError(f"v7 requires {FOOTMAP_OBSERVATIONS}D {group} observations")
        if kwargs.get("actor_obs_normalization") or kwargs.get("critic_obs_normalization"):
            raise ValueError("v7 checkpoint contract disables observation normalization")
        kwargs.setdefault("actor_hidden_dims", [400, 200, 100])
        kwargs.setdefault("critic_hidden_dims", [400, 200, 100])
        kwargs.setdefault("activation", "elu")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.actor[0] = nn.Linear(60 + EMBEDDING_SIZE, self.actor[0].out_features)
        self.critic[0] = nn.Linear(60 + EMBEDDING_SIZE, self.critic[0].out_features)
        self.actor_encoder = terrain_encoder()
        self.critic_encoder = terrain_encoder()
        self.register_buffer("input_mode_code", torch.tensor(INPUT_MODES.index(input_mode), dtype=torch.int64))
        self.input_mode = input_mode

    def spatial_channels(self, raw):
        if raw.ndim != 2 or raw.shape[-1] != FOOTMAP_OBSERVATIONS:
            raise ValueError("unexpected v7 observation shape")
        channels = raw.new_zeros((raw.shape[0], 6, SCAN_HEIGHT, SCAN_WIDTH))
        if self.input_mode != "blind":
            height = raw[:, 60:60 + SCAN_RAYS]
            valid = raw[:, 60 + SCAN_RAYS:60 + 2 * SCAN_RAYS]
            valid = torch.isfinite(height) & torch.isfinite(valid) & (valid > 0.5)
            height = torch.where(valid, height.clamp(-1, 1), torch.ones_like(height))
            channels[:, 0] = height.reshape(-1, SCAN_HEIGHT, SCAN_WIDTH)
            channels[:, 1] = valid.reshape(-1, SCAN_HEIGHT, SCAN_WIDTH)
        if self.input_mode == "footmap":
            feet = raw[:, 60 + 2 * SCAN_RAYS:].reshape(-1, 4, 3)
            channels[:, 2:] = foot_position_maps(feet[..., :2])
        return channels

    def get_actor_obs(self, obs):
        raw = super().get_actor_obs(obs)
        return torch.cat((raw[:, :60], self.actor_encoder(self.spatial_channels(raw))), dim=-1)

    def get_critic_obs(self, obs):
        raw = super().get_critic_obs(obs)
        return torch.cat((raw[:, :60], self.critic_encoder(self.spatial_channels(raw))), dim=-1)

    def load_state_dict(self, state_dict, strict=True):
        if not strict or "input_mode_code" not in state_dict:
            raise ValueError("v7 requires a strict, self-describing v7 checkpoint")
        code = state_dict["input_mode_code"]
        if code.shape != torch.Size([]) or code.dtype != torch.int64 or int(code) not in range(len(INPUT_MODES)):
            raise ValueError("invalid checkpoint input mode")
        result = super().load_state_dict(state_dict, strict=True)
        self.input_mode = INPUT_MODES[int(code)]
        return result


def register_footmap_policy():
    """RSL-RL3.0.1 resolves class_name in its runner module, not a plugin registry."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.FootMapActorCritic = FootMapActorCritic


def warmstart_from_60d(policy, source_state, std=0.2):
    """Copy the entire original MLP and zero only new feature columns."""
    if not torch.isfinite(torch.tensor(std)) or std <= 0:
        raise ValueError("std must be positive and finite")
    state = policy.state_dict()
    for key in state:
        if key.startswith(("actor_encoder.", "critic_encoder.")) or key == "input_mode_code":
            continue
        if key == "std":
            state[key].fill_(std)
        elif key in ("actor.0.weight", "critic.0.weight"):
            if source_state[key].shape != (state[key].shape[0], 60):
                raise ValueError("warmstart requires the 60D source network")
            state[key].zero_()
            state[key][:, :60].copy_(source_state[key])
        else:
            if state[key].shape != source_state[key].shape:
                raise ValueError(f"incompatible source layer {key}")
            state[key].copy_(source_state[key])
    policy.load_state_dict(state)
