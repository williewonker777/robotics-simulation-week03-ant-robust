"""Depth-aware v6 lanes, preserving every v5 benchmark surface and fall rule.

The sensor is an idealized yaw-stabilized terrain-height scan, not an RGB-D
renderer. Its local depth/validity features extend the 60D policy to 346D.
"""

import torch

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from week03_ant.depth_math import encode_height_scan
from .rough_v5_cfg import (
    FLAT_PLANE_HEIGHT,
    RoughLaneDemoAntEnvCfg,
    RoughLaneEvalAntEnvCfg,
    RoughLaneObservationCfg,
    RoughLaneRecoveryAntEnvCfg,
    RoughLaneSceneCfg,
    RoughLaneTrainObservationCfg,
)

DEPTH_RAYS = 143  # 13 longitudinal x 11 lateral rays
DEPTH_FEATURES = 2 * DEPTH_RAYS
DEPTH_OBSERVATIONS = 60 + DEPTH_FEATURES


def terrain_depth(env, mode: str = "actual", noise: float = 0.0) -> torch.Tensor:
    """Append finite local heights and validity, including the physical flat plane."""
    if mode not in ("actual", "zero", "shuffle"):
        raise ValueError(f"unknown depth ablation mode: {mode}")
    sensor = env.scene["height_scanner"]
    # Interval lane-wrap events teleport after scene.update. Recompute here, not
    # from a possibly cached pre-wrap sensor frame. Resets use the same path.
    sensor.update(0.0, force_recompute=True)
    data = sensor.data
    features = encode_height_scan(
        env.scene["robot"].data.root_pos_w[:, 2], data.ray_hits_w[..., 2],
        ray_offset_z=sensor.cfg.offset.pos[2], max_distance=sensor.cfg.max_distance,
        plane_height=FLAT_PLANE_HEIGHT,
    )
    if features.shape[1] != DEPTH_FEATURES:
        raise RuntimeError(f"expected {DEPTH_FEATURES} depth features, got {features.shape[1]}")
    if noise:
        features[:, :DEPTH_RAYS] = (
            features[:, :DEPTH_RAYS]
            + torch.empty_like(features[:, :DEPTH_RAYS]).uniform_(-noise, noise)
            * features[:, DEPTH_RAYS:]
        ).clamp(-1.0, 1.0)
    if mode == "zero":
        return torch.zeros_like(features)
    if mode == "shuffle":
        # Rotate across complete env blocks so each family receives another
        # family's scan in the balanced benchmark; no proprioception is moved.
        if env.num_envs < 2:
            raise ValueError("shuffle ablation requires at least two environments")
        return torch.roll(features, shifts=max(1, env.num_envs // 7), dims=0)
    return features


@configclass
class DepthLaneSceneCfg(RoughLaneSceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.8, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(2.4, 2.0), ordering="xy"),
        mesh_prim_paths=["/World/ground"],
        max_distance=4.0,
        update_period=0.0,
        debug_vis=False,
    )


@configclass
class DepthLaneObservationCfg(RoughLaneObservationCfg):
    @configclass
    class PolicyCfg(RoughLaneObservationCfg.PolicyCfg):
        terrain_depth = ObsTerm(func=terrain_depth, params={"mode": "actual", "noise": 0.0})

    policy: PolicyCfg = PolicyCfg()


@configclass
class DepthLaneTrainObservationCfg(RoughLaneTrainObservationCfg):
    @configclass
    class PolicyCfg(RoughLaneTrainObservationCfg.PolicyCfg):
        terrain_depth = ObsTerm(func=terrain_depth, params={"mode": "actual", "noise": 0.02})

    policy: PolicyCfg = PolicyCfg()


@configclass
class DepthLaneTrainAntEnvCfg(RoughLaneRecoveryAntEnvCfg):
    scene: DepthLaneSceneCfg = DepthLaneSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: DepthLaneTrainObservationCfg = DepthLaneTrainObservationCfg()


@configclass
class DepthLaneEvalAntEnvCfg(RoughLaneEvalAntEnvCfg):
    scene: DepthLaneSceneCfg = DepthLaneSceneCfg(num_envs=175, env_spacing=5.0, clone_in_fabric=False)
    observations: DepthLaneObservationCfg = DepthLaneObservationCfg()


@configclass
class DepthLaneDemoAntEnvCfg(RoughLaneDemoAntEnvCfg):
    scene: DepthLaneSceneCfg = DepthLaneSceneCfg(num_envs=7, env_spacing=5.0, clone_in_fabric=False)
    observations: DepthLaneObservationCfg = DepthLaneObservationCfg()
