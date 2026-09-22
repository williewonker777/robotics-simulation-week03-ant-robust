"""Separate v7 tasks: unchanged lanes, spatial terrain input and four foot centers."""

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from week03_ant.depth_math import encode_height_scan
from week03_ant.footmap_math import FOOT_NAMES, TIP_OFFSETS, SCAN_RAYS, feet_in_yaw_frame
from .depth_v6_cfg import DepthLaneSceneCfg
from .rough_v5_cfg import (
    FLAT_PLANE_HEIGHT, RoughLaneObservationCfg, RoughLaneTrainObservationCfg,
    RoughLaneRecoveryAntEnvCfg, RoughLaneEvalAntEnvCfg, RoughLaneDemoAntEnvCfg,
)


def spatial_terrain_depth(env, mode="actual", noise=0.0):
    if mode not in ("actual", "zero", "shuffle"):
        raise ValueError(f"unknown scan mode: {mode}")
    sensor = env.scene["height_scanner"]
    sensor.update(0.0, force_recompute=True)
    values = encode_height_scan(
        env.scene["robot"].data.root_pos_w[:, 2], sensor.data.ray_hits_w[..., 2],
        ray_offset_z=sensor.cfg.offset.pos[2], max_distance=sensor.cfg.max_distance,
        plane_height=FLAT_PLANE_HEIGHT,
    )
    if values.shape[-1] != 2 * SCAN_RAYS:
        raise RuntimeError("v7 requires a 17x13 scan")
    if noise:
        values[:, :SCAN_RAYS] = (values[:, :SCAN_RAYS] + torch.empty_like(values[:, :SCAN_RAYS])
                                .uniform_(-noise, noise) * values[:, SCAN_RAYS:]).clamp(-1, 1)
    if mode == "zero":
        return torch.zeros_like(values)
    if mode == "shuffle":
        if env.num_envs < 2:
            raise ValueError("shuffle requires multiple environments")
        return torch.roll(values, shifts=max(1, env.num_envs // 7), dims=0)
    return values


class distal_foot_positions(ManagerTermBase):
    """Kinematic distal centers, not ankle origins, terrain labels or ground truth footholds."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.ids, names = self.robot.find_bodies(list(FOOT_NAMES), preserve_order=True)
        if names != list(FOOT_NAMES):
            raise ValueError(f"unexpected feet: {names}")
        self.offsets = torch.tensor(TIP_OFFSETS, device=env.device).expand(env.num_envs, -1, -1)

    def __call__(self, env):
        data = self.robot.data
        tips = data.body_link_pos_w[:, self.ids] + quat_apply(data.body_link_quat_w[:, self.ids], self.offsets)
        return feet_in_yaw_frame(data.root_pos_w, data.root_quat_w, tips).flatten(1)


@configclass
class FootMapSceneCfg(DepthLaneSceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(3.2, 2.4), ordering="xy"),
        mesh_prim_paths=["/World/ground"], max_distance=4.0, update_period=0.0, debug_vis=False,
    )


@configclass
class FootMapObservationCfg(RoughLaneObservationCfg):
    @configclass
    class PolicyCfg(RoughLaneObservationCfg.PolicyCfg):
        terrain_depth = ObsTerm(func=spatial_terrain_depth, params={"noise": 0.0})
        foot_positions = ObsTerm(func=distal_foot_positions)

    policy: PolicyCfg = PolicyCfg()


@configclass
class FootMapTrainObservationCfg(RoughLaneTrainObservationCfg):
    @configclass
    class PolicyCfg(RoughLaneTrainObservationCfg.PolicyCfg):
        terrain_depth = ObsTerm(func=spatial_terrain_depth, params={"noise": 0.02})
        foot_positions = ObsTerm(func=distal_foot_positions)

    policy: PolicyCfg = PolicyCfg()


@configclass
class FootMapTrainAntEnvCfg(RoughLaneRecoveryAntEnvCfg):
    scene: FootMapSceneCfg = FootMapSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: FootMapTrainObservationCfg = FootMapTrainObservationCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match the frozen v6 run's explicit training overrides in every arm.
        self.events.lane_layout.params["family_weights"]["stepping_stones"] = 4.0
        self.rewards.stall.weight = -2.0
        self.rewards.stone_body_height.weight = -3.0
        self.rewards.stone_body_height.params["target_height"] = 0.55
        self.terminations.lane_departure.params["margin"] = 1.1


@configclass
class FootMapEvalAntEnvCfg(RoughLaneEvalAntEnvCfg):
    scene: FootMapSceneCfg = FootMapSceneCfg(num_envs=175, env_spacing=5.0, clone_in_fabric=False)
    observations: FootMapObservationCfg = FootMapObservationCfg()


@configclass
class FootMapDemoAntEnvCfg(RoughLaneDemoAntEnvCfg):
    scene: FootMapSceneCfg = FootMapSceneCfg(num_envs=7, env_spacing=5.0, clone_in_fabric=False)
    observations: FootMapObservationCfg = FootMapObservationCfg()
