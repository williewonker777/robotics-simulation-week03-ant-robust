"""Isolated scan-derived endpoint-hint tasks; physical benchmark is unchanged."""

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from week03_ant.depth_math import encode_height_scan
from week03_ant.foothold_math import GRID_RAYS, select_footholds, foothold_features, endpoint_support_cost
from .footmap_v7_cfg import (
    FootMapSceneCfg, FootMapTrainAntEnvCfg, FootMapEvalAntEnvCfg, FootMapDemoAntEnvCfg,
    distal_foot_positions,
)
from .rough_v5_cfg import (
    FLAT_PLANE_HEIGHT, RoughLaneObservationCfg, RoughLaneTrainObservationCfg, RoughLaneRecoveryRewardsCfg,
)
from .lanes import get_lane_state


class FootholdHints(ManagerTermBase):
    """Stateless and recomputed after resets/portal wraps; no hidden terrain map."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.feet = distal_foot_positions(cfg, env)

    def geometry(self, env):
        sensor = env.scene["height_scanner"]
        sensor.update(0.0, force_recompute=True)
        encoded = encode_height_scan(
            env.scene["robot"].data.root_pos_w[:, 2], sensor.data.ray_hits_w[..., 2],
            ray_offset_z=sensor.cfg.offset.pos[2], max_distance=sensor.cfg.max_distance,
            plane_height=FLAT_PLANE_HEIGHT,
        )
        if encoded.shape[-1] != 2 * GRID_RAYS:
            raise ValueError("v9 requires 33x25=825 scan rays")
        height = encoded[:, :GRID_RAYS]
        # Clipped range cannot support a metric-height target; abstain there.
        valid = (encoded[:, GRID_RAYS:] > .5) & (height.abs() < .99)
        surface_z = -height - .5
        feet = self.feet(env).reshape(-1, 4, 3)
        proposal, support, available, plateau = select_footholds(surface_z, valid, feet)
        return feet, proposal, support, available, plateau

    def __call__(self, env):
        feet, proposal, _, available, _ = self.geometry(env)
        return foothold_features(feet, proposal, available)


class EndpointSupportPenalty(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hints = FootholdHints(cfg, env)

    def __call__(self, env):
        feet, _, support, available, _ = self.hints.geometry(env)
        state = get_lane_state(env)
        family, _ = state.family_level()
        # Privileged family selection is training-only shaping. It is never used
        # by the proposal generator, observations, actor, or evaluation gate.
        on_stones = family == state.family_names.index("stepping_stones")
        return endpoint_support_cost(feet, support, available) * on_stones


@configclass
class FootholdSceneCfg(FootMapSceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso", offset=RayCasterCfg.OffsetCfg(pos=(.4, 0., 2.)),
        ray_alignment="yaw", pattern_cfg=patterns.GridPatternCfg(resolution=.1, size=(3.2, 2.4), ordering="xy"),
        mesh_prim_paths=["/World/ground"], max_distance=4., update_period=0., debug_vis=False,
    )


@configclass
class FootholdObservationCfg(RoughLaneObservationCfg):
    @configclass
    class PolicyCfg(RoughLaneObservationCfg.PolicyCfg):
        foot_positions = ObsTerm(func=distal_foot_positions)
        foothold_hints = ObsTerm(func=FootholdHints)

    policy: PolicyCfg = PolicyCfg()


@configclass
class FootholdTrainObservationCfg(RoughLaneTrainObservationCfg):
    @configclass
    class PolicyCfg(RoughLaneTrainObservationCfg.PolicyCfg):
        foot_positions = ObsTerm(func=distal_foot_positions)
        foothold_hints = ObsTerm(func=FootholdHints)

    policy: PolicyCfg = PolicyCfg()


@configclass
class FootholdRewardsCfg(RoughLaneRecoveryRewardsCfg):
    foothold_support = RewTerm(func=EndpointSupportPenalty, weight=0.)


@configclass
class FootholdTrainAntEnvCfg(FootMapTrainAntEnvCfg):
    scene: FootholdSceneCfg = FootholdSceneCfg(num_envs=4096, env_spacing=5., clone_in_fabric=False)
    observations: FootholdTrainObservationCfg = FootholdTrainObservationCfg()
    rewards: FootholdRewardsCfg = FootholdRewardsCfg()


@configclass
class FootholdEvalAntEnvCfg(FootMapEvalAntEnvCfg):
    scene: FootholdSceneCfg = FootholdSceneCfg(num_envs=175, env_spacing=5., clone_in_fabric=False)
    observations: FootholdObservationCfg = FootholdObservationCfg()


@configclass
class FootholdDemoAntEnvCfg(FootMapDemoAntEnvCfg):
    scene: FootholdSceneCfg = FootholdSceneCfg(num_envs=7, env_spacing=5., clone_in_fabric=False)
    observations: FootholdObservationCfg = FootholdObservationCfg()
