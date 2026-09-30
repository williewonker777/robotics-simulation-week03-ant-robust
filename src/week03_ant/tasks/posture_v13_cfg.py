"""Additive v13 reward; inherit all frozen v10 physical/observation settings."""

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from week03_ant.depth_math import encode_height_scan
from week03_ant.foothold_math import GRID_RAYS
from week03_ant.footmap_math import FOOT_NAMES, TIP_OFFSETS, feet_in_yaw_frame
from week03_ant.posture_math import terrain_posture_terms
from .foothold_v9_cfg import FootholdRewardsCfg, FootholdTrainAntEnvCfg
from .rough_v5_cfg import FLAT_PLANE_HEIGHT
from .lanes import get_lane_state


class PostureReward(ManagerTermBase):
    """Fresh sensor/kinematic diagnostics, also callable by read-only evaluation."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.ids, names = self.robot.find_bodies(list(FOOT_NAMES), preserve_order=True)
        if names != list(FOOT_NAMES):
            raise ValueError(f"unexpected feet: {names}")
        self.offsets = torch.tensor(TIP_OFFSETS, device=env.device).expand(env.num_envs, -1, -1)

    def geometry(self, env):
        data = self.robot.data
        sensor = env.scene["height_scanner"]
        sensor.update(0.0, force_recompute=True)
        encoded = encode_height_scan(
            data.root_pos_w[:, 2], sensor.data.ray_hits_w[..., 2],
            ray_offset_z=sensor.cfg.offset.pos[2], max_distance=sensor.cfg.max_distance,
            plane_height=FLAT_PLANE_HEIGHT,
        )
        if encoded.shape[-1] != 2 * GRID_RAYS:
            raise ValueError("v13 requires 33x25=825 scan rays")
        height = encoded[:, :GRID_RAYS]
        valid = (encoded[:, GRID_RAYS:] > .5) & (height.abs() < .99)
        surface_z = -height - .5
        offset = quat_apply(data.body_link_quat_w[:, self.ids], self.offsets)
        tips = data.body_link_pos_w[:, self.ids] + offset
        feet = feet_in_yaw_frame(data.root_pos_w, data.root_quat_w, tips)
        tip_velocity = data.body_link_lin_vel_w[:, self.ids] + torch.cross(
            data.body_link_ang_vel_w[:, self.ids], offset, dim=-1,
        )
        relative_velocity = tip_velocity - data.root_link_lin_vel_w[:, None]
        direction = get_lane_state(env).target_direction(data.root_pos_w)
        swing_speed = (relative_velocity[..., :2] * direction[:, None]).sum(-1)
        forward_speed = (data.root_link_lin_vel_w[:, :2] * direction).sum(-1)
        up = torch.zeros_like(data.root_pos_w)
        up[:, 2] = 1.
        upright = quat_apply(data.root_quat_w, up)[:, 2]
        return terrain_posture_terms(surface_z, valid, feet, swing_speed, forward_speed, upright)

    def __call__(self, env):
        return self.geometry(env)["reward"]


@configclass
class AdaptivePostureRewardsCfg(FootholdRewardsCfg):
    adaptive_posture = RewTerm(func=PostureReward, weight=1.0)


@configclass
class AdaptivePostureTrainEnvCfg(FootholdTrainAntEnvCfg):
    # Inherit FootMapTrainAntEnvCfg.__post_init__, including every reward override.
    rewards: AdaptivePostureRewardsCfg = AdaptivePostureRewardsCfg()
