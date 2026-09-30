"""Opt-in v15 additive directional reward; unchanged v14 observations and physics."""

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from week03_ant.direction_math import direction_metrics
from .command_v14_cfg import CommandTrainAntEnvCfg, CommandAntPPORunnerCfg
from .posture_v13_cfg import AdaptivePostureRewardsCfg
from .lanes import get_lane_state


class DirectionReward(ManagerTermBase):
    """Read current root COM motion; no caches, sensor queries, or physical writes."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']

    def geometry(self, env):
        data = self.robot.data
        forward = torch.zeros_like(data.root_pos_w)
        forward[:, 0] = 1.
        return direction_metrics(
            get_lane_state(env).target_direction(data.root_pos_w),
            quat_apply(data.root_quat_w, forward),
            data.root_com_lin_vel_w, data.root_com_ang_vel_w,
        )

    def __call__(self, env):
        return self.geometry(env)['reward']


@configclass
class DirectionalRewardsCfg(AdaptivePostureRewardsCfg):
    directional_stability = RewTerm(func=DirectionReward, weight=1.0)


@configclass
class DirectionTrainAntEnvCfg(CommandTrainAntEnvCfg):
    rewards: DirectionalRewardsCfg = DirectionalRewardsCfg()


@configclass
class DirectionAntPPORunnerCfg(CommandAntPPORunnerCfg):
    experiment_name = 'week03_ant_direction_v15'
