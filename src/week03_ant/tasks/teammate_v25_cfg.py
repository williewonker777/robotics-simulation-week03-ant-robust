"""Opt-in additive reward only; all original v21 physics/observations survive."""
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from week03_ant.teammate_recovery_v25 import recovery_terms, applied_rate
from .posture_v13_cfg import PostureReward
from .contact_continuation_v21_cfg import (
    ContinuationContactRewardsCfg, ContinuationContactTrainAntEnvCfg,
    ContinuationContactAntPPORunnerCfg,
)


class TeammateRecoveryReward(PostureReward):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.history = []

    def __call__(self, env, enabled=False):
        state = self.robot.data
        if not all(torch.isfinite(x).all() for x in (state.root_state_w, state.joint_pos, state.joint_vel,
                state.body_link_pos_w, state.body_link_quat_w, state.body_link_lin_vel_w, state.body_link_ang_vel_w)):
            raise ValueError('nonfinite physical state')
        geometry = self.geometry(env)
        up = torch.zeros_like(state.root_pos_w)
        up[:, 2] = 1
        upright = quat_apply(state.root_quat_w, up)[:, 2]
        terms = recovery_terms(geometry['body_clearance'], geometry['target_height'], geometry['valid'],
                               upright, env.action_manager.action, env.action_manager.prev_action,
                               state.root_ang_vel_b)
        rate = applied_rate(terms, enabled)
        self.history.append(dict(step=int(env.common_step_counter), enabled=enabled, dt=float(env.step_dt),
            raw_sum=float(terms['rate'].double().sum()), applied_sum=float(rate.double().sum()),
            abstained=int(terms['clearance_abstained'].sum()),
            **{key: float(terms[key].double().sum()) for key in
               ('clearance_risk', 'tilt_risk', 'action_delta_squared_sum', 'angular_xy_squared_sum')}))
        return rate


@configclass
class TeammateRewardsCfg(ContinuationContactRewardsCfg):
    teammate_recovery = RewTerm(func=TeammateRecoveryReward, weight=1., params={'enabled': False})


@configclass
class TeammateTrainAntEnvCfg(ContinuationContactTrainAntEnvCfg):
    rewards: TeammateRewardsCfg = TeammateRewardsCfg()


@configclass
class TeammateAntPPORunnerCfg(ContinuationContactAntPPORunnerCfg):
    experiment_name = 'week03_ant_teammate_port_v25'


def validate_config_parity():
    candidate = TeammateTrainAntEnvCfg().to_dict()
    candidate['rewards'].pop('teammate_recovery')
    if candidate != ContinuationContactTrainAntEnvCfg().to_dict():
        raise ValueError('v25 changed original environment beyond additive reward')
    return True
