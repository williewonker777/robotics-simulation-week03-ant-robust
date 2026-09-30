"""Opt-in no-cost continuation with unchanged v20 reward-time instrumentation."""
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from week03_ant.contact_continuation_v21 import coefficient
from .contact_v16_cfg import ContactSlipReward
from .contact_curriculum_v20_cfg import (
    CurriculumContactTrainAntEnvCfg, CurriculumContactRewardsCfg, CurriculumContactAntPPORunnerCfg,
)


class ContinuationContactSlipReward(ContactSlipReward):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.history = []

    def __call__(self, env, mode='no_cost', ramp_steps=4000):
        # Always execute identical sensors/geometry, even at coefficient zero.
        geometry = super().geometry(env)
        raw = geometry["reward"]
        factor = coefficient(mode, env.common_step_counter, ramp_steps)
        scaled = raw * factor
        if (not torch.isfinite(raw).all() or bool((raw < -1).any()) or bool((raw > 0).any())
                or not torch.isfinite(scaled).all()):
            raise ValueError('contact curriculum reward escaped finite [-1,0] bounds')
        self.history.append((env.common_step_counter, factor, float(raw.mean()), float(scaled.mean()),
                             int(geometry["valid"].sum()), float(env.step_dt),
                             float(raw.double().sum()) * env.step_dt,
                             float(scaled.double().sum()) * env.step_dt))
        return scaled

    def schedule(self, mode, ramp_steps):
        return dict(mode=mode, ramp_steps=ramp_steps, policy_steps=len(self.history),
                    common_step_counters=[row[0] for row in self.history],
                    coefficients=[row[1] for row in self.history],
                    raw_reward_mean=[row[2] for row in self.history],
                    scaled_reward_mean=[row[3] for row in self.history],
                    coefficient_sum=sum(row[1] for row in self.history),
                    valid_rows=[row[4] for row in self.history],
                    step_dt=[row[5] for row in self.history],
                    raw_reward_dt_sum=[row[6] for row in self.history],
                    scaled_reward_dt_sum=[row[7] for row in self.history],
                    realized_penalty_sum=-sum(row[7] for row in self.history))


@configclass
class ContinuationContactRewardsCfg(CurriculumContactRewardsCfg):
    contact_slip = RewTerm(func=ContinuationContactSlipReward, weight=1.,
                           params={'mode': 'no_cost', 'ramp_steps': 4000})


@configclass
class ContinuationContactTrainAntEnvCfg(CurriculumContactTrainAntEnvCfg):
    rewards: ContinuationContactRewardsCfg = ContinuationContactRewardsCfg()


@configclass
class ContinuationContactAntPPORunnerCfg(CurriculumContactAntPPORunnerCfg):
    experiment_name = 'week03_ant_contact_continuation_v21'


def validate_continuation_config_parity():
    candidate = ContinuationContactTrainAntEnvCfg().to_dict()
    reference = CurriculumContactTrainAntEnvCfg().to_dict()
    term, base = candidate['rewards']['contact_slip'], reference['rewards']['contact_slip']
    if term['weight'] != 1. or term['params'] != {'mode': 'no_cost', 'ramp_steps': 4000}:
        raise ValueError('continuation default coefficient contract differs')
    # Only the new reward entry point and zero-cost treatment are permitted.
    term['func'], term['params']['mode'] = base['func'], base['params']['mode']
    if candidate != reference:
        raise ValueError('undeclared continuation environment configuration difference')
    return True
