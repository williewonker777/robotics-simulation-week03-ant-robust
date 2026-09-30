"""Opt-in coefficient-only continuation of the unchanged v16 contact geometry."""
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from week03_ant.contact_curriculum_v20 import coefficient
from .contact_v16_cfg import ContactSlipReward, ContactTrainAntEnvCfg, ContactRewardsCfg, ContactAntPPORunnerCfg


class CurriculumContactSlipReward(ContactSlipReward):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.history = []

    def __call__(self, env, mode='immediate', ramp_steps=4000):
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
class CurriculumContactRewardsCfg(ContactRewardsCfg):
    contact_slip = RewTerm(func=CurriculumContactSlipReward, weight=1.,
                           params={'mode': 'immediate', 'ramp_steps': 4000})


@configclass
class CurriculumContactTrainAntEnvCfg(ContactTrainAntEnvCfg):
    rewards: CurriculumContactRewardsCfg = CurriculumContactRewardsCfg()


@configclass
class CurriculumContactAntPPORunnerCfg(ContactAntPPORunnerCfg):
    experiment_name = 'week03_ant_contact_curriculum_v20'


def validate_curriculum_config_parity():
    candidate, reference = CurriculumContactTrainAntEnvCfg().to_dict(), ContactTrainAntEnvCfg().to_dict()
    term, base = candidate['rewards']['contact_slip'], reference['rewards']['contact_slip']
    if term['weight'] != 1. or term['params'] != {'mode': 'immediate', 'ramp_steps': 4000}:
        raise ValueError('curriculum default coefficient contract differs')
    term['func'], term['params'] = base['func'], base['params']
    if candidate != reference:
        raise ValueError('undeclared curriculum configuration difference')
    return True
