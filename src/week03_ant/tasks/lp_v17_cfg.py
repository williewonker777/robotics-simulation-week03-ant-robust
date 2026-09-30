"""Reset-only signed learning-progress curriculum; observations/rewards unchanged."""

from isaaclab.utils import configclass
from week03_ant.lp_curriculum_v17 import lp_reset_root_state
from .contact_v16_cfg import ContactTrainAntEnvCfg, ContactAntPPORunnerCfg


@configclass
class LPTrainAntEnvCfg(ContactTrainAntEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.contact_slip.weight = 0.
        self.events.reset_base.func = lp_reset_root_state
        self.events.reset_base.params['mode'] = 'fixed'
        self.events.reset_base.params['stage_size'] = 1024


@configclass
class LPAntPPORunnerCfg(ContactAntPPORunnerCfg):
    experiment_name = 'week03_ant_lp_v17'
