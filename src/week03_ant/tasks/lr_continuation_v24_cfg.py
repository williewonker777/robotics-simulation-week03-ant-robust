"""Fixed-LR continuation: the complete v21 environment is inherited unchanged."""
from isaaclab.utils import configclass
from .contact_continuation_v21_cfg import (
    ContinuationContactTrainAntEnvCfg, ContinuationContactAntPPORunnerCfg,
)


@configclass
class LRContinuationTrainAntEnvCfg(ContinuationContactTrainAntEnvCfg):
    pass


@configclass
class LRContinuationAntPPORunnerCfg(ContinuationContactAntPPORunnerCfg):
    experiment_name = 'week03_ant_lr_continuation_v24'


def validate_lr_config_parity():
    if LRContinuationTrainAntEnvCfg().to_dict() != ContinuationContactTrainAntEnvCfg().to_dict():
        raise ValueError('LR study changed the v21 environment')
    return True
