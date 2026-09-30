"""Seed-only continuation: the complete v21 environment is inherited unchanged."""
from isaaclab.utils import configclass
from .contact_continuation_v21_cfg import (
    ContinuationContactTrainAntEnvCfg, ContinuationContactAntPPORunnerCfg,
)


@configclass
class SeedContinuationTrainAntEnvCfg(ContinuationContactTrainAntEnvCfg):
    pass


@configclass
class SeedContinuationAntPPORunnerCfg(ContinuationContactAntPPORunnerCfg):
    experiment_name = 'week03_ant_seed_continuation_v22'


def validate_seed_config_parity():
    if SeedContinuationTrainAntEnvCfg().to_dict() != ContinuationContactTrainAntEnvCfg().to_dict():
        raise ValueError('seed replication changed the v21 environment')
    return True
