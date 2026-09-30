"""Opt-in command observations; unchanged v13 training reward and eval physics."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from week03_ant.command_math import command_features
from .posture_v13_cfg import PostureReward, AdaptivePostureTrainEnvCfg
from .agents.prior_v10_cfg import PriorPolicyCfg, PriorAntPPORunnerCfg
from .foothold_v9_cfg import FootholdObservationCfg, FootholdTrainObservationCfg, FootholdEvalAntEnvCfg


class CommandObservation(PostureReward):
    def __call__(self, env):
        return command_features(self.geometry(env))


@configclass
class CommandObservationCfg(FootholdObservationCfg):
    @configclass
    class PolicyCfg(FootholdObservationCfg.PolicyCfg):
        posture_command = ObsTerm(func=CommandObservation)

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandTrainObservationCfg(FootholdTrainObservationCfg):
    @configclass
    class PolicyCfg(FootholdTrainObservationCfg.PolicyCfg):
        posture_command = ObsTerm(func=CommandObservation)

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandTrainAntEnvCfg(AdaptivePostureTrainEnvCfg):
    observations: CommandTrainObservationCfg = CommandTrainObservationCfg()


@configclass
class CommandEvalAntEnvCfg(FootholdEvalAntEnvCfg):
    observations: CommandObservationCfg = CommandObservationCfg()


@configclass
class CommandPolicyCfg(PriorPolicyCfg):
    class_name = "CommandPriorActorCritic"
    command_mode: str = "conditioned"
    require_config_match: bool = True


@configclass
class CommandAntPPORunnerCfg(PriorAntPPORunnerCfg):
    experiment_name = "week03_ant_command_v14"
    policy = CommandPolicyCfg(
        init_noise_std=.2, actor_obs_normalization=False, critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100], critic_hidden_dims=[400, 200, 100], activation="elu",
    )
