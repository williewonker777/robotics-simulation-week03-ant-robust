"""88D compact foothold hints; same PPO budget and hidden widths as v7/v8."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from .rsl_rl_ppo_cfg import FootMapAntPPORunnerCfg


@configclass
class FootholdPolicyCfg(RslRlPpoActorCriticCfg):
    class_name = "FootholdActorCritic"
    input_mode: str = "guided"
    require_config_match: bool = False


@configclass
class FootholdAntPPORunnerCfg(FootMapAntPPORunnerCfg):
    experiment_name = "week03_ant_foothold_v9"
    policy = FootholdPolicyCfg(
        init_noise_std=.2, actor_obs_normalization=False, critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100], critic_hidden_dims=[400, 200, 100], activation="elu",
    )
