"""v8 reuses the exact v7 PPO budget and environment."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import FootMapPolicyCfg, FootMapAntPPORunnerCfg


@configclass
class ResidualPolicyCfg(FootMapPolicyCfg):
    class_name = "ResidualActorCritic"
    residual_limit: float = 0.5
    require_config_match: bool = False


@configclass
class ResidualAntPPORunnerCfg(FootMapAntPPORunnerCfg):
    experiment_name = "week03_ant_residual_v8"
    policy = ResidualPolicyCfg(
        init_noise_std=0.2, actor_obs_normalization=False, critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100], critic_hidden_dims=[400, 200, 100], activation="elu",
    )
