"""Training-only v5 mean prior with the unchanged v9 PPO budget."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import FootMapAntPPORunnerCfg


@configclass
class PriorPolicyCfg(RslRlPpoActorCriticCfg):
    class_name = "PriorActorCritic"
    input_mode: str = "targets"
    prior_mode: str = "anchored"
    require_config_match: bool = False


@configclass
class PriorAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "PriorPPO"
    teacher_coef: float = .02


@configclass
class PriorAntPPORunnerCfg(FootMapAntPPORunnerCfg):
    experiment_name = "week03_ant_prior_v10"
    policy = PriorPolicyCfg(
        init_noise_std=.2, actor_obs_normalization=False, critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100], critic_hidden_dims=[400, 200, 100], activation="elu",
    )
    algorithm = PriorAlgorithmCfg(
        value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2, entropy_coef=.002,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-4, schedule="fixed",
        desired_kl=None, gamma=.995, lam=.95, max_grad_norm=1.,
    )
