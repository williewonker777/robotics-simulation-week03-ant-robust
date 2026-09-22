"""RSL-RL configuration shared by every experiment for a fair comparison."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Week03AntPPORunnerCfg(AntPPORunnerCfg):
    """Unmodified course PPO budget with a project-specific log namespace."""

    experiment_name = "week03_ant"


@configclass
class ExtremeRecoveryAntPPORunnerCfg(Week03AntPPORunnerCfg):
    """Stable fixed-step PPO settings for the v4 extreme-terrain curriculum.

    v3's adaptive scheduler reduced the learning rate to its floor while the
    discontinuous terrain was still being introduced.  The recovery run keeps a
    small fixed rate, restores exploration, and uses the same MLP widths so the
    v3 114D checkpoint can be warm-started without a shape conversion.
    """

    experiment_name = "week03_ant_recovery"
    max_iterations = 2500
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.12,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=7.5e-5,
        schedule="fixed",
        desired_kl=None,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
    )


@configclass
class RoughLaneAntPPORunnerCfg(Week03AntPPORunnerCfg):
    """Course PPO settings for the v5 lane task; only the log namespace differs.

    Keeping the course network and algorithm lets a v5 checkpoint be evaluated with
    the course ``AntPPORunnerCfg`` in an unseen environment.
    """

    experiment_name = "week03_ant_rough_v5"


@configclass
class DepthLaneAntPPORunnerCfg(RoughLaneAntPPORunnerCfg):
    """Separate depth-aware artifacts; the input width follows the v6 environment."""

    experiment_name = "week03_ant_depth_v6"


@configclass
class FootMapPolicyCfg(RslRlPpoActorCriticCfg):
    class_name = "FootMapActorCritic"
    input_mode: str = "footmap"


@configclass
class FootMapAntPPORunnerCfg(DepthLaneAntPPORunnerCfg):
    experiment_name = "week03_ant_footmap_v7"
    num_steps_per_env = 32
    max_iterations = 750
    save_interval = 250
    policy = FootMapPolicyCfg(
        init_noise_std=0.2, actor_obs_normalization=False, critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100], critic_hidden_dims=[400, 200, 100], activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.002,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-4, schedule="fixed",
        desired_kl=None, gamma=0.995, lam=0.95, max_grad_norm=1.0,
    )
