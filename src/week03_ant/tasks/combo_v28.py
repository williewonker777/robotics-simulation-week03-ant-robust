"""Explicit-import registrations for the v28 combination study.

Importing this module adds new task IDs only; existing task registrations and defaults are untouched.
Task IDs:

* ``Week03-Ant-Combo-v28-Play``: scoring and submission task (original rewards, swappable terrain).
* ``Week03-Ant-Combo-v28-<Terrain>-<D0|D1>-<Stock|Recovery>``: training variants, e.g.
  ``Week03-Ant-Combo-v28-Lim-D1-Stock``.

PPO entropy (Lim's factor) is selected with the Hydra override ``agent.algorithm.entropy_coef=0.005``.
"""

import gymnasium as gym

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .combo_v28_cfg import TRAIN_VARIANTS, train_cfg_name

EXPERIMENT_NAME = "week03_ant_combo_v28"
PLAY_TASK = "Week03-Ant-Combo-v28-Play"


@configclass
class ComboV28PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """The course ``AntPPORunnerCfg`` unchanged except for the log folder and checkpoint interval."""

    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 100
    experiment_name = EXPERIMENT_NAME
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
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
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


_RUNNER = f"{__name__}:ComboV28PPORunnerCfg"
_CFG_MODULE = "week03_ant.tasks.combo_v28_cfg"


def train_task_id(terrain: str, dr: str, reward: str) -> str:
    return f"Week03-Ant-Combo-v28-{terrain.capitalize()}-{dr.upper()}-{reward.capitalize()}"


def _register(task_id: str, cfg_name: str) -> None:
    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{_CFG_MODULE}:{cfg_name}", "rsl_rl_cfg_entry_point": _RUNNER},
    )


_register(PLAY_TASK, "ComboV28PlayEnvCfg")
for _variant in TRAIN_VARIANTS:
    _register(train_task_id(*_variant), train_cfg_name(*_variant))

__all__ = ["ComboV28PPORunnerCfg", "EXPERIMENT_NAME", "PLAY_TASK", "train_task_id"]
