"""Explicit training registration; deployment and evaluation stay on v16."""

import gymnasium as gym

if "Week03-Ant-Style-v18-Train-v0" not in gym.registry:
    gym.register(
        id="Week03-Ant-Style-v18-Train-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "week03_ant.tasks.style_v18_cfg:StyleTrainAntEnvCfg",
            "rsl_rl_cfg_entry_point": "week03_ant.tasks.style_v18_cfg:StyleAntPPORunnerCfg",
        },
    )
