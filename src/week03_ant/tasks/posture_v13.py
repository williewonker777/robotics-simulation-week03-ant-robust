"""Opt-in v13 training registration; evaluation physics remain frozen v10."""

import gymnasium as gym

name = "Week03-Ant-Adaptive-Posture-Train-v13"
if name not in gym.registry:
    gym.register(
        id=name, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "week03_ant.tasks.posture_v13_cfg:AdaptivePostureTrainEnvCfg",
            "rsl_rl_cfg_entry_point": "week03_ant.tasks.agents.prior_v10_cfg:PriorAntPPORunnerCfg",
        },
    )
