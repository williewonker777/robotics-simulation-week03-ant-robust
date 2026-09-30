"""Explicit-only registration of the v15 training task; retain the v14 eval task."""

import gymnasium as gym

name = 'Week03-Ant-Direction-v15-Train-v0'
if name not in gym.registry:
    gym.register(
        id=name, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': 'week03_ant.tasks.direction_v15_cfg:DirectionTrainAntEnvCfg',
            'rsl_rl_cfg_entry_point': 'week03_ant.tasks.direction_v15_cfg:DirectionAntPPORunnerCfg',
        },
    )
