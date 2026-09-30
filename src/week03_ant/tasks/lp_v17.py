"""Explicit training registration only; evaluation uses frozen v16 contact task."""

import gymnasium as gym

if 'Week03-Ant-LP-v17-Train-v0' not in gym.registry:
    gym.register(
        id='Week03-Ant-LP-v17-Train-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv',
        disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': 'week03_ant.tasks.lp_v17_cfg:LPTrainAntEnvCfg',
            'rsl_rl_cfg_entry_point': 'week03_ant.tasks.lp_v17_cfg:LPAntPPORunnerCfg',
        },
    )
