"""Explicit-only v16 task registration; no default controller change."""

import gymnasium as gym

for stage in ('Train', 'Eval'):
    name = f'Week03-Ant-Contact-v16-{stage}-v0'
    if name not in gym.registry:
        gym.register(
            id=name, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
            kwargs={
                'env_cfg_entry_point': f'week03_ant.tasks.contact_v16_cfg:Contact{stage}AntEnvCfg',
                'rsl_rl_cfg_entry_point': 'week03_ant.tasks.contact_v16_cfg:ContactAntPPORunnerCfg',
            },
        )
