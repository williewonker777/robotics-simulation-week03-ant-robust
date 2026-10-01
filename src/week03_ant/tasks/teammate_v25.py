"""Explicit import only: never change submission/default task registrations."""
import gymnasium as gym
NAME = 'Week03-Ant-Teammate-v25-Train-v0'
if NAME not in gym.registry:
    gym.register(id=NAME, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
        kwargs={'env_cfg_entry_point': 'week03_ant.tasks.teammate_v25_cfg:TeammateTrainAntEnvCfg',
                'rsl_rl_cfg_entry_point': 'week03_ant.tasks.teammate_v25_cfg:TeammateAntPPORunnerCfg'})
