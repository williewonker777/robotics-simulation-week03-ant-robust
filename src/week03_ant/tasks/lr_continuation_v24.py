"""Explicit-only training registration for fixed global-LR continuation sensitivity."""
import gymnasium as gym

NAME = 'Week03-Ant-LR-Continuation-v24-Train-v0'
if NAME not in gym.registry:
    gym.register(id=NAME, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': 'week03_ant.tasks.lr_continuation_v24_cfg:LRContinuationTrainAntEnvCfg',
                         'rsl_rl_cfg_entry_point': 'week03_ant.tasks.lr_continuation_v24_cfg:LRContinuationAntPPORunnerCfg'})
