"""Explicit-only training registration for fixed-policy continuation seed sensitivity."""
import gymnasium as gym

NAME = 'Week03-Ant-Seed-Continuation-v22-Train-v0'
if NAME not in gym.registry:
    gym.register(id=NAME, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': 'week03_ant.tasks.seed_continuation_v22_cfg:SeedContinuationTrainAntEnvCfg',
                         'rsl_rl_cfg_entry_point': 'week03_ant.tasks.seed_continuation_v22_cfg:SeedContinuationAntPPORunnerCfg'})
