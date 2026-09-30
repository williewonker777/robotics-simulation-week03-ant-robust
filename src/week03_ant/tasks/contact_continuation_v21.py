"""Explicit-only v21 training registration; defaults and frozen tasks unchanged."""
import gymnasium as gym

NAME = 'Week03-Ant-Contact-Continuation-v21-Train-v0'
if NAME not in gym.registry:
    gym.register(id=NAME, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': 'week03_ant.tasks.contact_continuation_v21_cfg:ContinuationContactTrainAntEnvCfg',
                         'rsl_rl_cfg_entry_point': 'week03_ant.tasks.contact_continuation_v21_cfg:ContinuationContactAntPPORunnerCfg'})
