"""Explicit-only v20 training registration; defaults and frozen tasks unchanged."""
import gymnasium as gym

NAME = 'Week03-Ant-Contact-Curriculum-v20-Train-v0'
if NAME not in gym.registry:
    gym.register(id=NAME, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': 'week03_ant.tasks.contact_curriculum_v20_cfg:CurriculumContactTrainAntEnvCfg',
                         'rsl_rl_cfg_entry_point': 'week03_ant.tasks.contact_curriculum_v20_cfg:CurriculumContactAntPPORunnerCfg'})
