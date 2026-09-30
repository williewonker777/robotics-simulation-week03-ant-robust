"""Explicit-only registration of the 91D v14 tasks."""

import gymnasium as gym

for stage in ("Train", "Eval"):
    name = f"Week03-Ant-Command-v14-{stage}-v0"
    if name not in gym.registry:
        gym.register(
            id=name, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"week03_ant.tasks.command_v14_cfg:Command{stage}AntEnvCfg",
                "rsl_rl_cfg_entry_point": "week03_ant.tasks.command_v14_cfg:CommandAntPPORunnerCfg",
            },
        )
