"""Opt-in registrations keep all archived v7 source hashes unchanged."""

import gymnasium as gym


for operation in ("Train", "Eval", "Demo"):
    task_id = f"Week03-Ant-Residual-Lanes-{operation}-v8"
    if task_id not in gym.registry:
        gym.register(
            id=task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"week03_ant.tasks.footmap_v7_cfg:FootMap{operation}AntEnvCfg",
                "rsl_rl_cfg_entry_point": "week03_ant.tasks.agents.residual_v8_cfg:ResidualAntPPORunnerCfg",
            },
        )
