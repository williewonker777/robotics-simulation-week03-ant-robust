"""Opt-in v10 registration; reuse the unchanged v9 physical/observation tasks."""

import gymnasium as gym


for operation in ("Train", "Eval", "Demo"):
    name = f"Week03-Ant-Prior-Lanes-{operation}-v10"
    if name not in gym.registry:
        gym.register(
            id=name, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"week03_ant.tasks.foothold_v9_cfg:Foothold{operation}AntEnvCfg",
                "rsl_rl_cfg_entry_point": "week03_ant.tasks.agents.prior_v10_cfg:PriorAntPPORunnerCfg",
            },
        )
