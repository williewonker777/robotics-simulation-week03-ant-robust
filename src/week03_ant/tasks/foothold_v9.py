"""Opt-in v9 registration without changing archived task registries."""

import gymnasium as gym

for operation in ("Train", "Eval", "Demo"):
    name = f"Week03-Ant-Foothold-Lanes-{operation}-v9"
    if name not in gym.registry:
        gym.register(
            id=name, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"week03_ant.tasks.foothold_v9_cfg:Foothold{operation}AntEnvCfg",
                "rsl_rl_cfg_entry_point": "week03_ant.tasks.agents.foothold_v9_cfg:FootholdAntPPORunnerCfg",
            },
        )
