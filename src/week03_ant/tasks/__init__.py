"""Gym task registrations for the Week 03 Ant experiments."""

import gymnasium as gym


_TASKS = {
    "Week03-Ant-Baseline-v0": "BaselineAntEnvCfg",
    "Week03-Ant-Friction-v0": "FrictionAntEnvCfg",
    "Week03-Ant-Robust-v0": "RobustAntEnvCfg",
    "Week03-Ant-Test-LowFriction-v0": "LowFrictionAntEnvCfg",
    "Week03-Ant-Test-Heavy-v0": "HeavyAntEnvCfg",
    "Week03-Ant-Test-Push-v0": "PushAntEnvCfg",
    "Week03-Ant-Terrain-v0": "TerrainAntEnvCfg",
    "Week03-Ant-Terrain-Train-v0": "TerrainRobustAntEnvCfg",
    "Week03-Ant-Terrain-Mild-Train-v0": "MildTerrainAntEnvCfg",
    "Week03-Ant-Terrain-Demo-v0": "TerrainDemoAntEnvCfg",
    "Week03-Ant-Terrain-Posture-v1": "TerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Posture-Mild-Train-v1": "MildTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Complex-Posture-v2": "ComplexTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Complex-Mild-Train-v2": "ComplexMildTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Extreme-Posture-v3": "ExtremeTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Extreme-Mild-Train-v3": "ExtremeMildTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Extreme-Approach-Train-v3": "ExtremeApproachTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Extreme-Ramp-Train-v3": "ExtremeRampTerrainPostureAntEnvCfg",
}

for task_id, cfg_name in _TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"week03_ant.tasks.ant_cfg:{cfg_name}",
                "rsl_rl_cfg_entry_point": (
                    "week03_ant.tasks.agents.rsl_rl_ppo_cfg:Week03AntPPORunnerCfg"
                ),
            },
        )

__all__ = ["_TASKS"]
