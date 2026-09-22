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
    "Week03-Ant-Terrain-Extreme-Recovery-v4": "ExtremeRecoveryTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Extreme-Recovery-Mild-Train-v4": "ExtremeRecoveryMildTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Extreme-Recovery-Approach-Train-v4": "ExtremeRecoveryApproachTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Extreme-Recovery-Ramp-Train-v4": "ExtremeRecoveryRampTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Extreme-Recovery-Focus-Train-v4": "ExtremeRecoveryFocusTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Extreme-Recovery-Pit-Focus-Train-v4": "ExtremeRecoveryPitFocusTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Extreme-Recovery-Geometry-Train-v4": "ExtremeRecoveryGeometryTerrainPostureAntEnvCfg",
    "Week03-Ant-Terrain-Extreme-Recovery-Robust-Train-v4": "ExtremeRecoveryRobustTrainTerrainPostureAntEnvCfg",
}

_RECOVERY_RUNNER = "week03_ant.tasks.agents.rsl_rl_ppo_cfg:ExtremeRecoveryAntPPORunnerCfg"

# v5: course 60D interface on treadmill terrain lanes (see tasks/lanes.py).
_ROUGH_V5_TASKS = {
    "Week03-Ant-Rough-Lanes-v5": "RoughLaneAntEnvCfg",
    "Week03-Ant-Rough-Lanes-Eval-v5": "RoughLaneEvalAntEnvCfg",
    "Week03-Ant-Rough-Lanes-Demo-v5": "RoughLaneDemoAntEnvCfg",
    "Week03-Ant-Rough-Lanes-Safe-Train-v5": "RoughLaneSafeAntEnvCfg",
    "Week03-Ant-Rough-Lanes-Traverse-Train-v5": "RoughLaneTraversalAntEnvCfg",
    "Week03-Ant-Rough-Lanes-Recovery-Train-v5": "RoughLaneRecoveryAntEnvCfg",
}
_ROUGH_V5_RUNNER = "week03_ant.tasks.agents.rsl_rl_ppo_cfg:RoughLaneAntPPORunnerCfg"

_DEPTH_V6_TASKS = {
    "Week03-Ant-Depth-Lanes-Train-v6": "DepthLaneTrainAntEnvCfg",
    "Week03-Ant-Depth-Lanes-Eval-v6": "DepthLaneEvalAntEnvCfg",
    "Week03-Ant-Depth-Lanes-Demo-v6": "DepthLaneDemoAntEnvCfg",
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
                    _RECOVERY_RUNNER
                    if "Extreme-Recovery" in task_id
                    else "week03_ant.tasks.agents.rsl_rl_ppo_cfg:Week03AntPPORunnerCfg"
                ),
            },
        )

for task_id, cfg_name in _ROUGH_V5_TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"week03_ant.tasks.rough_v5_cfg:{cfg_name}",
                "rsl_rl_cfg_entry_point": _ROUGH_V5_RUNNER,
            },
        )

for task_id, cfg_name in _DEPTH_V6_TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"week03_ant.tasks.depth_v6_cfg:{cfg_name}",
                "rsl_rl_cfg_entry_point": "week03_ant.tasks.agents.rsl_rl_ppo_cfg:DepthLaneAntPPORunnerCfg",
            },
        )

_FOOTMAP_V7_TASKS = {
    "Week03-Ant-FootMap-Lanes-Train-v7": "FootMapTrainAntEnvCfg",
    "Week03-Ant-FootMap-Lanes-Eval-v7": "FootMapEvalAntEnvCfg",
    "Week03-Ant-FootMap-Lanes-Demo-v7": "FootMapDemoAntEnvCfg",
}
for task_id, cfg_name in _FOOTMAP_V7_TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"week03_ant.tasks.footmap_v7_cfg:{cfg_name}",
                "rsl_rl_cfg_entry_point": "week03_ant.tasks.agents.rsl_rl_ppo_cfg:FootMapAntPPORunnerCfg",
            },
        )

__all__ = ["_TASKS", "_ROUGH_V5_TASKS", "_DEPTH_V6_TASKS", "_FOOTMAP_V7_TASKS"]
