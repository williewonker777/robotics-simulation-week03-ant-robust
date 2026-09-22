"""Record first-episode Ant kinematics on fixed lanes (diagnostic, not a score).

The NPZ contains pre-action states at every control step and an ``active`` mask
that excludes all samples after each environment's first termination/reset.
Foot body origins are ankle frames, not necessarily the collision tips; the USD
geometry metadata is saved alongside the recording to make that distinction explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

import cli_args


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Week03-Ant-Rough-Lanes-Eval-v5")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--family", default="stepping_stones")
parser.add_argument("--copies", type=int, default=5, help="Environments per difficulty level.")
parser.add_argument("--steps", type=int, default=960)
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--output", type=Path, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint or args_cli.copies < 1 or args_cli.steps < 1:
    parser.error("a checkpoint and positive copies/steps are required")
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import week03_ant  # noqa: F401
from week03_ant.tasks.lanes import get_lane_state, lane_assign


def collision_metadata(stage):
    """Describe actual collision geometry including instance-proxy descendants."""
    output = []
    robot = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        item = {"path": str(prim.GetPath()), "type": prim.GetTypeName()}
        for attr in prim.GetAttributes():
            if attr.GetName() in ("radius", "height", "axis", "size"):
                item[attr.GetName()] = attr.Get()
        body = prim
        while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if body.IsValid():
            item["body"] = str(body.GetPath())
            item["collider_to_body"] = np.asarray(cache.ComputeRelativeTransform(prim, body)[0]).tolist()
        output.append(item)
    return output


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    levels = env_cfg.scene.terrain.terrain_generator.lane_difficulties
    env_cfg.scene.num_envs = len(levels) * args_cli.copies
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    env_cfg.events.lane_layout.func = lane_assign
    env_cfg.events.lane_layout.params = {"family_levels": [(args_cli.family, i) for i in range(len(levels))]}
    checkpoint = Path(args_cli.checkpoint).resolve()
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        robot = env.unwrapped.scene["robot"]
        state = get_lane_state(env.unwrapped)
        metadata = {
            "task": args_cli.task, "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "seed": args_cli.seed, "terrain_generator_seed": env_cfg.scene.terrain.terrain_generator.seed,
            "step_dt": env.unwrapped.step_dt, "family": args_cli.family,
            "difficulties": levels, "level_indices": state.family_level()[1].cpu().tolist(),
            "body_names": robot.body_names, "joint_names": robot.joint_names,
            "joint_limits": robot.data.joint_pos_limits[0].cpu().tolist(),
            "collision_geometry": collision_metadata(env.unwrapped.sim.stage),
            "state_velocity_frames": {
                "root_state": "link pose, COM velocity",
                "body_state": "link pose, COM velocity",
                "root_link_velocity": "link-origin velocity",
                "body_link_velocity": "link-origin velocity (use for distal-tip omega cross offset)",
            },
            "note": "Pre-action states; use active mask. Body origins are not collision tips.",
        }
        buffers = {}
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = policy(obs)
                sample = {
                    "active": active, "actions": actions,
                    "root_state": robot.data.root_state_w,
                    "body_state": robot.data.body_state_w,
                    "root_link_velocity": robot.data.root_link_vel_w,
                    "body_link_velocity": robot.data.body_link_vel_w,
                    "joint_pos": robot.data.joint_pos, "joint_vel": robot.data.joint_vel,
                    "joint_wrench": robot.data.body_incoming_joint_wrench_b,
                    "ground_z": state.ground_height(robot.data.root_pos_w),
                    "distance": state.odometer(robot.data.root_pos_w[:, 0]) - state.spawn_x,
                }
                for name, value in sample.items():
                    buffers.setdefault(name, []).append(value.detach().cpu().numpy().copy())
                obs, _, dones, _ = env.step(actions)
                active &= ~dones.bool().reshape(-1)
                if not bool(active.any()):
                    break
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args_cli.output, **{key: np.stack(values) for key, values in buffers.items()})
        args_cli.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[GAIT] Recorded {len(buffers['active'])} steps to {args_cli.output}")
        print(f"[GAIT] body_names={robot.body_names}; joint_names={robot.joint_names}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
