# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2026, Robotics Simulation Week 03 Team.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a continuous multi-terrain GUI demo for a trained Ant checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Show Ant on each terrain family in the selected task.")
parser.add_argument("--task", default="Week03-Ant-Terrain-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of preview environments; defaults to one per configured terrain family.",
)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--family", help="Show only this v5 lane family (for focused diagnosis).")
parser.add_argument("--level", type=int, help="Override v5 difficulty level index for the selected families.")
parser.add_argument("--cycle-seconds", type=float, default=7.0, help="Seconds to follow each terrain environment.")
parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Wall-clock seconds before exit; zero keeps the demo open until the window closes.",
)
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Throttle simulation to real time for viewing.",
)
parser.add_argument(
    "--record",
    type=str,
    default=None,
    help="Write the followed camera to this MP4 (one cycle per environment) instead of running forever.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
if args_cli.record:
    args_cli.enable_cameras = True
    args_cli.real_time = False

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import week03_ant  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load the checkpoint and continuously render each configured terrain family."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is None:
        raise ValueError("the selected task must use a procedural terrain generator")
    # v5 lane tasks place one environment per terrain family (not per sub-terrain variant).
    lane_families = getattr(terrain_generator, "lane_families", None)
    terrain_names = list(lane_families) if lane_families else list(terrain_generator.sub_terrains)
    plane_family = getattr(terrain_generator, "plane_family", None)
    if lane_families and plane_family is not None:
        terrain_names.append(plane_family)
    if args_cli.family is not None or args_cli.level is not None:
        if not lane_families:
            raise ValueError("--family/--level selection requires a v5 lane task")
        if args_cli.family is not None:
            if args_cli.family not in terrain_names:
                raise ValueError(f"unknown family {args_cli.family!r}; choose from {terrain_names}")
            terrain_names = [args_cli.family]
        level = 3 if args_cli.level is None else args_cli.level
        if not 0 <= level < len(terrain_generator.lane_difficulties):
            raise ValueError("--level is outside the configured lane difficulty range")
        from week03_ant.tasks.lanes import lane_assign

        env_cfg.events.lane_layout.func = lane_assign
        env_cfg.events.lane_layout.params = {"family_levels": [(family, level) for family in terrain_names]}
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else len(terrain_names)
    if num_envs != len(terrain_names):
        raise ValueError(
            f"the terrain demo needs one environment per terrain family: "
            f"expected {len(terrain_names)}, got {num_envs}"
        )
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (7.5, 18.0, 10.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.4)

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.record else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"unsupported runner class: {agent_cfg.class_name}")
    if agent_cfg.policy.class_name == "FootMapActorCritic":
        from week03_ant.footmap_policy import register_footmap_policy
        register_footmap_policy()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    observations = env.get_observations()
    dt = env.unwrapped.step_dt
    started = time.monotonic()
    active_env = -1
    writer = None
    steps_per_env = int(round(args_cli.cycle_seconds / dt))
    if args_cli.record:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image, ImageDraw

        os.makedirs(os.path.dirname(os.path.abspath(args_cli.record)), exist_ok=True)
        # 60 Hz control, every second frame -> 30 fps real-time video
        writer = imageio.get_writer(args_cli.record, fps=int(round(0.5 / dt)), codec="libx264", quality=7)
    step = 0
    reset_counts = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)

    print("[DEMO] " + ", ".join(f"env {i}={name}" for i, name in enumerate(terrain_names)))
    print(f"[DEMO] checkpoint={checkpoint}")
    print("[DEMO] Close the Isaac Sim window to stop.")

    try:
        while simulation_app.is_running():
            step_started = time.monotonic()
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, _ = env.step(actions)
                reset_counts += dones.reshape(-1).to(torch.long)
            elapsed = time.monotonic() - started
            if writer is not None:
                # sim-time schedule so the recording does not depend on render speed
                elapsed = step * dt
                if step >= steps_per_env * num_envs:
                    break
            step += 1
            next_env = int(elapsed / args_cli.cycle_seconds) % num_envs
            if next_env != active_env:
                active_env = next_env
                print(f"[DEMO] Following env {active_env}: {terrain_names[active_env]}")
            robot_position = env.unwrapped.scene["robot"].data.root_pos_w[active_env].detach().cpu().tolist()
            eye = (robot_position[0] - 4.0, robot_position[1] + 4.0, robot_position[2] + 2.5)
            target = (robot_position[0], robot_position[1], robot_position[2] + 0.15)
            env.unwrapped.sim.set_camera_view(eye, target)
            if writer is not None and step % 2 == 0:
                frame = env.unwrapped.render()
                if frame is not None:
                    caption = f"{terrain_names[active_env]} | time {elapsed:.1f}s | resets {int(reset_counts[active_env])}"
                    lane_state = getattr(env.unwrapped, "_week03_lane_state", None)
                    if lane_state is not None:
                        distance = lane_state.odometer(env.unwrapped.scene["robot"].data.root_pos_w[:, 0])
                        distance = float((distance - lane_state.spawn_x)[active_env])
                        _, levels = lane_state.family_level()
                        difficulty = lane_state.difficulties[int(levels[active_env])]
                        caption += f" | difficulty {difficulty:g} | episode forward {distance:.1f}m"
                    annotated = Image.fromarray(frame)
                    draw = ImageDraw.Draw(annotated)
                    draw.rectangle((0, 0, annotated.width, 30), fill="black")
                    draw.text((10, 8), caption, fill="white")
                    frame = np.asarray(annotated)
                    writer.append_data(frame)
            if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
                break
            if args_cli.real_time:
                sleep_time = dt - (time.monotonic() - step_started)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        if writer is not None:
            writer.close()
            print(f"[DEMO] Wrote {args_cli.record}")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
