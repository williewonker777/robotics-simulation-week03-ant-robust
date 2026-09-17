# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2026, Robotics Simulation Week 03 Team.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a continuous four-terrain GUI demo for a trained Ant checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Show Ant on flat, rough, slope, and stair terrain patches.")
parser.add_argument("--task", default="Week03-Ant-Terrain-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=7)
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
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
if args_cli.num_envs != 4:
    parser.error("the terrain demo requires --num_envs 4")

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
    """Load the checkpoint and continuously render the four terrain families."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (7.5, 18.0, 10.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.4)

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"unsupported runner class: {agent_cfg.class_name}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    observations = env.get_observations()
    dt = env.unwrapped.step_dt
    started = time.monotonic()
    active_env = -1

    print("[DEMO] env 0=flat, env 1=random rough, env 2=slope, env 3=stairs")
    print(f"[DEMO] checkpoint={checkpoint}")
    print("[DEMO] Close the Isaac Sim window to stop.")

    try:
        while simulation_app.is_running():
            step_started = time.monotonic()
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, _, _ = env.step(actions)
            elapsed = time.monotonic() - started
            next_env = int(elapsed / args_cli.cycle_seconds) % args_cli.num_envs
            if next_env != active_env:
                active_env = next_env
                print(f"[DEMO] Following env {active_env}: {('flat', 'random rough', 'slope', 'stairs')[active_env]}")
            robot_position = env.unwrapped.scene["robot"].data.root_pos_w[active_env].detach().cpu().tolist()
            eye = (robot_position[0] - 4.0, robot_position[1] + 4.0, robot_position[2] + 2.5)
            target = (robot_position[0], robot_position[1], robot_position[2] + 0.15)
            env.unwrapped.sim.set_camera_view(eye, target)
            if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
                break
            if args_cli.real_time:
                sleep_time = dt - (time.monotonic() - step_started)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
