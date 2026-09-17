# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2026, Robotics Simulation Week 03 Team.
# SPDX-License-Identifier: BSD-3-Clause

"""Play and score one completed episode for every vectorized Ant environment.

This assignment-facing entry point extends the upstream viewer script by
waiting until every vectorized environment has completed once, then printing
the population mean and standard deviation of cumulative episode reward.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL Ant checkpoint.")
parser.add_argument("--task", required=True, help="Gym task ID used for evaluation.")
parser.add_argument(
    "--agent",
    default="rsl_rl_cfg_entry_point",
    help="Gym registration key for the RSL-RL agent configuration.",
)
parser.add_argument("--num_envs", type=int, default=100, help="Number of completed episodes to aggregate.")
parser.add_argument("--seed", type=int, default=24, help="Evaluation seed.")
parser.add_argument("--max_steps", type=int, default=960, help="Safety limit for the first episode.")
parser.add_argument("--output", type=Path, help="Optional JSON output path.")
parser.add_argument("--video", action="store_true", help="Record the rollout as MP4.")
parser.add_argument("--video_dir", type=Path, help="Video output directory.")
parser.add_argument("--real-time", action="store_true", help="Throttle playback to simulation time.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import week03_ant  # noqa: F401
from week03_ant.evaluation import FirstEpisodeAccumulator


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Run one episode in every environment and report population statistics."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (-4.0, 4.0, 2.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required for reproducible evaluation")
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_dir = args_cli.video_dir or Path(checkpoint).parent / "videos" / args_cli.task
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.max_steps,
            disable_logger=True,
            name_prefix=f"{args_cli.task}-seed{args_cli.seed}",
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading checkpoint: {checkpoint}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    observations = env.get_observations()
    accumulator = FirstEpisodeAccumulator(args_cli.num_envs, env.unwrapped.device)
    dt = env.unwrapped.step_dt

    try:
        for step in range(1, args_cli.max_steps + 1):
            started = time.time()
            with torch.inference_mode():
                actions = policy(observations)
                observations, rewards, dones, _ = env.step(actions)
            accumulator.update(rewards, dones)
            if accumulator.complete:
                print(f"[INFO] All {args_cli.num_envs} episodes completed at step {step}.")
                break
            if args_cli.real_time:
                sleep_time = dt - (time.time() - started)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        result = accumulator.summary().to_dict()
        result.update(
            {
                "task": args_cli.task,
                "checkpoint": str(Path(checkpoint).resolve()),
                "seed": args_cli.seed,
                "num_envs": args_cli.num_envs,
                "max_steps": args_cli.max_steps,
            }
        )
        serialized = json.dumps(result, indent=2, sort_keys=True)
        print("WEEK03_EVALUATION_JSON_BEGIN")
        print(serialized)
        print("WEEK03_EVALUATION_JSON_END")
        if args_cli.output:
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.output.write_text(serialized + "\n", encoding="utf-8")
            print(f"[INFO] Wrote evaluation: {args_cli.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
