"""Run the fixed v10 64-second stepping-stones horizon diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


TASK = "Week03-Ant-Prior-Lanes-Eval-v10"
MAX_STEPS = 3840
EPISODE_SECONDS = 64.0
SNAPSHOT_SECONDS = 16.0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path, help="New JSON path; existing files are refused.")
parser.add_argument("--geometry", required=True, type=int, help="Non-negative terrain generator seed.")
parser.add_argument("--seed", required=True, type=int, help="Non-negative environment/reset seed.")
parser.add_argument("--num_envs", type=int, default=10, help="First episodes on stones difficulty 1.0.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.geometry < 0 or args_cli.seed < 0:
    parser.error("--geometry and --seed must be non-negative")
if args_cli.output.exists():
    parser.error(f"refusing to overwrite existing output: {args_cli.output}")
if args_cli.device is None:
    args_cli.device = "cuda:1"

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
from week03_ant.evaluation import LaneTraversalGeometry
from week03_ant.horizon import HorizonEpisodeTracker
from week03_ant.prior_policy import register_prior_components
from week03_ant.tasks.lanes import ANT_FOOT_REACH, lane_assign
import week03_ant.tasks.prior_v10  # noqa: F401


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Evaluate exactly one 64-second first episode in each vectorized env."""
    started_utc = datetime.now(timezone.utc)
    wall_started = time.monotonic()
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = EPISODE_SECONDS
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.scene.terrain.terrain_generator.seed = args_cli.geometry
    env_cfg.events.lane_layout.func = lane_assign
    env_cfg.events.lane_layout.params = {"family_levels": [("stepping_stones", 4)]}

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_hash = sha256(checkpoint_path)
    env_cfg.log_dir = os.path.dirname(checkpoint)

    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        unwrapped = env.unwrapped
        dt = float(unwrapped.step_dt)
        if int(unwrapped.max_episode_length) != MAX_STEPS:
            raise RuntimeError(
                f"64-second diagnostic requires {MAX_STEPS} steps, got {unwrapped.max_episode_length}"
            )
        if abs(dt * MAX_STEPS - EPISODE_SECONDS) > 1.0e-6:
            raise RuntimeError(f"unexpected control timestep {dt}")

        lane_state = getattr(unwrapped, "_week03_lane_state", None)
        if lane_state is None:
            raise RuntimeError("v10 horizon diagnostic requires initialized LaneState")
        required = (
            "last_distance", "last_out_of_lane", "last_world_exit",
            "out_of_lane", "world_exit", "spawn_x", "odometer",
        )
        if any(not hasattr(lane_state, name) for name in required):
            raise RuntimeError("LaneState lacks first-episode auto-reset evidence")
        family, level = lane_state.family_level()
        stones_index = lane_state.family_names.index("stepping_stones")
        if not bool(torch.all(family == stones_index).item()) or not bool(torch.all(level == 4).item()):
            raise RuntimeError("every horizon environment must use stepping_stones level 4")
        if float(lane_state.difficulties[4]) != 1.0:
            raise RuntimeError("stepping_stones level 4 must be difficulty 1.0")

        geometry = LaneTraversalGeometry(
            tile_length=lane_state.tile_length,
            terrain_tiles=lane_state.terrain_tiles,
            foot_reach=ANT_FOOT_REACH,
        )
        if abs(geometry.one_tile_clearance - 13.1) > 1.0e-6:
            raise RuntimeError("unexpected one-tile threshold")
        if abs(geometry.all_tiles_clearance - 53.1) > 1.0e-6:
            raise RuntimeError("unexpected all-tiles threshold")

        if agent_cfg.class_name != "OnPolicyRunner":
            raise ValueError(f"unsupported runner class: {agent_cfg.class_name}")
        register_prior_components()
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        runner.load(str(checkpoint_path))
        policy = runner.get_inference_policy(device=unwrapped.device)
        observations = env.get_observations()
        tracker = HorizonEpisodeTracker(
            args_cli.num_envs,
            unwrapped.device,
            dt=dt,
            max_steps=MAX_STEPS,
            snapshot_seconds=SNAPSHOT_SECONDS,
            one_tile_threshold=geometry.one_tile_clearance,
            all_tiles_threshold=geometry.all_tiles_clearance,
        )

        for step in range(1, MAX_STEPS + 1):
            with torch.inference_mode():
                actions = policy(observations)
                observations, rewards, dones, _ = env.step(actions)
            root_x = unwrapped.scene["robot"].data.root_pos_w[:, 0]
            current_distance = lane_state.odometer(root_x) - lane_state.spawn_x
            tracker.update(
                step=step,
                rewards=rewards,
                dones=dones,
                reset_terminated=unwrapped.reset_terminated,
                current_distance=current_distance,
                last_distance=lane_state.last_distance,
                current_out_of_lane=lane_state.out_of_lane,
                last_out_of_lane=lane_state.last_out_of_lane,
                current_world_exit=lane_state.world_exit,
                last_world_exit=lane_state.last_world_exit,
            )
            if tracker.complete:
                print(f"[INFO] All {args_cli.num_envs} first episodes completed at step {step}.")
                break

        arrays_and_aggregates = tracker.to_dict()
        finished_utc = datetime.now(timezone.utc)
        result = {
            "schema": "week03_ant_prior_v10_horizon_v1",
            "task": TASK,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "policy_prior_mode": runner.alg.policy.prior_mode,
            "geometry_seed": args_cli.geometry,
            "reset_seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "device": str(unwrapped.device),
            "policy_observation_dimensions": int(observations["policy"].shape[-1]),
            "condition": {
                "terrain_family": "stepping_stones",
                "terrain_level_index": 4,
                "terrain_difficulty": 1.0,
                "episode_length_seconds": EPISODE_SECONDS,
                "max_steps": MAX_STEPS,
                "step_dt_seconds": dt,
                "distance_snapshot_seconds": SNAPSHOT_SECONDS,
                "one_tile_clearance_m": geometry.one_tile_clearance,
                "all_tiles_clearance_m": geometry.all_tiles_clearance,
            },
            "scope": {
                "promotion_metric": False,
                "primary_175_env_assignment_reused": False,
                "predeclared_holdout_pair": [args_cli.geometry, args_cli.seed] in ([66, 40], [67, 41]),
                "note": (
                    "Secondary diagnostic only. Policies share this 10-environment condition, but its reset "
                    "assignment differs from the primary 175-environment benchmark."
                ),
            },
            "success_definition": (
                "Final first-episode distance reaches the threshold with no posture termination, "
                "out-of-lane event, or world exit. Earlier threshold hits are diagnostic only."
            ),
            "snapshot_censoring": "distance_at_16s is null when the first episode ended at or before 16 s.",
            "started_utc": started_utc.isoformat(),
            "finished_utc": finished_utc.isoformat(),
            "wall_seconds": time.monotonic() - wall_started,
            **arrays_and_aggregates,
        }
        serialized = json.dumps(result, indent=2, sort_keys=True)
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        with args_cli.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
        print("WEEK03_PRIOR_HORIZON_JSON_BEGIN")
        print(serialized)
        print("WEEK03_PRIOR_HORIZON_JSON_END")
        print(f"[INFO] Wrote horizon diagnostic: {args_cli.output}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
