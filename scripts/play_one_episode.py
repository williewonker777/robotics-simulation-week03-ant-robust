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
import hashlib
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
parser.add_argument("--depth-mode", choices=("actual", "zero", "shuffle"),
                    help="Explicit v6 depth ablation; leaves the first 60 observations unchanged.")
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
from week03_ant.evaluation import (
    FirstEpisodeAccumulator,
    LaneTraversalGeometry,
    classify_lane_traversal,
)
from week03_ant.tasks.lanes import ANT_FOOT_REACH


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_episode_values(values: torch.Tensor, applicable: torch.Tensor) -> list[object]:
    values = values.detach().cpu()
    applicable = applicable.detach().cpu()
    return [value.item() if bool(is_applicable) else None for value, is_applicable in zip(values, applicable)]


def lane_breakdown(
    lane_state,
    summary,
    terminated,
    forward_distance,
    out_of_lane,
    world_exit,
    boundary_evidence_available: bool,
    dt: float,
    max_steps: int,
) -> dict:
    """Per terrain family and difficulty level statistics for the v5 lane tasks."""
    family, level = lane_state.family_level()
    speed = forward_distance / (summary.lengths.to(forward_distance.dtype) * dt)
    geometry = LaneTraversalGeometry(
        tile_length=lane_state.tile_length,
        terrain_tiles=lane_state.terrain_tiles,
        foot_reach=ANT_FOOT_REACH,
    )
    plane_family = lane_state.lane >= lane_state.num_mesh_lanes
    crossing = classify_lane_traversal(
        forward_distance,
        terminated,
        plane_family,
        geometry,
        out_of_lane=out_of_lane,
        world_exit=world_exit,
    )

    def crossing_stats(mask: torch.Tensor) -> dict:
        applicable = mask & crossing.applicable
        count = int(applicable.sum().item())
        if count == 0:
            return {
                "terrain_crossing_applicable": False,
                "terrain_crossing_episodes": 0,
                "one_terrain_tile_crossings": None,
                "one_terrain_tile_crossing_rate": None,
                "all_terrain_tiles_crossings": None,
                "all_terrain_tiles_crossing_rate": None,
                "complete_lane_loops": None,
                "distance_only_one_terrain_tile_clearances": None,
                "distance_only_one_terrain_tile_clearance_rate": None,
                "distance_only_all_terrain_tiles_clearances": None,
                "distance_only_all_terrain_tiles_clearance_rate": None,
                "distance_only_complete_lane_loops": None,
                "out_of_lane_exits": None,
                "world_exits": None,
            }

        def count_rate(values: torch.Tensor) -> tuple[int, float]:
            value_count = int(values[applicable].sum().item())
            return value_count, value_count / count

        one_count, one_rate = count_rate(crossing.cleared_one_tile)
        all_count, all_rate = count_rate(crossing.cleared_all_tiles)
        distance_one_count, distance_one_rate = count_rate(crossing.distance_cleared_one_tile)
        distance_all_count, distance_all_rate = count_rate(crossing.distance_cleared_all_tiles)
        return {
            "terrain_crossing_applicable": count > 0,
            "terrain_crossing_episodes": count,
            "one_terrain_tile_crossings": one_count,
            "one_terrain_tile_crossing_rate": one_rate,
            "all_terrain_tiles_crossings": all_count,
            "all_terrain_tiles_crossing_rate": all_rate,
            "complete_lane_loops": int(crossing.complete_lane_loops[applicable].sum().item()),
            "distance_only_one_terrain_tile_clearances": distance_one_count,
            "distance_only_one_terrain_tile_clearance_rate": distance_one_rate,
            "distance_only_all_terrain_tiles_clearances": distance_all_count,
            "distance_only_all_terrain_tiles_clearance_rate": distance_all_rate,
            "distance_only_complete_lane_loops": int(
                crossing.distance_complete_lane_loops[applicable].sum().item()
            ),
            "out_of_lane_exits": int(out_of_lane[applicable].sum().item()) if boundary_evidence_available else None,
            "world_exits": int(world_exit[applicable].sum().item()) if boundary_evidence_available else None,
        }

    all_episodes = torch.ones_like(terminated, dtype=torch.bool)
    output = {
        "lane_family_names": lane_state.family_names,
        "lane_difficulties": lane_state.difficulties,
        "lane_family_indices": family.detach().cpu().tolist(),
        "lane_level_indices": level.detach().cpu().tolist(),
        "forward_distance": forward_distance.detach().cpu().tolist(),
        "forward_distance_mean": float(forward_distance.mean().item()),
        "forward_speed_mean": float(speed.mean().item()),
        "fall_rate": float(terminated.float().mean().item()),
        "traversal_geometry": {
            "tile_length_m": geometry.tile_length,
            "terrain_tiles": geometry.terrain_tiles,
            "entrance_portal_half_length_m": geometry.entrance_half_length,
            "ant_foot_reach_m": geometry.foot_reach,
            "one_terrain_tile_clearance_m": geometry.one_tile_clearance,
            "all_terrain_tiles_clearance_m": geometry.all_tiles_clearance,
            "lane_period_m": geometry.lane_period,
            "lane_containment_foot_margin_m": lane_state.containment_foot_margin,
            "lane_containment_half_width_m": lane_state.lane_half_width - lane_state.containment_foot_margin,
        },
        "episode_terrain_crossing_applicable": crossing.applicable.detach().cpu().tolist(),
        "episode_distance_cleared_one_terrain_tile": _optional_episode_values(
            crossing.distance_cleared_one_tile, crossing.applicable
        ),
        "episode_distance_cleared_all_terrain_tiles": _optional_episode_values(
            crossing.distance_cleared_all_tiles, crossing.applicable
        ),
        "episode_distance_complete_lane_loops": _optional_episode_values(
            crossing.distance_complete_lane_loops, crossing.applicable
        ),
        "episode_cleared_one_terrain_tile": _optional_episode_values(
            crossing.cleared_one_tile, crossing.applicable
        ),
        "episode_cleared_all_terrain_tiles": _optional_episode_values(
            crossing.cleared_all_tiles, crossing.applicable
        ),
        "episode_complete_lane_loops": _optional_episode_values(crossing.complete_lane_loops, crossing.applicable),
        "episode_out_of_lane": out_of_lane.detach().cpu().tolist(),
        "episode_world_exit": world_exit.detach().cpu().tolist(),
        "boundary_evidence_available": boundary_evidence_available,
        "terrain_crossing_definition": (
            "Geometry threshold met with no terminal condition, no out-of-lane exit, and no world exit; "
            "plane-family episodes are inapplicable. Distance-only metrics are diagnostic, not successes."
        ),
        "family_breakdown": {},
    }
    output.update(crossing_stats(all_episodes))

    def stats(mask: torch.Tensor) -> dict:
        returns = summary.returns[mask]
        result = {
            "episodes": int(mask.sum().item()),
            "episode_return_mean": float(returns.mean().item()),
            "episode_return_std": float(returns.std(unbiased=False).item()),
            "episode_length_mean": float(summary.lengths[mask].float().mean().item()),
            "full_length_episodes": int((summary.lengths[mask] >= max_steps).sum().item()),
            "falls": int(terminated[mask].sum().item()),
            "fall_rate": float(terminated[mask].float().mean().item()),
            "forward_distance_mean": float(forward_distance[mask].mean().item()),
            "forward_speed_mean": float(speed[mask].mean().item()),
        }
        result.update(crossing_stats(mask))
        return result

    for family_index, family_name in enumerate(lane_state.family_names):
        family_mask = family == family_index
        if not bool(family_mask.any()):
            continue
        entry = stats(family_mask)
        entry["levels"] = {}
        for level_index, difficulty in enumerate(lane_state.difficulties):
            mask = family_mask & (level == level_index)
            if bool(mask.any()):
                entry["levels"][f"{level_index}:{difficulty:g}"] = stats(mask)
        output["family_breakdown"][family_name] = entry
    return output


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
    depth_term = getattr(env_cfg.observations.policy, "terrain_depth", None)
    if args_cli.depth_mode is not None:
        if depth_term is None:
            raise ValueError("--depth-mode requires a depth-aware task")
        depth_term.params["mode"] = args_cli.depth_mode

    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required for reproducible evaluation")
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    checkpoint_sha256 = _sha256(checkpoint)
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
    if agent_cfg.policy.class_name == "FootMapActorCritic":
        from week03_ant.footmap_policy import register_footmap_policy
        register_footmap_policy()
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
    # Why each first episode ended, and (lane tasks) how far it travelled.
    terminated = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    forward_distance = torch.full((args_cli.num_envs,), float("nan"), device=env.unwrapped.device)
    out_of_lane = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    world_exit = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    termination_reasons: list[str | None] = [None] * args_cli.num_envs
    lane_state = getattr(env.unwrapped, "_week03_lane_state", None)
    boundary_evidence_available = lane_state is not None and all(
        hasattr(lane_state, name) for name in ("last_out_of_lane", "last_world_exit")
    )

    try:
        if lane_state is not None and not boundary_evidence_available:
            raise RuntimeError("lane traversal evaluation requires per-episode boundary evidence")
        for step in range(1, args_cli.max_steps + 1):
            started = time.time()
            with torch.inference_mode():
                actions = policy(observations)
                observations, rewards, dones, _ = env.step(actions)
            newly_done = dones.reshape(-1).to(torch.bool) & ~accumulator.finished
            if bool(newly_done.any()):
                reset_terminated = env.unwrapped.reset_terminated.reshape(-1).to(torch.bool)
                terminated[newly_done] = reset_terminated[newly_done]
                lane_state = getattr(env.unwrapped, "_week03_lane_state", None)
                if lane_state is not None:
                    forward_distance[newly_done] = lane_state.last_distance[newly_done]
                    out_of_lane[newly_done] = lane_state.last_out_of_lane[newly_done]
                    world_exit[newly_done] = lane_state.last_world_exit[newly_done]

                # In this Isaac Lab version get_term() retains the manager's
                # last triggered label across reset.  If multiple terms trigger
                # in one compute(), only the last term in manager order remains.
                manager = env.unwrapped.termination_manager
                pending_ids = newly_done.nonzero().flatten().tolist()
                labels_by_id: dict[int, str] = {}
                for term_name in manager.active_terms:
                    term_mask = manager.get_term(term_name).reshape(-1).to(torch.bool) & newly_done
                    for env_id in term_mask.nonzero().flatten().tolist():
                        labels_by_id[env_id] = term_name
                for env_id in pending_ids:
                    termination_reasons[env_id] = labels_by_id.get(
                        env_id, "terminated" if bool(reset_terminated[env_id]) else "time_out"
                    )
            accumulator.update(rewards, dones)
            if accumulator.complete:
                print(f"[INFO] All {args_cli.num_envs} episodes completed at step {step}.")
                break
            if args_cli.real_time:
                sleep_time = dt - (time.time() - started)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        summary = accumulator.summary()
        result = summary.to_dict()
        result.update(
            {
                "task": args_cli.task,
                "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "seed": args_cli.seed,
                "num_envs": args_cli.num_envs,
                "max_steps": args_cli.max_steps,
                "step_dt_seconds": float(dt),
                "policy_observation_dimensions": int(observations["policy"].shape[-1]),
            }
        )
        if depth_term is not None:
            scanner = env_cfg.scene.height_scanner
            result["depth_observation"] = {
                "kind": "yaw-stabilized vertical raycast heights + validity (not rendered RGB-D)",
                "mode": depth_term.params.get("mode", "actual"),
                "noise": depth_term.params["noise"],
                "rays": int(env.unwrapped.scene["height_scanner"].num_rays),
                "grid_size_m": list(scanner.pattern_cfg.size),
                "grid_resolution_m": scanner.pattern_cfg.resolution,
                "offset_m": list(scanner.offset.pos),
                "max_range_m": scanner.max_distance,
                "flat_plane": "nearest downward intersection with existing plane at z=-50m",
            }
        if agent_cfg.policy.class_name == "FootMapActorCritic":
            result["footmap_policy"] = {"input_mode": runner.alg.policy.input_mode,
                                       "foot_order": ["front_left", "front_right", "back_left", "back_right"],
                                       "representation": "four distal-center Gaussian maps; no history"}
        result["terminated_episodes"] = int(terminated.sum().item())
        result["episode_terminated"] = terminated.detach().cpu().tolist()
        result["episode_termination_reason"] = termination_reasons
        result["termination_reason_semantics"] = (
            "Terminal manager label retained by get_term() after reset; when simultaneous terms trigger, "
            "this Isaac Lab manager retains only the last triggered term, not every simultaneous flag."
        )
        terrain = getattr(env.unwrapped.scene, "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        if generator_cfg is not None:
            result["terrain_generator_seed"] = generator_cfg.seed
        lane_state = getattr(env.unwrapped, "_week03_lane_state", None)
        if lane_state is not None:
            result.update(
                lane_breakdown(
                    lane_state,
                    summary,
                    terminated,
                    forward_distance,
                    out_of_lane,
                    world_exit,
                    boundary_evidence_available,
                    dt,
                    args_cli.max_steps,
                )
            )
        elif terrain_types is not None and generator_cfg is not None:
            type_names = list(generator_cfg.sub_terrains)
            result["terrain_type_indices"] = terrain_types.detach().cpu().tolist()
            result["terrain_levels"] = terrain_levels.detach().cpu().tolist()
            result["terrain_type_names"] = type_names
            result["terrain_breakdown"] = {}
            for type_index, type_name in enumerate(type_names):
                mask = terrain_types == type_index
                type_returns = summary.returns[mask]
                type_lengths = summary.lengths[mask]
                if type_returns.numel() == 0:
                    continue
                result["terrain_breakdown"][type_name] = {
                    "episodes": int(type_returns.numel()),
                    "episode_return_mean": float(type_returns.mean().item()),
                    "episode_return_std": float(type_returns.std(unbiased=False).item()),
                    "episode_length_mean": float(type_lengths.float().mean().item()),
                    "full_length_episodes": int((type_lengths >= args_cli.max_steps).sum().item()),
                }
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
