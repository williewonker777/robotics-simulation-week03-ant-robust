"""Matched first-episode evaluation of frozen v5, v10 and depth-gated v11."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

import cli_args


ROOT = Path(__file__).resolve().parents[1]
TASK = "Week03-Ant-Prior-Lanes-Eval-v10"
PARENT = ROOT / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
PARENT_SHA = "889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--geometry", required=True, type=int)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--mode", required=True, choices=("v5", "v10", "hybrid"))
parser.add_argument("--preset", default="balanced", choices=("cautious", "balanced", "selective"))
parser.add_argument("--policy-seed", type=int, choices=(42, 43, 44), default=42)
parser.add_argument("--seconds", type=int, choices=(16, 64), default=16)
parser.add_argument("--scenario", choices=("mixed", "stones"), default="mixed")
parser.add_argument("--phase", choices=("smoke", "development", "holdout", "video"), required=True)
parser.add_argument("--num_envs", type=int, default=175)
parser.add_argument("--record", type=Path)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint or args_cli.num_envs <= 0:
    parser.error("positive num_envs and --checkpoint are required")
if args_cli.geometry < 0 or args_cli.seed < 0:
    parser.error("nonnegative geometry/reset seeds required")
if args_cli.output.exists() or (args_cli.record and args_cli.record.exists()):
    parser.error("refusing to overwrite existing evidence")
if args_cli.record and args_cli.phase != "video":
    parser.error("rendered cases are separate qualitative videos, not benchmark data")
if hydra_args and any(not a.startswith("--kit_args") for a in hydra_args):
    parser.error("environment/PPO overrides are not permitted in the matched evaluator")
if args_cli.device is None:
    args_cli.device = "cuda:1"
if args_cli.record:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import week03_ant.tasks.prior_v10  # noqa: F401
from week03_ant.depth_math import encode_height_scan
from week03_ant.evaluation import LaneTraversalGeometry
from week03_ant.foothold_math import GRID_RAYS
from week03_ant.horizon import HorizonEpisodeTracker
from week03_ant.hybrid_gate import DepthPolicyGate, PRESETS
from week03_ant.hybrid_telemetry import HybridTelemetry
from week03_ant.prior_policy import register_prior_components
from week03_ant.tasks.lanes import ANT_FOOT_REACH, lane_assign
from week03_ant.tasks.rough_v5_cfg import FLAT_PLANE_HEIGHT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def scan(unwrapped):
    sensor = unwrapped.scene["height_scanner"]
    sensor.update(0.0, force_recompute=True)
    encoded = encode_height_scan(
        unwrapped.scene["robot"].data.root_pos_w[:, 2], sensor.data.ray_hits_w[..., 2],
        ray_offset_z=sensor.cfg.offset.pos[2], max_distance=sensor.cfg.max_distance,
        plane_height=FLAT_PLANE_HEIGHT,
    )
    if encoded.shape[-1] != 2 * GRID_RAYS:
        raise ValueError("v11 requires the unchanged 825-ray v10 scanner")
    height = encoded[:, :GRID_RAYS]
    valid = (encoded[:, GRID_RAYS:] > .5) & (height.abs() < .99)
    return -height - .5, valid


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    started = datetime.now(timezone.utc)
    wall_start = time.monotonic()
    checkpoint = Path(args_cli.checkpoint).resolve()
    archived = json.loads((ROOT / "artifacts/terrain_demo/prior_v10/frozen.json").read_text())
    expected = next(r for r in archived["runs"]
                    if r["group"] == "anchored" and r["training_seed"] == args_cli.policy_seed)
    if checkpoint != ROOT / expected["checkpoint"] or sha(checkpoint) != expected["sha256"]:
        raise ValueError("not the predeclared immutable anchored checkpoint")
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("immutable v5 checkpoint changed")
    if args_cli.mode == "v5" and args_cli.policy_seed != 42:
        raise ValueError("evaluate the identical v5 reference once, not once per student seed")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = float(args_cli.seconds)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.scene.terrain.terrain_generator.seed = args_cli.geometry
    env_cfg.log_dir = str(checkpoint.parent)
    if args_cli.scenario == "stones":
        env_cfg.events.lane_layout.func = lane_assign
        env_cfg.events.lane_layout.params = {"family_levels": [("stepping_stones", 4)]}
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.record else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    writer = None
    try:
        unwrapped = env.unwrapped
        dt = float(unwrapped.step_dt)
        max_steps = int(unwrapped.max_episode_length)
        if max_steps != args_cli.seconds * 60 or abs(dt - 1 / 60) > 1.e-8:
            raise ValueError("unexpected benchmark control time or episode length")
        lane = unwrapped._week03_lane_state
        geometry = LaneTraversalGeometry(lane.tile_length, lane.terrain_tiles, ANT_FOOT_REACH)
        if geometry.one_tile_clearance != 13.1 or geometry.all_tiles_clearance != 53.1:
            raise ValueError("strict benchmark distances changed")
        family, level = lane.family_level()
        family = family.clone()
        level = level.clone()
        register_prior_components()
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_optimizer=False)
        student = runner.get_inference_policy(device=unwrapped.device)
        policy = runner.alg.policy
        if policy.prior_mode != "anchored" or policy.input_mode != "targets":
            raise ValueError("v11 uses the frozen anchored targets policy only")
        parent = torch.load(PARENT, map_location=unwrapped.device, weights_only=False)["model_state_dict"]
        for key, value in policy.teacher.state_dict().items():
            if not torch.equal(value, parent[f"actor.{key}"]):
                raise ValueError(f"v5 teacher identity mismatch: {key}")
        policy.requires_grad_(False)
        observations = env.get_observations()
        if observations["policy"].shape != (args_cli.num_envs, 88):
            raise ValueError("unchanged 88D observation contract required")
        initial = {
            "root_state": tensor_sha(unwrapped.scene["robot"].data.root_state_w),
            "joint_pos": tensor_sha(unwrapped.scene["robot"].data.joint_pos),
            "joint_vel": tensor_sha(unwrapped.scene["robot"].data.joint_vel),
            "observations": tensor_sha(observations["policy"]),
        }
        config = PRESETS[args_cli.preset]
        gate = DepthPolicyGate(args_cli.num_envs, unwrapped.device, dt=dt, config=config)
        telemetry = HybridTelemetry(args_cli.num_envs, unwrapped.device, dt=dt)
        snapshot_seconds = 8.0 if args_cli.seconds == 16 else 16.0
        tracker = HorizonEpisodeTracker(
            args_cli.num_envs, unwrapped.device, dt=dt, max_steps=max_steps,
            snapshot_seconds=snapshot_seconds, one_tile_threshold=geometry.one_tile_clearance,
            all_tiles_threshold=geometry.all_tiles_clearance,
        )
        video_resets = 0
        if args_cli.record:
            import imageio.v2 as imageio
            import numpy as np
            from PIL import Image, ImageDraw
            args_cli.record.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args_cli.record), fps=30, codec="libx264", quality=7)

        for step in range(1, max_steps + 1):
            active = (~tracker.finished).clone()
            with torch.inference_mode():
                old_action = policy.teacher_mean(observations)
                new_action = student(observations)
                height, valid = scan(unwrapped)
                choice = gate.step(height, valid, old_action, new_action, mode=args_cli.mode)
                telemetry.before_step(step, active, choice, old_action, new_action)
                observations, rewards, dones, _ = env.step(choice["actions"])
            done = dones.reshape(-1).bool()
            terminated = unwrapped.reset_terminated.reshape(-1).bool()
            current_distance = lane.odometer(unwrapped.scene["robot"].data.root_pos_w[:, 0]) - lane.spawn_x
            tracker.update(
                step=step, rewards=rewards, dones=done, reset_terminated=terminated,
                current_distance=current_distance, last_distance=lane.last_distance,
                current_out_of_lane=lane.out_of_lane, last_out_of_lane=lane.last_out_of_lane,
                current_world_exit=lane.world_exit, last_world_exit=lane.last_world_exit,
            )
            telemetry.after_step(step, active, done, terminated)
            gate.reset(done)
            video_resets += int(done[0].item())
            if writer is not None and step % 2 == 0:
                pos = unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                unwrapped.sim.set_camera_view(pos + np.array([-4., -4., 2.5]), pos)
                frame = unwrapped.render()
                if frame is None:
                    raise RuntimeError("camera did not return a frame")
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 54), fill="black")
                draw.text((8, 6), f"{args_cli.mode} | t={step*dt:.2f}s | v10 alpha={float(choice['alpha'][0]):.2f} | resets={video_resets}", fill="white")
                draw.text((8, 30), f"current episode {float(current_distance[0]):.1f}m | uncut qualitative replay; resets are NOT benchmark successes", fill="white")
                writer.append_data(np.asarray(image))
            if tracker.complete:
                break

        data = tracker.to_dict()
        data["distance_at_snapshot_m"] = data.pop("distance_at_16s")
        data["episode_full_horizon_survival"] = data.pop("episode_full_64s_survival")
        data["aggregates"]["full_horizon_survivals"] = data["aggregates"].pop("full_64s_survivals")
        data.update(telemetry.to_dict(tracker.steps))
        result = {
            "schema": "week03_ant_depth_switch_v11_v1", "task": TASK,
            "mode": args_cli.mode, "preset": args_cli.preset, "gate_config": asdict(config),
            "policy_seed": None if args_cli.mode == "v5" else args_cli.policy_seed,
            "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint),
            "v5_sha256": PARENT_SHA, "teacher_tensor_identity_verified": True,
            "phase": args_cli.phase, "scenario": args_cli.scenario,
            "geometry_seed": args_cli.geometry, "reset_seed": args_cli.seed,
            "num_envs": args_cli.num_envs, "initial_state_sha256": initial,
            "condition": {
                "seconds": args_cli.seconds, "max_steps": max_steps, "dt": dt,
                "snapshot_seconds": snapshot_seconds, "one_threshold": geometry.one_tile_clearance,
                "six_threshold": geometry.all_tiles_clearance, "footprint_margin": ANT_FOOT_REACH,
                "observations": 88, "v5_observations": 60, "actions": 8,
            },
            "family_names": lane.family_names, "difficulties": lane.difficulties,
            "family_indices": family.cpu().tolist(), "level_indices": level.cpu().tolist(),
            "routing_inputs": "Only current ideal825-ray heights/validity; no terrain labels, maps or future state.",
            "unknown_handling": "Hold requested policy, clear confirmation counters; an existing fade may continue. Not a safety guarantee.",
            "switch_fall_interpretation": "Temporal association within0.5s, not proof of causation.",
            "record": str(args_cli.record.relative_to(ROOT)) if args_cli.record else None,
            "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.monotonic() - wall_start, **data,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        with args_cli.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"[v11] {args_cli.mode} {args_cli.scenario} {args_cli.seconds}s: {result['aggregates']}")
        print(f"[v11] wrote {args_cli.output}")
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
