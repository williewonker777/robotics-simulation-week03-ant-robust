"""Development-only fixed-pose depth routing diagnostic, not a locomotion test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--preset", choices=("cautious", "balanced", "selective"), default="balanced")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error("refusing to overwrite existing diagnostic evidence")
if args.device is None:
    args.device = "cuda:1"
sys.argv = [sys.argv[0]]
launcher = AppLauncher(args)

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import week03_ant.tasks.prior_v10  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from week03_ant.depth_math import encode_height_scan
from week03_ant.foothold_math import GRID_RAYS, foothold_grid
from week03_ant.hybrid_gate import DepthPolicyGate, PRESETS
from week03_ant.tasks.lanes import get_lane_state
from week03_ant.tasks.rough_v5_cfg import FLAT_PLANE_HEIGHT

TASK = "Week03-Ant-Prior-Lanes-Eval-v10"
OFFSETS = (0., 2., 3., 4., 6., 8.)


def strict_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: strict_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict_values(item) for item in value]
    return value


def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=35)
    cfg.seed = 24
    cfg.scene.terrain.terrain_generator.seed = 51
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset()
        robot, scanner = env.scene["robot"], env.scene["height_scanner"]
        lane = get_lane_state(env)
        family, levels = lane.family_level()
        flat = family == lane.family_names.index("flat")
        stones = family == lane.family_names.index("stepping_stones")
        torch.testing.assert_close(scanner.ray_starts[0, :, :2],
                                   foothold_grid(device=env.device).flatten(0, 1), atol=2e-6, rtol=0)
        config = PRESETS[args.preset]
        zero = torch.zeros((35, 8), device=env.device)
        one = torch.ones_like(zero)
        dt = float(env.step_dt)
        pose = robot.data.root_link_pose_w.clone()
        freshness_errors = []

        def capture(offset):
            # Lane metadata is used ONLY to place/report poses, never as gate input.
            pose[:, 0] = lane.x_entrance[lane.lane] + offset
            pose[:, 1] = lane.center_y[lane.lane]
            # Nominal lane surface, not torso-ground averaging over deep gaps.
            pose[:, 2] = lane.ground_z[lane.lane] + .65
            pose[:, 3:] = 0
            pose[:, 3] = 1
            robot.write_root_link_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(torch.zeros((35, 6), device=env.device))
            env.sim.forward()
            env.scene.update(dt)
            scanner.update(0., force_recompute=True)
            error = float((scanner.data.pos_w - robot.data.root_pos_w).abs().max())
            freshness_errors.append(error)
            torch.testing.assert_close(scanner.data.pos_w, robot.data.root_pos_w, atol=2e-4, rtol=0)
            encoded = encode_height_scan(
                robot.data.root_pos_w[:, 2], scanner.data.ray_hits_w[..., 2],
                ray_offset_z=scanner.cfg.offset.pos[2], max_distance=scanner.cfg.max_distance,
                plane_height=FLAT_PLANE_HEIGHT,
            )
            if encoded.shape != (35, GRID_RAYS * 2):
                raise ValueError("unchanged v10 scanner must produce 825 rays")
            height = encoded[:, :GRID_RAYS]
            valid = (encoded[:, GRID_RAYS:] > .5) & (height.abs() < .99)
            return -height - .5, valid

        def cells(result):
            records = []
            for i in range(35):
                record = {"env_id": i, "family": lane.family_names[int(family[i])],
                          "level": int(levels[i]), "root_position": robot.data.root_pos_w[i].tolist()}
                for key, value in result.items():
                    if key != "actions":
                        record[key] = value[i].item()
                records.append(record)
            return records

        snapshots = []
        flat_not_triggered = True
        flat_known = True
        stone_detected = False
        anticipated = False
        scans = {}
        for offset in OFFSETS:
            height, valid = capture(offset)
            scans[offset] = (height.clone(), valid.clone())
            gate = DepthPolicyGate(35, env.device, dt, config)
            for _ in range(config.enter_steps):
                result = gate.step(height, valid, zero, one)
            flat_not_triggered &= not bool(result["target_v10"][flat].any())
            flat_known &= not bool(result["uncertain"][flat].any())
            stone_detected |= bool(result["target_v10"][stones].any())
            if offset < 4:
                anticipated |= bool(result["target_v10"][~flat].any())
            snapshots.append({"offset_from_portal_center": offset, "gate_steps": config.enter_steps,
                              "before_first_tile_boundary": offset < 4, "cells": cells(result)})

        # Fresh captures, no controller reset between phases. Time here is virtual
        # gate time only: no physics steps, locomotion, survival, or performance.
        history_gate = DepthPolicyGate(35, env.device, dt, config)
        hold_steps = math.ceil((config.min_dwell_seconds + config.clear_seconds
                                + config.blend_seconds) / dt) + config.enter_steps + 2
        history = []
        entered = torch.zeros(35, dtype=torch.bool, device=env.device)
        returned = torch.zeros_like(entered)
        for phase, offset in (("initial_flat_portal", 0.), ("rough_pose_6", 6.),
                              ("rough_pose_8", 8.), ("return_flat_portal", 0.)):
            height, valid = capture(offset)
            events = []
            for step in range(hold_steps):
                result = history_gate.step(height, valid, zero, one)
                selected = result["switched"].nonzero().flatten()
                for i in selected.tolist():
                    events.append({"phase_step": step + 1, "phase_seconds": (step + 1) * dt,
                                   "env_id": i, "target_v10": bool(result["target_v10"][i])})
                entered |= result["switched"] & result["target_v10"]
                if phase == "return_flat_portal":
                    returned |= result["switched"] & ~result["target_v10"]
            history.append({"phase": phase, "offset_from_portal_center": offset,
                            "gate_steps": hold_steps, "switch_events": events, "final_cells": cells(result)})
        all_returned = bool(entered.any()) and bool((returned[entered]).all())
        scan_changed = bool(((scans[8.][0] - scans[0.][0]).abs().amax(-1)[stones] > .05).any())
        checks = {"flat_never_triggered": flat_not_triggered, "flat_coverage_known": flat_known,
                  "some_stones_detected": stone_detected, "rough_anticipated_before_boundary": anticipated,
                  "stone_scan_changes_after_teleport": scan_changed,
                  "some_history_entries": bool(entered.any()), "all_entered_return_to_v5": all_returned,
                  "final_target_and_alpha_v5": not bool(result["target_v10"].any()) and bool((result["alpha"] == 0).all())}
        report = {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
                  "created_utc": datetime.now(timezone.utc).isoformat(), "task": TASK,
                  "diagnostic_only": True, "geometry_seed": 51, "reset_seed": 24,
                  "num_envs": 35, "preset": args.preset, "gate_config": asdict(config),
                  "rays": GRID_RAYS, "dt": dt, "physics_steps": 0,
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "placement": "identity yaw; lane nominal ground_z + 0.65m; lateral lane center",
                  "limitations": ["Ideal vertical ray/analytic-plane scan, not RGB-D.",
                                  "Fixed teleports and virtual gate time, not policy actions or performance.",
                                  "Gate receives only encoded heights, validity and dummy endpoint actions.",
                                  "State is retained across history phases; snapshots use fresh gates."],
                  "max_sensor_root_position_error": max(freshness_errors),
                  "snapshots": snapshots, "history": history}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(strict_values(report), stream, indent=2, allow_nan=False)
            stream.write("\n")
        print("HYBRID_SCAN_PROBE", json.dumps({"status": report["status"], "checks": checks}), flush=True)
        if not all(checks.values()):
            raise RuntimeError("development scan diagnostic failed; inspect saved evidence")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        launcher.app.close()
