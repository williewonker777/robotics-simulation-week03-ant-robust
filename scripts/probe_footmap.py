"""Simulator checks for v7 spatial alignment, four foot centers, and teleport freshness."""

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
launcher = AppLauncher(args)

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import week03_ant  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, yaw_quat
from isaaclab_tasks.utils import parse_env_cfg
from week03_ant.footmap_math import scan_grid, foot_position_maps, FOOT_NAMES, TIP_OFFSETS
from week03_ant.tasks.footmap_v7_cfg import spatial_terrain_depth, distal_foot_positions
from week03_ant.tasks.lanes import get_lane_state, lane_wrap
from isaaclab.managers import ObservationTermCfg


def main():
    cfg = parse_env_cfg("Week03-Ant-FootMap-Lanes-Eval-v7", device=args.device or "cuda:0", num_envs=35)
    cfg.seed = 24
    env = gym.make("Week03-Ant-FootMap-Lanes-Eval-v7", cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (35, 514) and torch.isfinite(obs["policy"]).all()
        robot, sensor = env.scene["robot"], env.scene["height_scanner"]
        state = get_lane_state(env)
        term = distal_foot_positions(ObservationTermCfg(func=distal_foot_positions), env)
        grid = scan_grid(device=env.device).flatten(0, 1)
        torch.testing.assert_close(sensor.ray_starts[0, :, :2], grid, atol=1e-6, rtol=0)
        ids, _ = robot.find_bodies(list(FOOT_NAMES), preserve_order=True)
        offsets = torch.tensor(TIP_OFFSETS, device=env.device).expand(35, -1, -1)

        def check():
            scan, feet = spatial_terrain_depth(env).clone(), term(env).reshape(35, 4, 3).clone()
            assert torch.isfinite(scan).all() and torch.isfinite(feet).all()
            tips = robot.data.body_link_pos_w[:, ids] + quat_apply(robot.data.body_link_quat_w[:, ids], offsets)
            reconstructed = quat_apply(yaw_quat(robot.data.root_quat_w)[:, None].expand(-1, 4, -1), feet)
            reconstructed += robot.data.root_pos_w[:, None]
            torch.testing.assert_close(reconstructed, tips, atol=3e-5, rtol=0)
            return scan, feet

        initial_scan, initial_feet = check()
        family, _ = state.family_level()
        flat = family == state.family_names.index("flat")
        assert initial_scan[flat, 221:].eq(1).all()
        pose = robot.data.root_link_pose_w.clone()
        pose[:, 3:] = quat_from_euler_xyz(torch.full((35,), .2, device=env.device),
                                        torch.full((35,), -.15, device=env.device),
                                        torch.linspace(-2., 2., 35, device=env.device))
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        check()  # Nonzero roll/pitch/yaw must still match yaw-stabilized scan.
        pose[:, 0] = state.x_entrance[state.lane] + .2
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        entrance_scan, entrance_feet = check()
        mesh = state.lane < state.num_mesh_lanes
        pose[mesh, 0] = state.x_exit[state.lane[mesh]] + .2
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        check()
        lane_wrap(env, None)
        wrapped_scan, wrapped_feet = check()
        torch.testing.assert_close(entrance_scan, wrapped_scan, atol=1e-4, rtol=0)
        torch.testing.assert_close(entrance_feet, wrapped_feet, atol=3e-5, rtol=0)
        obs, _ = env.reset()
        reset_scan, reset_feet = check()
        torch.testing.assert_close(obs["policy"][:, 60:502], reset_scan)
        torch.testing.assert_close(obs["policy"][:, 502:], reset_feet.flatten(1))
        maps = foot_position_maps(initial_feet[..., :2])
        report = {
            "status": "pass", "observation_dimensions": 514, "ray_count": sensor.num_rays,
            "grid_order": "13 y rows x 17 x columns; xy flatten", "feet": list(FOOT_NAMES),
            "initial_foot_map_visible_per_foot": (maps.flatten(2).amax(2) > 0).sum(0).tolist(),
            "yaw_roll_pitch_alignment": True, "reset_freshness": True,
            "wrap_scan_max_error": float((entrance_scan - wrapped_scan).abs().max()),
            "wrap_feet_max_error": float((entrance_feet - wrapped_feet).abs().max()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("FOOTMAP_PROBE", report)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        launcher.app.close()
