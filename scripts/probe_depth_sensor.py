"""Check v6 sensor geometry, finite observations and reset/wrap freshness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Week03-Ant-Depth-Lanes-Eval-v6")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=24)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401
import week03_ant  # noqa: F401
from week03_ant.tasks.depth_v6_cfg import DEPTH_RAYS, terrain_depth
from week03_ant.tasks.lanes import get_lane_state, lane_wrap


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 35
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (35, 346)
        assert bool(torch.isfinite(obs["policy"]).all())
        robot = env.scene["robot"]
        scanner = env.scene["height_scanner"]
        state = get_lane_state(env)
        family, levels = state.family_level()

        def capture():
            features = terrain_depth(env).clone()
            torch.testing.assert_close(scanner.data.pos_w, robot.data.root_pos_w, atol=2e-4, rtol=0)
            assert features.shape == (35, 286) and bool(torch.isfinite(features).all())
            return features

        initial = capture()
        flat = family == state.family_names.index("flat")
        assert bool((initial[flat, DEPTH_RAYS:] == 1).all()), "flat-plane rays must hit"
        torch.testing.assert_close(
            initial[flat, :DEPTH_RAYS],
            (robot.data.root_pos_w[flat, 2:3] + 50.0 - 0.5).expand(-1, DEPTH_RAYS),
        )
        # Translate directly into each family's first rough tile, then observe
        # without advancing physics: this catches stale cached pre-teleport scans.
        pose = robot.data.root_link_pose_w.clone()
        pose[:, 0] = state.x_entrance[state.lane] + 8.0
        pose[:, 1] = state.center_y[state.lane]
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        terrain = capture()
        stone = family == state.family_names.index("stepping_stones")
        assert float(terrain[stone, :DEPTH_RAYS].std(dim=-1).max()) > 0.05, "stone geometry not visible"
        assert float((terrain[stone] - initial[stone]).abs().max()) > 0.1

        # Compare equal post-wrap coordinates with direct placement; no physics
        # step or reset is allowed between teleport and sensor acquisition.
        mesh = state.lane < state.num_mesh_lanes
        pose[:, 0] = state.x_entrance[state.lane] + 0.2
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        entrance = capture()
        pose[mesh, 0] = state.x_exit[state.lane[mesh]] + 0.2
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        capture()  # Deliberately populate a pre-wrap cache.
        lane_wrap(env, None)
        wrapped = capture()
        torch.testing.assert_close(entrance, wrapped, atol=1e-4, rtol=0)
        assert bool((state.wrap_count[mesh] == 1).all())
        obs, _ = env.reset()
        reset = capture()
        assert bool(torch.isfinite(obs["policy"]).all())
        assert bool((state.wrap_count == 0).all())
        assert bool((reset[flat, DEPTH_RAYS:] == 1).all())
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args_cli.output.with_suffix(".npz"),
            initial=initial.cpu().numpy(), terrain=terrain.cpu().numpy(),
            entrance=entrance.cpu().numpy(), wrapped=wrapped.cpu().numpy(), reset=reset.cpu().numpy(),
            family=family.cpu().numpy(), level=levels.cpu().numpy(),
        )
        report = {
            "status": "pass", "task": args_cli.task, "seed": args_cli.seed,
            "observations": 346, "rays": scanner.num_rays, "features": 286,
            "families": state.family_names, "difficulties": state.difficulties,
            "flat_plane_all_valid": True, "teleport_freshness": True,
            "wrap_equivalence_max_error": float((entrance - wrapped).abs().max()),
            "stone_spatial_std_per_level": terrain[stone, :DEPTH_RAYS].std(dim=-1).cpu().tolist(),
            "reset_freshness": True,
        }
        args_cli.output.write_text(json.dumps(report, indent=2) + "\n")
        print("[DEPTH PROBE]", json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
