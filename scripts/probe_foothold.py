"""Check v9 target geometry, dense scan ordering, and reset/wrap freshness."""

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, help="Optional warm-start policy for first-episode availability diagnostics.")
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
launcher = AppLauncher(args)

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import week03_ant.tasks.foothold_v9  # noqa: F401
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.utils import parse_env_cfg
from week03_ant.foothold_math import foothold_grid, foothold_features, endpoint_support_cost
from week03_ant.foothold_policy import FootholdActorCritic
from week03_ant.tasks.foothold_v9_cfg import FootholdHints
from week03_ant.tasks.lanes import get_lane_state, lane_wrap


def main():
    cfg = parse_env_cfg("Week03-Ant-Foothold-Lanes-Eval-v9", device=args.device or "cuda:1", num_envs=35)
    cfg.seed = 24
    env = gym.make("Week03-Ant-Foothold-Lanes-Eval-v9", cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (35, 88) and torch.isfinite(obs["policy"]).all()
        sensor, robot = env.scene["height_scanner"], env.scene["robot"]
        state = get_lane_state(env)
        torch.testing.assert_close(sensor.ray_starts[0, :, :2], foothold_grid(device=env.device).flatten(0, 1), atol=2e-6, rtol=0)
        term = FootholdHints(ObservationTermCfg(func=FootholdHints), env)

        def check():
            feet, proposal, support, available, plateau = term.geometry(env)
            hints = foothold_features(feet, proposal, available)
            assert torch.isfinite(hints).all()
            distance = (proposal[..., :2] - feet[..., :2]).norm(dim=-1)
            assert (distance[available] <= .45001).all()
            assert ((proposal[..., 2] - feet[..., 2]).abs()[available] <= .40001).all()
            cost = endpoint_support_cost(feet, support, available)
            assert torch.isfinite(cost).all() and (cost >= 0).all() and (cost <= 2).all()
            return hints.clone(), available.clone(), cost.clone(), plateau.sum(-1).clone()

        initial, initial_available, _, _ = check()
        initial_feet = term.feet(env).reshape(-1, 4, 3).clone()
        print("INITIAL_FEET", initial_feet[0].tolist(), flush=True)
        assert initial_available.all(), "flat reset must provide all four feet with candidates"
        torch.testing.assert_close(initial, obs["policy"][:, 72:])
        pose = robot.data.root_link_pose_w.clone()
        pose[:, 0] = state.x_entrance[state.lane] + 6.
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        _, terrain_available, terrain_cost, plateau_count = check()
        family, level = state.family_level()
        stones = family == state.family_names.index("stepping_stones")
        hard = stones & (level >= 3)
        assert terrain_available[hard].any(), "no observable hints on hard stones"
        pose[:, 3:] = quat_from_euler_xyz(torch.full((35,), .2, device=env.device),
                                        torch.full((35,), -.15, device=env.device),
                                        torch.linspace(-2., 2., 35, device=env.device))
        pose[:, 0] = state.x_entrance[state.lane] + .2
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        before, _, _, _ = check()
        mesh = state.lane < state.num_mesh_lanes
        pose[mesh, 0] = state.x_exit[state.lane[mesh]] + .2
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        check()
        lane_wrap(env, None)
        wrapped, _, _, _ = check()
        torch.testing.assert_close(before, wrapped, atol=2e-4, rtol=0)
        obs, _ = env.reset()
        after, _, _, _ = check()
        torch.testing.assert_close(obs["policy"][:, 72:], after)
        report = {"status": "pass", "observation_dimensions": 88, "rays": sensor.num_rays,
                  "initial_available_per_foot": initial_available.sum(0).tolist(),
                  "initial_feet_env0": initial_feet[0].tolist(),
                  "stone_available_by_level": terrain_available[stones].tolist(),
                  "stone_support_cost_by_level": terrain_cost[stones].tolist(),
                  "stone_plateau_cells_by_level": plateau_count[stones].tolist(),
                  "reset_freshness": True, "yaw_roll_pitch_finite": True,
                  "wrap_max_hint_error": float((before - wrapped).abs().max())}
        if args.checkpoint:
            model = FootholdActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 8).to(env.device)
            model.load_state_dict(torch.load(args.checkpoint, map_location=env.device, weights_only=False)["model_state_dict"])
            model.eval()
            active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            count = torch.zeros(env.num_envs, device=env.device)
            available_sum = torch.zeros(env.num_envs, 4, device=env.device)
            cost_sum = torch.zeros_like(count)
            for _ in range(args.steps):
                with torch.inference_mode():
                    _, available, cost, _ = check()
                    count += active
                    available_sum += available * active[:, None]
                    cost_sum += cost * active
                    obs, _, terminated, truncated, _ = env.step(model.act_inference(obs))
                    active &= ~(terminated | truncated)
                    assert torch.isfinite(obs["policy"]).all()
            diagnostics = {}
            for index, name in enumerate(state.family_names):
                for difficulty in range(5):
                    cell = (family == index) & (level == difficulty)
                    samples = int(count[cell].sum())
                    diagnostics[f"{name}:{difficulty}"] = {
                        "active_env_steps": samples,
                        "available_per_foot": available_sum[cell].sum(0).tolist(),
                        "support_cost_sum": float(cost_sum[cell].sum()),
                    }
            report["first_episode_availability"] = diagnostics
            report["diagnostic_steps"] = args.steps
            report["diagnostic_note"] = "Training-geometry scan availability; not benchmark or holdout success."
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("FOOTHOLD_PROBE", report)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        launcher.app.close()
