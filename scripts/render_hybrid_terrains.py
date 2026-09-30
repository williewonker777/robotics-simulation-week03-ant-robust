"""Uncut qualitative v5/v10/hybrid replays on five other rough terrain families.

This is separate from the frozen v11 benchmark: one environment per family,
fixed difficulty 1.0, and all resets remain visible for the full 16 seconds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TASK = "Week03-Ant-Prior-Lanes-Eval-v10"
FAMILIES = ("rough", "slope", "stairs", "waves", "obstacles")
PARENT = ROOT / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
PARENT_SHA = "889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e"
CHECKPOINT_SHA = "fa1ec87ef2b89b698f250d91452685cef0a7f76e340e790650116138aee875c1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def caption(mode, family, step, dt, alpha, resets, distance):
    return (
        f"{mode} | {family} difficulty=1.0 | t={step * dt:.2f}s | applied v10 alpha={alpha:.2f} | resets={resets}",
        f"current episode {distance:.1f}m | qualitative replay; resets are NOT benchmark successes",
    )


def verify_sources(frozen):
    for name, expected in frozen["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    return dict(frozen["source_sha256"])


def launch():
    # Imports must remain after entry so helper tests do not launch Isaac Sim.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("v5", "v10", "hybrid"))
    parser.add_argument("--output-dir", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory must be new; refusing to overwrite evidence")
    args.device = args.device or "cuda:1"
    args.enable_cameras = True
    frozen = json.loads((ROOT / "artifacts/terrain_demo/hybrid_v11/frozen.json").read_text())
    sources = verify_sources(frozen)
    checkpoint = ROOT / frozen["checkpoints"]["42"]["checkpoint"]
    if sha(checkpoint) != CHECKPOINT_SHA or sha(PARENT) != PARENT_SHA:
        raise ValueError("immutable model checksum mismatch")
    if frozen["preset"] != "cautious":
        raise ValueError("expected the previously frozen cautious gate")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.argv = [sys.argv[0]]
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        import gymnasium as gym
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image, ImageDraw
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.hydra import hydra_task_config
        import isaaclab_tasks  # noqa: F401
        import week03_ant.tasks.prior_v10  # noqa: F401
        from week03_ant.depth_math import encode_height_scan
        from week03_ant.foothold_math import GRID_RAYS
        from week03_ant.hybrid_gate import DepthPolicyGate, PRESETS
        from week03_ant.prior_policy import register_prior_components
        from week03_ant.tasks.lanes import lane_assign
        from week03_ant.tasks.rough_v5_cfg import FLAT_PLANE_HEIGHT

        @hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
        def run(env_cfg, agent_cfg):
            started = datetime.now(timezone.utc).isoformat()
            env_cfg.scene.num_envs = len(FAMILIES)
            env_cfg.episode_length_s = 16.0
            env_cfg.seed = 42
            env_cfg.sim.device = args.device
            agent_cfg.seed = 42
            agent_cfg.device = args.device
            agent_cfg.load_checkpoint = str(checkpoint)
            env_cfg.scene.terrain.terrain_generator.seed = 68
            env_cfg.log_dir = str(checkpoint.parent)
            env_cfg.events.lane_layout.func = lane_assign
            env_cfg.events.lane_layout.params = {"family_levels": [(name, 4) for name in FAMILIES]}
            env_cfg.viewer.resolution = (1280, 720)
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.asset_name = "robot"
            env_cfg.viewer.env_index = 0
            env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg, render_mode="rgb_array"),
                                    clip_actions=agent_cfg.clip_actions)
            writers = []
            try:
                unwrapped = env.unwrapped
                dt = float(unwrapped.step_dt)
                if abs(dt - 1 / 60) > 1.e-8 or int(unwrapped.max_episode_length) != 960:
                    raise ValueError("expected 60Hz and 16-second episodes")
                lane = unwrapped._week03_lane_state
                family, level = lane.family_level()
                assignment = [{"env_id": i, "family": lane.family_names[int(family[i])],
                               "family_index": int(family[i]), "level_index": int(level[i]),
                               "difficulty": float(lane.difficulties[int(level[i])])}
                              for i in range(len(FAMILIES))]
                if any(row["family"] != FAMILIES[i] or row["level_index"] != 4
                       or row["difficulty"] != 1.0 for i, row in enumerate(assignment)):
                    raise ValueError("actual lanes do not match the fixed five-family assignment")
                register_prior_components()
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                runner.load(str(checkpoint), load_optimizer=False)
                student = runner.get_inference_policy(device=unwrapped.device)
                policy = runner.alg.policy
                if policy.prior_mode != "anchored" or policy.input_mode != "targets":
                    raise ValueError("expected anchored targets checkpoint")
                parent = torch.load(PARENT, map_location=unwrapped.device, weights_only=False)["model_state_dict"]
                for key, value in policy.teacher.state_dict().items():
                    if not torch.equal(value, parent[f"actor.{key}"]):
                        raise ValueError(f"v5 teacher tensor mismatch: {key}")
                policy.requires_grad_(False)
                policy.eval()
                observations = env.get_observations()
                if observations["policy"].shape != (5, 88):
                    raise ValueError("expected unchanged 88D policy input")
                robot = unwrapped.scene["robot"]
                initial = {"root_state": tensor_sha(robot.data.root_state_w),
                           "joint_pos": tensor_sha(robot.data.joint_pos),
                           "joint_vel": tensor_sha(robot.data.joint_vel),
                           "observations": tensor_sha(observations["policy"])}
                config = PRESETS["cautious"]
                if asdict(config) != frozen["gate_config"]:
                    raise ValueError("gate configuration differs from frozen experiment")
                gate = DepthPolicyGate(5, unwrapped.device, dt=dt, config=config)
                files = [args.output_dir / f"{name}.mp4" for name in FAMILIES]
                for path in files:
                    writers.append(imageio.get_writer(str(path), fps=30, codec="libx264", quality=7))
                resets = [0] * 5
                first = [None] * 5
                events = [[] for _ in FAMILIES]
                frames = 0
                # Warm up the shared render product before per-environment camera captures.
                unwrapped.render()
                for step in range(1, 961):
                    with torch.inference_mode():
                        old_action = policy.teacher_mean(observations)
                        new_action = student(observations)
                        sensor = unwrapped.scene["height_scanner"]
                        sensor.update(0.0, force_recompute=True)
                        encoded = encode_height_scan(robot.data.root_pos_w[:, 2], sensor.data.ray_hits_w[..., 2],
                                                     ray_offset_z=sensor.cfg.offset.pos[2],
                                                     max_distance=sensor.cfg.max_distance,
                                                     plane_height=FLAT_PLANE_HEIGHT)
                        if encoded.shape != (5, GRID_RAYS * 2) or GRID_RAYS != 825:
                            raise ValueError("expected unchanged 825-ray scanner")
                        heights = encoded[:, :GRID_RAYS]
                        valid = (encoded[:, GRID_RAYS:] > .5) & (heights.abs() < .99)
                        choice = gate.step(-heights - .5, valid, old_action, new_action, mode=args.mode)
                        observations, _, dones, _ = env.step(choice["actions"])
                    done = dones.reshape(-1).bool()
                    current_distance = lane.odometer(robot.data.root_pos_w[:, 0]) - lane.spawn_x
                    for i in range(5):
                        if bool(done[i]):
                            event = {"step": step, "seconds": step * dt,
                                     "terminated": bool(unwrapped.reset_terminated[i]),
                                     "distance_m": float(lane.last_distance[i]),
                                     "out_of_lane": bool(lane.last_out_of_lane[i]),
                                     "world_exit": bool(lane.last_world_exit[i])}
                            events[i].append(event)
                            if first[i] is None:
                                first[i] = event.copy()
                            resets[i] += 1
                    gate.reset(done)
                    if step % 2 == 0:
                        # Rendering does not advance physics; verify this for every five-view batch.
                        state_before = robot.data.root_state_w.clone()
                        physics_step = unwrapped.sim.current_time_step_index
                        for i, writer in enumerate(writers):
                            pos = robot.data.root_pos_w[i].detach().cpu().numpy()
                            unwrapped.sim.set_camera_view(pos + np.array([-4., -4., 2.5]), pos)
                            unwrapped.sim.render()
                            frame = unwrapped.render(recompute=True)
                            if frame is None or frame.shape != (720, 1280, 3):
                                raise RuntimeError("camera did not return the expected RGB frame")
                            image = Image.fromarray(frame)
                            draw = ImageDraw.Draw(image)
                            draw.rectangle((0, 0, image.width, 54), fill="black")
                            lines = caption(args.mode, FAMILIES[i], step, dt, float(choice["alpha"][i]),
                                            resets[i], float(current_distance[i]))
                            for line, y in zip(lines, (6, 30)):
                                draw.text((8, y), line, fill="white")
                            writer.append_data(np.asarray(image))
                        if physics_step != unwrapped.sim.current_time_step_index or not torch.equal(state_before, robot.data.root_state_w):
                            raise RuntimeError("camera rendering unexpectedly advanced physics")
                        frames += 1
                for i in range(5):
                    if first[i] is None:
                        first[i] = {"step": 960, "seconds": 16.0, "terminated": False,
                                    "distance_m": float(current_distance[i]), "censored": True}
                for writer in writers:
                    writer.close()
                writers.clear()
                verify_sources(frozen)
                manifest = {"schema": "week03_ant_other_terrain_qualitative_v1", "mode": args.mode,
                            "purpose": "Qualitative only; all first episodes and resets shown, not frozen benchmark results.",
                            "geometry_seed": 68, "reset_seed": 42, "policy_seed": 42,
                            "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint),
                            "v5_sha256": sha(PARENT), "teacher_tensor_identity_verified": True,
                            "initial_state_sha256": initial, "gate_config": asdict(config),
                            "source_sha256": sources, "renderer_sha256": sha(__file__),
                            "routing_inputs": "Only ideal 825-ray heights/validity; no family labels.",
                            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                            "physics_steps": 960, "dt": dt, "frames_per_video": frames,
                            "fps": 30, "duration_seconds": frames / 30, "resolution": [1280, 720],
                            "render_batches_physics_unchanged": True,
                            "videos": [{**assignment[i], "path": path.name, "sha256": sha(path),
                                        "resets": resets[i], "first_episode": first[i], "reset_events": events[i]}
                                       for i, path in enumerate(files)]}
                with (args.output_dir / "manifest.json").open("x") as stream:
                    stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
                print(f"[other terrains] wrote {frames} frames each for all five {args.mode} replays")
            finally:
                for writer in writers:
                    writer.close()
                env.close()
        run()
    finally:
        app.close()


if __name__ == "__main__":
    launch()
