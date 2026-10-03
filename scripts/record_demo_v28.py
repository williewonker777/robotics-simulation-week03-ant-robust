# Copyright (c) 2026, Robotics Simulation Week 03 Team.
# SPDX-License-Identifier: BSD-3-Clause

"""Record a qualitative 16 s rollout of one checkpoint on one v28 demo terrain (env 0 of the batch).

The rollout uses the scored Play task, the same terrain strips and the same first reset as the demo
evaluation; rendering never advances physics.  Videos illustrate behaviour and are not scores.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Week03-Ant-Combo-v28-Play")
parser.add_argument("--terrain", required=True)
parser.add_argument("--terrain_seed", type=int, default=2028)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label", required=True, help="Caption shown in the video.")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=16, help="Batch size; env 0 is filmed.")
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--fps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import week03_ant.tasks.combo_v28  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from week03_ant.tasks.combo_v28_cfg import apply_eval_condition  # noqa: E402

FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def font(size: int):
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=1)  # index 1: KR face
    return ImageFont.load_default()


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    task = args_cli.task
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    condition = apply_eval_condition(env_cfg, args_cli.terrain, args_cli.terrain_seed)
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0

    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg, render_mode="rgb_array"), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    obs = env.get_observations()

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args_cli.output), fps=args_cli.fps, macro_block_size=None)
    big, small = font(30), font(22)
    start_x = float(robot.data.root_pos_w[0, 0])
    episode_return, falls, frames = 0.0, 0, 0
    first_done_step = None
    stride = max(1, round(1.0 / raw.step_dt / args_cli.fps))
    max_steps = int(raw.max_episode_length)
    for step in range(1, max_steps + 1):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)
        if first_done_step is None:
            episode_return += float(rewards[0])
            if bool(dones[0]):
                first_done_step = step
                falls += int(bool(raw.reset_terminated[0]))
        if step % stride == 0:
            pos = robot.data.root_pos_w[0].detach().cpu().numpy()
            raw.sim.set_camera_view(pos + np.array([-3.2, -3.2, 2.0]), pos)
            raw.sim.render()
            frame = raw.render(recompute=True)
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rectangle((0, 0, 1280, 84), fill=(12, 12, 12, 200))
            draw.text((18, 8), args_cli.label, font=big, fill=(255, 255, 255))
            status = "넘어짐" if falls and first_done_step else ("에피소드 진행 중" if first_done_step is None else "16초 완주")
            distance = float(pos[0]) - start_x if first_done_step is None else None
            line = f"{condition.description}  ·  t={step * raw.step_dt:4.1f}s  ·  첫 episode 보상 {episode_return:6.1f}"
            if distance is not None:
                line += f"  ·  전진 {distance:5.1f} m"
            draw.text((18, 48), line + f"  ·  {status}", font=small, fill=(220, 220, 220))
            writer.append_data(np.asarray(image))
            frames += 1
    writer.close()
    meta = {
        "terrain": args_cli.terrain, "terrain_seed": args_cli.terrain_seed, "checkpoint": args_cli.checkpoint,
        "checkpoint_sha256": sha256(args_cli.checkpoint), "label": args_cli.label, "num_envs": args_cli.num_envs,
        "env_index": 0, "frames": frames, "fps": args_cli.fps, "episode_return_env0": episode_return,
        "first_episode_end_step": first_done_step, "fell": bool(falls), "video": str(args_cli.output),
        "video_sha256": sha256(args_cli.output),
    }
    args_cli.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("V28_VIDEO " + json.dumps(meta, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
