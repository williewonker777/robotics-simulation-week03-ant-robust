# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2026, Robotics Simulation Week 03 Team.
# SPDX-License-Identifier: BSD-3-Clause

"""Score checkpoints on one demo-evaluation terrain with the course ``play_one_episode.py`` rule.

Course rule (IsaacLab_RS e83a5d2): ``num_envs`` environments, evaluation seed, the first episode of
every environment, its cumulative environment reward (the original seven Ant terms) and the
population standard deviation.  Only the terrain of the Play task changes.

One Isaac Sim process scores every listed checkpoint on one terrain.  The random-number state
right after environment creation is captured; before each checkpoint it is restored and the
environment reset, which reproduces the official script's first reset (same initial states as a
fresh ``play_one_episode.py`` run) for every policy.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v28 demo evaluation on one terrain.")
parser.add_argument("--task", default="Week03-Ant-Combo-v28-Play")
parser.add_argument("--terrain", required=True, help="Condition name from combo_v28_cfg.EVAL_CONDITIONS.")
parser.add_argument("--terrain_seed", type=int, required=True)
parser.add_argument("--checkpoints", required=True, help="JSON list of {id, path, sha256?}.")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--repeat_first", action="store_true", help="Score the first checkpoint again at the end.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import week03_ant.tasks.combo_v28  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from week03_ant.tasks.combo_v28_cfg import apply_eval_condition  # noqa: E402


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng(device: str) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.startswith("cuda") else None,
    }


def restore_rng(state: dict, device: str) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


@torch.inference_mode()
def score(env, policy, num_envs: int, rng_state: dict) -> dict:
    """Run until every environment has finished its first episode (official loop).

    The whole rollout, including the reset, runs in inference mode: buffers created while stepping
    are inference tensors and may not be updated in place outside it.
    """
    unwrapped = env.unwrapped
    restore_rng(rng_state, str(unwrapped.device))
    unwrapped.reset()
    robot = unwrapped.scene["robot"]
    # The official script scores from the environment's first reset, before PhysX has stepped; there the
    # reset event's joint-velocity offsets do not take effect and every joint starts at rest.  Reproduce
    # that exact initial state (joint positions are already identical) for every checkpoint.
    robot.write_joint_velocity_to_sim(torch.zeros_like(robot.data.joint_vel))
    obs = env.get_observations()
    device = unwrapped.device
    start_xy = robot.data.root_pos_w[:, :2].clone()
    last_xy = start_xy.clone()
    rewards_sum = torch.zeros(num_envs, dtype=torch.float64, device=device)
    steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    distance = torch.zeros(num_envs, dtype=torch.float64, device=device)
    timestep = 0
    while simulation_app.is_running():
        last_xy = torch.where(finished[:, None], last_xy, robot.data.root_pos_w[:, :2])
        actions = policy(obs)
        obs, rewards, dones, extras = env.step(actions)
        active = ~finished
        rewards_sum[active] += rewards[active]
        steps[active] += 1
        done = dones.bool()
        newly = done & active
        fell |= newly & unwrapped.reset_terminated
        distance[newly] = (last_xy[newly, 0] - start_xy[newly, 0]).double()
        finished |= done
        timestep += 1
        if finished.all().item():
            break
        if timestep >= unwrapped.max_episode_length:
            break
    unfinished = ~finished
    if unfinished.any():
        distance[unfinished] = (robot.data.root_pos_w[unfinished, 0] - start_xy[unfinished, 0]).double()
    return {
        "return_mean": rewards_sum.mean().item(),
        "return_std": rewards_sum.std(unbiased=False).item(),
        "steps_mean": steps.double().mean().item(),
        "fall_rate": fell.double().mean().item(),
        "full_length_rate": (steps >= unwrapped.max_episode_length).double().mean().item(),
        "distance_mean": distance.mean().item(),
        "completed": int(finished.sum().item()),
        "timesteps": timestep,
        "returns": [round(v, 6) for v in rewards_sum.tolist()],
        "steps": steps.tolist(),
        "fell": fell.tolist(),
        "distance": [round(v, 4) for v in distance.tolist()],
    }


def main():
    checkpoints = json.loads(Path(args_cli.checkpoints).read_text(encoding="utf-8"))
    out_dir = Path(args_cli.output_dir) / args_cli.terrain
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    condition = apply_eval_condition(env_cfg, args_cli.terrain, args_cli.terrain_seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    # The wrapper resets once on construction; that first reset is the official initial state.
    rng_state = capture_rng(env_cfg.sim.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    meta = {
        "task": args_cli.task,
        "terrain": args_cli.terrain,
        "category": condition.category,
        "description": condition.description,
        "terrain_seed": args_cli.terrain_seed,
        "friction": condition.friction,
        "combine_mode": condition.combine_mode,
        "num_envs": args_cli.num_envs,
        "eval_seed": args_cli.seed,
        "max_episode_length": int(env.unwrapped.max_episode_length),
        "step_dt": float(env.unwrapped.step_dt),
        "reward_terms": list(env.unwrapped.reward_manager.active_terms),
        "termination_terms": list(env.unwrapped.termination_manager.active_terms),
        "observation_dim": int(env.get_observations()["policy"].shape[-1]),
    }
    jobs = list(checkpoints)
    if args_cli.repeat_first and jobs:
        jobs.append(dict(jobs[0], id=jobs[0]["id"] + "__repeat"))
    for job in jobs:
        target = out_dir / f"{job['id']}.json"
        if target.exists():
            print(f"[V28] skip existing {target}")
            continue
        digest = sha256(job["path"])
        if job.get("sha256") and job["sha256"] != digest:
            raise RuntimeError(f"checkpoint SHA mismatch for {job['id']}: {digest} != {job['sha256']}")
        runner.load(job["path"], load_optimizer=False, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        started = time.time()
        result = score(env, policy, args_cli.num_envs, rng_state)
        result.update(meta)
        result.update({"checkpoint_id": job["id"], "checkpoint": os.path.abspath(job["path"]),
                       "checkpoint_sha256": digest, "seconds": time.time() - started})
        target.write_text(json.dumps(result) + "\n", encoding="utf-8")
        print(
            f"[V28] {args_cli.terrain:>18s} {job['id']:<34s} return {result['return_mean']:8.3f}"
            f" +- {result['return_std']:7.3f} fall {result['fall_rate']:.2f} dist {result['distance_mean']:7.2f}"
            f" ({result['seconds']:.1f}s)",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
