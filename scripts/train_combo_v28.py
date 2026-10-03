# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2026, Robotics Simulation Week 03 Team.
# SPDX-License-Identifier: BSD-3-Clause

"""Train one v28 combination run with RSL-RL (the course ``train.py`` plus a weights-only start).

``--init_checkpoint`` starts a NEW run from another run's policy/value weights (fresh optimizer,
iteration counter 0), as in Lim's ``train_finetune.py``.  After training, the final checkpoint is
re-loaded, checked for finite tensors and hashed into ``v28_run.json`` in the run directory.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train a v28 Ant combination run.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--init_checkpoint", type=str, default=None, help="Weights-only start for a continuation run.")
parser.add_argument("--run_record", type=str, default=None, help="Optional copy of v28_run.json.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import week03_ant.tasks.combo_v28  # noqa: E402,F401  # registers v28 task IDs
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_integrity(path: str) -> dict:
    """Re-load a saved checkpoint and confirm that every tensor is finite."""
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    state = loaded["model_state_dict"]
    nonfinite = [name for name, tensor in state.items() if not torch.isfinite(tensor).all()]
    return {
        "path": os.path.abspath(path),
        "sha256": sha256(path),
        "bytes": os.path.getsize(path),
        "iter": int(loaded.get("iter", -1)),
        "tensors": len(state),
        "nonfinite_tensors": nonfinite,
        "action_std": state["std"].tolist() if "std" in state else None,
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Keep the PPO networks on the simulation GPU (the runner config defaults to cuda:0).
    agent_cfg.device = env_cfg.sim.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_dir)
    env_cfg.log_dir = log_dir
    print(f"[INFO] v28 run directory: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    init_record = None
    if args_cli.init_checkpoint:
        init_path = os.path.abspath(args_cli.init_checkpoint)
        loaded = torch.load(init_path, map_location=agent_cfg.device, weights_only=False)
        # Weights only: a fresh optimizer and iteration 0 make this a new continuation run.
        result = torch.nn.Module.load_state_dict(runner.alg.policy, loaded["model_state_dict"], strict=True)
        runner.current_learning_iteration = 0
        init_record = {"path": init_path, "sha256": sha256(init_path), "iter": int(loaded.get("iter", -1)),
                       "load_result": str(result)}
        print(f"[INFO] Initialised weights from {init_path}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    started = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    seconds = time.time() - started

    final = os.path.join(log_dir, f"model_{agent_cfg.max_iterations - 1}.pt")
    record = {
        "task": args_cli.task,
        "seed": agent_cfg.seed,
        "run_name": agent_cfg.run_name,
        "run_dir": log_dir,
        "num_envs": env_cfg.scene.num_envs,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": agent_cfg.max_iterations,
        "transitions": env_cfg.scene.num_envs * agent_cfg.num_steps_per_env * agent_cfg.max_iterations,
        "entropy_coef": agent_cfg.algorithm.entropy_coef,
        "learning_rate": agent_cfg.algorithm.learning_rate,
        "schedule": agent_cfg.algorithm.schedule,
        "init_checkpoint": init_record,
        "train_seconds": seconds,
        "final_checkpoint": checkpoint_integrity(final),
        "finished_utc": datetime.utcnow().isoformat() + "Z",
    }
    text = json.dumps(record, indent=2)
    with open(os.path.join(log_dir, "v28_run.json"), "w", encoding="utf-8") as stream:
        stream.write(text + "\n")
    if args_cli.run_record:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.run_record)), exist_ok=True)
        with open(args_cli.run_record, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print("V28_RUN_JSON " + json.dumps(record))
    if record["final_checkpoint"]["nonfinite_tensors"]:
        raise RuntimeError(f"non-finite tensors in {final}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
