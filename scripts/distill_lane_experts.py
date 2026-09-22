"""Rehearse stable non-stone behavior while retaining learned stone recovery.

Two frozen teachers label simulated states using training-only lane identity.
The exported student is the unchanged single 60D/8D course MLP: no teacher,
terrain label, runtime switch, extra sensor or network is used at inference.
Teacher/student rollouts are aggregated across rounds (DAgger-style rehearsal).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Week03-Ant-Rough-Lanes-v5")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--reference-checkpoint", type=Path, required=True)
parser.add_argument("--stone-checkpoint", type=Path, required=True)
parser.add_argument("--output-run", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--steps", type=int, default=960)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--stone-start-distance", type=float, default=0.0,
                    help="training-only stone-teacher onset from entrance centre; 0 disables the gate")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint must identify the student's initial checkpoint")
if min(args_cli.num_envs, args_cli.rounds, args_cli.steps, args_cli.epochs, args_cli.batch_size) < 1:
    parser.error("environment/round/step/epoch/batch counts must be positive")
if not 0.0 < args_cli.learning_rate < float("inf"):
    parser.error("--learning-rate must be finite and positive")
if not 0.0 <= args_cli.stone_start_distance < float("inf"):
    parser.error("--stone-start-distance must be finite and nonnegative")
if args_cli.output_run.exists():
    parser.error("--output-run must be a new directory (existing experiments are never overwritten)")
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import week03_ant  # noqa: F401
from week03_ant.tasks.lanes import get_lane_state


def checkpoint_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    env_cfg.log_dir = str(args_cli.output_run)
    args_cli.output_run.mkdir(parents=True, exist_ok=False)
    dump_yaml(str(args_cli.output_run / "params/env.yaml"), env_cfg)
    dump_yaml(str(args_cli.output_run / "params/agent.yaml"), agent_cfg)
    provenance = {
        "method": "aggregated expert-labelled actor rehearsal (not PPO)",
        "reference": checkpoint_record(args_cli.reference_checkpoint),
        "stone": checkpoint_record(args_cli.stone_checkpoint),
        "student_initial": checkpoint_record(args_cli.checkpoint),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args_cli).items()},
        "teacher_routing": (
            "stone teacher on stepping_stones, optionally after entrance-distance gate; "
            "reference elsewhere, training only"
        ),
        "inference": "single 60D/8D student MLP; no lane identity or teachers",
    }
    (args_cli.output_run / "distillation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    writer = SummaryWriter(log_dir=str(args_cli.output_run))
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
        runner.load(args_cli.checkpoint, load_optimizer=False)
        student = runner.alg.policy
        if student.is_recurrent or student.actor_obs_normalization or student.critic_obs_normalization:
            raise ValueError("this bounded rehearsal requires the course feed-forward unnormalized policy")
        if student.actor[0].in_features != 60 or student.actor[-1].out_features != 8:
            raise ValueError("the student must keep the 60D/8D course interface")
        teachers = []
        for path in (args_cli.reference_checkpoint, args_cli.stone_checkpoint):
            teacher = copy.deepcopy(student)
            teacher.load_state_dict(torch.load(path, map_location=env.unwrapped.device, weights_only=False)["model_state_dict"])
            teacher.requires_grad_(False)
            teacher.eval()
            teachers.append(teacher)
        state = get_lane_state(env.unwrapped)
        families = state.family_level()[0]
        on_stones = families == state.family_names.index("stepping_stones")
        family_counts = {name: int((families == index).sum()) for index, name in enumerate(state.family_names)}
        if not bool(on_stones.any()) or bool(on_stones.all()):
            raise ValueError("rehearsal needs both stone and non-stone environments")
        optimizer = torch.optim.Adam(student.actor.parameters(), lr=args_cli.learning_rate)
        inputs, targets, records = [], [], []
        for round_index in range(1, args_cli.rounds + 1):
            observations, _ = env.reset()
            student.eval()
            # Fix teacher-versus-student behavior per environment for this round.
            # The optional ingress gate can still switch between the two teachers.
            teacher_rollout = torch.rand(env.num_envs, device=env.unwrapped.device) < 0.5
            stone_label_count = torch.zeros((), dtype=torch.long, device=env.unwrapped.device)
            x = torch.empty(args_cli.steps, env.num_envs, 60, device=env.unwrapped.device)
            y = torch.empty(args_cli.steps, env.num_envs, 8, device=env.unwrapped.device)
            with torch.no_grad():
                for step in range(args_cli.steps):
                    reference_actions = teachers[0].act_inference(observations)
                    stone_actions = teachers[1].act_inference(observations)
                    stone_labels = on_stones
                    if args_cli.stone_start_distance > 0.0:
                        # The common flat entrance cannot be identified as stones
                        # from course observations. Use the stable teacher there.
                        entrance_distance = (
                            env.unwrapped.scene["robot"].data.root_pos_w[:, 0]
                            - state.x_entrance[state.lane]
                        )
                        stone_labels = on_stones & (entrance_distance >= args_cli.stone_start_distance)
                    desired = torch.where(stone_labels[:, None], stone_actions, reference_actions)
                    stone_label_count += stone_labels.sum()
                    student_actions = student.act_inference(observations)
                    x[step] = observations["policy"]
                    y[step] = desired
                    behavior = torch.where(teacher_rollout[:, None], desired, student_actions)
                    observations, _, _, _ = env.step(behavior)
            if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
                raise RuntimeError("non-finite rehearsal data; refusing to train or save it")
            inputs.append(x.flatten(0, 1))
            targets.append(y.flatten(0, 1))
            all_x, all_y = torch.cat(inputs), torch.cat(targets)
            student.train()
            loss_total, batches = 0.0, 0
            for _ in range(args_cli.epochs):
                order = torch.randperm(len(all_x), device=env.unwrapped.device)
                for ids in order.split(args_cli.batch_size):
                    prediction = student.act_inference({"policy": all_x[ids]})
                    loss = F.smooth_l1_loss(prediction, all_y[ids])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.actor.parameters(), 1.0)
                    optimizer.step()
                    loss_total += loss.item()
                    batches += 1
            student.eval()
            if not all(bool(torch.isfinite(value).all()) for value in student.state_dict().values()):
                raise RuntimeError("non-finite student parameters; checkpoint not saved")
            # Standard RSL checkpoint, but no stale PPO optimizer moments/iteration.
            runner.alg.optimizer.state.clear()
            path = args_cli.output_run / f"model_round_{round_index}.pt"
            # OnPolicyRunner.save also uses logging initialized by PPO.learn;
            # this actor-only trainer writes its documented checkpoint fields directly.
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                    "iter": 0,
                    "infos": {**provenance, "round": round_index, "ppo_optimizer_state_reset": True},
                },
                path,
            )
            record = {
                "round": round_index, "samples": len(all_x), "loss": loss_total / batches,
                "checkpoint": str(path), "checkpoint_sha256": checkpoint_record(path)["sha256"],
                "family_counts": family_counts,
                "teacher_rollout_envs": int(teacher_rollout.sum()),
                "student_rollout_envs": int((~teacher_rollout).sum()),
                "round_stone_labels": int(stone_label_count),
                "round_reference_labels": args_cli.steps * env.num_envs - int(stone_label_count),
            }
            records.append(record)
            writer.add_scalar("Rehearsal/smooth_l1", record["loss"], round_index)
            writer.flush()
            (args_cli.output_run / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
            print(f"[REHEARSAL] {record}", flush=True)
            del all_x, all_y
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
