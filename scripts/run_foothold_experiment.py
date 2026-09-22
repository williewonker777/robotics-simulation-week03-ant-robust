"""Run the frozen, paired v9 foothold study without post-holdout selection."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import torch
import yaml

from week03_ant.foothold_policy import INPUT_MODES


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs/foothold_v9_20260922"
ARTIFACT = ROOT / "artifacts/terrain_demo/foothold_v9"
PYTHON = ROOT.parent / "run-python"
PARENT = ROOT / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
MODES = ("feet", "targets", "guided")
TRAIN_PAIRS = ((42, 51), (43, 58), (44, 59))
EVAL_PAIRS = ((64, 38), (65, 39))
ITERATIONS = 750
NUM_ENVS = 4096
STEPS_PER_ENV = 32
TASK_TRAIN = "Week03-Ant-Foothold-Lanes-Train-v9"
TASK_EVAL = "Week03-Ant-Foothold-Lanes-Eval-v9"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def source_hashes():
    previous = json.loads((ROOT / "artifacts/terrain_demo/residual_v8/source_at_train_start.json").read_text())
    additions = [
        "src/week03_ant/depth_math.py",
        "src/week03_ant/evaluation.py",
        "src/week03_ant/tasks/__init__.py",
        "src/week03_ant/tasks/depth_v6_cfg.py",
        "src/week03_ant/foothold_math.py",
        "src/week03_ant/foothold_policy.py",
        "src/week03_ant/tasks/foothold_v9.py",
        "src/week03_ant/tasks/foothold_v9_cfg.py",
        "src/week03_ant/tasks/agents/foothold_v9_cfg.py",
        "scripts/foothold_v9.py",
        "scripts/prepare_foothold_checkpoint.py",
        "scripts/probe_foothold.py",
        "scripts/run_foothold_experiment.py",
        "scripts/summarize_foothold.py",
        "scripts/cli_args.py",
        "scripts/demo_terrains.py",
        "scripts/render_foothold_comparison.py",
        "scripts/build_foothold_viewer.py",
    ]
    current = {name: sha(ROOT / name) for name in [*previous, *additions]}
    for name, expected in previous.items():
        if current[name] != expected:
            raise ValueError(f"archived v7/v8 implementation changed: {name}")
    return current


def check_sources(expected):
    for name, value in expected.items():
        if sha(ROOT / name) != value:
            raise ValueError(f"source changed during frozen experiment: {name}")


def run(command, label):
    log = WORK / f"{label}.log"
    if log.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {log}")
    started = datetime.now(timezone.utc).isoformat()
    write_json(WORK / "status.json", {"phase": label, "started": started, "command": command})
    print(f"START {label} {started}", flush=True)
    begin = time.monotonic()
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = {
        "label": label, "command": command, "started": started,
        "seconds": time.monotonic() - begin, "returncode": result.returncode, "log": str(log),
    }
    with (ARTIFACT / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if result.returncode:
        raise RuntimeError(f"{label} failed: {log}")
    print(f"DONE {label} {record['seconds']:.1f}s", flush=True)


def _int(value):
    return int(value)


def _float(value):
    return float(value)


def validate_training_config(params, mode, seed, geometry):
    """Fail closed on the saved configuration, not only the requested CLI."""
    agent = yaml.load((params / "agent.yaml").read_text(), Loader=yaml.BaseLoader)
    env = yaml.load((params / "env.yaml").read_text(), Loader=yaml.BaseLoader)
    policy, algorithm = agent["policy"], agent["algorithm"]
    expected_reward = -1.0 if mode == "guided" else 0.0
    checks = {
        "agent seed": _int(agent["seed"]) == seed,
        "steps": _int(agent["num_steps_per_env"]) == STEPS_PER_ENV,
        "iterations": _int(agent["max_iterations"]) == ITERATIONS,
        "experiment": agent["experiment_name"] == "week03_ant_foothold_v9",
        "policy class": policy["class_name"] == "FootholdActorCritic",
        "policy mode": policy["input_mode"] == mode,
        "mode guard": policy["require_config_match"] == "true",
        "actor normalization": policy["actor_obs_normalization"] == "false",
        "critic normalization": policy["critic_obs_normalization"] == "false",
        "actor widths": [_int(v) for v in policy["actor_hidden_dims"]] == [400, 200, 100],
        "critic widths": [_int(v) for v in policy["critic_hidden_dims"]] == [400, 200, 100],
        "environment seed": _int(env["seed"]) == seed,
        "environment count": _int(env["scene"]["num_envs"]) == NUM_ENVS,
        "terrain seed": _int(env["scene"]["terrain"]["terrain_generator"]["seed"]) == geometry,
        "guided reward": _float(env["rewards"]["foothold_support"]["weight"]) == expected_reward,
        "learning rate": _float(algorithm["learning_rate"]) == 1e-4,
        "fixed schedule": algorithm["schedule"] == "fixed",
        "gamma": _float(algorithm["gamma"]) == .995,
        "lambda": _float(algorithm["lam"]) == .95,
        "entropy": _float(algorithm["entropy_coef"]) == .002,
        "epochs": _int(algorithm["num_learning_epochs"]) == 5,
        "mini batches": _int(algorithm["num_mini_batches"]) == 4,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"saved training config mismatch: {', '.join(failed)}")
    return checks


def _validate_reference(state, parent):
    for name, source in parent.items():
        if name == "std":
            if not torch.equal(state[name], torch.full_like(state[name], .2)):
                raise ValueError("reference exploration std mismatch")
        elif name in ("actor.0.weight", "critic.0.weight"):
            if not torch.equal(state[name][:, :60], source) or state[name][:, 60:].count_nonzero():
                raise ValueError(f"reference expansion changed v5 layer: {name}")
        elif not torch.equal(state[name], source):
            raise ValueError(f"reference changed v5 layer: {name}")


def checkpoint_record(path, mode, seed, geometry, group, reference=False, **extra):
    path = path.resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    code = state.get("input_mode_code")
    if (code is None or code.shape != torch.Size([]) or code.dtype != torch.int64
            or int(code) != INPUT_MODES.index(mode)):
        raise ValueError("checkpoint input mode mismatch")
    if not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite checkpoint")
    if state["actor.0.weight"].shape != (400, 88) or state["critic.0.weight"].shape != (400, 88):
        raise ValueError("checkpoint is not the 88D v9 architecture")
    if state["actor.6.weight"].shape != (8, 100) or state["critic.6.weight"].shape != (1, 100):
        raise ValueError("checkpoint output architecture mismatch")
    if reference:
        parent = torch.load(PARENT, map_location="cpu", weights_only=False)["model_state_dict"]
        _validate_reference(state, parent)
        if checkpoint["iter"] != 0:
            raise ValueError("reference must be the untrained zero-expanded checkpoint")
    return {
        "checkpoint": str(path.relative_to(ROOT)), "sha256": sha(path), "mode": mode,
        "training_seed": seed, "training_geometry": geometry, "iteration": checkpoint["iter"],
        "group": group, "policy_class": "FootholdActorCritic", **extra,
    }


def train(device):
    if (ARTIFACT / "frozen.json").exists():
        raise FileExistsError("experiment already frozen")
    sources = source_hashes()
    source_file = ARTIFACT / "source_at_train_start.json"
    if source_file.exists() or (ARTIFACT / "commands.jsonl").exists():
        raise FileExistsError("training already started; inspect immutable evidence before retry")
    write_json(source_file, sources)
    parent_sha = sha(PARENT)
    records = []
    for seed, geometry in TRAIN_PAIRS:
        for mode in MODES:
            check_sources(sources)
            label = f"v9_{mode}_seed{seed}"
            init_dir = ROOT / f"logs/rsl_rl/week03_ant_foothold_v9/v9_init_{mode}_{seed}"
            run([
                str(PYTHON), "scripts/prepare_foothold_checkpoint.py", str(PARENT),
                str(init_dir / "model_0.pt"), "--mode", mode, "--seed", str(seed),
            ], f"prepare_{label}")
            reward = "-1.0" if mode == "guided" else "0.0"
            run([
                str(PYTHON), "scripts/foothold_v9.py", "train", "--task", TASK_TRAIN,
                "--headless", "--device", device, "--num_envs", str(NUM_ENVS),
                "--max_iterations", str(ITERATIONS), "--seed", str(seed), "--run_name", label,
                "--resume", "--load_run", init_dir.name, "--checkpoint", "model_0.pt",
                f"agent.policy.input_mode={mode}",
                f"env.rewards.foothold_support.weight={reward}",
                f"env.scene.terrain.terrain_generator.seed={geometry}", f"agent.device={device}",
            ], f"train_{label}")
            matches = list((ROOT / "logs/rsl_rl/week03_ant_foothold_v9").glob(f"*_{label}"))
            if len(matches) != 1:
                raise RuntimeError(f"ambiguous run directory: {matches}")
            source = matches[0]
            validate_training_config(source / "params", mode, seed, geometry)
            destination = ARTIFACT / "runs" / label
            destination.mkdir(parents=True, exist_ok=False)
            filename = f"model_{ITERATIONS - 1}.pt"
            shutil.copy2(source / filename, destination / filename)
            shutil.copytree(source / "params", destination / "params")
            for event in source.glob("events.out.tfevents.*"):
                shutil.copy2(event, destination / event.name)
            record = checkpoint_record(
                destination / filename, mode, seed, geometry, mode,
                transitions=NUM_ENVS * STEPS_PER_ENV * ITERATIONS, source_run=str(source),
                foothold_support_weight=-1.0 if mode == "guided" else 0.0,
            )
            if record["iteration"] != ITERATIONS - 1:
                raise ValueError("not the predeclared final iteration")
            write_json(destination / "manifest.json", record)
            records.append(record)
    reference = ARTIFACT / "runs/frozen_v5_reference/model_0.pt"
    run([
        str(PYTHON), "scripts/prepare_foothold_checkpoint.py", str(PARENT), str(reference),
        "--mode", "feet", "--seed", "42",
    ], "prepare_frozen_reference")
    check_sources(sources)
    if sha(PARENT) != parent_sha:
        raise ValueError("parent artifact mutated")
    write_json(ARTIFACT / "frozen.json", {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "parent_sha256": parent_sha,
        "selection": "all final749 only; representative seed42 predeclared; no post-holdout selection",
        "training_pairs": TRAIN_PAIRS, "heldout_pairs": EVAL_PAIRS, "runs": records, "device": device,
        "reference": checkpoint_record(
            reference, "feet", None, None, "frozen_v5", reference=True, transitions=0,
        ),
        "source_sha256": sources,
    })


def evaluate(device):
    frozen = json.loads((ARTIFACT / "frozen.json").read_text())
    check_sources(frozen["source_sha256"])
    for record in [frozen["reference"], *frozen["runs"]]:
        checkpoint = ROOT / record["checkpoint"]
        if sha(checkpoint) != record["sha256"]:
            raise ValueError("checkpoint changed after freeze")
        checkpoint_record(
            checkpoint, record["mode"], record["training_seed"], record["training_geometry"],
            record["group"], reference=record["group"] == "frozen_v5",
        )
        label = checkpoint.parent.name
        for geometry, seed in EVAL_PAIRS:
            check_sources(frozen["source_sha256"])
            stem = f"{label}__geometry{geometry}_reset{seed}"
            output = ARTIFACT / "evaluations" / f"{stem}.json"
            if output.exists():
                raise FileExistsError(output)
            run([
                str(PYTHON), "scripts/foothold_v9.py", "evaluate", "--task", TASK_EVAL,
                "--headless", "--device", device, "--num_envs", "175", "--seed", str(seed),
                "--checkpoint", str(checkpoint), "--output", str(output),
                f"env.scene.terrain.terrain_generator.seed={geometry}", f"agent.device={device}",
            ], f"eval_{stem}")
            result = json.loads(output.read_text())
            if (result["checkpoint_sha256"] != record["sha256"]
                    or result["policy_observation_dimensions"] != 88
                    or result["task"] != TASK_EVAL):
                raise ValueError("evaluation policy/task identity mismatch")
    write_json(WORK / "status.json", {
        "phase": "evaluations_complete", "finished": datetime.now(timezone.utc).isoformat(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    with (WORK / "gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args.device) if args.phase == "train" else evaluate(args.device)


if __name__ == "__main__":
    main()
