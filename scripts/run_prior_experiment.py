"""Fixed-budget v10 paired prior study; freeze all final policies before holdout."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time

import torch
import yaml
from rsl_rl.algorithms import PPO

from week03_ant.prior_policy import PRIOR_COEFFICIENTS, PRIOR_MODES
from week03_ant.prior_ppo import PINNED_PPO_SHA256, PINNED_PPO_VERSION
from run_foothold_experiment import _validate_reference, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs/prior_v10_20260922"
ARTIFACT = ROOT / "artifacts/terrain_demo/prior_v10"
PYTHON = ROOT.parent / "run-python"
PARENT = ROOT / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
PARENT_SHA = "889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e"
MODES = ("free", "anchored")
TRAIN_PAIRS = ((42, 51), (43, 58), (44, 59))
EVAL_PAIRS = ((66, 40), (67, 41))
ITERATIONS, NUM_ENVS, STEPS_PER_ENV = 750, 4096, 32
TASK_TRAIN = "Week03-Ant-Prior-Lanes-Train-v10"
TASK_EVAL = "Week03-Ant-Prior-Lanes-Eval-v10"


def upstream_provenance():
    source = Path(inspect.getsourcefile(PPO))
    version = importlib.metadata.version("rsl-rl-lib")
    digest = sha(source)
    if version != PINNED_PPO_VERSION or digest != PINNED_PPO_SHA256:
        raise ValueError("installed PPO differs from the parity-tested pinned implementation")
    return {"rsl_rl_version": version, "ppo_source": str(source), "ppo_sha256": digest}


def source_hashes():
    previous = json.loads((ROOT / "artifacts/terrain_demo/foothold_v9/frozen.json").read_text())["source_sha256"]
    additions = [
        "src/week03_ant/prior_policy.py", "src/week03_ant/prior_ppo.py",
        "src/week03_ant/tasks/prior_v10.py", "src/week03_ant/tasks/agents/prior_v10_cfg.py",
        "scripts/prior_v10.py", "scripts/prepare_prior_checkpoint.py",
        "scripts/run_prior_experiment.py", "scripts/summarize_prior.py", "scripts/summarize_prior_horizon.py",
        "src/week03_ant/horizon.py", "scripts/probe_prior_horizon.py",
        "scripts/render_prior_comparison.py", "scripts/build_prior_viewer.py",
        "tests/test_prior_policy.py", "tests/test_prior_ppo.py", "tests/test_prior_horizon.py",
        "tests/test_prior_harness.py", "outputs/prior_v10_20260922/PLAN.md",
    ]
    current = {name: sha(ROOT / name) for name in [*previous, *additions]}
    for name, expected in previous.items():
        if current[name] != expected:
            raise ValueError(f"archived v9 source changed: {name}")
    upstream_provenance()
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("immutable v5 parent changed")
    return current


def check_sources(expected):
    upstream_provenance()
    for name, value in expected.items():
        if sha(ROOT / name) != value:
            raise ValueError(f"source changed during frozen experiment: {name}")
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("immutable v5 parent changed")


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
    record = {"label": label, "command": command, "started": started,
              "seconds": time.monotonic() - begin, "returncode": result.returncode, "log": str(log)}
    with (ARTIFACT / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if result.returncode:
        raise RuntimeError(f"{label} failed: {log}")
    print(f"DONE {label} {record['seconds']:.1f}s", flush=True)


def validate_training_config(params, mode, seed, geometry):
    agent = yaml.load((params / "agent.yaml").read_text(), Loader=yaml.BaseLoader)
    env = yaml.load((params / "env.yaml").read_text(), Loader=yaml.BaseLoader)
    policy, algorithm = agent["policy"], agent["algorithm"]
    checks = {
        "agent seed": int(agent["seed"]) == seed,
        "steps": int(agent["num_steps_per_env"]) == STEPS_PER_ENV,
        "iterations": int(agent["max_iterations"]) == ITERATIONS,
        "experiment": agent["experiment_name"] == "week03_ant_prior_v10",
        "policy class": policy["class_name"] == "PriorActorCritic",
        "input mode": policy["input_mode"] == "targets",
        "prior mode": policy["prior_mode"] == mode,
        "mode guard": policy["require_config_match"] == "true",
        "actor normalization": policy["actor_obs_normalization"] == "false",
        "critic normalization": policy["critic_obs_normalization"] == "false",
        "actor widths": [int(v) for v in policy["actor_hidden_dims"]] == [400, 200, 100],
        "critic widths": [int(v) for v in policy["critic_hidden_dims"]] == [400, 200, 100],
        "environment seed": int(env["seed"]) == seed,
        "environment count": int(env["scene"]["num_envs"]) == NUM_ENVS,
        "terrain seed": int(env["scene"]["terrain"]["terrain_generator"]["seed"]) == geometry,
        "zero support reward": float(env["rewards"]["foothold_support"]["weight"]) == 0.,
        "PPO class": algorithm["class_name"] == "PriorPPO",
        "teacher coefficient": float(algorithm["teacher_coef"]) == PRIOR_COEFFICIENTS[mode],
        "learning rate": float(algorithm["learning_rate"]) == 1e-4,
        "fixed schedule": algorithm["schedule"] == "fixed",
        "no adaptive KL": algorithm["desired_kl"] == "null",
        "gamma": float(algorithm["gamma"]) == .995,
        "lambda": float(algorithm["lam"]) == .95,
        "entropy": float(algorithm["entropy_coef"]) == .002,
        "epochs": int(algorithm["num_learning_epochs"]) == 5,
        "mini batches": int(algorithm["num_mini_batches"]) == 4,
    }
    # Compare entire physical/sensor/reward contract with the archived targets arm.
    old_params = ROOT / "artifacts/terrain_demo/foothold_v9/runs/v9_targets_seed42/params"
    old_env = yaml.load((old_params / "env.yaml").read_text(), Loader=yaml.BaseLoader)
    old_agent = yaml.load((old_params / "agent.yaml").read_text(), Loader=yaml.BaseLoader)
    for section in ("observations", "rewards", "actions", "terminations", "events", "curriculum"):
        checks[f"unchanged {section}"] = env[section] == old_env[section]
    for section in ("robot", "height_scanner"):
        checks[f"unchanged {section}"] = env["scene"][section] == old_env["scene"][section]
    checks["unchanged simulation"] = env["sim"] == old_env["sim"]
    old_terrain = old_env["scene"]["terrain"]
    old_terrain["terrain_generator"]["seed"] = str(geometry)
    checks["unchanged terrain except seed"] = env["scene"]["terrain"] == old_terrain
    for section in ("episode_length_s", "decimation"):
        checks[f"unchanged {section}"] = env[section] == old_env[section]
    old_algorithm = dict(old_agent["algorithm"])
    old_algorithm["class_name"] = "PriorPPO"
    old_algorithm["teacher_coef"] = str(PRIOR_COEFFICIENTS[mode])
    checks["unchanged PPO except prior"] = algorithm == old_algorithm
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"saved training config mismatch: {', '.join(failed)}")
    return checks


def checkpoint_record(path, mode, seed, geometry, group, reference=False, **extra):
    path = path.resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    for key, dtype, expected in (
        ("input_mode_code", torch.int64, 1),
        ("prior_mode_code", torch.int64, PRIOR_MODES.index(mode)),
        ("prior_teacher_coef", torch.float32, PRIOR_COEFFICIENTS[mode]),
        ("prior_reference_std", torch.float32, .2),
    ):
        value = state.get(key)
        if (value is None or value.shape != torch.Size([]) or value.dtype != dtype
                or not torch.equal(value, torch.tensor(expected, dtype=dtype))):
            raise ValueError(f"checkpoint treatment mismatch: {key}")
    if not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite checkpoint")
    for layer, shape in (("actor.0.weight", (400, 88)), ("critic.0.weight", (400, 88)),
                         ("actor.6.weight", (8, 100)), ("critic.6.weight", (1, 100))):
        if state[layer].shape != shape:
            raise ValueError("checkpoint architecture mismatch")
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("v5 parent changed")
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)["model_state_dict"]
    for key, value in parent.items():
        if key.startswith("actor.") and not torch.equal(state[key.replace("actor.", "teacher.", 1)], value):
            raise ValueError(f"frozen teacher differs from v5: {key}")
    if reference:
        _validate_reference(state, parent)
        if checkpoint["iter"] != 0:
            raise ValueError("reference must be untrained zero-expanded v5")
    return {
        "checkpoint": str(path.relative_to(ROOT)), "sha256": sha(path), "mode": mode,
        "training_seed": seed, "training_geometry": geometry, "iteration": checkpoint["iter"],
        "group": group, "policy_class": "PriorActorCritic", "teacher_coef": PRIOR_COEFFICIENTS[mode],
        "reference_std": .2, "teacher_parent_sha256": PARENT_SHA, "teacher_bit_identical": True, **extra,
    }


def train(device):
    if (ARTIFACT / "frozen.json").exists():
        raise FileExistsError("experiment already frozen")
    sources = source_hashes()
    source_file = ARTIFACT / "source_at_train_start.json"
    if source_file.exists() or (ARTIFACT / "commands.jsonl").exists():
        raise FileExistsError("training already started; inspect immutable evidence before retry")
    write_json(source_file, sources)
    records = []
    for seed, geometry in TRAIN_PAIRS:
        for mode in MODES:
            check_sources(sources)
            label = f"v10_{mode}_seed{seed}"
            init_dir = ROOT / f"logs/rsl_rl/week03_ant_prior_v10/v10_init_{mode}_{seed}"
            run([str(PYTHON), "scripts/prepare_prior_checkpoint.py", str(PARENT),
                 str(init_dir / "model_0.pt"), "--mode", mode, "--seed", str(seed)], f"prepare_{label}")
            run([
                str(PYTHON), "scripts/prior_v10.py", "train", "--task", TASK_TRAIN,
                "--headless", "--device", device, "--num_envs", str(NUM_ENVS),
                "--max_iterations", str(ITERATIONS), "--seed", str(seed), "--run_name", label,
                "--resume", "--load_run", init_dir.name, "--checkpoint", "model_0.pt",
                f"agent.policy.prior_mode={mode}", f"agent.algorithm.teacher_coef={PRIOR_COEFFICIENTS[mode]}",
                "env.rewards.foothold_support.weight=0.0",
                f"env.scene.terrain.terrain_generator.seed={geometry}", f"agent.device={device}",
            ], f"train_{label}")
            matches = list((ROOT / "logs/rsl_rl/week03_ant_prior_v10").glob(f"*_{label}"))
            if len(matches) != 1:
                raise RuntimeError(f"ambiguous run directory: {matches}")
            source = matches[0]
            config_checks = validate_training_config(source / "params", mode, seed, geometry)
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
                foothold_support_weight=0., saved_config_checks=config_checks,
            )
            if record["iteration"] != ITERATIONS - 1:
                raise ValueError("not the predeclared final iteration")
            write_json(destination / "manifest.json", record)
            records.append(record)
    reference = ARTIFACT / "runs/frozen_v5_reference/model_0.pt"
    run([str(PYTHON), "scripts/prepare_prior_checkpoint.py", str(PARENT), str(reference),
         "--mode", "free", "--seed", "42"], "prepare_frozen_reference")
    check_sources(sources)
    write_json(ARTIFACT / "frozen.json", {
        "frozen_at": datetime.now(timezone.utc).isoformat(), "parent_sha256": PARENT_SHA,
        "selection": "all final749 only; representative seed42 fixed before holdout; no tuning",
        "training_pairs": TRAIN_PAIRS, "heldout_pairs": EVAL_PAIRS, "runs": records, "device": device,
        "reference": checkpoint_record(reference, "free", None, None, "frozen_v5", reference=True, transitions=0),
        "source_sha256": sources, "upstream": upstream_provenance(),
        "secondary_horizon": {"seconds": 64, "max_steps": 3840, "num_envs": 10,
                              "family": "stepping_stones", "level": 4, "gating": False},
    })


def evaluate(device, horizon=False):
    if horizon:
        verification_path = ARTIFACT / "verification.json"
        if not verification_path.exists():
            raise ValueError("audit primary evaluation before the horizon diagnostic")
        verification = json.loads(verification_path.read_text())
        if verification.get("status") != "pass" or verification.get("total_primary_first_episodes") != 2450:
            raise ValueError("horizon diagnostic requires the complete primary audit")
    frozen = json.loads((ARTIFACT / "frozen.json").read_text())
    check_sources(frozen["source_sha256"])
    for record in [frozen["reference"], *frozen["runs"]]:
        checkpoint = ROOT / record["checkpoint"]
        if sha(checkpoint) != record["sha256"]:
            raise ValueError("checkpoint changed after freeze")
        checkpoint_record(checkpoint, record["mode"], record["training_seed"], record["training_geometry"],
                          record["group"], reference=record["group"] == "frozen_v5")
        label = checkpoint.parent.name
        for geometry, seed in EVAL_PAIRS:
            check_sources(frozen["source_sha256"])
            stem = f"{label}__geometry{geometry}_reset{seed}"
            output = ARTIFACT / ("horizon" if horizon else "evaluations") / f"{stem}.json"
            if output.exists():
                raise FileExistsError(output)
            if horizon:
                command = [str(PYTHON), "scripts/probe_prior_horizon.py", "--checkpoint", str(checkpoint),
                           "--output", str(output), "--geometry", str(geometry), "--seed", str(seed),
                           "--headless", "--device", device, "--num_envs", "10"]
            else:
                command = [str(PYTHON), "scripts/prior_v10.py", "evaluate", "--task", TASK_EVAL,
                           "--headless", "--device", device, "--num_envs", "175", "--seed", str(seed),
                           "--checkpoint", str(checkpoint), "--output", str(output),
                           f"env.scene.terrain.terrain_generator.seed={geometry}", f"agent.device={device}"]
            run(command, f"{'horizon' if horizon else 'eval'}_{stem}")
            result = json.loads(output.read_text())
            if result["checkpoint_sha256"] != record["sha256"]:
                raise ValueError("evaluation checkpoint identity mismatch")
            if not horizon and (result["policy_observation_dimensions"] != 88 or result["task"] != TASK_EVAL):
                raise ValueError("evaluation policy/task identity mismatch")
    write_json(WORK / "status.json", {
        "phase": "horizon_complete" if horizon else "evaluations_complete",
        "finished": datetime.now(timezone.utc).isoformat(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate", "horizon"), required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    with (WORK / "gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "train":
            train(args.device)
        else:
            evaluate(args.device, horizon=args.phase == "horizon")


if __name__ == "__main__":
    main()
