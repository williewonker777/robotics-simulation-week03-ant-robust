"""Sequential, equal-budget v7 experiment; freeze every final model before held-out evaluation."""

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

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs/footmap_v7_20260921"
ARTIFACT = ROOT / "artifacts/terrain_demo/footmap_v7"
PYTHON = ROOT.parent / "run-python"
PARENT = ROOT / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
MODES = ("blind", "height", "footmap")
TRAIN_PAIRS = ((42, 51), (43, 58), (44, 59))
EVAL_PAIRS = ((60, 34), (61, 35))
ITERATIONS = 750


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


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
    with (WORK / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if result.returncode:
        raise RuntimeError(f"{label} failed: {log}")
    print(f"DONE {label} {record['seconds']:.1f}s", flush=True)


def checkpoint_record(path, mode, seed, geometry, **extra):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    assert int(state["input_mode_code"]) == MODES.index(mode)
    assert all(torch.isfinite(value).all() for value in state.values())
    assert state["actor.0.weight"].shape == (400, 92)
    return {"checkpoint": str(path.relative_to(ROOT)), "sha256": sha(path), "mode": mode,
            "training_seed": seed, "training_geometry": geometry, "iteration": checkpoint["iter"], **extra}


def train(device):
    if (ARTIFACT / "frozen.json").exists():
        raise FileExistsError("experiment already frozen")
    records = []
    parent_sha = sha(PARENT)
    for seed, geometry in TRAIN_PAIRS:
        for mode in MODES:
            label = f"v7_{mode}_seed{seed}"
            init_dir = ROOT / f"logs/rsl_rl/week03_ant_footmap_v7/v7_init_{mode}_{seed}"
            run([str(PYTHON), "scripts/prepare_footmap_checkpoint.py", str(PARENT),
                 str(init_dir / "model_0.pt"), "--mode", mode, "--seed", str(seed)], f"prepare_{label}")
            run([str(PYTHON), "scripts/train.py", "--task", "Week03-Ant-FootMap-Lanes-Train-v7",
                 "--headless", "--device", device, "--num_envs", "4096", "--max_iterations", str(ITERATIONS),
                 "--seed", str(seed), "--run_name", label, "--resume", "--load_run", init_dir.name,
                 "--checkpoint", "model_0.pt", f"agent.policy.input_mode={mode}",
                 f"env.scene.terrain.terrain_generator.seed={geometry}", f"agent.device={device}"], f"train_{label}")
            matches = list((ROOT / "logs/rsl_rl/week03_ant_footmap_v7").glob(f"*_{label}"))
            if len(matches) != 1:
                raise RuntimeError(f"ambiguous run directory: {matches}")
            source = matches[0]
            destination = ARTIFACT / "runs" / label
            destination.mkdir(parents=True, exist_ok=False)
            filename = f"model_{ITERATIONS - 1}.pt"
            shutil.copy2(source / filename, destination / filename)
            shutil.copytree(source / "params", destination / "params")
            for event in source.glob("events.out.tfevents.*"):
                shutil.copy2(event, destination / event.name)
            record = checkpoint_record(destination / filename, mode, seed, geometry,
                                       transitions=4096 * 32 * ITERATIONS, source_run=str(source))
            assert record["iteration"] == ITERATIONS - 1
            write_json(destination / "manifest.json", record)
            records.append(record)
    reference = ARTIFACT / "runs/frozen_v5_reference/model_0.pt"
    run([str(PYTHON), "scripts/prepare_footmap_checkpoint.py", str(PARENT), str(reference),
         "--mode", "blind", "--seed", "42"], "prepare_frozen_reference")
    assert sha(PARENT) == parent_sha
    write_json(ARTIFACT / "frozen.json", {
        "frozen_at": datetime.now(timezone.utc).isoformat(), "parent_sha256": parent_sha,
        "selection": "final iteration only, all three seeds; representative seed42 predeclared",
        "training_pairs": TRAIN_PAIRS, "heldout_pairs": EVAL_PAIRS, "runs": records, "device": device,
        "reference": checkpoint_record(reference, "blind", None, None, transitions=0),
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in [
            ROOT / "src/week03_ant/footmap_policy.py", ROOT / "src/week03_ant/footmap_math.py",
            ROOT / "src/week03_ant/tasks/footmap_v7_cfg.py", ROOT / "scripts/run_footmap_experiment.py"]},
    })


def evaluate(device):
    frozen = json.loads((ARTIFACT / "frozen.json").read_text())
    for record in [frozen["reference"], *frozen["runs"]]:
        checkpoint = ROOT / record["checkpoint"]
        assert sha(checkpoint) == record["sha256"], "checkpoint changed after freeze"
        label = checkpoint.parent.name
        for geometry, seed in EVAL_PAIRS:
            stem = f"{label}__geometry{geometry}_reset{seed}"
            output = ARTIFACT / "evaluations" / f"{stem}.json"
            if output.exists():
                raise FileExistsError(output)
            run([str(PYTHON), "scripts/play_one_episode.py", "--task", "Week03-Ant-FootMap-Lanes-Eval-v7",
                 "--headless", "--device", device, "--num_envs", "175", "--seed", str(seed),
                 "--checkpoint", str(checkpoint), "--output", str(output),
                 f"env.scene.terrain.terrain_generator.seed={geometry}", f"agent.device={device}"], f"eval_{stem}")
            result = json.loads(output.read_text())
            assert result["checkpoint_sha256"] == record["sha256"]
            assert result["footmap_policy"]["input_mode"] == record["mode"]
            assert result["policy_observation_dimensions"] == 514
    write_json(WORK / "status.json", {"phase": "evaluations_complete", "finished": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args.device) if args.phase == "train" else evaluate(args.device)


if __name__ == "__main__":
    main()
