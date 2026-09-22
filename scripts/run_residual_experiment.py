"""Fixed-budget sequential v8 study; no held-out evaluation until every run is frozen."""

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

from week03_ant.footmap_policy import INPUT_MODES

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs/residual_v8_20260922"
ARTIFACT = ROOT / "artifacts/terrain_demo/residual_v8"
PYTHON = ROOT.parent / "run-python"
PARENT = ROOT / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
MODES = ("blind", "footmap")
TRAIN_PAIRS = ((42, 51), (43, 58), (44, 59))
EVAL_PAIRS = ((62, 36), (63, 37))
ITERATIONS = 750


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def source_hashes():
    previous = json.loads((ROOT / "artifacts/terrain_demo/footmap_v7/source_at_train_start.json").read_text())
    paths = [*previous, "src/week03_ant/residual_policy.py", "src/week03_ant/tasks/residual_v8.py",
             "src/week03_ant/tasks/agents/residual_v8_cfg.py", "scripts/residual_v8.py",
             "scripts/prepare_residual_checkpoint.py", "scripts/run_residual_experiment.py",
             "scripts/summarize_footmap.py", "scripts/summarize_residual.py"]
    current = {name: sha(ROOT / name) for name in paths}
    for name, expected in previous.items():
        if current[name] != expected:
            raise ValueError(f"archived v7 implementation changed: {name}")
    return current


def check_sources(expected):
    for name, value in expected.items():
        if sha(ROOT / name) != value:
            raise ValueError(f"source changed during frozen experiment: {name}")


def run(command, label):
    log = WORK / f"{label}.log"
    if log.exists():
        raise FileExistsError(log)
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


def checkpoint_record(path, mode, seed, geometry, group, residual=True, **extra):
    path = path.resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    if int(state["input_mode_code"]) != INPUT_MODES.index(mode):
        raise ValueError("checkpoint input mode mismatch")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite checkpoint")
    if residual:
        parent = torch.load(PARENT, map_location="cpu", weights_only=False)["model_state_dict"]
        for name, value in parent.items():
            if name.startswith("actor."):
                if not torch.equal(value, state[name.replace("actor.", "actor.base.", 1)]):
                    raise ValueError(f"frozen v5 base mutated: {name}")
        if float(state["actor.residual_limit"]) != .5 or state["actor.residual.0.weight"].shape != (400, 92):
            raise ValueError("invalid residual architecture")
        if group == "frozen_v5" and (state["actor.residual.6.weight"].count_nonzero()
                                     or state["actor.residual.6.bias"].count_nonzero()):
            raise ValueError("frozen v5 reference has a nonzero residual")
    return {"checkpoint": str(path.relative_to(ROOT)), "sha256": sha(path), "mode": mode,
            "training_seed": seed, "training_geometry": geometry, "iteration": checkpoint["iter"],
            "group": group, "policy_class": "ResidualActorCritic" if residual else "FootMapActorCritic",
            "base_bit_identical": True if residual else None, "residual_limit": .5 if residual else None, **extra}


def train(device):
    if (ARTIFACT / "frozen.json").exists():
        raise FileExistsError("experiment already frozen")
    sources = source_hashes()
    source_file = ARTIFACT / "source_at_train_start.json"
    if source_file.exists():
        raise FileExistsError("training already started; inspect evidence before retry")
    write_json(source_file, sources)
    parent_sha = sha(PARENT)
    records = []
    for seed, geometry in TRAIN_PAIRS:
        for mode in MODES:
            check_sources(sources)
            label = f"v8_{mode}_seed{seed}"
            init_dir = ROOT / f"logs/rsl_rl/week03_ant_residual_v8/v8_init_{mode}_{seed}"
            run([str(PYTHON), "scripts/prepare_residual_checkpoint.py", str(PARENT),
                 str(init_dir / "model_0.pt"), "--mode", mode, "--seed", str(seed)], f"prepare_{label}")
            run([str(PYTHON), "scripts/residual_v8.py", "train", "--task", "Week03-Ant-Residual-Lanes-Train-v8",
                 "--headless", "--device", device, "--num_envs", "4096", "--max_iterations", str(ITERATIONS),
                 "--seed", str(seed), "--run_name", label, "--resume", "--load_run", init_dir.name,
                 "--checkpoint", "model_0.pt", f"agent.policy.input_mode={mode}",
                 f"env.scene.terrain.terrain_generator.seed={geometry}", f"agent.device={device}"], f"train_{label}")
            matches = list((ROOT / "logs/rsl_rl/week03_ant_residual_v8").glob(f"*_{label}"))
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
            record = checkpoint_record(destination / filename, mode, seed, geometry, f"residual_{mode}",
                                       transitions=4096 * 32 * ITERATIONS, source_run=str(source))
            if record["iteration"] != ITERATIONS - 1:
                raise ValueError("not the predeclared final iteration")
            write_json(destination / "manifest.json", record)
            records.append(record)
    reference = ARTIFACT / "runs/frozen_v5_reference/model_0.pt"
    run([str(PYTHON), "scripts/prepare_residual_checkpoint.py", str(PARENT), str(reference),
         "--mode", "blind", "--seed", "42"], "prepare_frozen_reference")
    comparisons = []
    previous = json.loads((ROOT / "artifacts/terrain_demo/footmap_v7/frozen.json").read_text())
    for record in previous["runs"]:
        if record["mode"] == "footmap":
            path = ROOT / record["checkpoint"]
            if sha(path) != record["sha256"]:
                raise ValueError("v7 comparison artifact mutated")
            comparisons.append(checkpoint_record(path, "footmap", record["training_seed"],
                                                  record["training_geometry"], "v7_footmap", residual=False))
    check_sources(sources)
    if sha(PARENT) != parent_sha:
        raise ValueError("parent artifact mutated")
    write_json(ARTIFACT / "frozen.json", {
        "frozen_at": datetime.now(timezone.utc).isoformat(), "parent_sha256": parent_sha,
        "selection": "all final749 only; video and diagnostic seed42 predeclared; no post-eval selection",
        "training_pairs": TRAIN_PAIRS, "heldout_pairs": EVAL_PAIRS, "runs": records, "device": device,
        "reference": checkpoint_record(reference, "blind", None, None, "frozen_v5", transitions=0),
        "comparisons": comparisons, "source_sha256": sources,
    })


def evaluate(device, diagnostics=False):
    frozen = json.loads((ARTIFACT / "frozen.json").read_text())
    check_sources(frozen["source_sha256"])
    records = [frozen["reference"], *frozen["comparisons"], *frozen["runs"]]
    if diagnostics:
        records = [r for r in records if r["group"] == "residual_footmap" and r["training_seed"] == 42]
    for record in records:
        checkpoint = ROOT / record["checkpoint"]
        if sha(checkpoint) != record["sha256"]:
            raise ValueError("checkpoint changed after freeze")
        label = checkpoint.parent.name
        residual = record["policy_class"] == "ResidualActorCritic"
        entry = ["scripts/residual_v8.py", "evaluate"] if residual else ["scripts/play_one_episode.py"]
        task = "Week03-Ant-Residual-Lanes-Eval-v8" if residual else "Week03-Ant-FootMap-Lanes-Eval-v7"
        for geometry, seed in EVAL_PAIRS:
            for scan_mode in (("zero", "shuffle") if diagnostics else ("actual",)):
                check_sources(frozen["source_sha256"])
                stem = f"{label}__geometry{geometry}_reset{seed}"
                if diagnostics:
                    stem += f"__scan_{scan_mode}"
                output = ARTIFACT / ("diagnostics" if diagnostics else "evaluations") / f"{stem}.json"
                if output.exists():
                    raise FileExistsError(output)
                command = [str(PYTHON), *entry, "--task", task, "--headless", "--device", device,
                           "--num_envs", "175", "--seed", str(seed), "--checkpoint", str(checkpoint),
                           "--output", str(output), f"env.scene.terrain.terrain_generator.seed={geometry}",
                           f"agent.device={device}"]
                if diagnostics:
                    command += ["--depth-mode", scan_mode]
                run(command, f"eval_{stem}")
                result = json.loads(output.read_text())
                if (result["checkpoint_sha256"] != record["sha256"]
                        or result["policy_observation_dimensions"] != 514):
                    raise ValueError("evaluation policy identity mismatch")
    write_json(WORK / "status.json", {"phase": "diagnostics_complete" if diagnostics else "evaluations_complete",
                                      "finished": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate", "diagnostics"), required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    with (WORK / "gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "train":
            train(args.device)
        else:
            evaluate(args.device, diagnostics=args.phase == "diagnostics")


if __name__ == "__main__":
    main()
