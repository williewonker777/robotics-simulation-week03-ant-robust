#!/usr/bin/env python3
"""Run the v28 combination study: training queue, demo evaluation and confirmation.

Phases (run one at a time; a lock file keeps a single GPU job on the RTX 5070, ``cuda:1``):

* ``stage1``  : 1000-iteration runs for every recipe and seed (seed-major order).
* ``stage2``  : 600-iteration weights-only continuations of chosen parents with the original reward
                ("stock", Lim's longer training) or Stick's recovery reward ("recovery").
* ``eval``    : score every finished checkpoint and the reference checkpoints on all demo terrains.
* ``confirm`` : re-score chosen checkpoints on fresh terrain strips (confirmation seed).
* ``status``  : print progress.

Completed work is skipped, so every phase can be re-run after an interruption.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import mmap
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT.parent / "run-python")
OUT = ROOT / "outputs" / "combo_v28"
RECORDS = OUT / "run_records"
LOGS = OUT / "logs"
DEVICE = "cuda:1"
KIT = "--kit_args=--/renderer/multiGpu/enabled=false"

TERRAINS = ("flat", "stick", "lim", "sticklim", "mine", "all")
ENTROPY = {"e0": 0.0, "e5": 0.005}
DRS = ("d0", "d1")
SEEDS = (42, 43, 44)
STAGE1_ITERATIONS = 1000
STAGE2_ITERATIONS = 600
NUM_ENVS = 4096

# The provided-code baseline and the submitted Robust42 recipes already exist with the same budget
# and seeds (flat ground, entropy 0); on the plane the ground-relative height equals world z.
REUSED = {
    "flat_e0_d0": "artifacts/runs/baseline_seed{seed}/model_999.pt",
    "flat_e0_d1": "artifacts/runs/robust_seed{seed}/model_999.pt",
}

REFERENCES = {
    "ref_stick_flat": "outputs/combo_v28/references/stick/flat.pt",
    "ref_stick_rough": "outputs/combo_v28/references/stick/rough.pt",
    "ref_stick_control": "outputs/combo_v28/references/stick/control.pt",
    "ref_stick_recovery": "outputs/combo_v28/references/stick/recovery.pt",
    "ref_lim_e0": "outputs/combo_v28/references/lim/e0_flat_baseline.pt",
    "ref_lim_e4": "outputs/combo_v28/references/lim/e4_mixed_relheight_s42.pt",
    "ref_lim_e15": "outputs/combo_v28/references/lim/e15_boxes_entropy_s44.pt",
    "ref_lim_f3a": "outputs/combo_v28/references/lim/f3a_final_s47.pt",
}
REFERENCE_SHA256 = {
    "ref_stick_flat": "e0e0d2573346d47b94c2f09b86d93578dc705d2c28a43dead72abbdd551c76c3",
    "ref_stick_rough": "b6f44bc7fc4f8e73ccbaab0e063f9bbd3cf0ab1260f89ceef8ea5663ca25a920",
    "ref_stick_control": "8c12057bf392c9702e0acba9f5d195f4637785efdbce2d4b6b31357b24df8be7",
    "ref_stick_recovery": "c77f2c0fab3b67be64cd56b7c4043c7fd2ad68f2e2465e012f963bc6d0f2fa67",
    "ref_lim_e0": "0769d66f35f6bf33d7849401cd1a1a3d7d518c31c8b6559373ce0c5edd49dd1e",
    "ref_lim_e4": "95d34ee323b7dd9c39a1cb97f2c0684c1b7befe083d23502e5939b2964f657a3",
    "ref_lim_e15": "88fa4a35e6fb06d4a5a465e9aec8eb8e9c8b29bffbae26bd895e10a524e6b96e",
    "ref_lim_f3a": "6b5c28f1c2f24961794b49831b4fe40d55ebaf1d4230ced05f21e79981e0de61",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direct_sha256(path: Path) -> str:
    """SHA-256 read with O_DIRECT, bypassing the page cache."""
    block = 1 << 20
    buffer = mmap.mmap(-1, block)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    digest, offset = hashlib.sha256(), 0
    try:
        while True:
            count = os.preadv(fd, [buffer], offset)
            if count <= 0:
                break
            digest.update(buffer[:count])
            offset += count
            if count < block:
                break
    finally:
        os.close(fd)
        buffer.close()
    return digest.hexdigest()


def recipe_id(terrain: str, entropy: str, dr: str) -> str:
    return f"{terrain}_{entropy}_{dr}"


def parse_recipe(recipe: str) -> tuple[str, str, str]:
    terrain, entropy, dr = recipe.split("_")
    if terrain not in TERRAINS or entropy not in ENTROPY or dr not in DRS:
        raise ValueError(f"unknown recipe {recipe}")
    return terrain, entropy, dr


ALL_RECIPES = tuple(recipe_id(t, e, d) for t in TERRAINS for e in ENTROPY for d in DRS)
STAGE1_RECIPES = tuple(r for r in ALL_RECIPES if r not in REUSED)


def task_id(terrain: str, dr: str, reward: str) -> str:
    return f"Week03-Ant-Combo-v28-{terrain.capitalize()}-{dr.upper()}-{reward.capitalize()}"


def record_path(run_name: str) -> Path:
    return RECORDS / f"{run_name}.json"


def load_record(run_name: str) -> dict | None:
    path = record_path(run_name)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    final = record.get("final_checkpoint", {})
    if final.get("nonfinite_tensors"):
        return None
    return record


def stage1_name(recipe: str, seed: int) -> str:
    return f"v28s1_{recipe}_s{seed}"


def stage2_name(parent: str, arm: str, seed: int) -> str:
    return f"v28s2_{parent}+{arm}_s{seed}"


def checkpoint_for(recipe: str, seed: int) -> Path | None:
    if recipe in REUSED:
        return ROOT / REUSED[recipe].format(seed=seed)
    if "+" in recipe:
        parent, arm = recipe.split("+")
        record = load_record(stage2_name(parent, arm, seed))
    else:
        record = load_record(stage1_name(recipe, seed))
    return Path(record["final_checkpoint"]["path"]) if record else None


def run_logged(command: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as stream:
        stream.write(f"\n# {utc()} {' '.join(command)}\n")
        stream.flush()
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return process.returncode


def train(run_name: str, task: str, seed: int, entropy: float, iterations: int, init: Path | None) -> bool:
    if load_record(run_name):
        return True
    command = [
        PYTHON, "scripts/train_combo_v28.py", "--task", task, "--seed", str(seed),
        "--num_envs", str(NUM_ENVS), "--max_iterations", str(iterations), "--run_name", run_name,
        "--run_record", str(record_path(run_name)), "--headless", "--device", DEVICE, KIT,
        f"agent.algorithm.entropy_coef={entropy}",
    ]
    if init is not None:
        command += ["--init_checkpoint", str(init)]
    for attempt in (1, 2):
        started = time.time()
        log(f"train {run_name} (attempt {attempt})")
        code = run_logged(command, LOGS / f"{run_name}.log")
        record = load_record(run_name)
        if code == 0 and record:
            path = Path(record["final_checkpoint"]["path"])
            direct = direct_sha256(path)
            record["direct_sha256"] = direct
            record["direct_sha256_match"] = direct == record["final_checkpoint"]["sha256"]
            record["attempt"] = attempt
            record["wall_seconds"] = time.time() - started
            record_path(run_name).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            if not record["direct_sha256_match"]:
                log(f"WARNING cached/direct SHA mismatch for {path}")
            log(f"done {run_name}: {record['train_seconds']:.0f}s train, {record['wall_seconds']:.0f}s wall")
            return True
        with (OUT / "failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"run": run_name, "attempt": attempt, "code": code, "utc": utc()}) + "\n")
        log(f"FAILED {run_name} attempt {attempt} (exit {code})")
    return False


def phase_stage1(seeds, recipes) -> None:
    for seed in seeds:
        for recipe in recipes:
            terrain, entropy, dr = parse_recipe(recipe)
            train(stage1_name(recipe, seed), task_id(terrain, dr, "stock"), seed, ENTROPY[entropy],
                  STAGE1_ITERATIONS, None)


def phase_stage2(seeds, jobs) -> None:
    """``jobs``: ``parent+arm`` items, e.g. ``lim_e5_d0+stock``."""
    for seed in seeds:
        for job in jobs:
            parent, arm = job.split("+")
            terrain, entropy, dr = parse_recipe(parent)
            init = checkpoint_for(parent, seed)
            if init is None:
                log(f"skip {job} s{seed}: parent checkpoint missing")
                continue
            train(stage2_name(parent, arm, seed), task_id(terrain, dr, arm), seed, ENTROPY[entropy],
                  STAGE2_ITERATIONS, init)


def checkpoint_inventory() -> list[dict]:
    """Every finished v28 checkpoint plus the reference checkpoints, with SHA from creation."""
    items = []
    for recipe in ALL_RECIPES:
        for seed in SEEDS:
            path = checkpoint_for(recipe, seed)
            if path is not None and path.exists():
                items.append({"id": f"{recipe}_s{seed}", "path": str(path), "recipe": recipe, "seed": seed})
    for record_file in sorted(RECORDS.glob("v28s2_*.json")):
        record = json.loads(record_file.read_text(encoding="utf-8"))
        name = record["run_name"][len("v28s2_"):]
        recipe, seed = name.rsplit("_s", 1)
        if not record["final_checkpoint"]["nonfinite_tensors"]:
            items.append({"id": f"{recipe}_s{seed}", "path": record["final_checkpoint"]["path"],
                          "recipe": recipe, "seed": int(seed), "sha256": record["final_checkpoint"]["sha256"]})
    for ref, path in REFERENCES.items():
        items.append({"id": ref, "path": str(ROOT / path), "recipe": ref, "seed": None,
                      "sha256": REFERENCE_SHA256[ref]})
    return items


def phase_eval(output: Path, terrain_seed: int, ids: list[str] | None, conditions: list[str] | None) -> None:
    names = conditions or EVAL_NAMES
    inventory = checkpoint_inventory()
    if ids:
        wanted = set(ids)
        inventory = [item for item in inventory if item["id"] in wanted]
        missing = wanted - {item["id"] for item in inventory}
        if missing:
            raise SystemExit(f"unknown checkpoint ids: {sorted(missing)}")
    for name in names:
        todo = [item for item in inventory if not (output / name / f"{item['id']}.json").exists()]
        if not todo:
            continue
        job_file = output / "jobs" / f"{name}.json"
        job_file.parent.mkdir(parents=True, exist_ok=True)
        job_file.write_text(json.dumps(todo, indent=1) + "\n", encoding="utf-8")
        log(f"eval {name}: {len(todo)} checkpoints (terrain seed {terrain_seed})")
        command = [
            PYTHON, "scripts/evaluate_demo_v28.py", "--terrain", name, "--terrain_seed", str(terrain_seed),
            "--checkpoints", str(job_file), "--output_dir", str(output), "--headless", "--device", DEVICE, KIT,
        ]
        code = run_logged(command, output / "logs" / f"{name}.log")
        done = sum((output / name / f"{item['id']}.json").exists() for item in todo)
        if code != 0 or done != len(todo):
            log(f"eval {name} incomplete: exit {code}, {done}/{len(todo)}")


EVAL_NAMES = [
    "flat", "flat_mu05", "flat_mu01", "flat_mu02_mult",
    "boxes_5", "boxes_10", "boxes_15", "boxes_fine_10", "boxes_10_mu02", "boxes_10_mu02_mult",
    "rough_5", "rough_10", "stick_rough", "wave_15",
    "slope_20", "slope_inv_20", "stairs_10", "stairs_inv_10", "isaaclab_rough",
    "obstacles_10", "rails_8", "cylinders_10", "cones_12", "tilted_blocks_8", "platform_10",
    "gaps_20", "pits_15", "stones",
]


def phase_status() -> None:
    finished = sorted(p.stem for p in RECORDS.glob("*.json"))
    print(f"finished runs: {len(finished)}")
    for seed in SEEDS:
        done = [r for r in STAGE1_RECIPES if load_record(stage1_name(r, seed))]
        print(f"  stage1 seed {seed}: {len(done)}/{len(STAGE1_RECIPES)}")
    for output in (OUT / "eval", OUT / "eval_confirm"):
        if output.exists():
            counts = {name: len(list((output / name).glob("*.json"))) for name in EVAL_NAMES if (output / name).exists()}
            print(f"  {output.name}: {sum(counts.values())} results over {len(counts)} terrains")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=("stage1", "stage2", "eval", "confirm", "status"))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--recipes", nargs="*", default=None)
    parser.add_argument("--jobs", nargs="*", default=None, help="stage2: parent+arm items")
    parser.add_argument("--ids", nargs="*", default=None, help="eval/confirm: checkpoint ids")
    parser.add_argument("--conditions", nargs="*", default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RECORDS.mkdir(parents=True, exist_ok=True)
    if args.phase == "status":
        phase_status()
        return
    lock = open(OUT / "gpu.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another v28 GPU phase is running")
    log(f"phase {args.phase} started")
    if args.phase == "stage1":
        phase_stage1(args.seeds, args.recipes or STAGE1_RECIPES)
    elif args.phase == "stage2":
        if not args.jobs:
            raise SystemExit("--jobs is required for stage2")
        phase_stage2(args.seeds, args.jobs)
    elif args.phase == "eval":
        phase_eval(OUT / "eval", 2028, args.ids, args.conditions)
    elif args.phase == "confirm":
        if not args.ids:
            raise SystemExit("--ids is required for confirm")
        phase_eval(OUT / "eval_confirm", 2029, args.ids, args.conditions)
    log(f"phase {args.phase} finished")


if __name__ == "__main__":
    main()
