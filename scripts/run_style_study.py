"""Run guarded v18 paired development and final training, one GPU job at a time."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from week03_ant.command_study import validate_learning_log
from week03_ant.style_study_v18 import (
    ARMS, ART, ENVS, EXPERIMENT, ITERATIONS, ROOT, START, TRAIN_GEOMETRY, TRAIN_SEED,
    TRANSITIONS, V16_CONTROL_SHA, WORK, init_path, legacy_hashes,
    normalize_training_parameters, prepare_checkpoint, save_json, sha, source_hashes, utc,
    validate_training_start, verify_hashes,
)
from week03_ant.posture_study import cache_snapshot, validate_teacher

PYTHON = str(ROOT.parent / "run-python")
CACHE = Path("/tmp/isaaclab/terrains")


def read(path):
    return json.loads(Path(path).read_text())


def run(command, label):
    path = WORK / f"{label}.log"
    if path.exists():
        attempt = 2
        while (WORK / f"{label}_attempt{attempt}.log").exists():
            attempt += 1
        path = WORK / f"{label}_attempt{attempt}.log"
    started, begin = utc(), time.monotonic()
    with path.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = {"label": label, "command": command, "started_utc": started,
              "finished_utc": utc(), "seconds": time.monotonic() - begin,
              "returncode": result.returncode, "log": str(path.relative_to(ROOT)),
              "log_sha256": sha(path)}
    with (ART / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"[v18] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode:
        print("\n".join(path.read_text().splitlines()[-30:]), flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)
    for flag in ("--output", "--audit-output", "--exposure-output"):
        if flag in command and not Path(command[command.index(flag) + 1]).is_file():
            raise FileNotFoundError(f"v18 command omitted {flag} output")


def train_command(arm, audit, exposure, expected=None, *, envs=ENVS, iterations=ITERATIONS,
                  device="cuda:1"):
    if arm not in ARMS or (envs, iterations) not in ((256, 16), (4096, 2), (ENVS, ITERATIONS)):
        raise ValueError("invalid v18 arm/budget")
    phase = "preflight" if envs == 256 else "capacity" if iterations == 2 else "paired"
    command = [PYTHON, "scripts/style_v18.py", "train", "--arm", arm,
               "--audit-output", str(audit), "--exposure-output", str(exposure),
               "--headless", "--device", device, "--num_envs", str(envs),
               "--max_iterations", str(iterations), "--seed", str(TRAIN_SEED),
               "--run_name", f"v18_{phase}_{arm}", "--resume", "--load_run", "v18_init",
               "--checkpoint", "model_0.pt",
               f"env.scene.terrain.terrain_generator.seed={TRAIN_GEOMETRY}",
               f"agent.device={device}"]
    if expected is not None:
        command += ["--expected-initial", str(expected)]
    return command


def assert_cache(expected):
    if cache_snapshot(CACHE, expected["geometries"]) != expected:
        raise ValueError("v18 training terrain cache changed")


def prepare():
    if (ART / "initial_shared.json").exists():
        raise FileExistsError("v18 shared init already prepared")
    save_json(ART / "initial_shared.json", prepare_checkpoint(init_path()))


def prepare_cache(device):
    if not (ART / "initial_shared.json").is_file():
        raise ValueError("prepare v18 init first")
    output = WORK / "prepare_training_geometry104.json"
    run([PYTHON, "scripts/evaluate_contact_v16.py", "--headless", "--device", device,
         "--controller", "history_original", "--geometry", str(TRAIN_GEOMETRY),
         "--seed", str(TRAIN_SEED), "--phase", "prepare", "--scenario", "mixed",
         "--seconds", "16", "--num_envs", "35", "--output", str(output),
         "--checkpoint", str(START)], "prepare_training_geometry104")
    if read(output).get("scored_episodes") != 0:
        raise ValueError("v18 cache preparation scored episodes")
    save_json(ART / "training_cache.json", {"created_utc": utc(), "development_only": True,
              "preparation": str(output.relative_to(ROOT)), "preparation_sha256": sha(output),
              "cache": cache_snapshot(CACHE, [TRAIN_GEOMETRY])})


def validate_exposure(data, arm, envs, iterations, *, require_terrain_coverage=True):
    if data["arm"] != arm or data["policy_steps"] != iterations * 32 or data["transition_steps"] != envs * iterations * 32:
        raise ValueError("v18 exposure budget differs")
    counts, teacher = data["lane_transition_counts"], data["lane_teacher_counts"]
    if len(counts) != 35 or len(teacher) != 35 or any(c < 0 or t < 0 or t > c for c, t in zip(counts, teacher)):
        raise ValueError("v18 exposure counters invalid")
    if sum(counts) != data["transition_steps"] or sum(teacher) <= 0 or sum(teacher) == sum(counts):
        raise ValueError("v18 detector never visited both mask states")
    family_count, family_teacher = data["family_transition_counts"], data["family_teacher_counts"]
    if len(family_count) != 7 or len(family_teacher) != 7:
        raise ValueError("v18 family telemetry missing")
    if family_count[-1] == 0 or family_teacher[-1] / family_count[-1] < .95:
        raise ValueError("v18 flat family not teacher-anchored")
    if (require_terrain_coverage
            and sum(c - t > 0 for c, t in zip(family_count[:-1], family_teacher[:-1])) < 4):
        raise ValueError("v18 detector did not see four rough families")


def paired_development(device, phase):
    if phase not in ("preflight", "capacity"):
        raise ValueError("invalid v18 development phase")
    prior = read(ART / "training_cache.json")["cache"]
    assert_cache(prior)
    if phase == "capacity" and not read(ART / "preflight.json")["passed"]:
        raise ValueError("v18 preflight not passed")
    envs, iterations = (256, 16) if phase == "preflight" else (4096, 2)
    first, records = None, {}
    for arm in ARMS:
        assert_cache(prior)
        audit, exposure = WORK / f"{phase}_{arm}_initial.json", WORK / f"{phase}_{arm}_exposure.json"
        if not audit.exists() and not exposure.exists():
            run(train_command(arm, audit, exposure, first, envs=envs, iterations=iterations,
                              device=device), f"{phase}_{arm}")
        elif not audit.is_file() or not exposure.is_file():
            raise ValueError("incomplete prior v18 development arm")
        assert_cache(prior)
        initial, counts = read(audit), read(exposure)
        validate_exposure(counts, arm, envs, iterations,
                          require_terrain_coverage=phase == "preflight")
        first = first or audit
        records[arm] = {"initial": str(audit.relative_to(ROOT)), "initial_sha256": sha(audit),
                        "exposure": str(exposure.relative_to(ROOT)), "exposure_sha256": sha(exposure),
                        "teacher_fraction_by_family": [t / c if c else None for c, t in
                                                       zip(counts["family_transition_counts"],
                                                           counts["family_teacher_counts"])]}
    save_json(ART / f"{phase}.json", {"created_utc": utc(), "passed": True,
              "development_only": True, "paired_initialization": True,
              "terrain_cache": prior, "records": records})


def freeze_train():
    preflight, capacity = read(ART / "preflight.json"), read(ART / "capacity.json")
    if not preflight["passed"] or not capacity["passed"] or preflight["terrain_cache"] != capacity["terrain_cache"]:
        raise ValueError("v18 paired development not complete")
    assert_cache(capacity["terrain_cache"])
    save_json(ART / "training_frozen.json", {
        "frozen_at": utc(), "source_sha256": source_hashes(), "legacy": legacy_hashes(),
        "starting_checkpoint": str(init_path().relative_to(ROOT)),
        "starting_checkpoint_sha256": sha(init_path()),
        "source_checkpoint_sha256": V16_CONTROL_SHA,
        "seed": TRAIN_SEED, "geometry": TRAIN_GEOMETRY,
        "envs": ENVS, "iterations_per_arm": ITERATIONS, "steps_per_env": 32,
        "transitions_per_arm": TRANSITIONS, "arms": list(ARMS), "selection": "final model249 only",
        "preflight_sha256": sha(ART / "preflight.json"),
        "capacity_sha256": sha(ART / "capacity.json"), "terrain_cache": capacity["terrain_cache"],
    })


def verify_training_freeze():
    frozen = read(ART / "training_frozen.json")
    verify_hashes(frozen["source_sha256"])
    if frozen["legacy"] != legacy_hashes() or frozen["source_checkpoint_sha256"] != V16_CONTROL_SHA:
        raise ValueError("v18 frozen ancestry changed")
    verify_hashes({frozen["starting_checkpoint"]: frozen["starting_checkpoint_sha256"]})
    if sha(ART / "preflight.json") != frozen["preflight_sha256"] or sha(ART / "capacity.json") != frozen["capacity_sha256"]:
        raise ValueError("v18 development evidence changed")
    assert_cache(frozen["terrain_cache"])
    return frozen


def train(device):
    import torch

    if (ART / "trained_models.json").exists():
        raise FileExistsError("v18 paired models already complete")
    verify_training_freeze()
    first, models, audits = None, {}, {}
    for arm in ARMS:
        verify_training_freeze()
        initial = ART / "training" / f"{arm}_initial.json"
        exposure = ART / "training" / f"{arm}_exposure.json"
        initial.parent.mkdir(parents=True, exist_ok=True)
        if not initial.exists() and not exposure.exists():
            run(train_command(arm, initial, exposure, first, device=device), f"train_{arm}")
        elif not initial.is_file() or not exposure.is_file():
            raise ValueError("incomplete v18 arm; refusing overwrite")
        verify_training_freeze()
        data, counts = read(initial), read(exposure)
        validate_exposure(counts, arm, ENVS, ITERATIONS)
        first = first or initial
        logdir = ROOT / data["log_dir"]
        saved = normalize_training_parameters(logdir, arm)
        if saved["normalized"] != data["normalized_parameters"] or saved["sha256"] != data["saved_parameter_sha256"]:
            raise ValueError("v18 saved YAML changed")
        final = logdir / "model_249.pt"
        state = torch.load(final, map_location="cpu", weights_only=False)
        if state.get("iter") != 249 or not all(torch.isfinite(v).all() for v in state["model_state_dict"].values()):
            raise ValueError("v18 final checkpoint invalid")
        validate_teacher(state["model_state_dict"])
        for values in state["optimizer_state_dict"]["state"].values():
            for value in values.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError("nonfinite v18 Adam state")
        audits[arm] = validate_learning_log(logdir, WORK / f"train_{arm}.log", ITERATIONS, TRANSITIONS)
        destination = ART / "runs" / arm / "model_249.pt"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(final, destination)
            for name in ("env.yaml", "agent.yaml"):
                shutil.copyfile(logdir / "params" / name, destination.parent / name)
        if sha(destination) != sha(final):
            raise ValueError("v18 copied model differs")
        models[arm] = {"checkpoint": str(destination.relative_to(ROOT)), "sha256": sha(destination),
                       "source_checkpoint": str(final.relative_to(ROOT)), "iteration": 249,
                       "transitions": TRANSITIONS, "initial_audit_sha256": sha(initial),
                       "exposure_sha256": sha(exposure)}
    save_json(ART / "training_validation.json", {"created_utc": utc(), "records": audits})
    save_json(ART / "trained_models.json", {"completed_utc": utc(), "models": models,
              "total_training_transitions": 2 * TRANSITIONS,
              "training_freeze_sha256": sha(ART / "training_frozen.json"),
              "training_validation_sha256": sha(ART / "training_validation.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "prepare_cache", "preflight", "capacity", "freeze_train", "train"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/hybrid_v11_20260922/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"prepare": prepare, "prepare_cache": lambda: prepare_cache(args.device),
         "preflight": lambda: paired_development(args.device, "preflight"),
         "capacity": lambda: paired_development(args.device, "capacity"),
         "freeze_train": freeze_train, "train": lambda: train(args.device)}[args.phase]()


if __name__ == "__main__":
    main()
