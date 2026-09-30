"""Execute only v17 paired preparation, development checks, and training."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from week03_ant.lp_study_v17 import (
    ARMS, ART, ENVS, EXPERIMENT, ITERATIONS, ROOT, STEPS_PER_ENV,
    START, TRAINING_GEOMETRY, TRAINING_SEED, TRANSITIONS, V16_CONTROL_SHA, WORK,
    cache_snapshot, check_initial, init_path, legacy_hashes, prepare_checkpoint,
    save_json, sha, source_hashes, training_parameters, utc, validate_learning_log,
    validate_teacher, verify_hashes,
)

PYTHON = str(ROOT.parent / "run-python")
CACHE = Path("/tmp/isaaclab/terrains")


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def run(command: list[str], label: str) -> None:
    path = WORK / f"{label}.log"
    begin, started = time.monotonic(), utc()
    with path.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = {"label": label, "command": command, "started_utc": started,
              "finished_utc": utc(), "seconds": time.monotonic() - begin,
              "returncode": result.returncode, "log": str(path.relative_to(ROOT)),
              "log_sha256": sha(path)}
    with (ART / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"[v17] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode:
        print("\n".join(path.read_text().splitlines()[-30:]), flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)
    for flag in ("--audit-output", "--sampling-output", "--output"):
        if flag in command:
            output = Path(command[command.index(flag) + 1])
            if not output.is_file():
                raise FileNotFoundError(f"successful command omitted {flag}: {output}")


def train_command(arm: str, audit: Path, sampling: Path, expected: Path | None = None,
                  *, iterations: int = ITERATIONS, envs: int = ENVS,
                  development: bool = False, device: str = "cuda:1") -> list[str]:
    if arm not in ARMS or iterations <= 0 or envs <= 0:
        raise ValueError("invalid training command")
    if development and (envs != 256 or iterations != 300):
        raise ValueError("only the amended 256-env/300-iteration development run is allowed")
    if not development and (envs, iterations) not in ((ENVS, 2), (ENVS, ITERATIONS)):
        raise ValueError("invalid full/capacity budget")
    phase = "preflight" if development else "capacity" if iterations == 2 else "paired"
    command = [
        PYTHON, "scripts/lp_v17.py", "train", "--arm", arm,
        "--audit-output", str(audit), "--sampling-output", str(sampling),
        "--headless", "--device", device, "--num_envs", str(envs),
        "--max_iterations", str(iterations), "--seed", str(TRAINING_SEED),
        "--run_name", f"v17_{phase}_{arm}", "--resume", "--load_run", "v17_init",
        "--checkpoint", "model_0.pt",
        f"env.scene.terrain.terrain_generator.seed={TRAINING_GEOMETRY}",
        f"agent.device={device}",
    ]
    if development:
        command[3:3] = ["--development", "--stage-size", "16"]
    if expected is not None:
        command += ["--expected-initial", str(expected)]
    return command


def assert_cache(expected: dict) -> None:
    if cache_snapshot(CACHE, expected["geometries"]) != expected:
        raise ValueError("training terrain cache changed")


def validate_sampling(data: dict, arm: str, *, iterations: int, envs: int,
                      stage_size: int, require_two: bool) -> None:
    if (data.get("mode") != arm or data.get("stage_size") != stage_size
            or data.get("seed") != 49017 or data.get("invalid_nonfinite_count") != 0
            or data.get("policy_steps") != iterations * STEPS_PER_ENV
            or data.get("transition_steps") != iterations * STEPS_PER_ENV * envs):
        raise ValueError("sampling treatment, finite evidence, or transition budget differs")
    if (len(data.get("lane_transition_counts", [])) != 35
            or min(data["lane_transition_counts"]) < 0
            or sum(data["lane_transition_counts"]) != data["transition_steps"]
            or len(data.get("sampled_episode_counts", [])) != 35
            or len(data.get("completed_counts", [])) != 35
            or sum(data["sampled_episode_counts"]) != sum(data["completed_counts"])):
        raise ValueError("sampling occupancy or completed-episode accounting differs")
    if (data.get("first_episode_exclusions", -1) < 0
            or data["first_episode_exclusions"] > envs
            or data.get("stage_updates") != len(data.get("stages", []))):
        raise ValueError("invalid LP episode/window counts")
    if require_two and data["stage_updates"] < 2:
        raise ValueError("two signed-LP windows were not reached")
    if any((len(stage.get("counts", [])) != 30 or min(stage["counts"]) < 4)
           for stage in data["stages"]):
        raise ValueError("LP window missed the per-task coverage gate")


def prepare() -> None:
    if (ART / "initial_shared.json").exists():
        raise FileExistsError("v17 paired initialization already prepared")
    save_json(ART / "initial_shared.json", prepare_checkpoint(init_path()))


def prepare_cache(device: str) -> None:
    """Construct geometry101 without a single scored episode before snapshotting it."""
    if not (ART / "initial_shared.json").is_file():
        raise ValueError("prepare the shared checkpoint first")
    output = WORK / "prepare_training_geometry101.json"
    run([PYTHON, "scripts/evaluate_contact_v16.py", "--headless", "--device", device,
         "--controller", "history_original", "--geometry", str(TRAINING_GEOMETRY),
         "--seed", str(TRAINING_SEED), "--phase", "prepare", "--scenario", "mixed",
         "--seconds", "16", "--num_envs", "35", "--output", str(output),
         "--checkpoint", str(START)], "prepare_training_geometry101")
    if read(output).get("scored_episodes") != 0:
        raise ValueError("training geometry prewarm unexpectedly scored episodes")
    save_json(ART / "training_cache_preparation.json", {
        "created_utc": utc(), "phase": "prepare", "development_only": True,
        "path": str(output.relative_to(ROOT)), "sha256": sha(output),
        "cache": cache_snapshot(CACHE, [TRAINING_GEOMETRY]),
    })


def _paired_development(*, device: str, development: bool) -> None:
    if not (ART / "initial_shared.json").is_file():
        raise ValueError("prepare shared v17 initialization first")
    prepared = read(ART / "training_cache_preparation.json")
    assert_cache(prepared["cache"])
    if not development and read(ART / "preflight.json").get("passed") is not True:
        raise ValueError("paired small-environment preflight must pass first")
    cache = cache_snapshot(CACHE, [TRAINING_GEOMETRY])
    phase, iterations, envs, stage_size = (
        ("preflight_v2", 300, 256, 16) if development else ("capacity", 2, ENVS, 1024))
    first, records = None, {}
    for arm in ARMS:
        audit, sampling = WORK / f"{phase}_{arm}_initial.json", WORK / f"{phase}_{arm}_sampling.json"
        assert_cache(cache)
        run(train_command(arm, audit, sampling, first, iterations=iterations, envs=envs,
                          development=development, device=device), f"{phase}_{arm}")
        assert_cache(cache)
        initial, sample = read(audit), read(sampling)
        if first is not None:
            check_initial(initial, read(first))
            if initial["initial_lane_sha256"] != read(first)["initial_lane_sha256"]:
                raise ValueError("paired initial lane layout differs")
        validate_sampling(sample, arm, iterations=iterations, envs=envs,
                          stage_size=stage_size, require_two=development)
        first = first or audit
        records[arm] = {"initial": str(audit.relative_to(ROOT)), "initial_sha256": sha(audit),
                        "sampling": str(sampling.relative_to(ROOT)), "sampling_sha256": sha(sampling),
                        "stage_updates": sample["stage_updates"],
                        "transition_steps": sample["transition_steps"],
                        "initial_pairing_verified": True}
    manifest = "preflight" if development else phase
    save_json(ART / f"{manifest}.json", {"created_utc": utc(), "passed": True,
              "development_only": True, "records": records, "terrain_cache": cache,
              "paired_initialization": True, "stage_size": stage_size})


def freeze_train() -> None:
    preflight, capacity = read(ART / "preflight.json"), read(ART / "capacity.json")
    if not preflight.get("passed") or not capacity.get("passed"):
        raise ValueError("v17 development preflight/capacity incomplete")
    assert_cache(capacity["terrain_cache"])
    if preflight["terrain_cache"] != capacity["terrain_cache"]:
        raise ValueError("development geometry cache differed")
    save_json(ART / "training_frozen.json", {
        "frozen_at": utc(), "source_sha256": source_hashes(training=True),
        "starting_checkpoint": str(init_path().relative_to(ROOT)),
        "starting_checkpoint_sha256": sha(init_path()),
        "source_checkpoint_sha256": V16_CONTROL_SHA, "legacy": legacy_hashes(),
        "training_seed": TRAINING_SEED, "geometry": TRAINING_GEOMETRY,
        "iterations_per_arm": ITERATIONS, "envs": ENVS, "steps_per_env": STEPS_PER_ENV,
        "transitions_per_arm": TRANSITIONS, "arms": list(ARMS),
        "selection": "only final iteration249", "capacity_sha256": sha(ART / "capacity.json"),
        "preflight_sha256": sha(ART / "preflight.json"),
        "terrain_cache": capacity["terrain_cache"],
    })


def verify_training_freeze() -> dict:
    frozen = read(ART / "training_frozen.json")
    verify_hashes(frozen["source_sha256"])
    if frozen["legacy"] != legacy_hashes():
        raise ValueError("legacy graph changed")
    verify_hashes({frozen["starting_checkpoint"]: frozen["starting_checkpoint_sha256"]})
    if (frozen["source_checkpoint_sha256"] != V16_CONTROL_SHA
            or frozen["arms"] != list(ARMS)
            or frozen["preflight_sha256"] != sha(ART / "preflight.json")
            or frozen["capacity_sha256"] != sha(ART / "capacity.json")):
        raise ValueError("training freeze configuration/development proof changed")
    assert_cache(frozen["terrain_cache"])
    return frozen


def train(device: str) -> None:
    import torch

    if (ART / "trained_models.json").exists():
        raise FileExistsError("v17 paired training already complete")
    verify_training_freeze()
    first, models, audits, summaries = None, {}, {}, {}
    for arm in ARMS:
        verify_training_freeze()
        initial = ART / "training" / f"{arm}_initial.json"
        sampling = ART / "training" / f"{arm}_sampling.json"
        if not initial.exists() and not sampling.exists():
            run(train_command(arm, initial, sampling, first, device=device), f"train_{arm}")
        elif not initial.is_file() or not sampling.is_file():
            raise ValueError("incomplete prior v17 arm; refusing to overwrite evidence")
        verify_training_freeze()
        data, sample = read(initial), read(sampling)
        validate_sampling(sample, arm, iterations=ITERATIONS, envs=ENVS,
                          stage_size=1024, require_two=True)
        if first is not None:
            check_initial(data, read(first))
            if data["initial_lane_sha256"] != read(first)["initial_lane_sha256"]:
                raise ValueError("full paired lane initialization differs")
        first = first or initial
        logdir = ROOT / data["log_dir"]
        saved = training_parameters(logdir, arm)
        if saved["normalized"] != data["normalized_parameters"] or saved["sha256"] != data["saved_parameter_sha256"]:
            raise ValueError("saved v17 parameters changed during training")
        final = logdir / "model_249.pt"
        state = torch.load(final, map_location="cpu", weights_only=False)
        if (state.get("iter") != 249
                or not all(torch.isfinite(value).all() for value in state["model_state_dict"].values())):
            raise ValueError("invalid v17 final checkpoint")
        validate_teacher(state["model_state_dict"])
        for values in state["optimizer_state_dict"]["state"].values():
            for value in values.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError("nonfinite v17 final optimizer state")
        audits[arm] = validate_learning_log(logdir, WORK / f"train_{arm}.log", ITERATIONS, TRANSITIONS)
        destination = ART / "runs" / arm / "model_249.pt"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(final, destination)
            for name in ("env.yaml", "agent.yaml"):
                shutil.copyfile(logdir / "params" / name, destination.parent / name)
        if sha(destination) != sha(final):
            raise ValueError("copied v17 model differs from trained final checkpoint")
        summaries[arm] = {"path": str(sampling.relative_to(ROOT)), "sha256": sha(sampling),
                          "stage_updates": sample["stage_updates"], "stage_size": sample["stage_size"],
                          "first_episode_exclusions": sample["first_episode_exclusions"],
                          "invalid_nonfinite_count": sample["invalid_nonfinite_count"],
                          "transition_steps": sample["transition_steps"]}
        models[arm] = {"checkpoint": str(destination.relative_to(ROOT)), "sha256": sha(destination),
                       "source_checkpoint": str(final.relative_to(ROOT)), "iteration": 249,
                       "transitions": TRANSITIONS, "initial_audit_sha256": sha(initial),
                       "initial_pairing_verified": True, "sampling_sha256": sha(sampling)}
    save_json(ART / "training_validation.json", {"created_utc": utc(), "records": audits,
              "sampling_summaries": summaries})
    save_json(ART / "trained_models.json", {"completed_utc": utc(), "models": models,
              "total_training_transitions": 2 * TRANSITIONS,
              "training_freeze_sha256": sha(ART / "training_frozen.json"),
              "training_validation_sha256": sha(ART / "training_validation.json")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "prepare_cache", "preflight", "capacity", "freeze_train", "train"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/hybrid_v11_20260922/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"prepare": prepare, "prepare_cache": lambda: prepare_cache(args.device),
         "preflight": lambda: _paired_development(device=args.device, development=True),
         "capacity": lambda: _paired_development(device=args.device, development=False),
         "freeze_train": freeze_train, "train": lambda: train(args.device)}[args.phase]()


if __name__ == "__main__":
    main()
