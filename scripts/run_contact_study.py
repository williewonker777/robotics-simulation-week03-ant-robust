"""Run only the preregistered v16 paired training/provenance stages; no holdouts."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from week03_ant.contact_study import (
    ARMS, ART, ENVS, ITERATIONS, ROOT, STEPS_PER_ENV, TRAINING_GEOMETRY, TRAINING_SEED, TRANSITIONS,
    V15_SHA, WORK, cache_snapshot, check_initial, init_path, prepare_checkpoint, save_json, sha, source_hashes,
    legacy_hashes, training_parameters, utc, validate_learning_log, validate_teacher, verify_hashes,
)

PYTHON = str(ROOT.parent / "run-python")
CACHE = Path("/tmp/isaaclab/terrains")


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def run(command: list[str], label: str) -> None:
    path = WORK / f"{label}.log"
    started, begin = utc(), time.monotonic()
    with path.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = {"label": label, "command": command, "started_utc": started, "finished_utc": utc(),
              "seconds": time.monotonic() - begin, "returncode": result.returncode,
              "log": str(path.relative_to(ROOT)), "log_sha256": sha(path)}
    with (ART / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    if "--audit-output" in command:
        audit_output = Path(command[command.index("--audit-output") + 1])
        if not audit_output.is_file():
            raise FileNotFoundError(f"successful command did not create audit output: {audit_output}")


def train_command(arm: str, output: Path, expected: Path | None = None, *, iterations: int = ITERATIONS,
                  envs: int = ENVS, device: str = "cuda:1") -> list[str]:
    command = [
        PYTHON, "scripts/contact_v16.py", "train", "--arm", arm, "--audit-output", str(output),
        "--headless", "--device", device, "--num_envs", str(envs), "--max_iterations", str(iterations),
        "--seed", str(TRAINING_SEED), "--run_name", f"v16_{'capacity' if iterations == 2 else 'paired'}_{arm}",
        "--resume", "--load_run", "v16_init", "--checkpoint", "model_0.pt",
        f"env.scene.terrain.terrain_generator.seed={TRAINING_GEOMETRY}", f"agent.device={device}",
    ]
    if expected is not None:
        command += ["--expected-initial", str(expected)]
    return command


def assert_cache(expected: dict) -> None:
    if cache_snapshot(CACHE, expected["geometries"]) != expected:
        raise ValueError("terrain cache changed")


def validate_rollout(rollout: dict, expected_samples: int, stage: str) -> None:
    """Reject reset-only, short, invalid, or incomplete contact coverage evidence."""
    required = {"samples", "sampled_rows", "valid_rows", "invalid_rows", "invalid_fraction",
                "contact_fraction", "terrain_contact_fraction", "flat_contact_fraction",
                "per_foot_contact_fraction", "per_target_per_foot_contact_fraction"}
    if not required <= rollout.keys():
        raise ValueError(f"{stage} contact rollout schema differs")
    if rollout["samples"] != expected_samples:
        raise ValueError(f"{stage} contact rollout sample count differs: expected {expected_samples}")
    if rollout["sampled_rows"] <= 0 or rollout["valid_rows"] <= 0:
        raise ValueError(f"{stage} contact rollout has no usable post-action rows")
    if rollout["invalid_fraction"] > .01:
        raise ValueError(f"{stage} contact rollout invalid fraction exceeds one percent")
    if not .02 <= rollout["contact_fraction"] <= .98:
        raise ValueError(f"{stage} contact rollout has implausible pooled contact coverage")
    if min(rollout["terrain_contact_fraction"], rollout["flat_contact_fraction"]) <= 0.:
        raise ValueError(f"{stage} contact rollout did not observe both configured collision targets")
    feet = rollout["per_foot_contact_fraction"]
    targets = rollout["per_target_per_foot_contact_fraction"]
    if len(feet) != 4 or min(feet) <= 0.:
        raise ValueError(f"{stage} contact rollout did not observe every foot")
    if len(targets) != 4 or any(len(row) != 2 or min(row) <= 0. for row in targets):
        raise ValueError(f"{stage} contact rollout did not observe every foot on both targets")


def prepare() -> None:
    if (ART / "initial_shared.json").exists():
        raise FileExistsError("v16 initialization already prepared")
    initial = prepare_checkpoint(init_path())
    save_json(ART / "initial_shared.json", initial)


def capacity(device: str) -> None:
    """Paired 4096-env two-iteration sensor/reward preflight; never a selection run."""
    if not (ART / "preflight.json").is_file():
        raise ValueError("64-environment preflight must pass before capacity smoke")
    cache = cache_snapshot(CACHE, [TRAINING_GEOMETRY])
    first, records = None, {}
    for arm in ARMS:
        output = WORK / f"capacity_{arm}_initial.json"
        assert_cache(cache)
        run(train_command(arm, output, first, iterations=2, device=device), f"capacity_{arm}")
        assert_cache(cache)
        data = read(output)
        if data["contact_slip_weight"] != float(arm == "slip"):
            raise ValueError("contact treatment was not applied")
        rollout_path = output.with_name(output.stem + "_contact_rollout.json")
        rollout = read(rollout_path)["rollout"]
        validate_rollout(rollout, 2 * STEPS_PER_ENV, "capacity")
        if first is not None:
            check_initial(data, read(first))
        first = first or output
        records[arm] = {"initial": str(output.relative_to(ROOT)), "sha256": sha(output),
                        "contact_preflight": data["contact_preflight"], "contact_rollout": str(rollout_path.relative_to(ROOT)),
                        "contact_rollout_sha256": sha(rollout_path), "rollout_contact": rollout,
                        "initial_pairing_verified": True,
                        "saved_config_sha256": data["saved_parameter_sha256"]}
    save_json(ART / "capacity.json", {"created_utc": utc(), "records": records, "terrain_cache": cache,
              "development_only": True, "paired_initialization": True, "contact_geometry_called_in_control": True})


def preflight(device: str) -> None:
    """First paired 64-env two-iteration gate; it never contributes a trained model."""
    if not (ART / "initial_shared.json").is_file():
        raise ValueError("prepare shared initialization first")
    cache = cache_snapshot(CACHE, [TRAINING_GEOMETRY])
    first, records = None, {}
    for arm in ARMS:
        output = WORK / f"preflight_{arm}_initial.json"
        assert_cache(cache)
        run(train_command(arm, output, first, iterations=2, envs=64, device=device), f"preflight_{arm}")
        assert_cache(cache)
        data = read(output)
        rollout_path = output.with_name(output.stem + "_contact_rollout.json")
        rollout = read(rollout_path)["rollout"]
        validate_rollout(rollout, 2 * STEPS_PER_ENV, "preflight")
        if first is not None:
            check_initial(data, read(first))
        first = first or output
        records[arm] = {"path": str(output.relative_to(ROOT)), "sha256": sha(output),
                        "contact_preflight": data["contact_preflight"], "contact_rollout": str(rollout_path.relative_to(ROOT)),
                        "contact_rollout_sha256": sha(rollout_path), "rollout_contact": rollout,
                        "initial_pairing_verified": True}
    save_json(ART / "preflight.json", {"created_utc": utc(), "passed": True, "records": records,
              "terrain_cache": cache, "development_only": True, "contact_geometry_called_in_both_arms": True})


def freeze_train() -> None:
    preflight_data = read(ART / "preflight.json")
    if preflight_data.get("passed") is not True:
        raise ValueError("64-environment preflight has not passed")
    capacity_data = read(ART / "capacity.json")
    assert_cache(capacity_data["terrain_cache"])
    # Hashing source files deliberately fails until the independently owned task/sensor lane exists.
    sources = source_hashes()
    save_json(ART / "training_frozen.json", {
        "frozen_at": utc(), "source_sha256": sources,
        "starting_checkpoint": str(init_path().relative_to(ROOT)), "starting_checkpoint_sha256": sha(init_path()),
        "source_checkpoint_sha256": V15_SHA, "legacy": legacy_hashes(), "training_seed": TRAINING_SEED, "geometry": TRAINING_GEOMETRY,
        "iterations_per_arm": ITERATIONS, "envs": ENVS, "steps_per_env": STEPS_PER_ENV,
        "transitions_per_arm": TRANSITIONS, "arms": list(ARMS), "selection": "only final iteration249",
        "capacity_sha256": sha(ART / "capacity.json"), "preflight_sha256": sha(ART / "preflight.json"),
        "terrain_cache": capacity_data["terrain_cache"],
    })


def verify_training_freeze() -> dict:
    frozen = read(ART / "training_frozen.json")
    verify_hashes(frozen["source_sha256"])
    if frozen.get("legacy") != legacy_hashes():
        raise ValueError("legacy inventory changed")
    verify_hashes({frozen["starting_checkpoint"]: frozen["starting_checkpoint_sha256"]})
    if frozen["source_checkpoint_sha256"] != V15_SHA:
        raise ValueError("starting source hash differs")
    assert_cache(frozen["terrain_cache"])
    return frozen


def train(device: str) -> None:
    import torch

    if (ART / "trained_models.json").exists():
        raise FileExistsError("v16 paired training already complete")
    freeze = verify_training_freeze()
    first, models, audits = None, {}, {}
    for arm in ARMS:
        verify_training_freeze()
        output = ART / "training" / f"{arm}_initial.json"
        run(train_command(arm, output, first, device=device), f"train_{arm}")
        verify_training_freeze()
        data = read(output)
        rollout_path = output.with_name(output.stem + "_contact_rollout.json")
        rollout = read(rollout_path)["rollout"]
        validate_rollout(rollout, ITERATIONS * STEPS_PER_ENV, "training")
        if first is not None:
            check_initial(data, read(first))
        first = first or output
        logdir = ROOT / data["log_dir"]
        saved = training_parameters(logdir, arm)
        if saved["normalized"] != data["normalized_parameters"] or saved["sha256"] != data["saved_parameter_sha256"]:
            raise ValueError("saved parameters changed during training")
        final = logdir / "model_249.pt"
        state = torch.load(final, map_location="cpu", weights_only=False)
        if state.get("iter") != 249 or not all(torch.isfinite(value).all() for value in state["model_state_dict"].values()):
            raise ValueError("invalid final checkpoint")
        validate_teacher(state["model_state_dict"])
        for values in state["optimizer_state_dict"]["state"].values():
            for value in values.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError("nonfinite final optimizer state")
        audits[arm] = validate_learning_log(logdir, WORK / f"train_{arm}.log", ITERATIONS, TRANSITIONS)
        destination = ART / "runs" / arm / "model_249.pt"
        destination.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(final, destination)
        for name in ("env.yaml", "agent.yaml"):
            shutil.copyfile(logdir / "params" / name, destination.parent / name)
        models[arm] = {"checkpoint": str(destination.relative_to(ROOT)), "sha256": sha(destination),
                       "source_checkpoint": str(final.relative_to(ROOT)), "iteration": 249,
                       "transitions": TRANSITIONS, "initial_audit_sha256": sha(output),
                       "initial_pairing_verified": True,
                       "contact_rollout": str(rollout_path.relative_to(ROOT)),
                       "contact_rollout_sha256": sha(rollout_path),
                       "contact_rollout_invalid_fraction": rollout["invalid_fraction"]}
    save_json(ART / "training_validation.json", {"created_utc": utc(), "records": audits,
              "contact_rollouts": {arm: models[arm]["contact_rollout_sha256"] for arm in ARMS}})
    save_json(ART / "trained_models.json", {"completed_utc": utc(), "models": models,
              "total_training_transitions": 2 * TRANSITIONS,
              "training_freeze_sha256": sha(ART / "training_frozen.json"),
              "training_validation_sha256": sha(ART / "training_validation.json")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "preflight", "capacity", "freeze_train", "train"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/hybrid_v11_20260922/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"prepare": prepare, "preflight": lambda: preflight(args.device), "capacity": lambda: capacity(args.device), "freeze_train": freeze_train,
         "train": lambda: train(args.device)}[args.phase]()


if __name__ == "__main__":
    main()
