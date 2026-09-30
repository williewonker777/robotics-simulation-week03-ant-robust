"""One declared paired continuation experiment; exclusive evidence, no retries."""

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from week03_ant.posture_study import (
    ARMS, ART, CONTROLLERS, HOLDOUTS, ROOT, START, START_SHA, WORK,
    cache_snapshot, legacy_hashes, save_json, sha, source_hashes,
    training_parameters, utc, validate_teacher, verify_hashes, evaluation_inputs,
)

PYTHON = str(ROOT.parent / "run-python")
CACHE = Path("/tmp/isaaclab/terrains")
INIT = ROOT / "logs/rsl_rl/week03_ant_adaptive_posture_v13/v13_init/model_0.pt"


def read(path):
    return json.loads(Path(path).read_text())


def run(command, label):
    path = WORK / f"{label}.log"
    start, begin = utc(), time.monotonic()
    with path.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = {"label": label, "command": command, "started_utc": start, "finished_utc": utc(),
              "seconds": time.monotonic() - begin, "returncode": result.returncode,
              "log": str(path.relative_to(ROOT)), "log_sha256": sha(path)}
    if (ART / "evaluation_inputs.json").is_file():
        inputs = evaluation_inputs()
        record.update(evaluation_inputs_sha256=sha(ART / "evaluation_inputs.json"),
                      terrain_cache_manifest_sha256=inputs["terrain_cache_manifest_sha256"])
    with (ART / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"[v13] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode:
        print("\n".join(path.read_text().splitlines()[-30:]), flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)


def train_command(arm, output, expected=None, *, capacity=False, device="cuda:1"):
    command = [PYTHON, "scripts/posture_v13.py", "train", "--arm", arm, "--audit-output", str(output),
               "--headless", "--device", device, "--num_envs", "4096", "--max_iterations", "2" if capacity else "250",
               "--seed", "45", "--run_name", f"v13_{'capacity' if capacity else 'paired'}_{arm}",
               "--resume", "--load_run", "v13_init", "--checkpoint", "model_0.pt",
               "env.scene.terrain.terrain_generator.seed=75", f"agent.device={device}"]
    if expected:
        command += ["--expected-initial", str(expected)]
    return command


def eval_command(controller, geometry, reset, scenario, phase, output, checkpoint, device="cuda:1"):
    return [PYTHON, "scripts/evaluate_posture_v13.py", "--headless", "--device", device,
            "--controller", controller, "--geometry", str(geometry), "--seed", str(reset),
            "--scenario", scenario, "--seconds", "16" if scenario == "mixed" else "64",
            "--num_envs", "175" if scenario == "mixed" else "10", "--phase", phase,
            "--output", str(output), "--checkpoint", str(checkpoint)]


def assert_same_cache(expected):
    if cache_snapshot(CACHE, expected["geometries"]) != expected:
        raise ValueError("terrain cache changed")


def capacity(device):
    before = cache_snapshot(CACHE, [75])
    initial = None
    records = {}
    for arm in ARMS:
        output = WORK / f"capacity_{arm}_initial.json"
        assert_same_cache(before)
        run(train_command(arm, output, initial, capacity=True, device=device), f"capacity_{arm}")
        assert_same_cache(before)
        data = read(output)
        if data.get("default_config_parity") is not True:
            raise ValueError("actual config inheritance was not checked")
        initial = initial or output
        records[arm] = {"initial": str(output.relative_to(ROOT)), "sha256": sha(output),
                        "saved_config_sha256": data["saved_parameter_sha256"]}
    save_json(ART / "capacity.json", {"created_utc": utc(), "records": records,
              "default_config_parity": True, "paired_initial_and_saved_configs": True,
              "terrain_cache": before, "development_only": True})


def freeze_train():
    proof = ART / "preflight.json"
    capacity_path = ART / "capacity.json"
    if read(proof).get("passed") is not True:
        raise ValueError("preflight has not passed")
    capacity_data = read(capacity_path)
    assert_same_cache(capacity_data["terrain_cache"])
    save_json(ART / "training_frozen.json", {
        "frozen_at": utc(), "source_sha256": source_hashes(training=True), "legacy": legacy_hashes(),
        "initial_checkpoint": {"checkpoint": str(INIT.relative_to(ROOT)), "sha256": sha(INIT)},
        "starting_checkpoint_sha256": START_SHA, "training_seed": 45, "geometry": 75,
        "iterations_per_arm": 250, "envs": 4096, "steps_per_env": 32,
        "transitions_per_arm": 32768000, "selection": "only final iteration249; no seed/model selection",
        "posture_config": asdict(PostureConfig()), "history_config": asdict(HistoryGateConfig()),
        "arms": {"control": 0., "adaptive": 1.}, "terrain_cache": capacity_data["terrain_cache"],
        "capacity_sha256": sha(capacity_path), "preflight_sha256": sha(proof),
    })


def verify_training():
    frozen = read(ART / "training_frozen.json")
    verify_hashes(frozen["source_sha256"])
    if frozen["legacy"] != legacy_hashes():
        raise ValueError("legacy inventory changed")
    entry = frozen["initial_checkpoint"]
    verify_hashes({entry["checkpoint"]: entry["sha256"]})
    assert_same_cache(frozen["terrain_cache"])
    return frozen


def train(device):
    import torch

    if (ART / "trained_models.json").exists():
        raise FileExistsError("paired training already complete")
    first, models = None, {}
    for arm in ARMS:
        verify_training()
        output = ART / "training" / f"{arm}_initial.json"
        run(train_command(arm, output, first, device=device), f"train_{arm}")
        verify_training()
        data = read(output)
        first = first or output
        logdir = ROOT / data["log_dir"]
        saved = training_parameters(logdir, arm)
        if saved["normalized"] != data["normalized_parameters"] or saved["sha256"] != data["saved_parameter_sha256"]:
            raise ValueError("saved parameters changed during training")
        final = logdir / "model_249.pt"
        state = torch.load(final, map_location="cpu", weights_only=False)
        if state["iter"] != 249 or not all(torch.isfinite(v).all() for v in state["model_state_dict"].values()):
            raise ValueError("invalid final checkpoint")
        validate_teacher(state["model_state_dict"])
        destination = ART / "runs" / arm / "model_249.pt"
        destination.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(final, destination)
        for name in ("env.yaml", "agent.yaml"):
            shutil.copyfile(logdir / "params" / name, destination.parent / name)
        models[arm] = {"checkpoint": str(destination.relative_to(ROOT)), "sha256": sha(destination),
                       "source_checkpoint": str(final.relative_to(ROOT)), "iteration": 249,
                       "transitions": 32768000, "initial_audit_sha256": sha(output)}
    save_json(ART / "trained_models.json", {"completed_utc": utc(), "models": models,
              "total_training_transitions": 65536000, "training_freeze_sha256": sha(ART / "training_frozen.json")})


def freeze_eval():
    verify_training()
    trained = read(ART / "trained_models.json")
    models = {"original": {"checkpoint": str(START.relative_to(ROOT)), "sha256": START_SHA}, **trained["models"]}
    for entry in models.values():
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
    save_json(ART / "frozen.json", {"frozen_at": utc(), "source_sha256": source_hashes(),
              "legacy": legacy_hashes(), "models": models, "holdouts": HOLDOUTS, "controllers": CONTROLLERS,
              "gate_config": asdict(HistoryGateConfig()), "posture_config": asdict(PostureConfig()),
              "training_freeze_sha256": sha(ART / "training_frozen.json"),
              "trained_models_sha256": sha(ART / "trained_models.json"),
              "primary": {"files": 14, "episodes": 2450, "seconds": 16, "envs": 175},
              "secondary": {"files": 14, "episodes": 140, "seconds": 64, "envs": 10},
              "selection": "all predeclared controllers/maps; only both final249 checkpoints"})


def verify_eval():
    frozen = read(ART / "frozen.json")
    verify_training()
    verify_hashes(frozen["source_sha256"])
    for entry in frozen["models"].values():
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
    if sha(ART / "training_frozen.json") != frozen["training_freeze_sha256"]:
        raise ValueError("training freeze changed")
    return frozen


def prepare_eval(device):
    verify_eval()
    records = []
    for geom, reset in HOLDOUTS:
        output = WORK / f"prepare_holdout_geometry{geom}.json"
        run(eval_command("original", geom, reset, "mixed", "prepare", output, START, device), f"prepare_holdout_geometry{geom}")
        if read(output)["scored_episodes"] != 0:
            raise ValueError("preparation scored episodes")
        records.append({"path": str(output.relative_to(ROOT)), "sha256": sha(output)})
    verify_eval()
    save_json(ART / "terrain_cache.json", {"created_utc": utc(), "preparation": records,
              "cache": cache_snapshot(CACHE, [g for g, _ in HOLDOUTS])})
    save_json(ART / "evaluation_inputs.json", {"created_utc": utc(),
              "experiment_freeze_sha256": sha(ART / "frozen.json"),
              "terrain_cache_manifest_sha256": sha(ART / "terrain_cache.json")})


def evaluate(device, horizon=False):
    from summarize_posture_v13 import audit
    from summarize_history_hybrid import check_initial

    folder, scenario = ("horizon", "stones") if horizon else ("evaluations", "mixed")
    evaluation_inputs()
    cache = read(ART / "terrain_cache.json")["cache"]
    initial = {}
    for controller in CONTROLLERS:
        for geom, reset in HOLDOUTS:
            frozen = verify_eval()
            evaluation_inputs()
            assert_same_cache(cache)
            key = controller.removeprefix("history_")
            if key == "v5":
                key = "original"
            label = f"{controller}__geometry{geom}_reset{reset}"
            output = ART / folder / f"{label}.json"
            run(eval_command(controller, geom, reset, scenario, "holdout", output,
                             ROOT / frozen["models"][key]["checkpoint"], device), f"{folder}_{label}")
            evaluation_inputs()
            assert_same_cache(cache)
            verify_eval()
            data = read(output)
            audit(data)
            check_initial(initial, (geom, reset), data)


def report():
    from summarize_posture_v13 import summarize, markdown

    verify_eval()
    evaluation_inputs()
    result = summarize(ART)
    save_json(ART / "summary.json", result)
    with (ART / "summary.md").open("x") as stream:
        stream.write(markdown(result))
    print(markdown(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capacity", "freeze_train", "train", "freeze_eval", "prepare_eval", "evaluate", "horizon", "report"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/hybrid_v11_20260922/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"capacity": lambda: capacity(args.device), "freeze_train": freeze_train,
         "train": lambda: train(args.device), "freeze_eval": freeze_eval,
         "prepare_eval": lambda: prepare_eval(args.device), "evaluate": lambda: evaluate(args.device),
         "horizon": lambda: evaluate(args.device, True), "report": report}[args.phase]()


if __name__ == "__main__":
    main()
