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
from week03_ant.direction_math import DirectionConfig
from week03_ant.direction_study import (
    ARMS, ART, CONTROLLERS, HOLDOUTS, ROOT, START, START_SHA, WORK, V14, V14_SHA, init_path, model_key,
    cache_snapshot, legacy_hashes, save_json, sha, source_hashes,
    training_parameters, utc, validate_teacher, verify_hashes, evaluation_inputs, validate_learning_log,
)

PYTHON = str(ROOT.parent / "run-python")
CACHE = Path("/tmp/isaaclab/terrains")



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
    print(f"[v15] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode:
        print("\n".join(path.read_text().splitlines()[-30:]), flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)


def train_command(arm, output, expected=None, *, capacity=False, device="cuda:1"):
    command = [PYTHON, "scripts/direction_v15.py", "train", "--arm", arm, "--audit-output", str(output),
               "--headless", "--device", device, "--num_envs", "4096", "--max_iterations", "2" if capacity else "250",
               "--seed", "47", "--run_name", f"v15_{'capacity' if capacity else 'paired'}_{arm}",
               "--resume", "--load_run", "v15_init", "--checkpoint", "model_0.pt",
               "env.scene.terrain.terrain_generator.seed=95", f"agent.device={device}"]
    if expected:
        command += ["--expected-initial", str(expected)]
    return command


def eval_command(controller, geometry, reset, scenario, phase, output, checkpoint, device="cuda:1"):
    return [PYTHON, "scripts/evaluate_direction_v15.py", "--headless", "--device", device,
            "--controller", controller, "--geometry", str(geometry), "--seed", str(reset),
            "--scenario", scenario, "--seconds", "16" if scenario == "mixed" else "64",
            "--num_envs", "175" if scenario == "mixed" else "10", "--phase", phase,
            "--output", str(output), "--checkpoint", str(checkpoint)]


def assert_same_cache(expected):
    if cache_snapshot(CACHE, expected["geometries"]) != expected:
        raise ValueError("terrain cache changed")


def capacity(device):
    before = cache_snapshot(CACHE, [95])
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
        "initial_checkpoints": {arm: {"checkpoint": str(init_path(arm).relative_to(ROOT)), "sha256": sha(init_path(arm))} for arm in ARMS},
        "starting_checkpoint_sha256": V14_SHA, "training_seed": 47, "geometry": 95,
        "iterations_per_arm": 250, "envs": 4096, "steps_per_env": 32,
        "transitions_per_arm": 32768000, "selection": "only final iteration249; no seed/model selection",
        "direction_config": asdict(DirectionConfig()), "posture_config": asdict(PostureConfig()), "history_config": asdict(HistoryGateConfig()),
        "arms": list(ARMS), "terrain_cache": capacity_data["terrain_cache"],
        "capacity_sha256": sha(capacity_path), "preflight_sha256": sha(proof),
    })


def verify_training():
    frozen = read(ART / "training_frozen.json")
    verify_hashes(frozen["source_sha256"])
    if frozen["legacy"] != legacy_hashes():
        raise ValueError("legacy inventory changed")
    for entry in frozen["initial_checkpoints"].values():
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
    assert_same_cache(frozen["terrain_cache"])
    return frozen


def train(device):
    import torch

    if (ART / "trained_models.json").exists():
        raise FileExistsError("paired training already complete")
    first, models, validation = None, {}, {}
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
        for values in state["optimizer_state_dict"]["state"].values():
            for value in values.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError("nonfinite final optimizer")
        if state["model_state_dict"]["command_mode_code"].item() != 1:
            raise ValueError("trained policy lost conditioned mode")
        validation[arm] = validate_learning_log(logdir, WORK / f"train_{arm}.log")
        if arm == "stable" and "Episode_Reward/directional_stability" not in validation[arm]["scalar_counts"]:
            raise ValueError("treated reward was not logged")
        destination = ART / "runs" / arm / "model_249.pt"
        destination.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(final, destination)
        for name in ("env.yaml", "agent.yaml"):
            shutil.copyfile(logdir / "params" / name, destination.parent / name)
        models[arm] = {"checkpoint": str(destination.relative_to(ROOT)), "sha256": sha(destination),
                       "source_checkpoint": str(final.relative_to(ROOT)), "iteration": 249,
                       "transitions": 32768000, "initial_audit_sha256": sha(output)}
    save_json(ART / "training_validation.json", {"created_utc": utc(), "records": validation})
    save_json(ART / "trained_models.json", {"completed_utc": utc(), "models": models,
              "training_validation_sha256": sha(ART / "training_validation.json"),
              "total_training_transitions": 65536000, "training_freeze_sha256": sha(ART / "training_frozen.json")})


def freeze_eval():
    verify_training()
    if read(ART / "final_model_smokes.json").get("passed") is not True:
        raise ValueError("final checkpoint inference smokes have not passed")
    trained = read(ART / "trained_models.json")
    models = {"original": {"checkpoint": str(START.relative_to(ROOT)), "sha256": START_SHA}, "v14": {"checkpoint": str(V14.relative_to(ROOT)), "sha256": V14_SHA}, **trained["models"]}
    for entry in models.values():
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
    save_json(ART / "frozen.json", {"frozen_at": utc(), "source_sha256": source_hashes(),
              "legacy": legacy_hashes(), "models": models, "holdouts": HOLDOUTS, "controllers": CONTROLLERS,
              "gate_config": asdict(HistoryGateConfig()), "direction_config": asdict(DirectionConfig()), "posture_config": asdict(PostureConfig()),
              "training_freeze_sha256": sha(ART / "training_frozen.json"),
              "trained_models_sha256": sha(ART / "trained_models.json"),
              "final_model_smokes_sha256": sha(ART / "final_model_smokes.json"),
              "primary": {"files": 12, "episodes": 2100, "seconds": 16, "envs": 175},
              "secondary": {"files": 12, "episodes": 120, "seconds": 64, "envs": 10},
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
        run(eval_command("history_original", geom, reset, "mixed", "prepare", output, START, device), f"prepare_holdout_geometry{geom}")
        if read(output)["scored_episodes"] != 0:
            raise ValueError("preparation scored episodes")
        records.append({"path": str(output.relative_to(ROOT)), "sha256": sha(output)})
    verify_eval()
    save_json(ART / "terrain_cache.json", {"created_utc": utc(), "preparation": records,
              "cache": cache_snapshot(CACHE, [g for g, _ in HOLDOUTS])})
    ledger = ART / "commands.jsonl"
    save_json(ART / "preholdout_ledger.json", {"created_utc": utc(), "path": str(ledger.relative_to(ROOT)),
        "lines": len(ledger.read_bytes().splitlines()), "bytes": ledger.stat().st_size, "sha256": sha(ledger)})
    save_json(ART / "evaluation_inputs.json", {"created_utc": utc(),
              "experiment_freeze_sha256": sha(ART / "frozen.json"),
              "preholdout_ledger_sha256": sha(ART / "preholdout_ledger.json"),
              "terrain_cache_manifest_sha256": sha(ART / "terrain_cache.json")})


def evaluate(device, horizon=False):
    from summarize_direction_v15 import audit
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
            key = model_key(controller)
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
            extra_key = (geom, reset, "prefix_rng")
            extra = (data["initial_prefix_sha256"], data["initial_rng_sha256"])
            if initial.setdefault(extra_key, extra) != extra:
                raise ValueError("evaluation prefix/RNG initialization differs")


def report():
    from summarize_direction_v15 import summarize, markdown

    verify_eval()
    evaluation_inputs()
    result = summarize(ART)
    save_json(ART / "summary.json", result)
    with (ART / "summary.md").open("x") as stream:
        stream.write(markdown(result))
    print(markdown(result), flush=True)


def development(device):
    """Cache warmup, 64env paired smoke, then legacy-prefix/outcome parity."""
    from week03_ant.direction_study import prepare_checkpoint
    from summarize_direction_v15 import audit

    result = prepare_checkpoint(init_path())
    save_json(ART / "initial_shared.json", result)
    for geom, reset in ((51, 24), (95, 47)):
        output = WORK / f"prepare_geometry{geom}.json"
        run(eval_command("history_original", geom, reset, "mixed", "prepare", output, START, device), f"prepare_geometry{geom}")
        if read(output)["scored_episodes"] != 0:
            raise ValueError("cache preparation scored episodes")
    smoke(device)


def smoke(device):
    from summarize_direction_v15 import audit

    # Historical development geometry51 has multiple old config variants. Do not
    # alter them. Pairing is proven by exact saved prefix/outcomes; new95 is pinned.
    cache = cache_snapshot(CACHE, [95])
    first = None
    for arm in ARMS:
        output = WORK / f"smoke_{arm}_initial.json"
        command = train_command(arm, output, first, capacity=True, device=device)
        command[command.index("--num_envs")+1] = "64"
        command[command.index("--run_name")+1] = f"v15_smoke_{arm}"
        assert_same_cache(cache)
        run(command, f"smoke_{arm}")
        assert_same_cache(cache)
        first = first or output
    output = WORK / "legacy_prefix_parity.json"
    cmd = eval_command("v14", 51, 24, "mixed", "smoke", output, V14, device)
    cmd[cmd.index("--num_envs")+1] = "35"
    run(cmd, "legacy_prefix_parity")
    verify_parity(output)


def verify_parity(output):
    from summarize_direction_v15 import audit

    data = read(output)
    audit(data)
    reference_path = ROOT / "outputs/command_conditioning_v14_20260922/final_smoke_conditioned.json"
    old = read(reference_path)
    keys = ("forward_distance", "episode_lengths", "episode_terminated", "episode_out_of_lane", "episode_world_exit",
            "episode_strict_one_tile_success", "episode_strict_all_tiles_success", "episode_full_horizon_survival",
            "episode_alpha_sum", "episode_v10_target_steps", "episode_switch_steps", "episode_switch_to_v10",
            "history_switch_events", "episode_return")
    for key in keys:
        if data[key] != old[key]:
            raise ValueError(f"legacy evaluator parity differs: {key}")
    for key in ("initial_state_sha256", "initial_prefix_sha256", "initial_rng_sha256", "posture_telemetry"):
        if data[key] != old[key]:
            raise ValueError(f"v14 evaluator initial/telemetry parity differs: {key}")
    save_json(ART / "evaluator_parity.json", {"created_utc": utc(), "reference": str(reference_path.relative_to(ROOT)),
        "reference_sha256": sha(reference_path), "new": str(output.relative_to(ROOT)), "new_sha256": sha(output),
        "equal_fields": list(keys) + ["initial_state_sha256", "initial_prefix_sha256", "initial_rng_sha256", "posture_telemetry"],
        "exact_initial_state_prefix": True, "development_only": True,
        "interpretation": "Unchanged v14 conditioned inference outcomes, returns, routing, posture telemetry and full initial91D/prefix/RNG match exactly. Directional diagnostics are passive."})


def final_smoke(device):
    from summarize_direction_v15 import audit

    verify_training()
    trained = read(ART / "trained_models.json")
    reference = read(WORK / "legacy_prefix_parity.json")
    records = []
    for controller in ("control", "stable", "history_stable"):
        entry = trained["models"][model_key(controller)]
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
        output = WORK / f"final_smoke_{controller}.json"
        command = eval_command(controller, 51, 24, "mixed", "smoke", output, ROOT / entry["checkpoint"], device)
        command[command.index("--num_envs")+1] = "35"
        run(command, f"final_smoke_{controller}")
        data = read(output)
        audit(data)
        for key in ("initial_state_sha256", "initial_prefix_sha256", "initial_rng_sha256"):
            if data[key] != reference[key]:
                raise ValueError(f"final checkpoint smoke initialization differs: {key}")
        records.append({"path": str(output.relative_to(ROOT)), "sha256": sha(output)})
    verify_training()
    save_json(ART / "final_model_smokes.json", {"created_utc": utc(), "passed": True,
        "development_only": True, "records": records, "exact_initial_pairing": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "smoke", "capacity", "freeze_train", "train", "final_smoke", "freeze_eval", "prepare_eval", "evaluate", "horizon", "report"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/hybrid_v11_20260922/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"development": lambda: development(args.device), "smoke": lambda: smoke(args.device), "capacity": lambda: capacity(args.device), "freeze_train": freeze_train,
         "train": lambda: train(args.device), "final_smoke": lambda: final_smoke(args.device), "freeze_eval": freeze_eval,
         "prepare_eval": lambda: prepare_eval(args.device), "evaluate": lambda: evaluate(args.device),
         "horizon": lambda: evaluate(args.device, True), "report": report}[args.phase]()


if __name__ == "__main__":
    main()
