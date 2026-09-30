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
from week03_ant.command_study import (
    ARMS, ART, CONTROLLERS, HOLDOUTS, ROOT, START, START_SHA, WORK, V13, V13_SHA, init_path, model_key,
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
    print(f"[v14] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode:
        print("\n".join(path.read_text().splitlines()[-30:]), flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)


def train_command(arm, output, expected=None, *, capacity=False, device="cuda:1"):
    command = [PYTHON, "scripts/command_v14.py", "train", "--arm", arm, "--audit-output", str(output),
               "--headless", "--device", device, "--num_envs", "4096", "--max_iterations", "2" if capacity else "250",
               "--seed", "46", "--run_name", f"v14_{'capacity' if capacity else 'paired'}_{arm}",
               "--resume", "--load_run", f"v14_init_{arm}", "--checkpoint", "model_0.pt",
               "env.scene.terrain.terrain_generator.seed=85", f"agent.device={device}"]
    if expected:
        command += ["--expected-initial", str(expected)]
    return command


def eval_command(controller, geometry, reset, scenario, phase, output, checkpoint, device="cuda:1"):
    return [PYTHON, "scripts/evaluate_command_v14.py", "--headless", "--device", device,
            "--controller", controller, "--geometry", str(geometry), "--seed", str(reset),
            "--scenario", scenario, "--seconds", "16" if scenario == "mixed" else "64",
            "--num_envs", "175" if scenario == "mixed" else "10", "--phase", phase,
            "--output", str(output), "--checkpoint", str(checkpoint)]


def assert_same_cache(expected):
    if cache_snapshot(CACHE, expected["geometries"]) != expected:
        raise ValueError("terrain cache changed")


def capacity(device):
    before = cache_snapshot(CACHE, [85])
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
        "starting_checkpoint_sha256": START_SHA, "training_seed": 46, "geometry": 85,
        "iterations_per_arm": 250, "envs": 4096, "steps_per_env": 32,
        "transitions_per_arm": 32768000, "selection": "only final iteration249; no seed/model selection",
        "posture_config": asdict(PostureConfig()), "history_config": asdict(HistoryGateConfig()),
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
        if arm == "masked" and any(not state["model_state_dict"][name].eq(0).all()
                for name in ("actor.0.command_weight", "critic.0.command_weight")):
            raise ValueError("masked command weights changed")
        validation[arm] = validate_learning_log(logdir, WORK / f"train_{arm}.log")
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
    models = {"original": {"checkpoint": str(START.relative_to(ROOT)), "sha256": START_SHA}, "v13": {"checkpoint": str(V13.relative_to(ROOT)), "sha256": V13_SHA}, **trained["models"]}
    for entry in models.values():
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
    save_json(ART / "frozen.json", {"frozen_at": utc(), "source_sha256": source_hashes(),
              "legacy": legacy_hashes(), "models": models, "holdouts": HOLDOUTS, "controllers": CONTROLLERS,
              "gate_config": asdict(HistoryGateConfig()), "posture_config": asdict(PostureConfig()),
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
    save_json(ART / "evaluation_inputs.json", {"created_utc": utc(),
              "experiment_freeze_sha256": sha(ART / "frozen.json"),
              "terrain_cache_manifest_sha256": sha(ART / "terrain_cache.json")})


def evaluate(device, horizon=False):
    from summarize_command_v14 import audit
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
    from summarize_command_v14 import summarize, markdown

    verify_eval()
    evaluation_inputs()
    result = summarize(ART)
    save_json(ART / "summary.json", result)
    with (ART / "summary.md").open("x") as stream:
        stream.write(markdown(result))
    print(markdown(result), flush=True)


def development(device):
    """Cache warmup, 64env paired smoke, then legacy-prefix/outcome parity."""
    from week03_ant.command_study import prepare_checkpoint
    from summarize_command_v14 import audit

    for arm in ARMS:
        result = prepare_checkpoint(init_path(arm), arm)
        save_json(ART / f"initial_{arm}.json", result)
    for geom, reset in ((51, 24), (85, 46)):
        output = WORK / f"prepare_geometry{geom}.json"
        run(eval_command("history_original", geom, reset, "mixed", "prepare", output, START, device), f"prepare_geometry{geom}")
        if read(output)["scored_episodes"] != 0:
            raise ValueError("cache preparation scored episodes")
    smoke(device)


def smoke(device):
    from summarize_command_v14 import audit

    # Historical development geometry51 has multiple old config variants. Do not
    # alter them. Pairing is proven by exact saved prefix/outcomes; new85 is pinned.
    cache = cache_snapshot(CACHE, [85])
    first = None
    for arm in ARMS:
        output = WORK / f"smoke_{arm}_initial.json"
        command = train_command(arm, output, first, capacity=True, device=device)
        command[command.index("--num_envs")+1] = "64"
        command[command.index("--seed")+1] = "24"
        command[command.index("--run_name")+1] = f"v14_smoke_{arm}"
        command[command.index("env.scene.terrain.terrain_generator.seed=85")] = "env.scene.terrain.terrain_generator.seed=51"
        assert_same_cache(cache)
        run(command, f"smoke_{arm}")
        assert_same_cache(cache)
        first = first or output
    output = WORK / "legacy_prefix_parity.json"
    cmd = eval_command("history_original", 51, 24, "mixed", "smoke", output, START, device)
    cmd[cmd.index("--num_envs")+1] = "35"
    run(cmd, "legacy_prefix_parity")
    verify_parity(output)


def verify_parity(output):
    from summarize_command_v14 import audit

    data = read(output)
    audit(data)
    reference_path = ROOT / "outputs/adaptive_posture_v13_20260922/eval_history_original_final_smoke.json"
    old = read(reference_path)
    keys = ("forward_distance", "episode_lengths", "episode_terminated", "episode_out_of_lane", "episode_world_exit",
            "episode_strict_one_tile_success", "episode_strict_all_tiles_success", "episode_full_horizon_survival",
            "episode_alpha_sum", "episode_v10_target_steps", "episode_switch_steps", "episode_switch_to_v10",
            "history_switch_events", "episode_return")
    for key in keys:
        if data[key] != old[key]:
            raise ValueError(f"legacy evaluator parity differs: {key}")
    if (data["initial_prefix_sha256"] != old["initial_state_sha256"]["observations"]
            or any(data["initial_state_sha256"][key] != old["initial_state_sha256"][key]
                   for key in ("root_state", "joint_pos", "joint_vel"))):
        raise ValueError("legacy evaluator initial state/prefix differs")
    # The plan requires exact prefix and outcomes, not old passive velocity-cache
    # telemetry. Keep the observed difference explicit rather than rounding it.
    differences = {}
    for key, value in old["posture_telemetry"].items():
        if key not in ("groups", "low_speed_steps") and value != data["posture_telemetry"][key]:
            raise ValueError(f"unexpected passive posture occupancy change: {key}")
    for group, fields in old["posture_telemetry"]["groups"].items():
        for key, a in fields.items():
            b = data["posture_telemetry"]["groups"][group][key]
            if a == b:
                continue
            if key not in ("forward_speed_sum", "flat_speed_bonus_sum", "reward_sum", "low_speed_steps"):
                raise ValueError(f"unexpected passive geometry change: {group}/{key}")
            differences[f"{group}/{key}"] = {"differing_episodes": sum(x != y for x, y in zip(a, b)),
                "max_absolute_sum_difference": max(abs(x-y) for x, y in zip(a, b))}
    differences["total_low_speed_count_difference"] = sum(data["posture_telemetry"]["low_speed_steps"]) - sum(old["posture_telemetry"]["low_speed_steps"])
    save_json(ART / "evaluator_parity.json", {"created_utc": utc(), "reference": str(reference_path.relative_to(ROOT)),
        "reference_sha256": sha(reference_path), "new": str(output.relative_to(ROOT)), "new_sha256": sha(output),
        "equal_fields": list(keys), "exact_initial_state_prefix": True, "development_only": True,
        "passive_velocity_telemetry_differences": differences,
        "interpretation": "Outcomes/returns/routing and height/foot geometry match exactly. Extra observation-time geometry reads change lazy root-link velocity cache timing around reset; passive speed sums are not byte-identical. New91D arms share the same ordering; flat performance speed uses distance/time, not this cached velocity. No rounding or scored process rerun."})


def probe(device):
    frozen = verify_eval()
    for controller in ("masked", "conditioned", "history_original"):
        key = model_key(controller)
        output = ART / "command_probe" / f"{controller}.json"
        command = eval_command(controller, 51, 24, "mixed", "probe", output, ROOT / frozen["models"][key]["checkpoint"], device)
        command[command.index("--num_envs")+1] = "35"
        run(command, f"command_probe_{controller}")
        verify_eval()


def final_smoke(device):
    from summarize_command_v14 import audit

    verify_training()
    trained = read(ART / "trained_models.json")
    reference = read(WORK / "legacy_prefix_parity.json")
    records = []
    for controller in ("masked", "conditioned", "history_conditioned"):
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
    parser.add_argument("phase", choices=("development", "smoke", "probe", "capacity", "freeze_train", "train", "final_smoke", "freeze_eval", "prepare_eval", "evaluate", "horizon", "report"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/hybrid_v11_20260922/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"development": lambda: development(args.device), "smoke": lambda: smoke(args.device), "probe": lambda: probe(args.device), "capacity": lambda: capacity(args.device), "freeze_train": freeze_train,
         "train": lambda: train(args.device), "final_smoke": lambda: final_smoke(args.device), "freeze_eval": freeze_eval,
         "prepare_eval": lambda: prepare_eval(args.device), "evaluate": lambda: evaluate(args.device),
         "horizon": lambda: evaluate(args.device, True), "report": report}[args.phase]()


if __name__ == "__main__":
    main()
