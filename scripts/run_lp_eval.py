"""Freeze, execute, and audit only the preregistered v17 fresh-map evaluations."""

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import subprocess
import time

from week03_ant.lp_study_v17 import (
    ART, CONTROLLERS, HOLDOUTS, ROOT, START, START_SHA, V16_CONTROL, V16_CONTROL_SHA, WORK,
    cache_snapshot, evaluation_inputs, legacy_hashes, model_key, save_json, sha,
    source_hashes, utc, verify_hashes,
)
from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from summarize_lp_v17 import (
    CONTACT_CONFIG, PARITY_FIELDS, audit, audit_development_proofs,
    _verify_training, summarize, markdown,
)

PYTHON = str(ROOT.parent / "run-python")
CACHE = Path("/tmp/isaaclab/terrains")


def verify_training_freeze():
    from run_lp_study import verify_training_freeze as verify
    return verify()


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
    print(f"[v17] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode or not Path(command[command.index("--output") + 1]).is_file():
        print("\n".join(path.read_text().splitlines()[-25:]), flush=True)
        raise RuntimeError(f"evaluation process failed or did not write declared output: {label}")


def eval_command(controller, geometry, reset, scenario, phase, output, checkpoint, *, device="cuda:1", envs=None):
    command = [PYTHON, "scripts/evaluate_lp_v17.py", "--headless", "--device", device,
               "--controller", controller, "--geometry", str(geometry), "--seed", str(reset),
               "--scenario", scenario, "--seconds", "16" if scenario == "mixed" else "64",
               "--num_envs", str(envs or (175 if scenario == "mixed" else 10)), "--phase", phase,
               "--output", str(output), "--checkpoint", str(checkpoint)]
    return command


def assert_cache(expected):
    if cache_snapshot(CACHE, expected["geometries"]) != expected:
        raise ValueError("terrain cache changed")


def parity(device):
    """Reproduce the frozen v16-control development episode before new scoring."""
    verify_training_freeze()
    old_path = ROOT / "outputs/contact_slip_v16_20260923/final_smoke_control.json"
    probe_path = ROOT / "outputs/contact_slip_v16_20260923/probe_stage_v3.json"
    old = read(old_path)
    if ((old.get("controller"), old.get("checkpoint_sha256"), old.get("phase"),
         old.get("scenario"), old.get("geometry_seed"), old.get("reset_seed"),
         old.get("num_envs")) !=
        ("control", V16_CONTROL_SHA, "smoke", "mixed", 51, 24, 35)):
        raise ValueError("frozen v16 control reference condition differs")
    output = WORK / "legacy_parity_v16_control.json"
    run(eval_command("v16_control", 51, 24, "mixed", "smoke", output, V16_CONTROL,
                     device=device, envs=35), "legacy_parity_v16_control")
    new = read(output)
    audit(new)
    if (new.get("checkpoint_sha256") != V16_CONTROL_SHA
            or any(field not in old or field not in new or old[field] != new[field]
                   for field in PARITY_FIELDS)):
        raise ValueError("v17 evaluator changed frozen v16 physical/parity fields")
    probe = read(probe_path)
    targets = ["/World/ground/terrain/mesh", "/World/flatPlane/GroundPlane/CollisionPlane"]
    if probe.get("configured_targets") != targets or probe.get("collision_prims") != targets:
        raise ValueError("v16 contact sensor targets differ")
    save_json(ART / "evaluator_parity.json", {
        "created_utc": utc(), "reference_path": str(old_path.relative_to(ROOT)),
        "reference_sha256": sha(old_path), "new_path": str(output.relative_to(ROOT)),
        "new_sha256": sha(output), "equal_fields": list(PARITY_FIELDS),
        "exact_reference_parity": True, "development_only": True,
        "sensor_probe_path": str(probe_path.relative_to(ROOT)),
        "sensor_probe_sha256": sha(probe_path), "two_live_collision_prims": targets,
    })


def final_smoke(device):
    """All six controllers on old development geometry before holdout freeze."""
    verify_training_freeze()
    trained = read(ART / "trained_models.json")
    parity = read(WORK / "legacy_parity_v16_control.json")
    references = {"original": {"checkpoint": str(START.relative_to(ROOT)), "sha256": START_SHA},
                  "v16_control": {"checkpoint": str(V16_CONTROL.relative_to(ROOT)), "sha256": V16_CONTROL_SHA},
                  **trained["models"]}
    records = []
    for controller in CONTROLLERS:
        entry = references[model_key(controller)]
        checkpoint = ROOT / entry["checkpoint"]
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
        output = WORK / f"final_smoke_{controller}.json"
        run(eval_command(controller, 51, 24, "mixed", "smoke", output, checkpoint,
                         device=device, envs=35), f"final_smoke_{controller}")
        data = read(output)
        audit(data)
        for key in ("initial_state_sha256", "initial_prefix_sha256", "initial_rng_sha256"):
            if data[key] != parity[key]:
                raise ValueError(f"final smoke initialization differs: {key}")
        records.append({"path": str(output.relative_to(ROOT)), "sha256": sha(output),
                        "controller": controller, "checkpoint_sha256": entry["sha256"]})
    verify_training_freeze()
    save_json(ART / "final_model_smokes.json", {
        "created_utc": utc(), "passed": True, "development_only": True,
        "exact_initial_pairing": True, "records": records,
        "v16_reference_parity_sha256": sha(ART / "evaluator_parity.json"),
    })


def freeze_eval():
    verify_training_freeze()
    smokes = read(ART / "final_model_smokes.json")
    if smokes.get("passed") is not True or set(record["controller"] for record in smokes["records"]) != set(CONTROLLERS):
        raise ValueError("all six final checkpoint smokes must pass")
    trained = read(ART / "trained_models.json")
    models = {"original": {"checkpoint": str(START.relative_to(ROOT)), "sha256": START_SHA},
              "v16_control": {"checkpoint": str(V16_CONTROL.relative_to(ROOT)), "sha256": V16_CONTROL_SHA},
              **trained["models"]}
    for entry in models.values():
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
    frozen = {
        "frozen_at": utc(), "source_sha256": source_hashes(), "legacy": legacy_hashes(),
        "models": models, "holdouts": HOLDOUTS, "controllers": CONTROLLERS,
        "gate_config": asdict(HistoryGateConfig()), "posture_config": asdict(PostureConfig()),
        "contact_config": CONTACT_CONFIG,
        "training_freeze_sha256": sha(ART / "training_frozen.json"),
        "trained_models_sha256": sha(ART / "trained_models.json"),
        "final_model_smokes_sha256": sha(ART / "final_model_smokes.json"),
        "evaluator_parity_sha256": sha(ART / "evaluator_parity.json"),
        "primary": {"files": 12, "episodes": 2100, "seconds": 16, "envs": 175},
        "secondary": {"files": 12, "episodes": 120, "seconds": 64, "envs": 10},
        "selection": "all predeclared controllers/maps; only paired final249 checkpoints",
    }
    audit_development_proofs(ART, frozen, models)
    _verify_training(ART, frozen)
    save_json(ART / "frozen.json", frozen)


def verify_eval():
    frozen = read(ART / "frozen.json")
    verify_training_freeze()
    verify_hashes(frozen["source_sha256"])
    if frozen["legacy"] != legacy_hashes():
        raise ValueError("legacy source/model inventory changed")
    for entry in frozen["models"].values():
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
    audit_development_proofs(ART, frozen, frozen["models"])
    _verify_training(ART, frozen)
    if sha(ART / "training_frozen.json") != frozen["training_freeze_sha256"]:
        raise ValueError("training freeze changed")
    return frozen


def prepare_eval(device):
    verify_eval()
    records = []
    for geometry, reset in HOLDOUTS:
        output = WORK / f"prepare_holdout_geometry{geometry}.json"
        run(eval_command("history_original", geometry, reset, "mixed", "prepare", output, START,
                         device=device), f"prepare_holdout_geometry{geometry}")
        if read(output)["scored_episodes"] != 0:
            raise ValueError("preparation scored episodes")
        records.append({"path": str(output.relative_to(ROOT)), "sha256": sha(output)})
    verify_eval()
    save_json(ART / "terrain_cache.json", {"created_utc": utc(), "preparation": records,
              "cache": cache_snapshot(CACHE, [geometry for geometry, _ in HOLDOUTS])})
    ledger = ART / "commands.jsonl"
    save_json(ART / "preholdout_ledger.json", {
        "created_utc": utc(), "path": str(ledger.relative_to(ROOT)),
        "lines": len(ledger.read_bytes().splitlines()), "bytes": ledger.stat().st_size,
        "sha256": sha(ledger),
    })
    save_json(ART / "evaluation_inputs.json", {
        "created_utc": utc(), "experiment_freeze_sha256": sha(ART / "frozen.json"),
        "preholdout_ledger_sha256": sha(ART / "preholdout_ledger.json"),
        "terrain_cache_manifest_sha256": sha(ART / "terrain_cache.json"),
    })


def evaluate(device, *, horizon=False):
    from summarize_history_hybrid import check_initial

    folder, scenario = ("horizon", "stones") if horizon else ("evaluations", "mixed")
    evaluation_inputs()
    cache = read(ART / "terrain_cache.json")["cache"]
    initial = {}
    for controller in CONTROLLERS:
        for geometry, reset in HOLDOUTS:
            frozen = verify_eval()
            evaluation_inputs()
            assert_cache(cache)
            key = model_key(controller)
            label = f"{controller}__geometry{geometry}_reset{reset}"
            output = ART / folder / f"{label}.json"
            run(eval_command(controller, geometry, reset, scenario, "holdout", output,
                             ROOT / frozen["models"][key]["checkpoint"], device=device),
                f"{folder}_{label}")
            evaluation_inputs()
            assert_cache(cache)
            verify_eval()
            data = read(output)
            audit(data)
            check_initial(initial, (geometry, reset), data)
            extra_key = (geometry, reset, "prefix_rng")
            prefix_rng = (data["initial_prefix_sha256"], data["initial_rng_sha256"])
            if initial.setdefault(extra_key, prefix_rng) != prefix_rng:
                raise ValueError("holdout prefix/RNG initialization differs")


def report():
    verify_eval()
    evaluation_inputs()
    result = summarize(ART)
    save_json(ART / "summary.json", result)
    with (ART / "summary.md").open("x") as stream:
        stream.write(markdown(result))
    print(markdown(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("parity", "final_smoke", "freeze_eval", "prepare_eval", "evaluate", "horizon", "report"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/hybrid_v11_20260922/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"parity": lambda: parity(args.device),
         "final_smoke": lambda: final_smoke(args.device), "freeze_eval": freeze_eval,
         "prepare_eval": lambda: prepare_eval(args.device), "evaluate": lambda: evaluate(args.device),
         "horizon": lambda: evaluate(args.device, horizon=True), "report": report}[args.phase]()


if __name__ == "__main__":
    main()
