"""Freeze, execute, and audit only the preregistered v16 fresh-map evaluations."""

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import subprocess
import time

from week03_ant.contact_study import (
    ART, CONTROLLERS, HOLDOUTS, ROOT, START, START_SHA, V15, V15_SHA, WORK,
    cache_snapshot, evaluation_inputs, legacy_hashes, model_key, save_json, sha,
    source_hashes, utc, verify_hashes,
)
from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from run_contact_study import verify_training_freeze
from summarize_contact_v16 import CONTACT_CONFIG, audit, audit_development_proofs, summarize, markdown

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
    print(f"[v16] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode or not Path(command[command.index("--output") + 1]).is_file():
        print("\n".join(path.read_text().splitlines()[-25:]), flush=True)
        raise RuntimeError(f"evaluation process failed or did not write declared output: {label}")


def eval_command(controller, geometry, reset, scenario, phase, output, checkpoint, *, device="cuda:1", envs=None):
    command = [PYTHON, "scripts/evaluate_contact_v16.py", "--headless", "--device", device,
               "--controller", controller, "--geometry", str(geometry), "--seed", str(reset),
               "--scenario", scenario, "--seconds", "16" if scenario == "mixed" else "64",
               "--num_envs", str(envs or (175 if scenario == "mixed" else 10)), "--phase", phase,
               "--output", str(output), "--checkpoint", str(checkpoint)]
    return command


def assert_cache(expected):
    if cache_snapshot(CACHE, expected["geometries"]) != expected:
        raise ValueError("terrain cache changed")


def final_smoke(device):
    """All six controllers on old development geometry before holdout freeze."""
    verify_training_freeze()
    trained = read(ART / "trained_models.json")
    parity = read(WORK / "legacy_parity_v15_control.json")
    references = {"original": {"checkpoint": str(START.relative_to(ROOT)), "sha256": START_SHA},
                  "v15_control": {"checkpoint": str(V15.relative_to(ROOT)), "sha256": V15_SHA},
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
        "v15_reference_parity_sha256": sha(ART / "evaluator_parity.json"),
    })


def freeze_eval():
    verify_training_freeze()
    smokes = read(ART / "final_model_smokes.json")
    if smokes.get("passed") is not True or set(record["controller"] for record in smokes["records"]) != set(CONTROLLERS):
        raise ValueError("all six final checkpoint smokes must pass")
    trained = read(ART / "trained_models.json")
    models = {"original": {"checkpoint": str(START.relative_to(ROOT)), "sha256": START_SHA},
              "v15_control": {"checkpoint": str(V15.relative_to(ROOT)), "sha256": V15_SHA},
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
    parser.add_argument("phase", choices=("final_smoke", "freeze_eval", "prepare_eval", "evaluate", "horizon", "report"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/hybrid_v11_20260922/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"final_smoke": lambda: final_smoke(args.device), "freeze_eval": freeze_eval,
         "prepare_eval": lambda: prepare_eval(args.device), "evaluate": lambda: evaluate(args.device),
         "horizon": lambda: evaluate(args.device, horizon=True), "report": report}[args.phase]()


if __name__ == "__main__":
    main()
