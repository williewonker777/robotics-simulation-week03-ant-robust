"""Freeze and execute v18 fresh-map evaluation with one serialized GPU job."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import subprocess
import time

from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from week03_ant.posture_study import cache_snapshot
from week03_ant.style_eval_v18 import evaluation_inputs, source_hashes
from week03_ant.style_study_v18 import (
    ARMS, ART, CONTROLLERS, HOLDOUTS, ROOT, START, START_SHA,
    V16_CONTROL, V16_CONTROL_SHA, WORK, legacy_hashes, model_key,
    save_json, sha, utc, verify_hashes,
)
from run_style_study import verify_training_freeze
from summarize_style_v18 import CONTACT_CONFIG, PARITY_FIELDS, audit, markdown, summarize

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
    if (ART / "evaluation_inputs.json").exists():
        inputs = evaluation_inputs()
        record.update(evaluation_inputs_sha256=sha(ART / "evaluation_inputs.json"),
                      terrain_cache_manifest_sha256=inputs["terrain_cache_manifest_sha256"])
    with (ART / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"[v18] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode or not Path(command[command.index("--output") + 1]).is_file():
        print("\n".join(path.read_text().splitlines()[-25:]), flush=True)
        raise RuntimeError(f"v18 evaluator did not produce declared output: {label}")


def eval_command(controller, geometry, reset, scenario, phase, output, checkpoint,
                 *, device="cuda:1", envs=None):
    return [PYTHON, "scripts/evaluate_style_v18.py", "--headless", "--device", device,
            "--controller", controller, "--geometry", str(geometry), "--seed", str(reset),
            "--scenario", scenario, "--seconds", "16" if scenario == "mixed" else "64",
            "--num_envs", str(envs or (175 if scenario == "mixed" else 10)),
            "--phase", phase, "--output", str(output), "--checkpoint", str(checkpoint)]


def assert_cache(expected):
    if cache_snapshot(CACHE, expected["geometries"]) != expected:
        raise ValueError("v18 evaluation terrain cache changed")


def models():
    trained = read(ART / "trained_models.json")
    return {"original": {"checkpoint": str(START.relative_to(ROOT)), "sha256": START_SHA},
            "v16_control": {"checkpoint": str(V16_CONTROL.relative_to(ROOT)),
                            "sha256": V16_CONTROL_SHA}, **trained["models"]}


def parity(device):
    verify_training_freeze()
    old_path = ROOT / "outputs/contact_slip_v16_20260923/final_smoke_control.json"
    old = read(old_path)
    if ((old.get("controller"), old.get("checkpoint_sha256"), old.get("phase"),
         old.get("scenario"), old.get("geometry_seed"), old.get("reset_seed"),
         old.get("num_envs")) !=
        ("control", V16_CONTROL_SHA, "smoke", "mixed", 51, 24, 35)):
        raise ValueError("v18 frozen v16 parity reference condition changed")
    output = WORK / "legacy_parity_v16_control.json"
    run(eval_command("v16_control", 51, 24, "mixed", "smoke", output, V16_CONTROL,
                     device=device, envs=35), "legacy_parity_v16_control")
    new = read(output)
    audit(new)
    if new.get("checkpoint_sha256") != V16_CONTROL_SHA or any(
            field not in old or field not in new or old[field] != new[field]
            for field in PARITY_FIELDS):
        raise ValueError("v18 evaluator differs from frozen v16 physical fields")
    save_json(ART / "evaluator_parity.json", {
        "created_utc": utc(), "development_only": True, "exact_reference_parity": True,
        "reference_path": str(old_path.relative_to(ROOT)), "reference_sha256": sha(old_path),
        "new_path": str(output.relative_to(ROOT)), "new_sha256": sha(output),
        "equal_fields": list(PARITY_FIELDS),
    })


def final_smoke(device):
    verify_training_freeze()
    references = models()
    parity_data = read(WORK / "legacy_parity_v16_control.json")
    records, initial = [], None
    for controller in CONTROLLERS:
        entry = references[model_key(controller)]
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
        if controller == "v16_control":
            output = WORK / "legacy_parity_v16_control.json"
        else:
            output = WORK / f"final_smoke_{controller}.json"
            run(eval_command(controller, 51, 24, "mixed", "smoke", output,
                             ROOT / entry["checkpoint"], device=device, envs=35),
                f"final_smoke_{controller}")
        data = read(output)
        audit(data)
        hashes = {key: data[key] for key in ("initial_state_sha256", "initial_prefix_sha256",
                                            "initial_rng_sha256")}
        if initial is None:
            initial = hashes
        if hashes != initial or hashes != {key: parity_data[key] for key in hashes}:
            raise ValueError("v18 final smoke initial state/prefix/RNG differs")
        records.append({"path": str(output.relative_to(ROOT)), "sha256": sha(output),
                        "controller": controller, "checkpoint_sha256": entry["sha256"]})
    save_json(ART / "final_model_smokes.json", {
        "created_utc": utc(), "passed": True, "development_only": True,
        "exact_initial_pairing": True, "records": records,
        "v16_reference_parity_sha256": sha(ART / "evaluator_parity.json"),
    })


def freeze_eval():
    verify_training_freeze()
    smoke = read(ART / "final_model_smokes.json")
    if not smoke["passed"] or {record["controller"] for record in smoke["records"]} != set(CONTROLLERS):
        raise ValueError("v18 six-controller smoke incomplete")
    references = models()
    for entry in references.values():
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
    save_json(ART / "frozen.json", {
        "frozen_at": utc(), "source_sha256": source_hashes(), "legacy": legacy_hashes(),
        "models": references, "holdouts": HOLDOUTS, "controllers": CONTROLLERS,
        "gate_config": asdict(HistoryGateConfig()),
        "posture_config": asdict(PostureConfig()), "contact_config": CONTACT_CONFIG,
        "training_freeze_sha256": sha(ART / "training_frozen.json"),
        "trained_models_sha256": sha(ART / "trained_models.json"),
        "training_validation_sha256": sha(ART / "training_validation.json"),
        "final_model_smokes_sha256": sha(ART / "final_model_smokes.json"),
        "evaluator_parity_sha256": sha(ART / "evaluator_parity.json"),
        "primary": {"files": 12, "episodes": 2100, "seconds": 16, "envs": 175},
        "secondary": {"files": 12, "episodes": 120, "seconds": 64, "envs": 10},
        "selection": "all predeclared controllers/maps; paired final249 only",
    })


def verify_eval():
    frozen = read(ART / "frozen.json")
    verify_training_freeze()
    if frozen["source_sha256"] != source_hashes() or frozen["legacy"] != legacy_hashes():
        raise ValueError("v18 source/legacy freeze changed")
    verify_hashes(frozen["source_sha256"])
    for entry in frozen["models"].values():
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
    for key, path in (("training_freeze_sha256", "training_frozen.json"),
                      ("trained_models_sha256", "trained_models.json"),
                      ("training_validation_sha256", "training_validation.json"),
                      ("final_model_smokes_sha256", "final_model_smokes.json"),
                      ("evaluator_parity_sha256", "evaluator_parity.json")):
        if frozen[key] != sha(ART / path):
            raise ValueError(f"v18 frozen evidence changed: {path}")
    return frozen


def prepare_eval(device):
    verify_eval()
    records = []
    for geometry, reset in HOLDOUTS:
        output = WORK / f"prepare_holdout_geometry{geometry}.json"
        run(eval_command("history_original", geometry, reset, "mixed", "prepare", output,
                         START, device=device), f"prepare_holdout_geometry{geometry}")
        if read(output).get("scored_episodes") != 0:
            raise ValueError("v18 holdout cache preparation scored episodes")
        records.append({"path": str(output.relative_to(ROOT)), "sha256": sha(output)})
    verify_eval()
    save_json(ART / "terrain_cache.json", {"created_utc": utc(), "preparation": records,
              "cache": cache_snapshot(CACHE, [geometry for geometry, _ in HOLDOUTS])})
    ledger = ART / "commands.jsonl"
    data = ledger.read_bytes()
    save_json(ART / "preholdout_ledger.json", {
        "created_utc": utc(), "path": str(ledger.relative_to(ROOT)),
        "lines": len(data.splitlines()), "bytes": len(data),
        "sha256": __import__("hashlib").sha256(data).hexdigest(),
    })
    save_json(ART / "evaluation_inputs.json", {
        "created_utc": utc(), "experiment_freeze_sha256": sha(ART / "frozen.json"),
        "preholdout_ledger_sha256": sha(ART / "preholdout_ledger.json"),
        "terrain_cache_manifest_sha256": sha(ART / "terrain_cache.json"),
    })


def evaluate(device, *, horizon=False):
    folder, scenario = ("horizon", "stones") if horizon else ("evaluations", "mixed")
    inputs = evaluation_inputs()
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
            output.parent.mkdir(parents=True, exist_ok=True)
            run(eval_command(controller, geometry, reset, scenario, "holdout", output,
                             ROOT / frozen["models"][key]["checkpoint"], device=device),
                f"{folder}_{label}")
            evaluation_inputs()
            assert_cache(cache)
            data = read(output)
            audit(data)
            pair = (geometry, reset)
            hashes = {name: data[name] for name in ("initial_state_sha256", "initial_prefix_sha256",
                                                  "initial_rng_sha256")}
            if pair in initial and initial[pair] != hashes:
                raise ValueError("v18 holdout paired initial state/prefix/RNG differs")
            initial[pair] = hashes
    if inputs != evaluation_inputs():
        raise ValueError("v18 evaluation input manifest changed during scoring")


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
    parser.add_argument("phase", choices=("parity", "final_smoke", "freeze_eval", "prepare_eval",
                                          "evaluate", "horizon", "report"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/hybrid_v11_20260922/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"parity": lambda: parity(args.device), "final_smoke": lambda: final_smoke(args.device),
         "freeze_eval": freeze_eval, "prepare_eval": lambda: prepare_eval(args.device),
         "evaluate": lambda: evaluate(args.device), "horizon": lambda: evaluate(args.device, horizon=True),
         "report": report}[args.phase]()


if __name__ == "__main__":
    main()
