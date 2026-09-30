"""Run the bounded v20 common-map evaluation, serially and without overwrites."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from week03_ant.curriculum_study_v20 import (
    ART, WORK, ROOT, CONTROLLERS, HOLDOUTS, models, model_key, source_hashes,
    legacy_hashes, verify_frozen, evaluation_inputs, save_json, sha, utc, TRANSITIONS,
)
from week03_ant.posture_study import cache_snapshot
from summarize_lp_v17 import PARITY_FIELDS
from summarize_contact_v16 import _paired, _flat_identity
from summarize_curriculum_v20 import audit, summarize, markdown, audit_training

PYTHON = str(ROOT.parent / "run-python")
CACHE = Path("/tmp/isaaclab/terrains")


def read(path):
    return json.loads(Path(path).read_text())


def verify_model_inventory():
    manifest = models()
    if set(manifest) != {'parent', 'immediate', 'ramped'}:
        raise ValueError('complete trained model inventory required')
    for name, entry in manifest.items():
        if sha(ROOT / entry['checkpoint']) != entry['sha256']:
            raise ValueError('model checkpoint bytes changed')
        if name != 'parent' and (entry.get('transitions') != TRANSITIONS or entry.get('iteration') != 249):
            raise ValueError('only final full-budget checkpoints are allowed')
    return manifest


def eval_command(controller, geometry, reset, scenario, phase, output, device="cuda:1"):
    entry = models()[model_key(controller)]
    return [PYTHON, "scripts/evaluate_curriculum_v20.py", "--headless", "--device", device,
            "--controller", controller, "--geometry", str(geometry), "--seed", str(reset),
            "--scenario", scenario, "--seconds", "16" if scenario == "mixed" else "64",
            "--num_envs", str(35 if phase in ("smoke", "prepare_development") else 175 if scenario == "mixed" else 10),
            "--phase", phase, "--output", str(output),
            "--checkpoint", str(ROOT / entry["checkpoint"])]


def run(command, label):
    log = WORK / f"{label}.log"
    output = Path(command[command.index("--output") + 1])
    if log.exists() or output.exists():
        raise FileExistsError(f"no implicit retries/overwrites: {label}")
    started, begin = utc(), time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = {"label": label, "command": command, "started_utc": started,
              "finished_utc": utc(), "seconds": time.monotonic() - begin,
              "returncode": result.returncode, "log": str(log.relative_to(ROOT)),
              "log_sha256": sha(log)}
    if (ART / "evaluation_inputs.json").exists():
        inputs = evaluation_inputs()
        record.update(evaluation_inputs_sha256=sha(ART / "evaluation_inputs.json"),
                      terrain_cache_manifest_sha256=inputs["terrain_cache_manifest_sha256"])
    with (ART / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"[v20] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode or not output.is_file():
        print("\n".join(log.read_text().splitlines()[-30:]), flush=True)
        raise RuntimeError(f"evaluator failed; preserve evidence and stop: {label}")
    return output


def development(device):
    audit_training()
    before, legacy, model_manifest = source_hashes(), legacy_hashes(), verify_model_inventory()
    if set(model_manifest) != {"parent", "immediate", "ramped"}:
        raise ValueError("complete trained model inventory required")
    preparation = WORK / "prepare_development.json"
    run(eval_command("parent", 51, 24, "mixed", "prepare_development", preparation, device),
        "prepare_development")
    if read(preparation).get("scored_episodes") != 0:
        raise ValueError("development preparation scored episodes")
    cache = cache_snapshot(CACHE, [51])
    initial, flats, records, parity = {}, {}, [], []
    references = {
        "parent": ROOT / "outputs/rebaseline_v19_20260928/attempt02/smoke_v16_control.json",
        "history_parent": ROOT / "outputs/rebaseline_v19_20260928/attempt02/smoke_history_control.json",
    }
    for controller in CONTROLLERS:
        if verify_model_inventory() != model_manifest:
            raise ValueError("development model inventory changed")
        if cache_snapshot(CACHE, [51]) != cache:
            raise ValueError("development cache changed")
        output = WORK / f"smoke_{controller}.json"
        run(eval_command(controller, 51, 24, "mixed", "smoke", output, device), f"smoke_{controller}")
        data = read(output)
        rows = audit(data)
        if cache_snapshot(CACHE, [51]) != cache:
            raise ValueError("development cache changed")
        _paired(initial, (51, 24), data)
        if controller.startswith("history_"):
            _flat_identity(flats, (51, 24), data, rows)
        if controller in references:
            old = read(references[controller])
            if any(old.get(field) != data.get(field) for field in PARITY_FIELDS):
                raise ValueError(f"development parity failed: {controller}")
            parity.append({"controller": controller, "equal_fields": list(PARITY_FIELDS),
                           "path": str(references[controller].relative_to(ROOT)),
                           "sha256": sha(references[controller])})
        records.append({"controller": controller, "path": str(output.relative_to(ROOT)),
                        "sha256": sha(output)})
    if source_hashes() != before or legacy_hashes() != legacy or verify_model_inventory() != model_manifest:
        raise ValueError("development source changed during execution")
    save_json(ART / "development.json", {
        "created_utc": utc(), "passed": True, "exact_initial_pairing": True,
        "exact_flat_identity": True, "records": records, "parity": parity,
        "source_sha256": before, "models": model_manifest, "terrain_cache": cache,
        "preparation": {"path": str(preparation.relative_to(ROOT)), "sha256": sha(preparation)},
    })


def freeze():
    audit_training()
    verify_model_inventory()
    data = read(ART / "development.json")
    if (not data["passed"] or len(data["records"]) != 6 or len(data["parity"]) != 2
            or {row["controller"] for row in data["records"]} != set(CONTROLLERS)
            or data["source_sha256"] != source_hashes() or data["models"] != models()
            or data["terrain_cache"] != cache_snapshot(CACHE, [51])):
        raise ValueError("incomplete development proof")
    preparation = data["preparation"]
    if (sha(ROOT / preparation["path"]) != preparation["sha256"]
            or read(ROOT / preparation["path"]).get("scored_episodes") != 0):
        raise ValueError("development preparation proof changed")
    # Re-open the raw data, rather than trusting success flags alone.
    initial, flats = {}, {}
    for record in data["records"]:
        path = ROOT / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("development raw result changed")
        raw = read(path)
        rows = audit(raw)
        _paired(initial, (51, 24), raw)
        if raw["controller"].startswith("history_"):
            _flat_identity(flats, (51, 24), raw, rows)
    for reference in data["parity"]:
        if sha(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError("legacy development reference changed")
        new = read(WORK / f"smoke_{reference['controller']}.json")
        old = read(ROOT / reference["path"])
        if any(old[field] != new[field] for field in PARITY_FIELDS):
            raise ValueError("legacy parity proof changed")
    save_json(ART / "frozen.json", {
        "frozen_at": utc(), "source_sha256": source_hashes(), "legacy": legacy_hashes(),
        "models": models(), "controllers": list(CONTROLLERS), "holdouts": HOLDOUTS,
        "new_training_transitions": 2 * TRANSITIONS, "development_sha256": sha(ART / "development.json"),
        "training_freeze_sha256": sha(ART / "training_frozen.json"),
        "training_validation_sha256": sha(ART / "training_validation.json"),
        "trained_models_sha256": sha(ART / "trained_models.json"),
        "primary": {"files": 12, "episodes": 2100, "seconds": 16, "envs": 175},
        "secondary": {"files": 12, "episodes": 120, "seconds": 64, "envs": 10},
    })
    verify_frozen()


def prepare(device):
    verify_frozen()
    records = []
    for geometry, reset in HOLDOUTS:
        label = f"prepare_geometry{geometry}"
        output = WORK / f"{label}.json"
        run(eval_command("parent", geometry, reset, "mixed", "prepare", output, device), label)
        if read(output).get("scored_episodes") != 0:
            raise ValueError("initialization-only preparation scored an episode")
        records.append({"path": str(output.relative_to(ROOT)), "sha256": sha(output)})
    save_json(ART / "terrain_cache.json", {
        "created_utc": utc(), "preparation": records,
        "cache": cache_snapshot(CACHE, [g for g, _ in HOLDOUTS]),
    })
    ledger = ART / "commands.jsonl"
    raw = ledger.read_bytes()
    save_json(ART / "preholdout_ledger.json", {
        "created_utc": utc(), "path": str(ledger.relative_to(ROOT)), "lines": len(raw.splitlines()),
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
    })
    save_json(ART / "evaluation_inputs.json", {
        "created_utc": utc(), "experiment_freeze_sha256": sha(ART / "frozen.json"),
        "preholdout_ledger_sha256": sha(ART / "preholdout_ledger.json"),
        "terrain_cache_manifest_sha256": sha(ART / "terrain_cache.json"),
    })


def assert_cache(expected):
    if cache_snapshot(CACHE, [g for g, _ in HOLDOUTS]) != expected:
        raise ValueError("terrain cache changed; do not mix attempts")


def evaluate(device, horizon=False):
    folder, scenario = ("horizon", "stones") if horizon else ("evaluations", "mixed")
    cache = read(ART / "terrain_cache.json")["cache"]
    initial, flats = {}, {}
    for controller in CONTROLLERS:
        for geometry, reset in HOLDOUTS:
            verify_frozen()
            evaluation_inputs()
            assert_cache(cache)
            label = f"{controller}__geometry{geometry}_reset{reset}"
            output = ART / folder / f"{label}.json"
            run(eval_command(controller, geometry, reset, scenario, "holdout", output, device),
                f"{folder}_{label}")
            verify_frozen()
            evaluation_inputs()
            assert_cache(cache)
            data = read(output)
            rows = audit(data)
            _paired(initial, (geometry, reset), data)
            if scenario == "mixed" and controller.startswith("history_"):
                _flat_identity(flats, (geometry, reset), data, rows)


def report():
    result = summarize(ART)
    save_json(ART / "summary.json", result)
    with (ART / "summary.md").open("x") as stream:
        stream.write(markdown(result))
    print(markdown(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "freeze", "prepare", "evaluate", "horizon", "report"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.device != "cuda:1":
        parser.error("v20 is assigned to cuda:1")
    ART.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/hybrid_v11_20260922/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"development": lambda: development(args.device), "freeze": freeze,
         "prepare": lambda: prepare(args.device), "evaluate": lambda: evaluate(args.device),
         "horizon": lambda: evaluate(args.device, True), "report": report}[args.phase]()


if __name__ == "__main__":
    main()
