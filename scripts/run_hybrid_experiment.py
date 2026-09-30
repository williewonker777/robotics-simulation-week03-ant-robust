"""Bounded development followed by immutable, paired fresh-map v11 tests."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from week03_ant.hybrid_gate import PRESETS
from summarize_hybrid import aggregate, audit, markdown, summarize


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/terrain_demo/hybrid_v11"
WORK = ROOT / "outputs/hybrid_v11_20260922"
PYTHON = ROOT.parent / "run-python"
HOLDOUTS = ((68, 42), (69, 43))
POLICY_SEEDS = (42, 43, 44)
PRESET_ORDER = ("cautious", "balanced", "selective")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def source_hashes():
    previous = json.loads((ROOT / "artifacts/terrain_demo/prior_v10/frozen.json").read_text())["source_sha256"]
    current = {name: sha(ROOT / name) for name in previous}
    if current != previous:
        raise ValueError("archived v10 source/PLAN changed")
    additions = [
        "src/week03_ant/hybrid_gate.py", "src/week03_ant/hybrid_telemetry.py",
        "scripts/evaluate_hybrid.py", "scripts/run_hybrid_experiment.py", "scripts/summarize_hybrid.py",
        "scripts/probe_hybrid_scan.py", "tests/test_hybrid_gate.py", "tests/test_hybrid_telemetry.py",
        "tests/test_hybrid_summary.py", "outputs/hybrid_v11_20260922/PLAN.md",
    ]
    current.update({p: sha(ROOT / p) for p in additions})
    return current


def check_sources(expected):
    if source_hashes() != expected:
        raise ValueError("experiment source changed")


def checkpoints():
    previous = json.loads((ROOT / "artifacts/terrain_demo/prior_v10/frozen.json").read_text())
    result = {}
    for seed in POLICY_SEEDS:
        record = next(r for r in previous["runs"] if r["group"] == "anchored" and r["training_seed"] == seed)
        if sha(ROOT / record["checkpoint"]) != record["sha256"]:
            raise ValueError("frozen anchored model changed")
        result[str(seed)] = {"checkpoint": record["checkpoint"], "sha256": record["sha256"]}
    parent = ROOT / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
    if sha(parent) != previous["parent_sha256"]:
        raise ValueError("frozen v5 model changed")
    return result, previous["parent_sha256"]


def run(command, label):
    path = WORK / f"{label}.log"
    started = datetime.now(timezone.utc).isoformat()
    begin = time.monotonic()
    with path.open("x") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record = {"label": label, "command": command, "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(), "returncode": process.returncode,
              "seconds": time.monotonic() - begin, "log": str(path.relative_to(ROOT))}
    with (ARTIFACT / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"[v11] {label}: exit{process.returncode}, {record['seconds']:.1f}s", flush=True)
    if process.returncode:
        print("\n".join(path.read_text().splitlines()[-35:]), flush=True)
        raise subprocess.CalledProcessError(process.returncode, command)


def command(mode, seed, preset, geom, reset, scenario, seconds, phase, output, device, record=None):
    models, _ = checkpoints()
    result = [str(PYTHON), "scripts/evaluate_hybrid.py", "--headless", "--device", device,
              "--mode", mode, "--policy-seed", str(seed), "--preset", preset,
              "--geometry", str(geom), "--seed", str(reset), "--scenario", scenario,
              "--seconds", str(seconds), "--num_envs", "175" if scenario == "mixed" else "10",
              "--phase", phase, "--checkpoint", models[str(seed)]["checkpoint"], "--output", str(output)]
    if record:
        result += ["--record", str(record)]
    return result


def develop(device):
    if (ARTIFACT / "selection.json").exists() or (ARTIFACT / "frozen.json").exists():
        raise FileExistsError("development already selected/frozen")
    expected_sources = source_hashes()
    write_json(ARTIFACT / "source_at_development_start.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": expected_sources,
        "presets": {p: asdict(PRESETS[p]) for p in PRESET_ORDER},
        "selection_order": "one desc, falls asc, lane asc, six desc, cautious/balanced/selective order",
        "geometry": 51, "reset": 24, "policy_seed": 42,
    })
    trials = []
    initial = None
    for mode, preset in [("v5", "balanced"), ("v10", "balanced"), *[("hybrid", p) for p in PRESET_ORDER]]:
        check_sources(expected_sources)
        path = ARTIFACT / "development" / f"{mode}_{preset}.json"
        run(command(mode, 42, preset, 51, 24, "mixed", 16, "development", path, device), f"dev_{mode}_{preset}")
        data = json.loads(path.read_text())
        rows = audit(data)
        if initial is None:
            initial = data["initial_state_sha256"]
        if initial != data["initial_state_sha256"]:
            raise ValueError("development initial states are not paired")
        trials.append({"mode": mode, "preset": preset, "path": str(path.relative_to(ROOT)),
                       "sha256": sha(path), "metrics": aggregate(rows)})
        print(json.dumps(trials[-1]["metrics"]), flush=True)
    candidates = [r for r in trials if r["mode"] == "hybrid"]
    def rank(record):
        m = record["metrics"]
        return (m["one"], -m["falls"], -m["lane"], m["six"], -PRESET_ORDER.index(record["preset"]))
    winner = max(candidates, key=rank)
    write_json(ARTIFACT / "selection.json", {
        "selected_at": datetime.now(timezone.utc).isoformat(), "preset": winner["preset"],
        "gate_config": asdict(PRESETS[winner["preset"]]), "trials": trials,
        "note": "Development only; selected even if no candidate beats standalone policies. Fresh holdouts determine result.",
    })


def freeze():
    selection = json.loads((ARTIFACT / "selection.json").read_text())
    dev_sources = json.loads((ARTIFACT / "source_at_development_start.json").read_text())["source_sha256"]
    check_sources(dev_sources)
    models, parent = checkpoints()
    write_json(ARTIFACT / "frozen.json", {
        "frozen_at": datetime.now(timezone.utc).isoformat(), "preset": selection["preset"],
        "gate_config": selection["gate_config"], "selection_sha256": sha(ARTIFACT / "selection.json"),
        "checkpoints": models, "v5_sha256": parent, "source_sha256": source_hashes(),
        "holdouts": HOLDOUTS, "new_training_transitions": 0,
        "primary": {"scenario": "mixed", "seconds": 16, "envs": 175, "files": 14, "episodes": 2450},
        "secondary": {"scenario": "stones", "seconds": 64, "envs": 10, "files": 14, "episodes": 140},
        "promotion": "one/six>=both; terrain falls/lane<=v5;world0;flatfalls<=v5;64s stone six>v5 and falls<=v5",
        "video_condition": {"policy_seed": 42, "geometry": 68, "reset": 42, "scenario": "stones", "seconds": 16},
    })


def evaluate(device, horizon=False):
    frozen = json.loads((ARTIFACT / "frozen.json").read_text())
    folder, scenario, seconds = ("horizon", "stones", 64) if horizon else ("evaluations", "mixed", 16)
    initial = {}
    for mode, seed in [("v5", 42), *[(m, s) for s in POLICY_SEEDS for m in ("v10", "hybrid")]]:
        for geom, reset in HOLDOUTS:
            check_sources(frozen["source_sha256"])
            if checkpoints()[0] != frozen["checkpoints"]:
                raise ValueError("frozen models changed")
            label = f"{mode}_seed{seed}__geometry{geom}_reset{reset}"
            output = ARTIFACT / folder / f"{label}.json"
            run(command(mode, seed, frozen["preset"], geom, reset, scenario, seconds, "holdout", output, device), f"{folder}_{label}")
            data = json.loads(output.read_text())
            rows = audit(data)
            initial.setdefault((geom, reset), data["initial_state_sha256"])
            if data["initial_state_sha256"] != initial[(geom, reset)]:
                raise ValueError("heldout initial states are not paired")
            print(json.dumps(aggregate(rows)), flush=True)


def report():
    frozen = json.loads((ARTIFACT / "frozen.json").read_text())
    check_sources(frozen["source_sha256"])
    result = summarize(ARTIFACT)
    write_json(ARTIFACT / "summary.json", result)
    with (ARTIFACT / "summary.md").open("x") as stream:
        stream.write(markdown(result))
    print(markdown(result), flush=True)


def videos(device):
    frozen = json.loads((ARTIFACT / "frozen.json").read_text())
    if not (ARTIFACT / "summary.json").exists():
        raise ValueError("complete primary and64s evaluation before videos")
    records = []
    for mode in ("v5", "v10", "hybrid"):
        check_sources(frozen["source_sha256"])
        movie = ARTIFACT / "videos" / f"{mode}_seed42_stones10.mp4"
        evidence = ARTIFACT / "videos" / f"{mode}_seed42_stones10.json"
        run(command(mode, 42, frozen["preset"], 68, 42, "stones", 16, "video", evidence, device, record=movie), f"video_{mode}")
        run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(movie), "-f", "null", "-"], f"decode_{mode}")
        records.append({"mode": mode, "path": str(movie.relative_to(ROOT)), "sha256": sha(movie),
                        "evidence": str(evidence.relative_to(ROOT)), "full_decode_passed": True})
    write_json(ARTIFACT / "video_manifest.json", {"selection": "All predeclared seed42/geometry68/reset42 modes, followenv0; no favorable selection.",
                                                "note": "Qualitative rendered rollouts excluded from benchmark counts; resets shown.", "videos": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("develop", "freeze", "evaluate", "horizon", "report", "videos"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    with (WORK / "gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"develop": lambda: develop(args.device), "freeze": freeze,
         "evaluate": lambda: evaluate(args.device), "horizon": lambda: evaluate(args.device, True),
         "report": report, "videos": lambda: videos(args.device)}[args.phase]()
