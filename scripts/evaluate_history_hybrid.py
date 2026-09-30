"""Versioned temporal selector adapter; reuse the byte-frozen v11 simulator loop.

Only in-process gate construction and NEW result metadata change. Existing
evaluation files, policies, observations, rewards and simulator code do not.
The original backend result is retained under ignored outputs/ for provenance.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/terrain_demo/history_v12"
V11 = ROOT / "artifacts/terrain_demo/hybrid_v11/frozen.json"
BACKEND = ROOT / "scripts/evaluate_hybrid.py"
BASE_SCHEMA = "week03_ant_depth_switch_v11_v1"
SCHEMA = "week03_ant_history_switch_v12_v1"
NEW_SOURCES = (
    "src/week03_ant/history_gate.py", "scripts/evaluate_history_hybrid.py",
    "scripts/summarize_history_hybrid.py", "tests/test_history_gate.py",
    "tests/test_history_harness.py", "tests/test_history_summary.py",
)
SPATIAL_FIELDS = (
    "enter_span", "enter_edge", "retain_span", "retain_edge",
    "enter_coverage", "retain_coverage", "enter_edge_coverage", "retain_edge_coverage",
)
BASE_FIELDS = set(SPATIAL_FIELDS) | {
    "actions", "alpha", "target_v10", "switched", "uncertain", "coverage",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("history", "instant", "v5", "v10"), default="history")
    parser.add_argument("--geometry", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, choices=(42,), default=42)
    parser.add_argument("--scenario", choices=("mixed", "stones"), default="mixed")
    parser.add_argument("--seconds", type=int, choices=(16, 64), default=16)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--phase", choices=("smoke", "holdout", "video"), default="smoke")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_backend-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_events-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.num_envs is None:
        args.num_envs = 175 if args.scenario == "mixed" else 10
    if args.num_envs <= 0 or min(args.geometry, args.seed) < 0:
        parser.error("positive num-envs and nonnegative geometry/reset seeds required")
    if args.record and args.phase != "video":
        parser.error("recording is qualitative only: use --phase video")
    if args._worker != bool(args._backend_output and args._events_output):
        parser.error("internal worker requires both private output paths")
    if not args._worker and (args._backend_output or args._events_output):
        parser.error("private output paths require worker mode")
    return args


def validate_paths(args):
    for path in (args.output, args.record, args._backend_output, args._events_output):
        if path is not None and path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
    if args.record and not args.record.resolve().is_relative_to(ROOT):
        raise ValueError("record path must be inside this repository")
    for path in (args._backend_output, args._events_output):
        if path and not path.resolve().is_relative_to(ROOT / "outputs/history_v12_20260922/backend"):
            raise ValueError("private backend evidence must remain under ignored outputs")


def backend_argv(args, backend_output):
    mode = args.controller if args.controller in ("v5", "v10") else "hybrid"
    checkpoint = ROOT / "artifacts/terrain_demo/prior_v10/runs/v10_anchored_seed42/model_749.pt"
    command = [str(BACKEND), "--mode", mode, "--preset", "cautious",
               "--geometry", str(args.geometry), "--seed", str(args.seed),
               "--policy-seed", str(args.policy_seed), "--scenario", args.scenario,
               "--seconds", str(args.seconds), "--num_envs", str(args.num_envs),
               "--phase", args.phase, "--device", args.device,
               "--checkpoint", str(checkpoint), "--output", str(backend_output)]
    if args.headless:
        command.append("--headless")
    if args.record:
        command += ["--record", str(args.record.resolve()), "--kit_args=--/renderer/multiGpu/enabled=false"]
    return command


@contextmanager
def patched_runtime(module, factory, config, argv):
    """Restore exact imported objects and argv/path even after simulator failure."""
    old_factory, old_presets = module.DepthPolicyGate, module.PRESETS
    old_argv, old_path = sys.argv, sys.path.copy()
    try:
        module.DepthPolicyGate = factory
        module.PRESETS = {"cautious": config}
        sys.argv = argv
        sys.path.insert(0, str(ROOT / "scripts"))
        yield
    finally:
        module.DepthPolicyGate, module.PRESETS = old_factory, old_presets
        sys.argv = old_argv
        sys.path[:] = old_path


class RecordedGate:
    """Record terminal-step decisions, never post-reset episodes, without feedback."""

    def __init__(self, gate, num_envs, on_complete=None, max_steps=None):
        self.gate = gate
        self.events = [[] for _ in range(num_envs)]
        self.finished = None
        self.step_index = 0
        self.on_complete, self.max_steps = on_complete, max_steps
        self._saved = False

    def step(self, *args, **kwargs):
        output = self.gate.step(*args, **kwargs)
        self.step_index += 1
        if self.finished is None:
            self.finished = output["target_v10"].new_zeros(output["target_v10"].shape)
        indices = (output["switched"] & ~self.finished).nonzero().flatten().cpu().tolist()
        if indices:
            vectors = {key: value.detach().cpu().tolist() for key, value in output.items()
                       if hasattr(value, "shape") and value.shape == self.finished.shape}
            for index in indices:
                values = {key: value[index] for key, value in vectors.items()}
                values = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                          for key, value in values.items()}
                self.events[index].append({
                    "step": self.step_index, "target_v10": values["target_v10"],
                    "features": {key: values[key] for key in SPATIAL_FIELDS if key in values},
                    "history": {key: value for key, value in values.items() if key not in BASE_FIELDS},
                })
        return output

    def reset(self, done_mask=None):
        import torch

        with torch.inference_mode():
            if self.finished is not None:
                if done_mask is None:
                    self.finished.fill_(True)
                else:
                    self.finished |= done_mask
            self.gate.reset(done_mask)
            completed = self.finished is not None and bool(self.finished.all())
            at_limit = self.max_steps is not None and self.step_index >= self.max_steps
            if self.on_complete and not self._saved and (completed or at_limit):
                self.on_complete(self.events)
                self._saved = True


def verify_originals():
    frozen = json.loads(V11.read_text())
    if len(frozen["source_sha256"]) != 63:
        raise ValueError("expected exactly 63 original frozen sources")
    for path, digest in frozen["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"frozen source changed: {path}")
    for entry in frozen["checkpoints"].values():
        if sha(ROOT / entry["checkpoint"]) != entry["sha256"]:
            raise ValueError("original v10 model changed")
    parent = ROOT / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
    if sha(parent) != frozen["v5_sha256"]:
        raise ValueError("original v5 model changed")
    return frozen


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    validate_paths(args)
    if args.dry_run:
        print(json.dumps({"controller": args.controller, "gate_injection": args.controller == "history",
                          "backend_argv": backend_argv(args, Path("<new-ignored-backend-output>"))}, indent=2))
        return

    # Bind the original spatial class inside history_gate BEFORE injection.
    import week03_ant.hybrid_gate as spatial
    from week03_ant.history_gate import HistoryDepthPolicyGate, HistoryGateConfig

    originals = verify_originals()
    new_sources = {path: sha(ROOT / path) for path in NEW_SOURCES}
    history_config = HistoryGateConfig()
    config = history_config if args.controller == "history" else spatial.PRESETS["cautious"]
    frozen_path = ART / "frozen.json"
    if args.phase != "smoke":
        frozen = json.loads(frozen_path.read_text())
        if (frozen["source_sha256"] != new_sources or frozen["history_config"] != asdict(history_config)
                or frozen["v11_frozen_sha256"] != sha(V11)):
            raise ValueError("history source/config differs from predeclared freeze")

    if args._worker:
        # SimulationApp.close may terminate the whole worker process. Save the
        # diagnostic sidecar before shutdown; publication belongs to the parent.
        original_class = spatial.DepthPolicyGate
        instances = []

        def completed(events):
            with args._events_output.open("x") as stream:
                json.dump({"events": events, "source_sha256": new_sources,
                           "controller": args.controller, "gate_config": asdict(config),
                           "num_envs": args.num_envs, "completed_step": instances[0].step_index,
                           "invocation": args._backend_output.stem},
                          stream, indent=2, allow_nan=False)
                stream.write("\n")

        def factory(num_envs, device, dt, config):
            if instances:
                raise ValueError("expected exactly one gate instance")
            cls = HistoryDepthPolicyGate if args.controller == "history" else original_class
            wrapped = RecordedGate(cls(num_envs, device, dt, config), num_envs,
                                   on_complete=completed, max_steps=args.seconds * 60)
            instances.append(wrapped)
            return wrapped

        with patched_runtime(spatial, factory, config, backend_argv(args, args._backend_output)):
            runpy.run_path(str(BACKEND), run_name="__main__")
        return

    backend_dir = ROOT / "outputs/history_v12_20260922/backend"
    backend_dir.mkdir(parents=True, exist_ok=True)
    token = f"{args.output.stem}_{uuid.uuid4().hex}"
    backend_output = backend_dir / f"{token}.json"
    events_output = backend_dir / f"{token}.events.json"
    command = [sys.executable, str(Path(__file__).resolve()), *argv, "--_worker",
               "--_backend-output", str(backend_output), "--_events-output", str(events_output)]
    process = subprocess.run(command, cwd=ROOT)
    if process.returncode:
        raise RuntimeError(f"simulation worker failed ({process.returncode}); evidence retained")
    data = json.loads(backend_output.read_text())
    if data["schema"] != BASE_SCHEMA or data["gate_config"] != asdict(config):
        raise ValueError("backend contract/config mismatch")
    diagnostic = json.loads(events_output.read_text())
    if (diagnostic["source_sha256"] != new_sources or diagnostic["controller"] != args.controller
            or diagnostic["gate_config"] != asdict(config) or diagnostic["num_envs"] != args.num_envs
            or diagnostic["invocation"] != backend_output.stem
            or diagnostic["completed_step"] != max(data["episode_lengths"])):
        raise ValueError("worker diagnostic provenance mismatch")
    events = diagnostic["events"]
    if len(events) != args.num_envs:
        raise ValueError("worker diagnostic environment count mismatch")
    for index, rows in enumerate(events):
        if ([event["step"] for event in rows] != data["episode_switch_steps"][index]
                or [event["target_v10"] for event in rows] != data["episode_switch_to_v10"][index]):
            raise ValueError("history events differ from first-episode telemetry")
    data.update({
        "schema": SCHEMA, "base_contract_schema": BASE_SCHEMA, "controller": args.controller,
        "selector_variant": "v12_history" if args.controller == "history" else
                            ("v11_instant" if args.controller == "instant" else "fixed_policy"),
        "routing_inputs": "Only causal history of depth-derived spatial features and validity; no family labels, maps, rewards or future states."
                          if args.controller == "history" else "Unchanged v11 current-depth selector or fixed policy.",
        "unknown_handling": "Unknown never means flat; hold target, reset continuous confirmations; expire stale evidence after configured unknown duration. Fade may complete."
                            if args.controller == "history" else data["unknown_handling"],
        "history_switch_events": events, "source_sha256": new_sources,
        "original_source_sha256": originals["source_sha256"], "v11_frozen_sha256": sha(V11),
        "backend_evaluator_sha256": sha(BACKEND), "backend_result_sha256": sha(backend_output),
        "backend_result_local": str(backend_output.relative_to(ROOT)),
        "backend_events_sha256": sha(events_output),
        "backend_events_local": str(events_output.relative_to(ROOT)),
        "new_training_transitions": 0,
        "adapter_finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    from summarize_history_hybrid import audit

    audit(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"[v12 adapter] {args.controller}: {args.output}")


if __name__ == "__main__":
    main()
