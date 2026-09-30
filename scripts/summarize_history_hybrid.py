"""Audit v12 first episodes and temporal switch evidence without retraining claims."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct

from week03_ant.history_gate import HistoryGateConfig
from week03_ant.hybrid_gate import PRESETS

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("frozen_v11_summary", ROOT / "scripts/summarize_hybrid.py")
BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BASE)
SCHEMA = "week03_ant_history_switch_v12_v1"
BASE_SCHEMA = "week03_ant_depth_switch_v11_v1"
CONTROLLERS = ("v5", "v10", "instant", "history")
INITIAL_KEYS = ("root_state", "joint_pos", "joint_vel", "observations")
NEW_SOURCES = (
    "src/week03_ant/history_gate.py", "scripts/evaluate_history_hybrid.py",
    "scripts/summarize_history_hybrid.py", "tests/test_history_gate.py",
    "tests/test_history_harness.py", "tests/test_history_summary.py",
)
FEATURES = {f"{roi}_{field}" for roi in ("enter", "retain")
            for field in ("span", "edge", "coverage", "edge_coverage")}
HISTORY_FIELDS = {"rough_votes", "clear_votes", "rough_fraction", "clear_fraction",
                  "enter_history_samples", "exit_history_samples", "history_samples",
                  "rough_evidence_seconds", "clear_evidence_seconds",
                  "consecutive_rough_seconds", "consecutive_clear_seconds",
                  "unknown_seconds", "history_expired"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hashes(value, keys=None, count=None):
    if (not isinstance(value, dict) or (keys is not None and set(value) != set(keys))
            or (count is not None and len(value) != count)
            or any(not isinstance(k, str) or not isinstance(v, str)
                   or re.fullmatch(r"[0-9a-f]{64}", v) is None for k, v in value.items())):
        raise ValueError("malformed SHA256 inventory")


def check_initial(initial, pair, data):
    hashes = data["initial_state_sha256"]
    _hashes(hashes, INITIAL_KEYS)
    if pair in initial and initial[pair] != hashes:
        raise ValueError("paired controllers have different initial states/observations")
    initial[pair] = dict(hashes)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("expected finite nonnegative evidence")
    return value


def _count(value, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("invalid history sample/vote count")
    return value


def _steps(seconds, dt):
    return max(1, math.ceil(seconds / dt - 1.e-9))


def _duration_steps(value, dt, maximum):
    value = _number(value)
    count = round(value / dt)
    if count > maximum or not math.isclose(value, count * dt, abs_tol=1.e-5, rel_tol=1.e-6):
        raise ValueError("duration is not a valid discrete sample count")
    return count


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _event(event, controller, dt, previous):
    if not isinstance(event, dict) or set(event) != {"step", "target_v10", "features", "history"}:
        raise ValueError("malformed switch evidence")
    step, target, features, history = (event[k] for k in ("step", "target_v10", "features", "history"))
    if type(step) is not int or step <= 0 or type(target) is not bool:
        raise ValueError("invalid switch evidence identity")
    if not isinstance(features, dict) or set(features) != FEATURES:
        raise ValueError("missing spatial switch evidence")
    for key, value in features.items():
        _number(value)
        if "coverage" in key and value > 1:
            raise ValueError("coverage exceeds one")
    cfg = HistoryGateConfig()
    spatial = cfg.spatial
    roi = "enter" if target else "retain"
    if any(features[f"{roi}_{key}"] < _f32(spatial.min_coverage) for key in ("coverage", "edge_coverage")):
        raise ValueError("switch occurred without known relevant depth coverage")
    if target:
        spatial_ok = (features["enter_span"] >= _f32(spatial.span_enter)
                      or features["enter_edge"] >= _f32(spatial.edge_enter))
    else:
        spatial_ok = (features["retain_span"] <= _f32(spatial.exit_ratio * spatial.span_enter)
                      and features["retain_edge"] <= _f32(spatial.exit_ratio * spatial.edge_enter))
    if not spatial_ok:
        raise ValueError("switch contradicts spatial thresholds")
    if previous is not None and step - previous < _steps(cfg.min_dwell_seconds, dt):
        raise ValueError("switch violates minimum dwell")
    if controller != "history":
        if history != {}:
            raise ValueError("non-history controller has temporal history metadata")
        if target and step < spatial.enter_steps:
            raise ValueError("instant switch precedes spatial confirmation")
        return
    if not isinstance(history, dict) or set(history) != HISTORY_FIELDS:
        raise ValueError("missing temporal switch evidence")
    samples = _count(history["history_samples"], min(step, _steps(cfg.history_seconds, dt)))
    if samples == 0 or history["history_expired"] is not False or _number(history["unknown_seconds"]) != 0:
        raise ValueError("switch used empty/expired/unknown history")
    for direction, word in (("enter", "rough"), ("exit", "clear")):
        window = _steps(getattr(cfg, f"{direction}_window_seconds"), dt)
        count = _count(history[f"{direction}_history_samples"], min(samples, window))
        if count != min(samples, window):
            raise ValueError("recorded window denominator inconsistent with history occupancy")
        votes = _count(history[f"{word}_votes"], count)
        fraction = _number(history[f"{word}_fraction"])
        if not math.isclose(fraction, votes / count, abs_tol=1.e-6, rel_tol=1.e-6):
            raise ValueError("history vote fraction mismatch")
        if _duration_steps(history[f"{word}_evidence_seconds"], dt, window) != votes:
            raise ValueError("history support duration mismatch")
        continuous = _duration_steps(history[f"consecutive_{word}_seconds"], dt, step)
        if votes < min(continuous, count):
            raise ValueError("continuous confirmation contradicts recorded votes")
        if (direction == "enter") == target:
            if (votes < _steps(getattr(cfg, f"{direction}_min_seconds"), dt)
                    or fraction < _f32(getattr(cfg, f"{direction}_fraction"))
                    or continuous < _steps(getattr(cfg, f"{direction}_confirm_seconds"), dt)):
                raise ValueError("insufficient sustained switch evidence")


def _audit_routing(data):
    targets = data.get("episode_v10_target_steps")
    if not isinstance(targets, list) or len(targets) != data["num_envs"]:
        raise ValueError("missing target occupancy array")
    dt = data["condition"]["dt"]
    blend = data["gate_config"]["blend_seconds"]
    for i, length in enumerate(data["episode_lengths"]):
        events = dict(zip(data["episode_switch_steps"][i], data["episode_switch_to_v10"][i]))
        target = False
        alpha, alpha_sum, target_steps = 0.0, 0.0, 0
        for step in range(1, length + 1):
            if data["mode"] == "hybrid":
                target = events.get(step, target)
                # Gate state is float32; telemetry accumulates those values in float64.
                alpha = min(1., max(0., _f32(alpha + _f32((1 if target else -1) * dt / blend))))
            else:
                target = data["mode"] == "v10"
                alpha = float(target)
            target_steps += int(target)
            alpha_sum += alpha
        if (not math.isclose(alpha_sum, data["episode_alpha_sum"][i], abs_tol=1.e-5, rel_tol=1.e-8)
                or not math.isclose(alpha_sum / length, data["episode_v10_duty"][i], abs_tol=1.e-8, rel_tol=1.e-8)):
            raise ValueError("reconstructed alpha/duty disagrees with switch events")
        if type(targets[i]) is not int or targets[i] != target_steps:
            raise ValueError("reconstructed target occupancy disagrees with switch events")


def audit(data):
    controller = data.get("controller")
    if data.get("task") != "Week03-Ant-Prior-Lanes-Eval-v10" or data.get("difficulties") != [.2, .4, .6, .8, 1.]:
        raise ValueError("changed task/difficulty contract")
    condition = data.get("condition", {})
    expected_snapshot = 8.0 if condition.get("seconds") == 16 else 16.0
    if condition.get("snapshot_seconds") != expected_snapshot:
        raise ValueError("changed snapshot contract")
    _hashes({key: data.get(key) for key in ("backend_evaluator_sha256", "backend_result_sha256")})
    if data.get("schema") != SCHEMA or data.get("base_contract_schema") != BASE_SCHEMA or controller not in CONTROLLERS:
        raise ValueError("unexpected version/controller contract")
    mode = controller if controller in ("v5", "v10") else "hybrid"
    variant = "v12_history" if controller == "history" else ("v11_instant" if controller == "instant" else "fixed_policy")
    config = asdict(HistoryGateConfig()) if controller == "history" else asdict(PRESETS["cautious"])
    if data.get("mode") != mode or data.get("selector_variant") != variant or data.get("gate_config") != config:
        raise ValueError("controller/mode/config identity mismatch")
    if (data.get("teacher_tensor_identity_verified") is not True
            or type(data.get("new_training_transitions")) is not int or data["new_training_transitions"] != 0):
        raise ValueError("frozen teacher/no-training contract violated")
    _hashes(data.get("source_sha256"), NEW_SOURCES)
    _hashes(data.get("original_source_sha256"), count=63)
    _hashes(data.get("initial_state_sha256"), INITIAL_KEYS)
    base_view = dict(data, schema=BASE_SCHEMA)
    rows = BASE.audit(base_view)
    _audit_routing(data)
    events = data.get("history_switch_events")
    if not isinstance(events, list) or len(events) != len(rows):
        raise ValueError("missing per-episode switch evidence")
    for i, entries in enumerate(events):
        if not isinstance(entries, list) or len(entries) != data["episode_switch_count"][i]:
            raise ValueError("switch evidence count mismatch")
        previous = None
        for j, event in enumerate(entries):
            _event(event, controller, data["condition"]["dt"], previous)
            if event["step"] != data["episode_switch_steps"][i][j] or event["target_v10"] != data["episode_switch_to_v10"][i][j]:
                raise ValueError("switch evidence differs from first-episode telemetry")
            previous = event["step"]
    return rows


def aggregate(rows):
    result = BASE.aggregate(rows)
    terrain = [r for r in rows if r["family"] != "flat"]
    flat = [r for r in rows if r["family"] == "flat"]
    result["active_steps"] = sum(r["steps"] for r in terrain)
    result["active_seconds"] = result["active_steps"] / 60
    result["switches_per_100_active_seconds"] = (100 * result["switches"] / result["active_seconds"]
                                                   if result["active_seconds"] else None)
    result["flat_active_seconds"] = sum(r["steps"] for r in flat) / 60
    result["flat_switches"] = sum(r["switches"] for r in flat)
    return result


def improvement(history, instant, primary=True):
    def rate(group, key, denominator="n"):
        if group[denominator] <= 0:
            raise ValueError("empty comparison denominator")
        return Fraction(group[key], group[denominator])
    checks = {"fewer_switches_per_active_time": rate(history, "switches", "active_steps") < rate(instant, "switches", "active_steps"),
              "world_zero": history["world"] + history["flat_world"] == 0}
    checks.update({f"{key}_not_lower": rate(history, key) >= rate(instant, key) for key in ("one", "six")})
    checks.update({f"{key}_not_higher": rate(history, key) <= rate(instant, key) for key in ("falls", "lane")})
    if primary:
        checks["flat_falls_not_higher"] = rate(history, "flat_falls", "flat_n") <= rate(instant, "flat_falls", "flat_n")
    return {"passed": all(checks.values()), "checks": checks}


def _timestamp(value):
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("timestamps must include timezone")
    return dt


def summarize(directory):
    directory = Path(directory)
    frozen = json.loads((directory / "frozen.json").read_text())
    if frozen["holdouts"] != [[72, 46], [73, 47]] or frozen["controllers"] != list(CONTROLLERS):
        raise ValueError("changed predeclared holdout inventory")
    if frozen["history_config"] != asdict(HistoryGateConfig()) or frozen["instant_config"] != asdict(PRESETS["cautious"]):
        raise ValueError("changed frozen selector configuration")
    _hashes(frozen["source_sha256"], NEW_SOURCES)
    _hashes(frozen["v11_source_sha256"], count=63)
    v11_path = ROOT / "artifacts/terrain_demo/hybrid_v11/frozen.json"
    v11 = json.loads(v11_path.read_text())
    if sha(v11_path) != frozen["v11_frozen_sha256"] or frozen["v11_source_sha256"] != v11["source_sha256"]:
        raise ValueError("v11 provenance mismatch")
    if frozen["checkpoints"]["42"] != v11["checkpoints"]["42"] or frozen["v5_sha256"] != v11["v5_sha256"]:
        raise ValueError("checkpoint freeze differs from v11")
    for name, digest in {**frozen["source_sha256"], **frozen["v11_source_sha256"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"source changed after freeze: {name}")
    checkpoint = frozen["checkpoints"]["42"]
    parent = ROOT / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
    if sha(ROOT / checkpoint["checkpoint"]) != checkpoint["sha256"] or sha(parent) != frozen["v5_sha256"]:
        raise ValueError("frozen model bytes changed")
    result = {"files": []}
    families = ("flat", "rough", "slope", "stairs", "waves", "obstacles", "stepping_stones")
    for scenario, folder, envs, seconds in (("mixed", "evaluations", 175, 16), ("stones", "horizon", 10, 64)):
        groups, initial = defaultdict(list), {}
        expected_paths = {f"{c}__geometry{g}_reset{r}.json" for c in CONTROLLERS for g, r in frozen["holdouts"]}
        if {p.name for p in (directory / folder).glob("*.json")} != expected_paths:
            raise ValueError("missing or unexpected evaluation file")
        for controller in CONTROLLERS:
            for geom, reset in frozen["holdouts"]:
                path = directory / folder / f"{controller}__geometry{geom}_reset{reset}.json"
                data = json.loads(path.read_text())
                expected = (controller, None if controller == "v5" else 42, geom, reset, scenario, envs, seconds, "holdout")
                actual = (data["controller"], data["policy_seed"], data["geometry_seed"], data["reset_seed"],
                          data["scenario"], data["num_envs"], data["condition"]["seconds"], data["phase"])
                if actual != expected or data["checkpoint_sha256"] != checkpoint["sha256"] or data["v5_sha256"] != frozen["v5_sha256"]:
                    raise ValueError("evaluation/model contract differs from freeze")
                if (data["source_sha256"] != frozen["source_sha256"] or data["original_source_sha256"] != frozen["v11_source_sha256"]
                        or data["v11_frozen_sha256"] != frozen["v11_frozen_sha256"]):
                    raise ValueError("evaluation source provenance mismatch")
                if _timestamp(data["started_utc"]) <= _timestamp(frozen["frozen_at"]):
                    raise ValueError("evaluation precedes freeze")
                rows = audit(data)
                if data["backend_evaluator_sha256"] != frozen["v11_source_sha256"]["scripts/evaluate_hybrid.py"]:
                    raise ValueError("backend evaluator provenance mismatch")
                backend_name = data.get("backend_result_local")
                if not isinstance(backend_name, str):
                    raise ValueError("missing local backend provenance path")
                backend = (ROOT / backend_name).resolve()
                if Path(backend_name).is_absolute() or not backend.is_relative_to(ROOT.resolve()):
                    raise ValueError("backend provenance path escapes repository")
                if backend.exists() and sha(backend) != data["backend_result_sha256"]:
                    raise ValueError("local backend result digest mismatch")
                expected_counts = ({(f, level): 5 for f in families for level in range(5)}
                                   if scenario == "mixed" else {("stepping_stones", 4): 10})
                if Counter((r["family"], r["level"]) for r in rows) != expected_counts:
                    raise ValueError("family/difficulty inventory mismatch")
                check_initial(initial, (geom, reset), data)
                groups[controller].extend(rows)
                result["files"].append({"path": str(path.relative_to(directory)), "sha256": sha(path), "episodes": len(rows)})
        result[folder] = {
            "groups": {c: aggregate(rows) for c, rows in groups.items()},
            "family": {c: {f: aggregate([r for r in rows if r["family"] == f]) for f in sorted({r["family"] for r in rows})}
                       for c, rows in groups.items()},
            "family_level": {c: {f"{f}/level{level}": aggregate([r for r in rows if (r["family"], r["level"]) == (f, level)])
                                  for f, level in sorted({(r["family"], r["level"]) for r in rows})} for c, rows in groups.items()},
        }
        result[folder]["history_vs_instant"] = improvement(result[folder]["groups"]["history"],
                                                           result[folder]["groups"]["instant"], primary=scenario == "mixed")
    result["interpretation"] = ("Diagnostic comparison: one frozen policy seed42 on two maps, not a universal-best claim. "
                                "16s and64s outcomes are separate. Ideal ray depth, no recurrent policy training. "
                                "Switch evidence is checked for consistency; raw depth is not independently reconstructed.")
    return result


def markdown(result):
    lines = ["# v12: sustained depth-history selector", "", result["interpretation"], ""]
    for folder, title in (("evaluations", "16s mixed terrain"), ("horizon", "64s stones, difficulty1.0")):
        section = result[folder]
        lines += [f"## {title}", "", "| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat duty | Switches | Active seconds | Switches/100s |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for controller in CONTROLLERS:
            d = section["groups"][controller]
            flat_duty = "—" if d["flat_mean_v10_duty"] is None else f"{d['flat_mean_v10_duty']:.3f}"
            lines.append(f"| {controller} | {d['one']}/{d['n']} | {d['six']}/{d['n']} | {d['falls']}/{d['n']} | {d['lane']}/{d['n']} | {d['world'] + d['flat_world']} | {d['flat_falls']}/{d['flat_n']} | {flat_duty} | {d['switches']} | {d['active_seconds']:.2f} | {d['switches_per_100_active_seconds']:.3f} |")
        assessment = section["history_vs_instant"]
        lines += ["", f"History improvement criteria: **{'PASS' if assessment['passed'] else 'FAIL'}**", ""]
        lines += [f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in assessment["checks"].items()]
        lines += ["", "### Per-family history minus instant (counts; negative crossing or positive failure is a regression)", "",
                  "| Family | Δone | Δsix | Δfalls | Δlane | Δworld incl.flat | Δflat falls |", "|---|---:|---:|---:|---:|---:|---:|"]
        for family, h in section["family"]["history"].items():
            old = section["family"]["instant"][family]
            changes = [h[k] - old[k] for k in ("one", "six", "falls", "lane")]
            changes += [h["world"] + h["flat_world"] - old["world"] - old["flat_world"], h["flat_falls"] - old["flat_falls"]]
            lines.append(f"| {family} | " + " | ".join(f"{v:+d}" for v in changes) + " |")
        lines.append("")
    lines.append("See JSON for complete per-family/level aggregates and source-file digests. No tuning on these holdouts.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = [args.directory / name for name in ("summary.json", "summary.md")]
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite summary evidence")
    result = summarize(args.directory)
    for path, content in zip(paths, (json.dumps(result, indent=2, allow_nan=False) + "\n", markdown(result))):
        with path.open("x") as stream:
            stream.write(content)
