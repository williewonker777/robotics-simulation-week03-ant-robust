"""Independent raw first-episode audit and preregistered v18 comparisons."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from week03_ant.style_eval_v18 import evaluation_inputs, source_hashes
from week03_ant.style_study_v18 import (
    ARMS, ART, CONTROLLERS, HOLDOUTS, ROOT, SCHEMA, START_SHA, V16_CONTROL_SHA,
    EVAL_TASK, legacy_hashes, sha, verify_hashes,
)
from summarize_lp_v17 import (
    CONTACT_CONFIG, FLAT_IDENTITY_FIELDS, FAMILIES, HISTORY, PARITY_FIELDS,
    _flat_identity, _paired, aggregate as aggregate_physical,
    audit as audit_v17_physical, improvement,
)
from summarize_direction_v15 import audit_preholdout_ledger
from week03_ant.lp_study_v17 import SCHEMA as V17_SCHEMA

PROJECT = {
    "v16_control": "v16_control", "always": "fixed", "gated": "lp",
    "history_original": "history_original", "history_always": "history_fixed",
    "history_gated": "history_lp",
}


def audit(data):
    """Check v18 conditions, then reuse the frozen v17 physical/telemetry audit."""
    controller = data.get("controller")
    if data.get("schema") != SCHEMA or data.get("task") != EVAL_TASK or controller not in CONTROLLERS:
        raise ValueError("invalid v18 schema/task/controller")
    if data.get("phase") not in ("smoke", "holdout") or data.get("scenario") not in ("mixed", "stones"):
        raise ValueError("invalid v18 scoring phase/scenario")
    if data["phase"] == "holdout" and (data.get("geometry_seed"), data.get("reset_seed")) not in HOLDOUTS:
        raise ValueError("v18 holdout map differs")
    if data.get("evaluation_plan_sha256") != sha(ROOT / "docs/experiment_plans/terrain_style_v18.md"):
        raise ValueError("v18 evaluation plan changed")
    projected = deepcopy(data)
    projected.update(schema=V17_SCHEMA, controller=PROJECT[controller], phase="smoke")
    return audit_v17_physical(projected)


def _read(path):
    return json.loads(Path(path).read_text())


def _inside(name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("v18 provenance path must be repository-relative")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("v18 provenance path escapes repository")
    return path


def _verify_frozen(directory):
    frozen = _read(directory / "frozen.json")
    if (frozen["holdouts"] != [list(pair) for pair in HOLDOUTS]
            or frozen["controllers"] != list(CONTROLLERS)
            or frozen["gate_config"] != asdict(HistoryGateConfig())
            or frozen["posture_config"] != asdict(PostureConfig())
            or frozen["contact_config"] != CONTACT_CONFIG):
        raise ValueError("v18 frozen evaluation condition changed")
    if frozen["source_sha256"] != source_hashes() or frozen["legacy"] != legacy_hashes():
        raise ValueError("v18 source/legacy inventory changed")
    verify_hashes({**frozen["source_sha256"], **frozen["legacy"]["sources"],
                   **frozen["legacy"]["models"]})
    for model in frozen["models"].values():
        verify_hashes({model["checkpoint"]: model["sha256"]})
    if (frozen["models"]["original"]["sha256"] != START_SHA
            or frozen["models"]["v16_control"]["sha256"] != V16_CONTROL_SHA
            or set(frozen["models"]) != {"original", "v16_control", *ARMS}):
        raise ValueError("v18 frozen model inventory changed")
    if (sha(directory / "training_frozen.json") != frozen["training_freeze_sha256"]
            or sha(directory / "trained_models.json") != frozen["trained_models_sha256"]
            or sha(directory / "training_validation.json") != frozen["training_validation_sha256"]
            or sha(directory / "final_model_smokes.json") != frozen["final_model_smokes_sha256"]
            or sha(directory / "evaluator_parity.json") != frozen["evaluator_parity_sha256"]):
        raise ValueError("v18 training/development proof changed")
    training, trained = _read(directory / "training_frozen.json"), _read(directory / "trained_models.json")
    validation = _read(directory / "training_validation.json")
    if (training["source_sha256"] != {k: frozen["source_sha256"][k] for k in training["source_sha256"]}
            or trained["models"] != {arm: frozen["models"][arm] for arm in ARMS}
            or trained["total_training_transitions"] != 65536000
            or trained["training_validation_sha256"] != frozen["training_validation_sha256"]
            or set(validation.get("records", {})) != set(ARMS)):
        raise ValueError("v18 training contract changed")
    preflight, capacity = _read(directory / "preflight.json"), _read(directory / "capacity.json")
    if (sha(directory / "preflight.json") != training["preflight_sha256"]
            or sha(directory / "capacity.json") != training["capacity_sha256"]
            or preflight.get("terrain_cache") != training["terrain_cache"]
            or capacity.get("terrain_cache") != training["terrain_cache"]
            or preflight.get("passed") is not True or capacity.get("passed") is not True):
        raise ValueError("v18 development terrain-cache evidence changed")
    paired_start = None
    for arm in ARMS:
        model = frozen["models"][arm]
        if model["iteration"] != 249 or model["transitions"] != 32768000:
            raise ValueError("v18 selected nonfinal checkpoint")
        if sha(directory / "training" / f"{arm}_initial.json") != model["initial_audit_sha256"]:
            raise ValueError("v18 paired initialization evidence changed")
        if sha(directory / "training" / f"{arm}_exposure.json") != model["exposure_sha256"]:
            raise ValueError("v18 mask exposure evidence changed")
        initial = _read(directory / "training" / f"{arm}_initial.json")
        exposure = _read(directory / "training" / f"{arm}_exposure.json")
        if (initial.get("arm") != arm or initial.get("num_envs") != 4096
                or initial.get("optimizer_state_empty") is not True or initial.get("iteration") != 0
                or initial.get("initial_policy_parity") != {"actor": True, "critic": True, "teacher": True}):
            raise ValueError("v18 actual starting learner contract differs")
        start = {key: initial[key] for key in ("initial_state_sha256", "initial_prefix_sha256",
                                             "policy_state_sha256", "rng_sha256",
                                             "initial_lane_sha256", "normalized_parameters")}
        if paired_start is not None and paired_start != start:
            raise ValueError("v18 paired learner initialization differs")
        paired_start = start
        logdir = _inside(initial["log_dir"])
        for name, digest in initial["saved_parameter_sha256"].items():
            if sha(logdir / "params" / f"{name}.yaml") != digest:
                raise ValueError("v18 actual saved training config changed")
        counts, teacher = exposure.get("lane_transition_counts"), exposure.get("lane_teacher_counts")
        families, family_teacher = exposure.get("family_transition_counts"), exposure.get("family_teacher_counts")
        if (exposure.get("arm") != arm or exposure.get("policy_steps") != 8000
                or exposure.get("transition_steps") != 32768000
                or not isinstance(counts, list) or not isinstance(teacher, list)
                or len(counts) != 35 or len(teacher) != 35
                or any(type(c) is not int or type(t) is not int or not 0 <= t <= c
                       for c, t in zip(counts, teacher))
                or sum(counts) != 32768000
                or not isinstance(families, list) or not isinstance(family_teacher, list)
                or len(families) != 7 or len(family_teacher) != 7
                or families != [sum(counts[i:i + 5]) for i in range(0, 35, 5)]
                or family_teacher != [sum(teacher[i:i + 5]) for i in range(0, 35, 5)]
                or families[-1] == 0 or family_teacher[-1] / families[-1] < .95
                or sum(c - t > 0 for c, t in zip(families[:-1], family_teacher[:-1])) < 4):
            raise ValueError("v18 raw mask exposure/budget invalid")
        record = validation["records"][arm]
        if (record.get("iterations_logged") != 250
                or record.get("transitions_logged") != 32768000
                or record.get("all_scalars_finite") is not True
                or sha(_inside(record["text_log"])) != record["text_log_sha256"]
                or any(count != 250 for count in record["scalar_counts"].values())):
            raise ValueError("v18 training log validation differs")
        for path, digest in record["event_files_sha256"].items():
            if sha(_inside(path)) != digest:
                raise ValueError("v18 TensorBoard scalar evidence changed")
    parity, smokes = _read(directory / "evaluator_parity.json"), _read(directory / "final_model_smokes.json")
    if (parity.get("exact_reference_parity") is not True or smokes.get("passed") is not True
            or smokes.get("development_only") is not True
            or smokes.get("exact_initial_pairing") is not True
            or smokes.get("v16_reference_parity_sha256") != frozen["evaluator_parity_sha256"]
            or len(smokes.get("records", [])) != len(CONTROLLERS)):
        raise ValueError("v18 evaluator development proofs incomplete")
    old_path, new_path = _inside(parity["reference_path"]), _inside(parity["new_path"])
    old, new = _read(old_path), _read(new_path)
    if (old_path != (ROOT / "outputs/contact_slip_v16_20260923/final_smoke_control.json").resolve()
            or new_path != (ROOT / "outputs/terrain_style_v18_20260923/legacy_parity_v16_control.json").resolve()
            or parity.get("equal_fields") != list(PARITY_FIELDS)
            or (old.get("controller"), old.get("checkpoint_sha256"), old.get("phase"),
                old.get("scenario"), old.get("geometry_seed"), old.get("reset_seed"),
                old.get("num_envs")) != ("control", V16_CONTROL_SHA, "smoke", "mixed", 51, 24, 35)
            or new.get("controller") != "v16_control"
            or new.get("checkpoint_sha256") != V16_CONTROL_SHA
            or sha(old_path) != parity["reference_sha256"]
            or sha(new_path) != parity["new_sha256"]
            or any(old[key] != new[key] for key in PARITY_FIELDS)):
        raise ValueError("v18 evaluator v16 reference parity differs")
    initial, controllers = {}, set()
    for record in smokes["records"]:
        controller = record.get("controller")
        if (controller not in CONTROLLERS or controller in controllers
                or set(record) != {"path", "sha256", "controller", "checkpoint_sha256"}):
            raise ValueError("v18 smoke controller inventory differs")
        controllers.add(controller)
        path = _inside(record["path"])
        if sha(path) != record["sha256"]:
            raise ValueError("v18 final smoke changed")
        data = _read(path)
        model = frozen["models"][controller.removeprefix("history_")]
        if ((data.get("controller"), data.get("phase"), data.get("scenario"),
             data.get("geometry_seed"), data.get("reset_seed"), data.get("num_envs"),
             data.get("condition", {}).get("seconds")) !=
            (controller, "smoke", "mixed", 51, 24, 35, 16)
                or data.get("checkpoint") != model["checkpoint"]
                or data.get("checkpoint_sha256") != model["sha256"]
                or record["checkpoint_sha256"] != model["sha256"]
                or not HISTORY._timestamp(data["started_utc"]) <=
                HISTORY._timestamp(data["finished_utc"]) <= HISTORY._timestamp(smokes["created_utc"])):
            raise ValueError("v18 smoke model/condition/chronology differs")
        audit(data)
        _paired(initial, (51, 24), data)
    if (controllers != set(CONTROLLERS)
            or initial[(51, 24)] != {key: new[key] for key in
                                      ("initial_state_sha256", "initial_prefix_sha256", "initial_rng_sha256")}):
        raise ValueError("v18 smoke inventory or v16 parity initialization differs")
    return frozen


def summarize(directory=ART):
    directory = Path(directory)
    frozen = _verify_frozen(directory)
    inputs = evaluation_inputs(directory)
    if (inputs["experiment_freeze_sha256"] != sha(directory / "frozen.json")
            or inputs["terrain_cache_manifest_sha256"] != sha(directory / "terrain_cache.json")):
        raise ValueError("v18 holdout input linkage differs")
    if frozen["primary"] != {"files": 12, "episodes": 2100, "seconds": 16, "envs": 175}:
        raise ValueError("v18 primary inventory changed")
    if frozen["secondary"] != {"files": 12, "episodes": 120, "seconds": 64, "envs": 10}:
        raise ValueError("v18 secondary inventory changed")
    result = {"files": [], "provenance": {
        "experiment_freeze_sha256": sha(directory / "frozen.json"),
        "training_freeze_sha256": frozen["training_freeze_sha256"],
        "trained_models_sha256": frozen["trained_models_sha256"],
        "evaluation_inputs_sha256": sha(directory / "evaluation_inputs.json"),
        "terrain_cache_manifest_sha256": inputs["terrain_cache_manifest_sha256"],
        "preholdout_ledger": audit_preholdout_ledger(directory, inputs),
    }}
    for scenario, folder, envs, seconds in (("mixed", "evaluations", 175, 16),
                                             ("stones", "horizon", 10, 64)):
        expected = {f"{controller}__geometry{geometry}_reset{reset}.json"
                    for controller in CONTROLLERS for geometry, reset in HOLDOUTS}
        if {path.name for path in (directory / folder).glob("*.json")} != expected:
            raise ValueError("missing or unexpected v18 raw result")
        groups, initial, flats = defaultdict(list), {}, {}
        for controller in CONTROLLERS:
            model = frozen["models"][controller.removeprefix("history_")]
            for geometry, reset in HOLDOUTS:
                path = directory / folder / f"{controller}__geometry{geometry}_reset{reset}.json"
                data = _read(path)
                if ((data.get("controller"), data.get("geometry_seed"), data.get("reset_seed"),
                     data.get("scenario"), data.get("num_envs"), data.get("condition", {}).get("seconds"),
                     data.get("phase")) !=
                    (controller, geometry, reset, scenario, envs, seconds, "holdout")):
                    raise ValueError("v18 raw condition differs")
                if (data.get("checkpoint") != model["checkpoint"]
                        or data.get("checkpoint_sha256") != model["sha256"]
                        or data.get("experiment_freeze_sha256") != result["provenance"]["experiment_freeze_sha256"]
                        or data.get("evaluation_inputs_sha256") != result["provenance"]["evaluation_inputs_sha256"]
                        or data.get("terrain_cache_manifest_sha256") != inputs["terrain_cache_manifest_sha256"]):
                    raise ValueError("v18 raw source/model/input linkage differs")
                if not (HISTORY._timestamp(inputs["created_utc"]) < HISTORY._timestamp(data["started_utc"])
                        <= HISTORY._timestamp(data["finished_utc"])):
                    raise ValueError("v18 score chronology differs from pre-holdout freeze")
                rows = audit(data)
                counts = ({(family, level): 5 for family in FAMILIES for level in range(5)}
                          if scenario == "mixed" else {("stepping_stones", 4): 10})
                if Counter((row["family"], row["level"]) for row in rows) != counts:
                    raise ValueError("v18 family/level inventory differs")
                _paired(initial, (geometry, reset), data)
                if scenario == "mixed" and controller.startswith("history_"):
                    _flat_identity(flats, (geometry, reset), data, rows)
                groups[controller].extend(rows)
                result["files"].append({"path": str(path.relative_to(directory)),
                                        "sha256": sha(path), "episodes": len(rows)})
        section = {
            "groups": {name: aggregate_physical(rows) for name, rows in groups.items()},
            "family": {name: {family: aggregate_physical([row for row in rows if row["family"] == family])
                              for family in sorted({row["family"] for row in rows})}
                       for name, rows in groups.items()},
            "family_level": {name: {f"{family}/level{level}": aggregate_physical(
                [row for row in rows if (row["family"], row["level"]) == (family, level)])
                for family, level in sorted({(row["family"], row["level"]) for row in rows})}
                for name, rows in groups.items()},
        }
        section["actor_gated_vs_always"] = improvement(
            section["groups"]["gated"], section["groups"]["always"], primary=scenario == "mixed")
        section["hybrid_gated_vs_always"] = improvement(
            section["groups"]["history_gated"], section["groups"]["history_always"],
            hybrid=True, primary=scenario == "mixed")
        if scenario == "mixed":
            identity = {"verified": len(flats) == len(HOLDOUTS), "maps": len(flats),
                        "flat_episodes_per_controller": sum(len(value["indices"]) for value in flats.values()),
                        "fields": list(FLAT_IDENTITY_FIELDS)}
            section["hybrid_flat_raw_identity"] = identity
            check = section["hybrid_gated_vs_always"]["checks"]
            check["fixed_v5_per_environment_flat_raw_identity"] = identity["verified"]
            section["hybrid_gated_vs_always"]["passed"] = all(check.values())
        result[folder] = section
    if len(result["files"]) != 24 or sum(item["episodes"] for item in result["files"]) != 2220:
        raise ValueError("incomplete v18 24-file/2220-episode audit")
    result["interpretation"] = (
        "Single training seed and two fresh maps: exploratory, not universal superiority. "
        "Only paired causal change is a depth-derived binary mask on the frozen-v5 teacher loss. "
        "Actor still receives 91D features, not the full scan. v16 and original history are descriptive. "
        "64s hardest-stones evidence cannot override 16s mixed primary failure. "
        "Ideal simulator rays and visited-state contact telemetry do not demonstrate real-camera or robot safety."
    )
    return result


def markdown(result):
    lines = ["# v18 depth-gated teacher style: paired result", "", result["interpretation"], ""]
    for folder, title in (("evaluations", "Primary: 16s mixed"), ("horizon", "Secondary: 64s hardest stones")):
        section = result[folder]
        lines += [f"## {title}", "", "| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat speed |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for controller in CONTROLLERS:
            group = section["groups"][controller]
            speed = group["flat_mean_episode_speed"]
            lines.append(f"| {controller} | {group['one']}/{group['n']} | {group['six']}/{group['n']} | "
                         f"{group['falls']}/{group['n']} | {group['lane']}/{group['n']} | "
                         f"{group['world'] + group['flat_world']} | {group['flat_falls']}/{group['flat_n']} | "
                         f"{'—' if speed is None else f'{speed:.4f}'} |")
        for label in ("actor_gated_vs_always", "hybrid_gated_vs_always"):
            judgment = section[label]
            lines += ["", f"{label}: **{'PASS' if judgment['passed'] else 'FAIL'}**"]
            lines += [f"- {name}: {'PASS' if ok else 'FAIL'}" for name, ok in judgment["checks"].items()]
        lines.append("")
    lines += ["See summary.json and 24 raw JSON files for all family/level outcomes, paired initialization, "
              "routing and telemetry. No default-controller switch without both primary promotions.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=ART)
    args = parser.parse_args()
    result = summarize(args.directory)
    print(markdown(result))
