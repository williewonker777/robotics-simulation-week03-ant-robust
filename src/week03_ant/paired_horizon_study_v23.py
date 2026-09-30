"""Frozen-policy, same-episode two-window study contracts and provenance."""
from __future__ import annotations

import json
from pathlib import Path

from . import seed_study_v22 as prior
from .lp_study_v17 import evaluation_inputs as _inputs
from .posture_study import (
    ROOT as ROOT, START as START, V5_SHA as V5_SHA, save_json as save_json,
    sha as sha, tensor_sha as tensor_sha, utc as utc,
    validate_teacher as validate_teacher, verify_hashes as verify_hashes,
)

# Explicit public re-exports consumed by the evaluator rather than this module.
__all__ = ("START", "V5_SHA", "tensor_sha", "validate_teacher")

ART = ROOT / "artifacts/terrain_demo/paired_horizon_v23"
WORK = ROOT / "outputs/paired_horizon_v23_20260928"
PLAN = ROOT / "docs/experiment_plans/paired_horizon_v23.md"
SCHEMA = "week03_ant_paired_horizon_v23_window_v1"
BUNDLE_SCHEMA = "week03_ant_paired_horizon_v23_bundle_v1"
TASK = prior.TASK
CONTROLLERS = prior.CONTROLLERS
TRAIN_SEEDS = prior.TRAIN_SEEDS
RUNS = prior.RUNS
COMPARISONS = prior.COMPARISONS
HOLDOUTS = ((117, 77), (118, 78))
WINDOWS = (16, 64)
PHYSICAL_SECONDS = 64
ENVS = 175
PHYSICAL_ROLLOUTS = len(CONTROLLERS) * len(HOLDOUTS)
FIRST_EPISODES = PHYSICAL_ROLLOUTS * ENVS
WINDOW_OBSERVATIONS = FIRST_EPISODES * len(WINDOWS)
REFERENCE_CONTROLLERS = ("parent", "history_parent")
EXPECTED_GPU_COMMANDS = 29
SOURCE_FILES = (
    "src/week03_ant/paired_horizon_study_v23.py",
    "src/week03_ant/paired_horizon_v23.py",
    "scripts/evaluate_paired_horizon_v23.py",
    "tests/test_paired_horizon_v23.py",
    "scripts/run_paired_horizon_v23.py",
    "scripts/summarize_paired_horizon_v23.py",
    "tests/test_paired_horizon_study_v23.py",
    "docs/experiment_plans/paired_horizon_v23.md",
)

model_key = prior.model_key
training_seed = prior.training_seed
policy_mode = prior.policy_mode
command_mode = prior.command_mode


def read(path):
    return json.loads(Path(path).read_text())


def inside(name):
    path = Path(name)
    resolved = (ROOT / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("v23 provenance path escapes repository")
    return resolved


def models():
    result = prior.models()
    if set(result) != {"parent", *RUNS}:
        raise ValueError("v23 requires all three frozen seeds and their parent")
    for name, entry in result.items():
        if sha(inside(entry["checkpoint"])) != entry["sha256"]:
            raise ValueError("v23 model bytes changed")
        if name != "parent" and (entry.get("iteration") != 249
                                 or entry.get("transitions") != 32768000
                                 or entry.get("training_seed") != training_seed(name)):
            raise ValueError("v23 must not select another seed or checkpoint")
    return result


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def legacy_hashes():
    frozen = prior.verify_frozen()
    post = read(prior.ART / "static_validation.json")["post_experiment_verifier_sha256"]
    result = {
        "sources": {**frozen["legacy"]["sources"], **frozen["source_sha256"], **post},
        "models": {**frozen["legacy"]["models"],
                   **{entry["checkpoint"]: entry["sha256"] for entry in frozen["models"].values()}},
    }
    verify_hashes({**result["sources"], **result["models"]})
    return result


def pin_references():
    """Pin all eight existing 16s development records before any new rollout."""
    prior.verify_frozen()
    development = read(prior.ART / "development.json")
    records = development["records"]
    if len(records) != 8 or {row["controller"] for row in records} != set(CONTROLLERS):
        raise ValueError("v22 development controller inventory changed")
    inventory = {str((prior.ART / name).relative_to(ROOT)): sha(prior.ART / name)
                 for name in ("frozen.json", "development.json", "trained_models.json",
                              "final_validation.json")}
    for row in records:
        path = inside(row["path"])
        if sha(path) != row["sha256"]:
            raise ValueError("historical development evidence changed")
        inventory[row["path"]] = row["sha256"]
    save_json(ART / "historical_references.json", dict(
        created_utc=utc(), records=records, sha256=inventory,
        v22_frozen_sha256=sha(prior.ART / "frozen.json"),
        disclosure="Existing v22 16s development records, not new scored episodes."))


def verify_references():
    result = read(ART / "historical_references.json")
    records = result["records"]
    if (len(records) != 8 or {row["controller"] for row in records} != set(CONTROLLERS)
            or result["v22_frozen_sha256"] != sha(prior.ART / "frozen.json")):
        raise ValueError("v23 historical reference inventory differs")
    verify_hashes(result["sha256"])
    for row in records:
        inside(row["path"])
        if result["sha256"].get(row["path"]) != row["sha256"]:
            raise ValueError("v23 reference linkage changed")
    if records != read(prior.ART / "development.json")["records"]:
        raise ValueError("v23 references no longer match frozen v22 development")
    return result


def validate_request(controller, geometry, seed, scenario, seconds, envs, phase,
                     *, instrumented=True):
    if (controller not in CONTROLLERS or scenario != "mixed"
            or type(seconds) is not int or seconds != PHYSICAL_SECONDS
            or type(geometry) is not int or type(seed) is not int
            or type(envs) is not int or type(instrumented) is not bool):
        raise ValueError("undeclared v23 physical/controller contract")
    if phase in ("prepare_development", "reference", "smoke"):
        valid = (geometry, seed, envs) == (51, 24, 35)
        if phase == "reference":
            valid &= controller in REFERENCE_CONTROLLERS and not instrumented
        else:
            valid &= instrumented
            if phase == "prepare_development":
                valid &= controller == "parent"
    elif phase in ("prepare", "holdout"):
        valid = (geometry, seed) in HOLDOUTS and envs == ENVS and instrumented
        if phase == "prepare":
            valid &= controller == "parent"
    else:
        valid = False
    if not valid:
        raise ValueError("undeclared v23 phase/population or disabled holdout prefix")


def frozen_budget():
    return dict(controllers=list(CONTROLLERS), holdouts=[list(pair) for pair in HOLDOUTS],
                windows=list(WINDOWS), physical_seconds=PHYSICAL_SECONDS,
                new_training_transitions=0, physical_rollouts=PHYSICAL_ROLLOUTS,
                first_episodes=FIRST_EPISODES, window_observations=WINDOW_OBSERVATIONS)


def verify_frozen():
    frozen = read(ART / "frozen.json")
    if (frozen["source_sha256"] != source_hashes() or frozen["legacy"] != legacy_hashes()
            or frozen["models"] != models()
            or any(frozen.get(key) != value for key, value in frozen_budget().items())
            or frozen["v22_frozen_sha256"] != sha(prior.ART / "frozen.json")):
        raise ValueError("v23 frozen source/model/population/budget changed")
    for key, filename in (("development_sha256", "development.json"),
                          ("historical_references_sha256", "historical_references.json")):
        if frozen[key] != sha(ART / filename):
            raise ValueError("v23 frozen development/reference proof changed")
    verify_references()
    development = read(ART / "development.json")
    if (development.get("passed") is not True or len(development["records"]) != 8
            or len(development["controls"]) != 2
            or {row["controller"] for row in development["records"]} != set(CONTROLLERS)
            or {row["controller"] for row in development["controls"]} != set(REFERENCE_CONTROLLERS)
            or development["source_sha256"] != frozen["source_sha256"]
            or development["models"] != frozen["models"]):
        raise ValueError("v23 incomplete development inventory")
    for record in (*development["records"], *development["controls"], development["preparation"]):
        if sha(inside(record["path"])) != record["sha256"]:
            raise ValueError("v23 development raw evidence changed")
    return frozen


def evaluation_inputs():
    return _inputs(ART)
