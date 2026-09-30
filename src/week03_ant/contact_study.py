"""Immutable initialization and provenance helpers for the v16 contact-slip study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from week03_ant.command_study import check_initial, validate_learning_log
from week03_ant.posture_study import (
    ROOT,
    START,
    START_SHA,
    V5_SHA,
    cache_snapshot,
    save_json,
    sha,
    state_hashes,
    tensor_sha,
    utc,
    validate_teacher,
    verify_hashes,
)

ART = ROOT / "artifacts/terrain_demo/contact_slip_v16"
WORK = ROOT / "outputs/contact_slip_v16_20260923"
V15 = ART.parent / "directional_stability_v15/runs/control/model_249.pt"
V15_SHA = "4aa169eba098e7ab69e82958485879db2f34910c6cb4a6d0367f33efef3bc3ec"
ARMS = ("control", "slip")
CONTROLLERS = ("v15_control", *ARMS, "history_original", "history_control", "history_slip")
HOLDOUTS = ((99, 62), (100, 63))
TASK = "Week03-Ant-Contact-v16-Eval-v0"
TRAIN_TASK = "Week03-Ant-Contact-v16-Train-v0"
EXPERIMENT = "week03_ant_contact_v16"
SCHEMA = "week03_ant_contact_slip_v16_v1"
TRAINING_GEOMETRY = 98
TRAINING_SEED = 48
ITERATIONS = 250
ENVS = 4096
STEPS_PER_ENV = 32
TRANSITIONS = ITERATIONS * ENVS * STEPS_PER_ENV


def init_path(arm: str | None = None) -> Path:
    if arm is not None and arm not in ARMS:
        raise ValueError("unknown v16 arm")
    return ROOT / f"logs/rsl_rl/{EXPERIMENT}/v16_init/model_0.pt"


def model_key(controller: str) -> str:
    if controller not in CONTROLLERS:
        raise ValueError("unknown v16 controller")
    return controller.removeprefix("history_")


def command_mode(controller: str) -> str | None:
    return None if model_key(controller) == "original" else "conditioned"


def evaluation_inputs(directory: Path = ART) -> dict:
    """Validate the pre-holdout command-ledger prefix before evaluation appends."""
    directory = Path(directory)
    record = json.loads((directory / "evaluation_inputs.json").read_text())
    if (record.get("experiment_freeze_sha256") != sha(directory / "frozen.json")
            or record.get("terrain_cache_manifest_sha256") != sha(directory / "terrain_cache.json")):
        raise ValueError("pinned evaluation input manifest changed")
    pin_path = directory / "preholdout_ledger.json"
    if record.get("preholdout_ledger_sha256") != sha(pin_path):
        raise ValueError("preholdout command ledger manifest changed")
    pin = json.loads(pin_path.read_text())
    path_name = Path(pin.get("path", ""))
    ledger = (ROOT / path_name).resolve()
    if (path_name.is_absolute() or not ledger.is_relative_to(ROOT.resolve())
            or ledger != (directory / "commands.jsonl").resolve() or not ledger.is_file()):
        raise ValueError("invalid command ledger path")
    if type(pin.get("bytes")) is not int or type(pin.get("lines")) is not int or pin["bytes"] <= 0 or pin["lines"] <= 0:
        raise ValueError("invalid command ledger prefix size")
    prefix = ledger.read_bytes()[:pin["bytes"]]
    if (len(prefix) != pin["bytes"] or not prefix.endswith(b"\n") or len(prefix.splitlines()) != pin["lines"]
            or hashlib.sha256(prefix).hexdigest() != pin.get("sha256")):
        raise ValueError("preholdout command ledger prefix changed")
    return record


def legacy_hashes() -> dict:
    """Pin the v15 graph of old source/model references before v16 starts."""
    from week03_ant.direction_study import legacy_hashes as v15_legacy

    frozen = json.loads((ROOT / "artifacts/terrain_demo/directional_stability_v15/frozen.json").read_text())
    old = v15_legacy()
    if frozen.get("legacy") != old:
        raise ValueError("v15 legacy inventory changed")
    sources = {**old["sources"], **frozen["source_sha256"]}
    models = {**old["models"], **{entry["checkpoint"]: entry["sha256"] for entry in frozen["models"].values()}}
    if models.get(str(V15.relative_to(ROOT))) != V15_SHA:
        raise ValueError("v15 control reference inventory differs")
    verify_hashes({**sources, **models})
    return {"sources": sources, "models": models}


def _source_state():
    import torch

    if sha(V15) != V15_SHA:
        raise ValueError("pinned v15 control source changed")
    return torch.load(V15, map_location="cpu", weights_only=False)


def validate_training_start(state, optimizer, arm: str | None = None) -> None:
    """Accept exactly v15-control's learned state with only scalar std reset."""
    import torch

    if arm is not None and arm not in ARMS:
        raise ValueError("unknown v16 arm")
    expected = _source_state()["model_state_dict"]
    expected["std"] = torch.full_like(expected["std"], 0.2)
    if state.keys() != expected.keys():
        raise ValueError("actual initial policy keys differ")
    for name, value in state.items():
        if value.dtype != expected[name].dtype or not torch.equal(value.detach().cpu(), expected[name]):
            raise ValueError(f"actual initial policy tensor differs: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite initial policy tensor: {name}")
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("expected a fresh Adam optimizer")
    saved = optimizer.state_dict()
    if saved["state"] or len(saved["param_groups"]) != 1:
        raise ValueError("initial optimizer must have one empty Adam parameter group")
    expected_options = {"lr": 1.e-4, "betas": (.9, .999), "eps": 1.e-8,
                        "weight_decay": 0., "amsgrad": False, "maximize": False}
    if any(saved["param_groups"][0][key] != value for key, value in expected_options.items()):
        raise ValueError("initial Adam settings differ")
    validate_teacher(state)


def prepare_checkpoint(destination: Path, seed: int = TRAINING_SEED) -> dict:
    """Copy the selected control once; leave learned command U and teacher intact."""
    import torch
    from week03_ant.command_policy import CommandPriorActorCritic

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    source = _source_state()
    torch.manual_seed(seed)
    policy = CommandPriorActorCritic(
        {"policy": torch.zeros(1, 91)}, {"policy": ["policy"], "critic": ["policy"]}, 8,
        command_mode="conditioned", prior_mode="anchored", input_mode="targets", require_config_match=True,
    )
    policy.load_state_dict(source["model_state_dict"])
    with torch.no_grad():
        policy.std.fill_(0.2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.e-4)
    state = policy.state_dict()
    validate_training_start(state, optimizer)
    checkpoint = {
        "model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(), "iter": 0,
        "infos": {"study": "contact_slip_v16", "command_mode": "conditioned",
                  "starting_sha256": V15_SHA, "source_iteration": source["iter"], "seed": seed,
                  "reset_std": .2, "reset_optimizer": "Adam 1e-4, empty state", "teacher_sha256": V5_SHA},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(checkpoint, stream)
    return {
        "checkpoint": str(destination.resolve().relative_to(ROOT)), "sha256": sha(destination),
        "state_sha256": state_hashes(state), "source_sha256": V15_SHA,
        "all_non_std_tensors_identical": True, "learned_command_weights_preserved": True,
        "optimizer_state_empty": True,
    }


def training_parameters(log_directory: Path, arm: str) -> dict:
    """Normalize only the predeclared contact-slip treatment and output names."""
    from week03_ant.posture_study import training_parameters as posture_parameters

    if arm not in ARMS:
        raise ValueError("unknown v16 arm")
    result = posture_parameters(log_directory, "adaptive")
    rewards = result["normalized"]["env"]["rewards"]
    if float(rewards["contact_slip"]["weight"]) != float(arm == "slip"):
        raise ValueError("saved contact-slip reward arm mismatch")
    rewards["contact_slip"]["weight"] = "<paired-contact-slip-treatment>"
    # This is a v16 control, not a second v15 directional treatment.
    if "directional_stability" in rewards and float(rewards["directional_stability"]["weight"]) != 0.:
        raise ValueError("v15 directional reward was inherited")
    policy = result["normalized"]["agent"]["policy"]
    if (policy.get("command_mode") != "conditioned" or policy.get("class_name") != "CommandPriorActorCritic"
            or policy.get("require_config_match") != "true"):
        raise ValueError("saved policy mode/class guard differs")
    agent = result["normalized"]["agent"]
    if agent.get("load_run") != "v16_init" or agent.get("load_checkpoint") != "model_0.pt":
        raise ValueError("saved shared starting checkpoint selection differs")
    return result


# The sensor/task files are owned by the companion implementation lane.  Keep them
# in the manifest so a training freeze cannot proceed before they exist and hash.
TRAINING_SOURCES = (
    "src/week03_ant/contact_math.py", "src/week03_ant/contact_study.py",
    "src/week03_ant/tasks/contact_v16.py", "src/week03_ant/tasks/contact_v16_cfg.py",
    "scripts/contact_v16.py", "scripts/run_contact_study.py", "tests/test_contact_math.py",
    "tests/test_contact_adapter.py", "tests/test_contact_harness.py", "tests/test_contact_training.py",
    "docs/experiment_plans/contact_slip_v16.md", "artifacts/terrain_demo/contact_slip_v16/research.md",
    "outputs/contact_slip_v16_20260923/PLAN.md",
)

SOURCE_FILES = (
    *TRAINING_SOURCES, "src/week03_ant/contact_telemetry.py", "scripts/evaluate_contact_v16.py",
    "scripts/run_contact_eval.py", "scripts/summarize_contact_v16.py", "tests/test_contact_telemetry.py",
    "tests/test_contact_eval.py", "tests/test_contact_summary.py",
)


def source_hashes(training: bool = False) -> dict[str, str]:
    return {name: sha(ROOT / name) for name in (TRAINING_SOURCES if training else SOURCE_FILES)}


__all__ = [
    "ARMS", "ART", "CONTROLLERS", "ENVS", "EXPERIMENT", "HOLDOUTS", "ITERATIONS", "ROOT", "SCHEMA",
    "START", "START_SHA", "STEPS_PER_ENV", "TASK", "TRAINING_GEOMETRY", "TRAINING_SEED", "TRAINING_SOURCES",
    "TRAIN_TASK", "TRANSITIONS", "V15", "V15_SHA", "V5_SHA", "WORK", "cache_snapshot", "check_initial",
    "command_mode", "evaluation_inputs", "init_path", "legacy_hashes", "model_key", "prepare_checkpoint", "save_json", "sha",
    "source_hashes", "state_hashes", "tensor_sha", "training_parameters", "utc", "validate_learning_log",
    "validate_teacher", "validate_training_start", "verify_hashes", "SOURCE_FILES",
]
