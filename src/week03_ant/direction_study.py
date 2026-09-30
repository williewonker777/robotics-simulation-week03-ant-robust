"""Opt-in v15 directional regularization with a shared learned 91D warm start."""

import hashlib
import json
from pathlib import Path

from week03_ant.posture_study import (
    ROOT, START, START_SHA, V5, V5_SHA, cache_snapshot, save_json, sha, state_hashes,
    tensor_sha, utc, validate_teacher, verify_hashes,
    training_parameters as posture_parameters,
)
from week03_ant.command_study import check_initial, validate_learning_log

ART = ROOT / "artifacts/terrain_demo/directional_stability_v15"
WORK = ROOT / "outputs/directional_stability_v15_20260923"
V14 = ROOT / "artifacts/terrain_demo/command_conditioning_v14/runs/conditioned/model_249.pt"
V14_SHA = "844e6a3b9c4bee914ce65faa08322f892ad91f6dc520b5bc5167aac5fd543b41"
ARMS = ("control", "stable")
CONTROLLERS = ("v14", *ARMS, "history_original", "history_control", "history_stable")
HOLDOUTS = ((96, 60), (97, 61))
TASK = "Week03-Ant-Command-v14-Eval-v0"
TRAIN_TASK = "Week03-Ant-Direction-v15-Train-v0"
SCHEMA = "week03_ant_directional_stability_v15_v1"
EXPERIMENT = "week03_ant_direction_v15"


def evaluation_inputs(directory=ART):
    from week03_ant.posture_study import evaluation_inputs as original
    record = original(directory)
    pin_path = Path(directory) / "preholdout_ledger.json"
    if record.get("preholdout_ledger_sha256") != sha(pin_path):
        raise ValueError("preholdout command ledger manifest changed")
    pin = json.loads(pin_path.read_text())
    name = Path(pin["path"])
    path = (ROOT / name).resolve()
    if (name.is_absolute() or not path.is_relative_to(ROOT.resolve())
            or path != (Path(directory) / "commands.jsonl").resolve() or not path.is_file()):
        raise ValueError("invalid command ledger path")
    if (type(pin.get("bytes")) is not int or pin["bytes"] <= 0
            or type(pin.get("lines")) is not int or pin["lines"] <= 0):
        raise ValueError("invalid command ledger prefix size")
    prefix = path.read_bytes()[:pin["bytes"]]
    if (len(prefix) != pin["bytes"] or not prefix.endswith(b"\n")
            or len(prefix.splitlines()) != pin["lines"]
            or hashlib.sha256(prefix).hexdigest() != pin["sha256"]):
        raise ValueError("preholdout command ledger prefix changed")
    return record


def legacy_hashes():
    from week03_ant.command_study import legacy_hashes as old_hashes

    frozen = json.loads((ROOT / "artifacts/terrain_demo/command_conditioning_v14/frozen.json").read_text())
    old = old_hashes()
    if frozen["legacy"] != old:
        raise ValueError("v14 legacy inventory changed")
    sources = {**old["sources"], **frozen["source_sha256"]}
    models = {**old["models"], **{v["checkpoint"]: v["sha256"] for v in frozen["models"].values()}}
    if len(sources) != 106 or len(models) != 8 or models.get(str(V14.relative_to(ROOT))) != V14_SHA:
        raise ValueError("unexpected v15 reference inventory")
    verify_hashes({**sources, **models})
    return {"sources": sources, "models": models}


def model_key(controller):
    if controller not in CONTROLLERS:
        raise ValueError("unknown v15 controller")
    return controller.removeprefix("history_")


def command_mode(controller):
    return None if model_key(controller) == "original" else "conditioned"


def init_path(arm=None):
    if arm is not None and arm not in ARMS:
        raise ValueError("unknown direction arm")
    return ROOT / f"logs/rsl_rl/{EXPERIMENT}/v15_init/model_0.pt"


def validate_training_start(state, optimizer, arm=None):
    """Every learned tensor, including command weights, must survive the load."""
    import torch

    if arm is not None and arm not in ARMS:
        raise ValueError("unknown direction arm")
    if sha(V14) != V14_SHA:
        raise ValueError("pinned v14 initial source changed")
    expected = torch.load(V14, map_location="cpu", weights_only=False)["model_state_dict"]
    expected["std"] = torch.full_like(expected["std"], .2)
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
    options = {"lr": 1.e-4, "betas": (.9, .999), "eps": 1.e-8,
               "weight_decay": 0., "amsgrad": False, "maximize": False}
    if any(saved["param_groups"][0][key] != value for key, value in options.items()):
        raise ValueError("initial Adam settings differ")


def prepare_checkpoint(destination, seed=47):
    import torch
    from week03_ant.command_policy import CommandPriorActorCritic

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if sha(V14) != V14_SHA:
        raise ValueError("v14 starting checkpoint changed")
    torch.manual_seed(seed)
    original = torch.load(V14, map_location="cpu", weights_only=False)
    policy = CommandPriorActorCritic({"policy": torch.zeros(1, 91)},
        {"policy": ["policy"], "critic": ["policy"]}, 8, command_mode="conditioned",
        prior_mode="anchored", input_mode="targets", require_config_match=True)
    policy.load_state_dict(original["model_state_dict"])
    with torch.no_grad():
        policy.std.fill_(.2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.e-4)
    state = policy.state_dict()
    validate_training_start(state, optimizer)
    validate_teacher(state)
    checkpoint = {"model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(), "iter": 0,
        "infos": {"study": "directional_stability_v15", "command_mode": "conditioned",
                  "starting_sha256": V14_SHA, "source_iteration": original["iter"], "seed": seed,
                  "reset_std": .2, "reset_optimizer": "Adam 1e-4, empty state", "teacher_sha256": V5_SHA}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(checkpoint, stream)
    return {"checkpoint": str(destination.resolve().relative_to(ROOT)), "sha256": sha(destination),
            "state_sha256": state_hashes(state), "source_sha256": V14_SHA,
            "all_non_std_tensors_identical": True, "learned_command_weights_preserved": True,
            "optimizer_state_empty": True}


def training_parameters(log_directory, arm):
    """Normalize only the new reward weight and output names, not policy access."""
    if arm not in ARMS:
        raise ValueError("unknown direction arm")
    result = posture_parameters(log_directory, "adaptive")
    rewards = result["normalized"]["env"]["rewards"]
    # The inherited posture term is fixed at one, not a v15 treatment dimension.
    rewards["adaptive_posture"]["weight"] = "1.0"
    if float(rewards["directional_stability"]["weight"]) != float(arm == "stable"):
        raise ValueError("saved direction reward arm mismatch")
    rewards["directional_stability"]["weight"] = "<paired-direction-treatment>"
    agent = result["normalized"]["agent"]
    policy = agent["policy"]
    if (policy.get("command_mode") != "conditioned" or policy.get("class_name") != "CommandPriorActorCritic"
            or policy.get("require_config_match") != "true"):
        raise ValueError("saved policy mode/class guard differs")
    if agent.get("load_run") != "v15_init" or agent.get("load_checkpoint") != "model_0.pt":
        raise ValueError("saved shared starting checkpoint selection differs")
    return result


TRAINING_SOURCES = (
    "src/week03_ant/direction_math.py", "src/week03_ant/direction_study.py",
    "src/week03_ant/tasks/direction_v15.py", "src/week03_ant/tasks/direction_v15_cfg.py",
    "scripts/direction_v15.py", "scripts/run_direction_study.py",
    "tests/test_direction_math.py", "tests/test_direction_adapter.py",
    "tests/test_direction_training.py", "tests/test_direction_harness.py",
    "docs/experiment_plans/directional_stability_v15.md",
    "outputs/directional_stability_v15_20260923/PLAN.md",
)
SOURCE_FILES = (*TRAINING_SOURCES, "src/week03_ant/direction_telemetry.py",
                "scripts/evaluate_direction_v15.py", "scripts/summarize_direction_v15.py",
                "tests/test_direction_telemetry.py", "tests/test_direction_summary.py")


def source_hashes(training=False):
    return {name: sha(ROOT / name) for name in (TRAINING_SOURCES if training else SOURCE_FILES)}
