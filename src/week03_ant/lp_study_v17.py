"""Frozen inputs and provenance helpers for the v17 terrain-sampling study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from week03_ant.command_study import check_initial, validate_learning_log
from week03_ant.contact_study import START, START_SHA, V5_SHA
from week03_ant.posture_study import (
    ROOT, cache_snapshot, save_json, sha, state_hashes, tensor_sha, utc,
    validate_teacher, verify_hashes,
)

ART = ROOT / "artifacts/terrain_demo/learning_progress_v17"
WORK = ROOT / "outputs/learning_progress_v17_20260923"
V16_CONTROL = ROOT / "artifacts/terrain_demo/contact_slip_v16/runs/control/model_249.pt"
V16_CONTROL_SHA = "1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc"
ARMS = ("fixed", "lp")
CONTROLLERS = ("v16_control", *ARMS, "history_original", "history_fixed", "history_lp")
HOLDOUTS = ((102, 64), (103, 65))
TRAIN_TASK = "Week03-Ant-LP-v17-Train-v0"
EVAL_TASK = "Week03-Ant-Contact-v16-Eval-v0"
EXPERIMENT = "week03_ant_lp_v17"
SCHEMA = "week03_ant_lp_curriculum_v17_v1"
TRAINING_GEOMETRY = 101
TRAINING_SEED = 49
ITERATIONS = 250
ENVS = 4096
STEPS_PER_ENV = 32
TRANSITIONS = ITERATIONS * ENVS * STEPS_PER_ENV

TRAINING_SOURCES = (
    "src/week03_ant/lp_curriculum_v17.py",
    "src/week03_ant/lp_study_v17.py",
    "src/week03_ant/tasks/lp_v17_cfg.py",
    "src/week03_ant/tasks/lp_v17.py",
    "scripts/lp_v17.py",
    "scripts/run_lp_study.py",
    "tests/test_lp_curriculum_v17.py",
    "tests/test_lp_training_v17.py",
    "docs/experiment_plans/learning_progress_v17.md",
    "artifacts/terrain_demo/learning_progress_v17/research.md",
    "outputs/learning_progress_v17_20260923/PLAN.md",
)
SOURCE_FILES = (*TRAINING_SOURCES,
                "scripts/evaluate_lp_v17.py", "scripts/run_lp_eval.py",
                "scripts/summarize_lp_v17.py", "tests/test_lp_eval_v17.py",
                "tests/test_lp_summary_v17.py")


def init_path() -> Path:
    return ROOT / f"logs/rsl_rl/{EXPERIMENT}/v17_init/model_0.pt"


def model_key(controller: str) -> str:
    if controller not in CONTROLLERS:
        raise ValueError("unknown v17 controller")
    return controller.removeprefix("history_")


def command_mode(controller: str) -> str | None:
    return None if model_key(controller) == "original" else "conditioned"


def source_hashes(training: bool = False) -> dict[str, str]:
    return {name: sha(ROOT / name) for name in (TRAINING_SOURCES if training else SOURCE_FILES)}


def legacy_hashes() -> dict:
    """Pin the complete, already reviewed v16 source/model graph."""
    from week03_ant.contact_study import legacy_hashes as v16_legacy

    frozen = json.loads((ROOT / "artifacts/terrain_demo/contact_slip_v16/frozen.json").read_text())
    old = v16_legacy()
    if frozen.get("legacy") != old:
        raise ValueError("v16 legacy inventory changed")
    sources = {**old["sources"], **frozen["source_sha256"]}
    models = {**old["models"],
              **{entry["checkpoint"]: entry["sha256"] for entry in frozen["models"].values()}}
    if len(sources) != 143 or len(models) != 12:
        raise ValueError("unexpected v17 legacy source/model inventory")
    if models.get(str(V16_CONTROL.relative_to(ROOT))) != V16_CONTROL_SHA:
        raise ValueError("v16 control source does not match pinned checkpoint")
    verify_hashes({**sources, **models})
    return {"sources": sources, "models": models}


def _source_state():
    import torch

    if sha(V16_CONTROL) != V16_CONTROL_SHA:
        raise ValueError("pinned v16 control checkpoint changed")
    return torch.load(V16_CONTROL, map_location="cpu", weights_only=False)


def validate_training_start(state, optimizer, arm: str | None = None) -> None:
    """Only scalar exploration std and empty Adam/iteration may differ from v16."""
    import torch

    if arm is not None and arm not in ARMS:
        raise ValueError("unknown v17 arm")
    expected = _source_state()["model_state_dict"]
    expected["std"] = torch.full_like(expected["std"], 0.2)
    if state.keys() != expected.keys():
        raise ValueError("actual initial policy keys differ")
    for name, value in state.items():
        if value.dtype != expected[name].dtype or not torch.equal(value.detach().cpu(), expected[name]):
            raise ValueError(f"initial v17 policy tensor differs: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite initial v17 policy tensor: {name}")
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("expected fresh Adam optimizer")
    saved = optimizer.state_dict()
    if saved["state"] or len(saved["param_groups"]) != 1:
        raise ValueError("initial Adam state must be empty")
    expected_options = {"lr": 1.e-4, "betas": (.9, .999), "eps": 1.e-8,
                        "weight_decay": 0., "amsgrad": False, "maximize": False}
    if any(saved["param_groups"][0][key] != value for key, value in expected_options.items()):
        raise ValueError("initial Adam settings differ")
    validate_teacher(state)


def prepare_checkpoint(destination: Path, seed: int = TRAINING_SEED) -> dict:
    """Make one shared paired initialization from v16 control, with no overwrite."""
    import torch
    from week03_ant.command_policy import CommandPriorActorCritic

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    source = _source_state()
    torch.manual_seed(seed)
    policy = CommandPriorActorCritic(
        {"policy": torch.zeros(1, 91)}, {"policy": ["policy"], "critic": ["policy"]}, 8,
        command_mode="conditioned", prior_mode="anchored", input_mode="targets",
        require_config_match=True,
    )
    policy.load_state_dict(source["model_state_dict"])
    with torch.no_grad():
        policy.std.fill_(0.2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.e-4)
    state = policy.state_dict()
    validate_training_start(state, optimizer)
    checkpoint = {
        "model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(), "iter": 0,
        "infos": {"study": "learning_progress_v17", "command_mode": "conditioned",
                  "starting_sha256": V16_CONTROL_SHA, "source_iteration": source["iter"],
                  "seed": seed, "reset_std": 0.2,
                  "reset_optimizer": "Adam 1e-4, empty state", "teacher_sha256": V5_SHA},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(checkpoint, stream)
    return {"checkpoint": str(destination.resolve().relative_to(ROOT)), "sha256": sha(destination),
            "state_sha256": state_hashes(state), "source_sha256": V16_CONTROL_SHA,
            "all_non_std_tensors_identical": True, "learned_command_weights_preserved": True,
            "optimizer_state_empty": True}


def training_parameters(log_directory: Path, arm: str, stage_size: int = 1024) -> dict:
    """Normalize only the reset-sampling treatment and output path/name."""
    from week03_ant.posture_study import training_parameters as posture_parameters

    if arm not in ARMS:
        raise ValueError("unknown v17 arm")
    result = posture_parameters(log_directory, "adaptive")
    normalized = result["normalized"]
    env, agent = normalized["env"], normalized["agent"]
    if float(env["rewards"]["contact_slip"]["weight"]) != 0.:
        raise ValueError("v17 must not inherit v16 slip treatment")
    reset = env["events"]["reset_base"]
    if reset["func"] != "week03_ant.lp_curriculum_v17:lp_reset_root_state":
        raise ValueError("v17 reset function changed")
    if reset["params"].get("mode") != arm:
        raise ValueError("v17 training sampling arm changed")
    if (isinstance(stage_size, bool) or not isinstance(stage_size, int) or stage_size <= 0
            or int(reset["params"].get("stage_size", "0")) != stage_size):
        raise ValueError("v17 training stage size changed")
    reset["params"]["mode"] = "<paired-lp-sampling-treatment>"
    policy = agent["policy"]
    if (policy.get("command_mode") != "conditioned"
            or policy.get("class_name") != "CommandPriorActorCritic"
            or policy.get("require_config_match") != "true"):
        raise ValueError("v17 policy mode/class guard differs")
    if (agent.get("load_run") != "v17_init" or agent.get("load_checkpoint") != "model_0.pt"
            or agent.get("experiment_name") != EXPERIMENT):
        raise ValueError("v17 shared initial checkpoint or experiment differs")
    return result


def evaluation_inputs(directory: Path = ART) -> dict:
    """Bind the pre-holdout command-ledger prefix and terrain cache manifest."""
    directory = Path(directory)
    record = json.loads((directory / "evaluation_inputs.json").read_text())
    if (record.get("experiment_freeze_sha256") != sha(directory / "frozen.json")
            or record.get("terrain_cache_manifest_sha256") != sha(directory / "terrain_cache.json")
            or record.get("preholdout_ledger_sha256") != sha(directory / "preholdout_ledger.json")):
        raise ValueError("pinned v17 evaluation inputs changed")
    pin = json.loads((directory / "preholdout_ledger.json").read_text())
    path_name = Path(pin.get("path", ""))
    ledger = (ROOT / path_name).resolve()
    if (path_name.is_absolute() or not ledger.is_relative_to(ROOT.resolve())
            or ledger != (directory / "commands.jsonl").resolve() or not ledger.is_file()):
        raise ValueError("invalid v17 command ledger path")
    if (type(pin.get("bytes")) is not int or type(pin.get("lines")) is not int
            or pin["bytes"] <= 0 or pin["lines"] <= 0):
        raise ValueError("invalid v17 command ledger prefix size")
    prefix = ledger.read_bytes()[:pin["bytes"]]
    if (len(prefix) != pin["bytes"] or not prefix.endswith(b"\n")
            or len(prefix.splitlines()) != pin["lines"]
            or hashlib.sha256(prefix).hexdigest() != pin.get("sha256")):
        raise ValueError("pre-holdout command ledger prefix changed")
    return record
