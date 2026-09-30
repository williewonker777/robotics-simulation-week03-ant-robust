"""Pinned source and paired-init helpers for the v18 teacher-style experiment."""

from __future__ import annotations

import json
from pathlib import Path

from .lp_study_v17 import V16_CONTROL, V16_CONTROL_SHA, validate_training_start
from .posture_study import ROOT, START, START_SHA, V5_SHA, save_json, sha, state_hashes, tensor_sha, utc, verify_hashes

ART = ROOT / "artifacts/terrain_demo/terrain_style_v18"
WORK = ROOT / "outputs/terrain_style_v18_20260923"
EXPERIMENT = "week03_ant_style_v18"
TRAIN_TASK = "Week03-Ant-Style-v18-Train-v0"
EVAL_TASK = "Week03-Ant-Contact-v16-Eval-v0"
ARMS = ("always", "gated")
CONTROLLERS = ("v16_control", *ARMS, "history_original", "history_always", "history_gated")
HOLDOUTS = ((105, 66), (106, 67))
TRAIN_GEOMETRY = 104
TRAIN_SEED = 50
ITERATIONS = 250
ENVS = 4096
STEPS_PER_ENV = 32
TRANSITIONS = ITERATIONS * ENVS * STEPS_PER_ENV
SCHEMA = "week03_ant_terrain_style_v18_v1"
TRAINING_SOURCES = (
    "src/week03_ant/style_math_v18.py", "src/week03_ant/style_policy_v18.py",
    "src/week03_ant/style_study_v18.py", "src/week03_ant/tasks/style_v18_cfg.py",
    "src/week03_ant/tasks/style_v18.py", "scripts/style_v18.py",
    "scripts/run_style_study.py", "tests/test_style_v18.py",
    "docs/experiment_plans/terrain_style_v18.md",
)


def init_path():
    return ROOT / f"logs/rsl_rl/{EXPERIMENT}/v18_init/model_0.pt"


def source_hashes():
    return {name: sha(ROOT / name) for name in TRAINING_SOURCES}


def legacy_hashes():
    """Verify the complete frozen v17 ancestry without modifying it."""
    from .lp_study_v17 import legacy_hashes as old_hashes

    frozen = json.loads((ROOT / "artifacts/terrain_demo/learning_progress_v17/frozen.json").read_text())
    old = old_hashes()
    if frozen["legacy"] != old:
        raise ValueError("frozen v17 ancestry differs")
    sources = {**old["sources"], **frozen["source_sha256"]}
    models = {**old["models"], **{v["checkpoint"]: v["sha256"] for v in frozen["models"].values()}}
    verify_hashes({**sources, **models})
    return {"sources": sources, "models": models}


def prepare_checkpoint(destination):
    """Create one shared policy state, preserving v16 except reset exploration."""
    import torch
    from .command_policy import CommandPriorActorCritic

    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if sha(V16_CONTROL) != V16_CONTROL_SHA:
        raise ValueError("v16 source checkpoint changed")
    source = torch.load(V16_CONTROL, map_location="cpu", weights_only=False)
    torch.manual_seed(TRAIN_SEED)
    policy = CommandPriorActorCritic(
        {"policy": torch.zeros(1, 91)}, {"policy": ["policy"], "critic": ["policy"]}, 8,
        input_mode="targets", prior_mode="anchored", command_mode="conditioned",
        require_config_match=True,
    )
    policy.load_state_dict(source["model_state_dict"])
    with torch.no_grad():
        policy.std.fill_(.2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.e-4)
    validate_training_start(policy.state_dict(), optimizer)
    checkpoint = {
        "model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "iter": 0, "infos": {"study": "terrain_style_v18", "seed": TRAIN_SEED,
                             "starting_sha256": V16_CONTROL_SHA, "source_iteration": source["iter"],
                             "reset_std": .2, "teacher_sha256": V5_SHA},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(checkpoint, stream)
    return {"checkpoint": str(destination.relative_to(ROOT)), "sha256": sha(destination),
            "state_sha256": state_hashes(policy.state_dict()), "source_sha256": V16_CONTROL_SHA,
            "optimizer_state_empty": True, "all_non_std_tensors_identical": True}


def normalize_training_parameters(log_directory, arm):
    """Read the actual dumped YAML; only style mode and output names may differ."""
    import yaml

    if arm not in ARMS:
        raise ValueError("invalid v18 arm")
    directory = Path(log_directory)
    files = {name: directory / "params" / f"{name}.yaml" for name in ("env", "agent")}
    values = {name: yaml.load(path.read_text(), Loader=yaml.BaseLoader) for name, path in files.items()}
    env, agent = values["env"], values["agent"]
    if (float(env["rewards"]["contact_slip"]["weight"]) != 0.
            or float(env["rewards"]["adaptive_posture"]["weight"]) != 1.):
        raise ValueError("v18 reward config changed")
    if "style_gate" not in env["observations"] or set(agent["obs_groups"]) != {"policy", "critic"}:
        raise ValueError("v18 separate observation group missing")
    if agent["obs_groups"] != {"policy": ["policy"], "critic": ["policy"]}:
        raise ValueError("v18 network received extra observation")
    policy = agent["policy"]
    if (policy["class_name"] != "StyleCommandPriorActorCritic" or policy["style_mode"] != arm
            or policy["command_mode"] != "conditioned" or policy["prior_mode"] != "anchored"
            or policy["input_mode"] != "targets"):
        raise ValueError("v18 style/policy config changed")
    if (agent["algorithm"]["class_name"] != "StylePriorPPO"
            or float(agent["algorithm"]["teacher_coef"]) != .02
            or agent["load_run"] != "v18_init" or agent["load_checkpoint"] != "model_0.pt"
            or agent["experiment_name"] != EXPERIMENT):
        raise ValueError("v18 PPO/start config changed")
    if "week03_ant.lp_curriculum_v17" in str(env["events"]):
        raise ValueError("v17 reset sampler leaked into v18")
    policy["style_mode"] = "<paired-style-treatment>"
    for field in ("log_dir", "io_descriptors_output_dir"):
        if Path(env.pop(field)).resolve() != directory.resolve():
            raise ValueError("v18 output path differs from dumped config")
    agent.pop("run_name")
    return {"normalized": values, "sha256": {name: sha(path) for name, path in files.items()}}


def model_key(controller):
    if controller not in CONTROLLERS:
        raise ValueError("unknown v18 controller")
    return controller.removeprefix("history_")


def command_mode(controller):
    return None if model_key(controller) == "original" else "conditioned"
