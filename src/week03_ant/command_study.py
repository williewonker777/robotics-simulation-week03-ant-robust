"""Opt-in v14 command-access study: immutable references and paired initialization."""

import json
from pathlib import Path

from week03_ant.posture_study import (
    ROOT, START, START_SHA, V5, V5_SHA, cache_snapshot,
    save_json, sha, state_hashes, tensor_sha, utc, validate_teacher, verify_hashes,
    training_parameters as posture_parameters,
)

ART = ROOT / "artifacts/terrain_demo/command_conditioning_v14"
WORK = ROOT / "outputs/command_conditioning_v14_20260922"
V13 = ROOT / "artifacts/terrain_demo/adaptive_posture_v13/runs/adaptive/model_249.pt"
V13_SHA = "0c803ffdf4fceddf9b86d05b318b7f8e0fc80c8316ed5cdc2cd042850e892031"
ARMS = ("masked", "conditioned")
CONTROLLERS = ("v13", *ARMS, "history_original", "history_masked", "history_conditioned")
HOLDOUTS = ((86, 58), (87, 59))
TASK = "Week03-Ant-Command-v14-Eval-v0"
TRAIN_TASK = "Week03-Ant-Command-v14-Train-v0"
SCHEMA = "week03_ant_command_conditioning_v14_v1"
EXPERIMENT = "week03_ant_command_v14"


def evaluation_inputs(directory=ART):
    from week03_ant.posture_study import evaluation_inputs as original

    return original(directory)


def legacy_hashes():
    from week03_ant.posture_study import legacy_hashes as old_hashes

    frozen = json.loads((ROOT / "artifacts/terrain_demo/adaptive_posture_v13/frozen.json").read_text())
    old = old_hashes()
    if frozen["legacy"] != old:
        raise ValueError("v13 legacy inventory changed")
    sources = {**old["sources"], **frozen["source_sha256"]}
    models = {**old["models"], **{v["checkpoint"]: v["sha256"] for v in frozen["models"].values()}}
    if len(sources) != 89 or len(models) != 6 or models.get(str(V13.relative_to(ROOT))) != V13_SHA:
        raise ValueError("unexpected v14 reference inventory")
    verify_hashes({**sources, **models})
    return {"sources": sources, "models": models}


def model_key(controller):
    if controller not in CONTROLLERS:
        raise ValueError("unknown v14 controller")
    return controller.removeprefix("history_")


def command_mode(controller):
    key = model_key(controller)
    return key if key in ARMS else None


def init_path(arm):
    if arm not in ARMS:
        raise ValueError("unknown command mode")
    return ROOT / f"logs/rsl_rl/{EXPERIMENT}/v14_init_{arm}/model_0.pt"


def validate_training_start(state, optimizer, arm):
    """Check the actual learner against every original88D tensor and fresh Adam."""
    import torch
    from week03_ant.posture_study import validate_training_start as validate_base

    if arm not in ARMS:
        raise ValueError("unknown command mode")
    additions = {"actor.0.command_weight", "critic.0.command_weight", "command_mode_code", "command_schema_code"}
    validate_base({k: v for k, v in state.items() if k not in additions}, optimizer)
    for key in ("actor.0.command_weight", "critic.0.command_weight"):
        value = state.get(key)
        if value is None or value.shape != (400, 3) or value.dtype != torch.float32 or not value.eq(0).all():
            raise ValueError("initial command weights must be registered float32 zeros")
    for key, expected in (("command_mode_code", ARMS.index(arm)), ("command_schema_code", 1)):
        value = state.get(key)
        if value is None or value.shape != () or value.dtype != torch.int64 or value.item() != expected:
            raise ValueError(f"initial command metadata differs: {key}")


def prepare_checkpoint(destination, arm, seed=46):
    import torch
    from week03_ant.command_policy import CommandPriorActorCritic, warmstart_command_from_prior

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if sha(START) != START_SHA:
        raise ValueError("v10 starting checkpoint changed")
    torch.manual_seed(seed)
    original = torch.load(START, map_location="cpu", weights_only=False)
    policy = CommandPriorActorCritic({"policy": torch.zeros(1, 91)},
        {"policy": ["policy"], "critic": ["policy"]}, 8, command_mode=arm,
        prior_mode="anchored", input_mode="targets", require_config_match=True)
    warmstart_command_from_prior(policy, original["model_state_dict"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.e-4)
    state = policy.state_dict()
    validate_training_start(state, optimizer, arm)
    validate_teacher(state)
    checkpoint = {"model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(), "iter": 0,
        "infos": {"study": "command_conditioning_v14", "command_mode": arm, "starting_sha256": START_SHA,
                  "source_iteration": original["iter"], "seed": seed, "reset_std": .2,
                  "reset_optimizer": "Adam 1e-4, empty state", "teacher_sha256": V5_SHA}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(checkpoint, stream)
    return {"checkpoint": str(destination.resolve().relative_to(ROOT)), "sha256": sha(destination),
            "state_sha256": state_hashes(state), "source_sha256": START_SHA,
            "old_non_std_tensors_identical": True, "optimizer_state_empty": True}


def training_parameters(log_directory, arm):
    """Identical reward; normalize only explicit access mode and output names."""
    if arm not in ARMS:
        raise ValueError("unknown command mode")
    result = posture_parameters(log_directory, "adaptive")
    policy = result["normalized"]["agent"]["policy"]
    if (policy.get("command_mode") != arm or policy.get("class_name") != "CommandPriorActorCritic"
            or policy.get("require_config_match") != "true"):
        raise ValueError("saved policy mode/class guard differs")
    policy["command_mode"] = "<paired-command-access>"
    agent = result["normalized"]["agent"]
    if agent.get("load_run") != f"v14_init_{arm}" or agent.get("load_checkpoint") != "model_0.pt":
        raise ValueError("saved starting checkpoint selection differs")
    agent["load_run"] = "<paired-mode-initial-checkpoint>"
    return result


def check_initial(data, expected):
    for key in ("initial_state_sha256", "initial_prefix_sha256", "policy_state_sha256", "rng_sha256",
                "num_envs", "dt", "normalized_parameters"):
        if data[key] != expected[key]:
            raise ValueError(f"paired training initialization differs: {key}")


def validate_learning_log(log_directory, text_log, iterations=250, transitions=32768000):
    """Independently check executed iterations, steps and every TensorBoard scalar."""
    import math
    import re
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    text = Path(text_log).read_text()
    pairs = [(int(a), int(b)) for a, b in re.findall(r"Learning iteration (\d+)/(\d+)", text)]
    totals = [int(v) for v in re.findall(r"Total timesteps:\s*(\d+)", text)]
    if pairs != [(i, iterations) for i in range(iterations)] or len(totals) != iterations or totals[-1] != transitions:
        raise ValueError("executed training log budget differs")
    if totals != [(i + 1) * (transitions // iterations) for i in range(iterations)]:
        raise ValueError("training log transition sequence differs")
    accumulator = EventAccumulator(str(log_directory), size_guidance={"scalars": 0}).Reload()
    tags = accumulator.Tags()["scalars"]
    required = {"Episode_Reward/adaptive_posture", "Loss/value_function", "Loss/surrogate", "Loss/prior_loss",
                "Train/mean_reward", "Policy/mean_noise_std"}
    if not required <= set(tags):
        raise ValueError("missing required training scalar logs")
    counts = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        if len(events) != iterations or not all(math.isfinite(e.value) for e in events):
            raise ValueError(f"missing/nonfinite training scalar: {tag}")
        if tag in required and [e.step for e in events] != list(range(iterations)):
            raise ValueError("training scalar iteration sequence differs")
        counts[tag] = len(events)
    events = sorted(Path(log_directory).glob("events.out.tfevents.*"))
    return {"iterations_logged": iterations, "transitions_logged": transitions, "all_scalars_finite": True,
            "scalar_counts": counts, "text_log": str(Path(text_log).relative_to(ROOT)), "text_log_sha256": sha(text_log),
            "event_files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in events}}


TRAINING_SOURCES = (
    "src/week03_ant/command_policy.py", "src/week03_ant/command_math.py", "src/week03_ant/command_study.py",
    "src/week03_ant/tasks/command_v14.py", "src/week03_ant/tasks/command_v14_cfg.py",
    "scripts/command_v14.py", "scripts/run_command_study.py", "tests/test_command_policy.py",
    "tests/test_command_math.py", "tests/test_command_adapter.py", "tests/test_command_training.py",
    "tests/test_command_harness.py", "docs/experiment_plans/command_conditioning_v14.md",
    "outputs/command_conditioning_v14_20260922/PLAN.md",
)
SOURCE_FILES = (*TRAINING_SOURCES, "scripts/evaluate_command_v14.py", "scripts/summarize_command_v14.py",
                "tests/test_command_summary.py")


def source_hashes(training=False):
    return {name: sha(ROOT / name) for name in (TRAINING_SOURCES if training else SOURCE_FILES)}
