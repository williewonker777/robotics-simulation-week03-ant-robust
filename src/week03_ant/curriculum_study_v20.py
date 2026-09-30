"""Pinned inputs for one paired contact-cost curriculum experiment."""

from __future__ import annotations

import json

from .command_study import check_initial as check_initial
from .lp_study_v17 import (
    V16_CONTROL as PARENT, V16_CONTROL_SHA as PARENT_SHA,
    evaluation_inputs as _inputs, validate_training_start as _validate_start,
)
from .posture_study import (
    ROOT as ROOT, START as START, V5_SHA as V5_SHA, save_json as save_json,
    sha as sha, state_hashes as state_hashes, tensor_sha as tensor_sha,
    utc as utc, validate_teacher as validate_teacher, verify_hashes as verify_hashes,
)

__all__ = ["ROOT", "START", "V5_SHA", "save_json", "sha", "state_hashes", "tensor_sha",
           "utc", "validate_teacher", "verify_hashes", "check_initial", "PARENT", "PARENT_SHA"]

ART = ROOT / "artifacts/terrain_demo/contact_curriculum_v20"
WORK = ROOT / "outputs/contact_curriculum_v20_20260928"
PLAN = ROOT / "docs/experiment_plans/contact_curriculum_v20.md"
EXPERIMENT = "week03_ant_contact_curriculum_v20"
TRAIN_TASK = "Week03-Ant-Contact-Curriculum-v20-Train-v0"
TASK = "Week03-Ant-Contact-v16-Eval-v0"
ARMS = ("immediate", "ramped")
CONTROLLERS = ("parent", *ARMS, "history_parent", "history_immediate", "history_ramped")
HOLDOUTS = ((111, 71), (112, 72))
COMPARISONS = (("ramped", "immediate"), ("history_ramped", "history_immediate"))
PARENT_COMPARISONS = (("immediate", "parent"), ("ramped", "parent"),
                      ("history_immediate", "history_parent"), ("history_ramped", "history_parent"))
TRAIN_SEED, TRAIN_GEOMETRY = 51, 110
ENVS, ITERATIONS, STEPS_PER_ENV, RAMP_STEPS = 4096, 250, 32, 4000
TRANSITIONS = ENVS * ITERATIONS * STEPS_PER_ENV
SCHEMA = "week03_ant_contact_curriculum_v20_v1"
TRAINING_SOURCES = (
    "src/week03_ant/contact_curriculum_v20.py", "src/week03_ant/curriculum_study_v20.py",
    "src/week03_ant/tasks/contact_curriculum_v20.py",
    "src/week03_ant/tasks/contact_curriculum_v20_cfg.py",
    "scripts/contact_curriculum_v20.py", "scripts/run_curriculum_training_v20.py",
    "tests/test_contact_curriculum_v20.py", "tests/test_curriculum_training_v20.py",
    "tests/test_curriculum_harness_v20.py", "docs/experiment_plans/contact_curriculum_v20.md",
)
EVALUATION_SOURCES = (
    "scripts/evaluate_curriculum_v20.py", "scripts/run_curriculum_eval_v20.py",
    "scripts/summarize_curriculum_v20.py", "tests/test_curriculum_eval_v20.py",
)


def source_hashes(training=False):
    paths = TRAINING_SOURCES if training else (*TRAINING_SOURCES, *EVALUATION_SOURCES)
    return {name: sha(ROOT / name) for name in paths}


def legacy_hashes():
    from .rebaseline_study_v19 import verify_frozen as verify_v19

    prior = verify_v19()
    return {"sources": {**prior["legacy"]["sources"], **prior["source_sha256"]},
            "models": {**prior["legacy"]["models"],
                       **{v["checkpoint"]: v["sha256"] for v in prior["models"].values()}}}


def init_path():
    return ROOT / f"logs/rsl_rl/{EXPERIMENT}/v20_init/model_0.pt"


def validate_training_start(state, optimizer, arm=None):
    if arm is not None and arm not in ARMS:
        raise ValueError("unknown v20 arm")
    _validate_start(state, optimizer)


def prepare_checkpoint(destination):
    import torch
    from pathlib import Path
    from .command_policy import CommandPriorActorCritic

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    verify_hashes({str(PARENT.relative_to(ROOT)): PARENT_SHA})
    source = torch.load(PARENT, map_location="cpu", weights_only=False)
    torch.manual_seed(TRAIN_SEED)
    policy = CommandPriorActorCritic(
        {"policy": torch.zeros(1, 91)}, {"policy": ["policy"], "critic": ["policy"]}, 8,
        command_mode="conditioned", prior_mode="anchored", input_mode="targets",
        require_config_match=True,
    )
    policy.load_state_dict(source["model_state_dict"])
    with torch.no_grad():
        policy.std.fill_(0.2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.e-4)
    validate_training_start(policy.state_dict(), optimizer)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save({"model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "iter": 0, "infos": {"study": "contact_curriculum_v20", "seed": TRAIN_SEED,
                                         "starting_sha256": PARENT_SHA, "teacher_sha256": V5_SHA,
                                         "reset_std": .2, "source_iteration": source["iter"]}}, stream)
    return {"checkpoint": str(destination.relative_to(ROOT)), "sha256": sha(destination),
            "source_sha256": PARENT_SHA, "state_sha256": state_hashes(policy.state_dict()),
            "all_non_std_tensors_identical": True, "optimizer_state_empty": True}


def training_parameters(log_directory, arm, ramp_steps=RAMP_STEPS):
    from .posture_study import training_parameters as posture_parameters

    if arm not in ARMS:
        raise ValueError("unknown v20 arm")
    result = posture_parameters(log_directory, "adaptive")
    env, agent = (result["normalized"][name] for name in ("env", "agent"))
    term = env["rewards"]["contact_slip"]
    if (float(term["weight"]) != 1.0 or term["params"]["mode"] != arm
            or int(term["params"]["ramp_steps"]) != ramp_steps
            or set(term["params"]) != {"mode", "ramp_steps"}):
        raise ValueError("v20 saved contact schedule/weight differs")
    term["params"]["mode"] = "<paired-coefficient-trajectory>"
    policy = agent["policy"]
    if (policy["command_mode"] != "conditioned" or policy["class_name"] != "CommandPriorActorCritic"
            or policy["prior_mode"] != "anchored" or policy["input_mode"] != "targets"
            or policy["require_config_match"] != "true"
            or agent["algorithm"]["class_name"] != "PriorPPO"
            or float(agent["algorithm"]["teacher_coef"]) != .02
            or agent["load_run"] != "v20_init" or agent["load_checkpoint"] != "model_0.pt"
            or agent["experiment_name"] != EXPERIMENT):
        raise ValueError("v20 saved network/prior/start differs")
    if "week03_ant.lp_curriculum_v17" in str(env["events"]):
        raise ValueError("unplanned reset curriculum")
    return result


def model_key(controller):
    if controller not in CONTROLLERS:
        raise ValueError("unknown v20 controller")
    return controller.removeprefix("history_")


def command_mode(controller):
    model_key(controller)
    return "conditioned"


def policy_mode(controller):
    model_key(controller)
    return "hybrid" if controller.startswith("history_") else "v10"


def models():
    parent = {"checkpoint": str(PARENT.relative_to(ROOT)), "sha256": PARENT_SHA}
    path = ART / "trained_models.json"
    return {"parent": parent, **(json.loads(path.read_text())["models"] if path.exists() else {})}


def evaluation_inputs(directory=ART):
    return _inputs(directory)


def validate_request(controller, geometry, reset, scenario, seconds, envs, phase):
    model_key(controller)
    if scenario not in ("mixed", "stones") or phase not in ("prepare_development", "prepare", "smoke", "holdout"):
        raise ValueError("invalid v20 evaluation phase/scenario")
    if phase in ("prepare_development", "smoke"):
        if (geometry, reset, scenario, seconds, envs) != (51, 24, "mixed", 16, 35):
            raise ValueError("development must use51/24/35env")
        if phase == "prepare_development" and controller != "parent":
            raise ValueError("prepare with frozen parent only")
    elif ((geometry, reset) not in HOLDOUTS
          or (seconds, envs) != ((16, 175) if scenario == "mixed" else (64, 10))
          or (phase == "prepare" and (controller != "parent" or scenario != "mixed"))):
        raise ValueError("changed v20 evaluation matrix")


def verify_frozen():
    frozen = json.loads((ART / "frozen.json").read_text())
    if (frozen["source_sha256"] != source_hashes() or frozen["legacy"] != legacy_hashes()
            or frozen["models"] != models() or frozen["controllers"] != list(CONTROLLERS)
            or frozen["holdouts"] != [list(pair) for pair in HOLDOUTS]
            or frozen["new_training_transitions"] != 2 * TRANSITIONS):
        raise ValueError("v20 evaluation freeze changed")
    for key, name in (("development_sha256", "development.json"),
                      ("training_freeze_sha256", "training_frozen.json"),
                      ("training_validation_sha256", "training_validation.json"),
                      ("trained_models_sha256", "trained_models.json")):
        if frozen[key] != sha(ART / name):
            raise ValueError(f"v20 frozen evidence changed: {name}")
    training = json.loads((ART / "training_frozen.json").read_text())
    trained = json.loads((ART / "trained_models.json").read_text())
    validated = json.loads((ART / "training_validation.json").read_text())
    if (training["source_sha256"] != source_hashes(training=True)
            or training["source_checkpoint_sha256"] != PARENT_SHA
            or trained["training_freeze_sha256"] != sha(ART / "training_frozen.json")
            or trained["training_validation_sha256"] != sha(ART / "training_validation.json")
            or trained["total_training_transitions"] != 2 * TRANSITIONS
            or set(trained["models"]) != set(ARMS) or set(validated["records"]) != set(ARMS)):
        raise ValueError("v20 training evidence linkage/budget changed")
    for arm in ARMS:
        model, record = trained["models"][arm], validated["records"][arm]
        if model["iteration"] != 249 or model["transitions"] != TRANSITIONS:
            raise ValueError("v20 trained checkpoint selection/budget changed")
        for key in ("initial", "schedule", "rollout"):
            item = record[key]
            verify_hashes({item["path"]: item["sha256"]})
        logs = record["learning_log"]
        verify_hashes({logs["text_log"]: logs["text_log_sha256"], **logs["event_files_sha256"]})
        if (model["initial_audit_sha256"] != record["initial"]["sha256"]
                or model["schedule_sha256"] != record["schedule"]["sha256"]):
            raise ValueError("v20 model/run audit links changed")
    for item in models().values():
        verify_hashes({item["checkpoint"]: item["sha256"]})
    development = json.loads((ART / "development.json").read_text())
    if development["source_sha256"] != frozen["source_sha256"]:
        raise ValueError("v20 development sources differ from scored sources")
    for record in development["records"]:
        verify_hashes({record["path"]: record["sha256"]})
    return frozen
