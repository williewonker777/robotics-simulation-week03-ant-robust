"""Immutable three-seed no-cost continuation study and provenance contracts."""
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

ART = ROOT / "artifacts/terrain_demo/seed_continuation_v22"
WORK = ROOT / "outputs/seed_continuation_v22_20260928"
PLAN = ROOT / "docs/experiment_plans/seed_continuation_v22.md"
EXPERIMENT = "week03_ant_seed_continuation_v22"
TRAIN_TASK = "Week03-Ant-Seed-Continuation-v22-Train-v0"
TASK = "Week03-Ant-Contact-v16-Eval-v0"
TRAIN_SEEDS = (51, 52, 53)
RUNS = ("seed51", "seed52", "seed53")
TRAIN_SEED, TRAIN_GEOMETRY = 51, 110
ENVS, ITERATIONS, STEPS_PER_ENV, RAMP_STEPS = 4096, 250, 32, 4000
TRANSITIONS = ENVS * ITERATIONS * STEPS_PER_ENV
TOTAL_TRANSITIONS = len(RUNS) * TRANSITIONS
DEVELOPMENT_TRANSITIONS = len(RUNS) * (256 + 4096) * 2 * STEPS_PER_ENV
V21_ART = ROOT / "artifacts/terrain_demo/contact_continuation_v21"
INIT_SOURCE = ROOT / "logs/rsl_rl/week03_ant_contact_continuation_v21/v21_init/model_0.pt"
INIT_SHA = "10d8485d6d18f1dd6c01724ff69fc6a2ced3bbcd3d7b0db1285b7097cb5c21bd"
CONTROLLERS = ("parent", *RUNS, "history_parent", *("history_" + run for run in RUNS))
HOLDOUTS = ((115, 75), (116, 76))
COMPARISONS = (
    ("seed51", "parent"), ("seed52", "parent"), ("seed53", "parent"),
    ("history_seed51", "history_parent"), ("history_seed52", "history_parent"),
    ("history_seed53", "history_parent"), ("seed52", "seed51"), ("seed53", "seed51"),
    ("seed53", "seed52"), ("history_seed52", "history_seed51"),
    ("history_seed53", "history_seed51"), ("history_seed53", "history_seed52"),
)
SCHEMA = "week03_ant_seed_continuation_v22_v1"
TRAINING_SOURCES = (
    "src/week03_ant/seed_study_v22.py",
    "src/week03_ant/tasks/seed_continuation_v22.py",
    "src/week03_ant/tasks/seed_continuation_v22_cfg.py",
    "scripts/seed_continuation_v22.py", "scripts/run_seed_training_v22.py",
    "tests/test_seed_training_v22.py", "tests/test_seed_harness_v22.py",
    "docs/experiment_plans/seed_continuation_v22.md",
)
EVALUATION_SOURCES = (
    "scripts/evaluate_seed_v22.py", "scripts/run_seed_eval_v22.py",
    "scripts/summarize_seed_v22.py", "tests/test_seed_eval_v22.py",
)


def training_seed(run):
    if run not in RUNS:
        raise ValueError("unknown v22 trained run")
    return TRAIN_SEEDS[RUNS.index(run)]


def source_hashes(training=False):
    paths = TRAINING_SOURCES if training else (*TRAINING_SOURCES, *EVALUATION_SOURCES)
    return {name: sha(ROOT / name) for name in paths}


def legacy_hashes():
    from .continuation_study_v21 import verify_frozen as verify_v21

    prior = verify_v21()
    post = json.loads((V21_ART / "static_validation.json").read_text())["post_experiment_verifier_sha256"]
    result = {"sources": {**prior["legacy"]["sources"], **prior["source_sha256"], **post},
              "models": {**prior["legacy"]["models"],
                         **{item["checkpoint"]: item["sha256"] for item in prior["models"].values()}}}
    verify_hashes({**result["sources"], **result["models"]})
    return result


def init_path():
    return ROOT / f"logs/rsl_rl/{EXPERIMENT}/v22_init/model_0.pt"


def validate_training_start(state, optimizer, seed=None):
    if seed is not None and (type(seed) is not int or seed not in TRAIN_SEEDS):
        raise ValueError("undeclared v22 training seed")
    _validate_start(state, optimizer)


def prepare_checkpoint(destination):
    """Copy the exact v21 init, including historical metadata, without reserialization."""
    from pathlib import Path
    import torch
    from .command_policy import CommandPriorActorCritic

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if sha(INIT_SOURCE) != INIT_SHA:
        raise ValueError("v21 initialization changed")
    source = torch.load(INIT_SOURCE, map_location="cpu", weights_only=False)
    torch.manual_seed(TRAIN_SEED)
    policy = CommandPriorActorCritic(
        {"policy": torch.zeros(1, 91)}, {"policy": ["policy"], "critic": ["policy"]}, 8,
        command_mode="conditioned", prior_mode="anchored", input_mode="targets", require_config_match=True)
    policy.load_state_dict(source["model_state_dict"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.e-4)
    optimizer.load_state_dict(source["optimizer_state_dict"])
    validate_training_start(policy.state_dict(), optimizer)
    if source["iter"] != 0:
        raise ValueError("initial iteration differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(INIT_SOURCE.read_bytes())
    if sha(destination) != INIT_SHA:
        raise ValueError("initialization copy differs")
    return {"checkpoint": str(destination.relative_to(ROOT)), "sha256": INIT_SHA,
            "source_checkpoint": str(INIT_SOURCE.relative_to(ROOT)), "source_sha256": INIT_SHA,
            "parent_sha256": PARENT_SHA, "state_sha256": state_hashes(policy.state_dict()),
            "all_non_std_tensors_identical": True, "optimizer_state_empty": True,
            "historical_infos_preserved": True}


def training_parameters(log_directory, seed, ramp_steps=RAMP_STEPS):
    """Keep actual seeds; project only exact opt-in identities to v21's canonical format."""
    from .posture_study import training_parameters as posture_parameters

    if type(seed) is not int or seed not in TRAIN_SEEDS:
        raise ValueError("undeclared v22 training seed")
    result = posture_parameters(log_directory, "adaptive")
    env, agent = (result["normalized"][name] for name in ("env", "agent"))
    term = env["rewards"]["contact_slip"]
    if (float(term["weight"]) != 1.0 or term["params"] != {"mode": "no_cost", "ramp_steps": str(ramp_steps)}
            or term["func"] != "week03_ant.tasks.contact_continuation_v21_cfg:ContinuationContactSlipReward"):
        raise ValueError("v22 unchanged no-cost reward contract differs")
    if int(env["seed"]) != seed or int(agent["seed"]) != seed:
        raise ValueError("requested training seed did not reach simulator and agent")
    if int(env["scene"]["terrain"]["terrain_generator"]["seed"]) != TRAIN_GEOMETRY:
        raise ValueError("training terrain seed changed")
    policy = agent["policy"]
    if (policy["command_mode"] != "conditioned" or policy["class_name"] != "CommandPriorActorCritic"
            or policy["prior_mode"] != "anchored" or policy["input_mode"] != "targets"
            or policy["require_config_match"] != "true"
            or agent["algorithm"]["class_name"] != "PriorPPO"
            or float(agent["algorithm"]["teacher_coef"]) != .02
            or agent["load_run"] != "v22_init" or agent["load_checkpoint"] != "model_0.pt"
            or agent["experiment_name"] != EXPERIMENT):
        raise ValueError("v22 saved network/prior/start differs")
    term["func"] = "week03_ant.tasks.contact_curriculum_v20_cfg:CurriculumContactSlipReward"
    term["params"]["mode"] = "<paired-coefficient-trajectory>"
    agent["experiment_name"] = "week03_ant_contact_curriculum_v20"
    agent["load_run"] = "v20_init"
    if "week03_ant.lp_curriculum_v17" in str(env["events"]):
        raise ValueError("unplanned reset curriculum")
    return result


def model_key(controller):
    if controller not in CONTROLLERS:
        raise ValueError("unknown v22 controller")
    return controller.removeprefix("history_")


def command_mode(controller):
    model_key(controller)
    return "conditioned"


def policy_mode(controller):
    model_key(controller)
    return "hybrid" if controller.startswith("history_") else "v10"


def models():
    from .continuation_study_v21 import models as prior_models

    path = ART / "trained_models.json"
    return {"parent": prior_models()["parent"],
            **(json.loads(path.read_text())["models"] if path.exists() else {})}


def evaluation_inputs(directory=ART):
    return _inputs(directory)


def validate_request(controller, geometry, reset, scenario, seconds, envs, phase):
    model_key(controller)
    if scenario not in ("mixed", "stones") or phase not in ("prepare_development", "prepare", "smoke", "holdout"):
        raise ValueError("invalid v22 evaluation phase/scenario")
    if phase in ("prepare_development", "smoke"):
        if (geometry, reset, scenario, seconds, envs) != (51, 24, "mixed", 16, 35):
            raise ValueError("development must use51/24/35env")
        if phase == "prepare_development" and controller != "parent":
            raise ValueError("prepare with frozen parent only")
    elif ((geometry, reset) not in HOLDOUTS
          or (seconds, envs) != ((16, 175) if scenario == "mixed" else (64, 10))
          or (phase == "prepare" and (controller != "parent" or scenario != "mixed"))):
        raise ValueError("changed v22 evaluation matrix")


def verify_frozen():
    frozen = json.loads((ART / "frozen.json").read_text())
    if (frozen["source_sha256"] != source_hashes() or frozen["legacy"] != legacy_hashes()
            or frozen["models"] != models() or frozen["controllers"] != list(CONTROLLERS)
            or frozen["holdouts"] != [list(pair) for pair in HOLDOUTS]
            or frozen["new_training_transitions"] != TOTAL_TRANSITIONS
            or frozen["reused_training_transitions"] != 0):
        raise ValueError("v22 evaluation freeze changed")
    for key, name in (("development_sha256", "development.json"),
                      ("training_freeze_sha256", "training_frozen.json"),
                      ("training_validation_sha256", "training_validation.json"),
                      ("trained_models_sha256", "trained_models.json"),
                      ("full_replay_sha256", "full_replay.json")):
        if frozen[key] != sha(ART / name):
            raise ValueError(f"v22 frozen evidence changed: {name}")
    trained = json.loads((ART / "trained_models.json").read_text())
    validation = json.loads((ART / "training_validation.json").read_text())
    if (set(trained["models"]) != set(RUNS) or set(validation["records"]) != set(RUNS)
            or trained["total_training_transitions"] != TOTAL_TRANSITIONS
            or trained["training_freeze_sha256"] != sha(ART / "training_frozen.json")
            or trained["training_validation_sha256"] != sha(ART / "training_validation.json")
            or validation["same_seed_capacity_initial_pairing"] is not True
            or validation["actual_random_horizons_paired"] is not True
            or validation["historical_references_sha256"] != sha(ART / "historical_references.json")
            or validation["full_replay_sha256"] != sha(ART / "full_replay.json")):
        raise ValueError("v22 training evidence linkage/budget changed")
    for run in RUNS:
        model, record = trained["models"][run], validation["records"][run]
        if (model["iteration"] != 249 or model["transitions"] != TRANSITIONS
                or model["training_seed"] != training_seed(run)
                or model["checkpoint"] != str((ART / "runs" / run / "model_249.pt").relative_to(ROOT))):
            raise ValueError("v22 final fresh checkpoint identity/budget changed")
        for name in ("initial", "schedule", "rollout"):
            item = record[name]
            verify_hashes({item["path"]: item["sha256"]})
        logs = record["learning_log"]
        verify_hashes({logs["text_log"]: logs["text_log_sha256"], **logs["event_files_sha256"]})
        if (model["initial_audit_sha256"] != record["initial"]["sha256"]
                or model["schedule_sha256"] != record["schedule"]["sha256"]):
            raise ValueError("v22 model/raw audit links changed")
    for item in models().values():
        verify_hashes({item["checkpoint"]: item["sha256"]})
    development = json.loads((ART / "development.json").read_text())
    if development["source_sha256"] != frozen["source_sha256"]:
        raise ValueError("v22 development sources differ from scored sources")
    for record in development["records"]:
        verify_hashes({record["path"]: record["sha256"]})
    training = json.loads((ART / "training_frozen.json").read_text())
    if (training["source_sha256"] != source_hashes(training=True)
            or training["all_source_sha256"] != source_hashes()
            or training["legacy"] != frozen["legacy"]
            or training["source_checkpoint_sha256"] != PARENT_SHA
            or training["historical_references_sha256"] != sha(ART / "historical_references.json")
            or training["seeds"] != list(TRAIN_SEEDS) or training["runs"] != list(RUNS)
            or training["total_training_transitions"] != TOTAL_TRANSITIONS
            or training["development_training_transitions"] != DEVELOPMENT_TRANSITIONS):
        raise ValueError("v22 main-start frozen definitions/reference links changed")
    expected_paths = {str((ART / "expected_main" / f"{run}_initial.json").relative_to(ROOT)) for run in RUNS}
    if set(training["expected_main_sha256"]) != expected_paths:
        raise ValueError("v22 same-seed initialization expectation inventory changed")
    verify_hashes(training["expected_main_sha256"])
    for key, name in (("preflight_sha256", "preflight.json"), ("capacity_sha256", "capacity.json"),
                      ("initial_shared_sha256", "initial_shared.json"), ("training_cache_sha256", "training_cache.json")):
        if training[key] != sha(ART / name):
            raise ValueError("v22 development/init provenance changed")
    references = json.loads((ART / "historical_references.json").read_text())
    verify_hashes(references["sha256"])
    return frozen
