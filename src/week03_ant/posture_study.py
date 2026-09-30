"""Small provenance/warm-start helpers for the additive v13 posture study."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/terrain_demo/adaptive_posture_v13"
WORK = ROOT / "outputs/adaptive_posture_v13_20260922"
START = ROOT / "artifacts/terrain_demo/prior_v10/runs/v10_anchored_seed42/model_749.pt"
START_SHA = "fa1ec87ef2b89b698f250d91452685cef0a7f76e340e790650116138aee875c1"
V5 = ROOT / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
V5_SHA = "889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e"
CONTROLLERS = ("v5", "original", "control", "adaptive", "history_original",
               "history_control", "history_adaptive")
ARMS = ("control", "adaptive")
HOLDOUTS = ((78, 52), (79, 53))
TASK = "Week03-Ant-Prior-Lanes-Eval-v10"
TRAIN_TASK = "Week03-Ant-Adaptive-Posture-Train-v13"
SCHEMA = "week03_ant_adaptive_posture_v13_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def state_hashes(state):
    return {name: tensor_sha(value) for name, value in sorted(state.items())}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def legacy_hashes():
    history = json.loads((ROOT / "artifacts/terrain_demo/history_v12/frozen.json").read_text())
    videos = json.loads((ROOT / "artifacts/terrain_demo/hybrid_v11/other_terrains/predeclared.json").read_text())
    sources = {**history["v11_source_sha256"], **history["source_sha256"], **videos["source_sha256"]}
    models = {entry["checkpoint"]: entry["sha256"] for entry in history["checkpoints"].values()}
    models[str(V5.relative_to(ROOT))] = V5_SHA
    if len(sources) != 72 or len(models) != 4:
        raise ValueError("unexpected legacy inventory")
    verify_hashes({**sources, **models})
    return {"sources": sources, "models": models}


def verify_hashes(inventory):
    for name, digest in inventory.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"pinned file changed: {name}")


def cache_snapshot(cache_root, geometries):
    """Same tested inventory rule as v12 preparation; no cache writes or imports."""
    cache_root = Path(cache_root)
    expected = {g * 10007 + col * 131 + row
                for g in geometries for col in range(30) for row in range(8)}
    matches = {seed: [] for seed in expected}
    for path in cache_root.glob("*/cfg.yaml"):
        match = re.search(r"^seed: (\d+)\s*$", path.read_text(), re.MULTILINE)
        if match and int(match[1]) in expected:
            matches[int(match[1])].append(path.parent)
    if any(len(paths) != 1 for paths in matches.values()):
        raise ValueError("missing or ambiguous terrain-cache tile")
    tiles = []
    for seed, (directory,) in sorted(matches.items()):
        if directory.is_symlink():
            raise ValueError("indirect terrain cache directory")
        hashes = {}
        for name in ("cfg.yaml", "mesh.obj", "origin.csv"):
            path = directory / name
            if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                raise ValueError("missing/indirect/empty terrain cache file")
            hashes[name] = sha(path)
        tiles.append({"tile_seed": seed, "cache_key": directory.name, "sha256": hashes})
    return {"geometries": list(geometries), "tiles_per_geometry": 240, "tiles": tiles}


def validate_teacher(state):
    import torch

    if sha(V5) != V5_SHA:
        raise ValueError("original v5 changed")
    parent = torch.load(V5, map_location="cpu", weights_only=False)["model_state_dict"]
    teacher = {name: value for name, value in state.items() if name.startswith("teacher.")}
    expected = {name.replace("actor.", "teacher.", 1): value
                for name, value in parent.items() if name.startswith("actor.")}
    if teacher.keys() != expected.keys() or any(not torch.equal(teacher[k].cpu(), expected[k]) for k in teacher):
        raise ValueError("frozen teacher differs from original v5")


def prepare_checkpoint(destination, seed=45):
    """Keep every pretrained network tensor; only reset std and Adam/iteration."""
    import torch
    from week03_ant.prior_policy import PriorActorCritic

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if sha(START) != START_SHA:
        raise ValueError("unexpected v10 starting checkpoint")
    torch.manual_seed(seed)
    original = torch.load(START, map_location="cpu", weights_only=False)
    source = original["model_state_dict"]
    policy = PriorActorCritic({"policy": torch.zeros(1, 88)},
                             {"policy": ["policy"], "critic": ["policy"]}, 8,
                             prior_mode="anchored", input_mode="targets", require_config_match=True)
    policy.load_state_dict(source)
    with torch.no_grad():
        policy.std.fill_(.2)
    state = policy.state_dict()
    if state.keys() != source.keys() or any(not torch.equal(state[k], source[k]) for k in state if k != "std"):
        raise ValueError("warm-start changed a non-exploration tensor")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite initialization")
    validate_teacher(state)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.e-4)
    checkpoint = {"model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(), "iter": 0,
                  "infos": {"study": "adaptive_posture_v13", "starting_sha256": START_SHA,
                            "source_iteration": original["iter"], "seed": seed,
                            "reset_std": .2, "reset_optimizer": "Adam 1e-4, empty state",
                            "teacher_sha256": V5_SHA}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(checkpoint, stream)
    return {"checkpoint": str(destination.resolve().relative_to(ROOT)), "sha256": sha(destination),
            "state_sha256": state_hashes(state), "source_sha256": START_SHA,
            "all_non_std_tensors_identical": True, "optimizer_state_empty": True}


def validate_training_start(state, optimizer):
    """Check the actual loaded learner, not merely an argument or checkpoint name."""
    import torch

    if sha(START) != START_SHA:
        raise ValueError("pinned initial source changed")
    expected = torch.load(START, map_location="cpu", weights_only=False)["model_state_dict"]
    expected["std"] = torch.full_like(expected["std"], .2)
    if state.keys() != expected.keys():
        raise ValueError("actual initial policy keys differ")
    for name, value in state.items():
        if not torch.equal(value.detach().cpu(), expected[name]):
            raise ValueError(f"actual initial policy tensor differs: {name}")
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("expected a fresh Adam optimizer")
    saved = optimizer.state_dict()
    if saved["state"] or len(saved["param_groups"]) != 1:
        raise ValueError("initial optimizer must have one empty Adam parameter group")
    group = saved["param_groups"][0]
    expected_options = {"lr": 1.e-4, "betas": (.9, .999), "eps": 1.e-8,
                        "weight_decay": 0., "amsgrad": False, "maximize": False}
    if any(group[key] != value for key, value in expected_options.items()):
        raise ValueError("initial Adam settings differ")


def training_parameters(log_directory, arm):
    """Read actual dumped configs, normalizing ONLY treatment and output names."""
    import yaml

    directory = Path(log_directory)
    files = {name: directory / "params" / f"{name}.yaml" for name in ("env", "agent")}
    # BaseLoader preserves all scalar text without constructing Python objects.
    values = {name: yaml.load(path.read_text(), Loader=yaml.BaseLoader) for name, path in files.items()}
    env, agent = values["env"], values["agent"]
    if float(env["rewards"]["adaptive_posture"]["weight"]) != float(arm == "adaptive"):
        raise ValueError("saved config reward arm mismatch")
    env["rewards"]["adaptive_posture"]["weight"] = "<paired-treatment>"
    for field in ("log_dir", "io_descriptors_output_dir"):
        if Path(env.pop(field)).resolve() != directory.resolve():
            raise ValueError("saved env output path does not match run directory")
    agent.pop("run_name")
    return {"normalized": values, "sha256": {name: sha(path) for name, path in files.items()}}


TRAINING_SOURCES = (
    "src/week03_ant/posture_math.py", "src/week03_ant/posture_study.py",
    "src/week03_ant/tasks/posture_v13.py", "src/week03_ant/tasks/posture_v13_cfg.py",
    "scripts/posture_v13.py", "scripts/run_posture_study.py",
    "tests/test_posture_math.py", "tests/test_posture_adapter.py", "tests/test_posture_training.py",
    "tests/test_posture_harness.py", "docs/experiment_plans/adaptive_posture_v13.md",
    "outputs/adaptive_posture_v13_20260922/PLAN.md",
)
SOURCE_FILES = (*TRAINING_SOURCES, "src/week03_ant/posture_telemetry.py",
                "scripts/evaluate_posture_v13.py", "scripts/summarize_posture_v13.py",
                "tests/test_posture_telemetry.py", "tests/test_posture_summary.py")


def source_hashes(training=False):
    return {name: sha(ROOT / name) for name in (TRAINING_SOURCES if training else SOURCE_FILES)}


def evaluation_inputs(directory=ART):
    """Bind pre-materialized cache evidence to one frozen evaluation matrix."""
    directory = Path(directory)
    record = json.loads((directory / "evaluation_inputs.json").read_text())
    if (record["experiment_freeze_sha256"] != sha(directory / "frozen.json")
            or record["terrain_cache_manifest_sha256"] != sha(directory / "terrain_cache.json")):
        raise ValueError("pinned evaluation input manifest changed")
    return record
