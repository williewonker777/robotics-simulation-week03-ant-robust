"""Fixed, evaluation-only common-map comparison; no training or policy changes."""

from __future__ import annotations

import json

from .lp_study_v17 import V16_CONTROL, V16_CONTROL_SHA, evaluation_inputs as _inputs
from .posture_study import (
    ROOT, START, START_SHA, V5, V5_SHA, cache_snapshot, save_json, sha,
    tensor_sha, utc, validate_teacher, verify_hashes,
)

ART = ROOT / "artifacts/terrain_demo/rebaseline_v19"
WORK = ROOT / "outputs/rebaseline_v19_20260928/attempt02"
PLAN = ROOT / "docs/experiment_plans/rebaseline_v19.md"
SCHEMA = "week03_ant_rebaseline_v19_v1"
TASK = "Week03-Ant-Contact-v16-Eval-v0"
CONTROLLERS = ("v5", "history_original", "v16_control", "history_control")
HOLDOUTS = ((107, 68), (108, 69), (109, 70))
COMPARISONS = (("v16_control", "v5"), ("history_control", "history_original"))
SOURCES = (
    "src/week03_ant/rebaseline_study_v19.py",
    "scripts/evaluate_rebaseline_v19.py",
    "scripts/run_rebaseline_v19.py",
    "scripts/summarize_rebaseline_v19.py",
    "tests/test_rebaseline_v19.py",
    "tests/test_rebaseline_summary_v19.py",
    "docs/experiment_plans/rebaseline_v19.md",
    "docs/REBASELINE_V19_REFERENCES.md",
)


def model_key(controller):
    if controller not in CONTROLLERS:
        raise ValueError("unknown v19 controller")
    return "original" if controller in ("v5", "history_original") else "control"


def command_mode(controller):
    return None if model_key(controller) == "original" else "conditioned"


def policy_mode(controller):
    model_key(controller)
    if controller.startswith("history_"):
        return "hybrid"
    return "v5" if controller == "v5" else "v10"


def models():
    return {
        "original": {"checkpoint": str(START.relative_to(ROOT)), "sha256": START_SHA},
        "control": {"checkpoint": str(V16_CONTROL.relative_to(ROOT)), "sha256": V16_CONTROL_SHA},
        "v5": {"checkpoint": str(V5.relative_to(ROOT)), "sha256": V5_SHA},
    }


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def legacy_hashes():
    """Preserve all frozen v0-v18 source/model bytes, including v18 itself."""
    from .style_study_v18 import legacy_hashes as previous

    frozen = json.loads((ROOT / "artifacts/terrain_demo/terrain_style_v18/frozen.json").read_text())
    old = previous()
    if old != frozen["legacy"]:
        raise ValueError("v18 legacy graph changed")
    sources = {**old["sources"], **frozen["source_sha256"]}
    checkpoints = {**old["models"], **{v["checkpoint"]: v["sha256"] for v in frozen["models"].values()}}
    verify_hashes({**sources, **checkpoints})
    return {"sources": sources, "models": checkpoints}


def verify_frozen():
    frozen = json.loads((ART / "frozen.json").read_text())
    if (frozen["source_sha256"] != source_hashes() or frozen["legacy"] != legacy_hashes()
            or frozen["models"] != models() or frozen["controllers"] != list(CONTROLLERS)
            or frozen["holdouts"] != [list(pair) for pair in HOLDOUTS]
            or frozen["new_training_transitions"] != 0
            or frozen["development_sha256"] != sha(ART / "development.json")):
        raise ValueError("v19 frozen inputs changed")
    for entry in models().values():
        verify_hashes({entry["checkpoint"]: entry["sha256"]})
    development = json.loads((ART / "development.json").read_text())
    if development["source_sha256"] != frozen["source_sha256"]:
        raise ValueError("development did not validate frozen sources")
    for entry in development["records"]:
        verify_hashes({entry["path"]: entry["sha256"]})
    return frozen


def evaluation_inputs(directory=ART):
    return _inputs(directory)


def validate_request(controller, geometry, reset, scenario, seconds, envs, phase):
    model_key(controller)
    if scenario not in ("mixed", "stones") or phase not in ("prepare_development", "prepare", "smoke", "holdout"):
        raise ValueError("unknown scenario/phase")
    expected = (16, 175) if scenario == "mixed" else (64, 10)
    if phase in ("prepare_development", "smoke"):
        if (geometry, reset, scenario, seconds, envs) != (51, 24, "mixed", 16, 35):
            raise ValueError("development is restricted to geometry51/reset24/35env")
        if phase == "prepare_development" and controller != "history_original":
            raise ValueError("development preparation uses only the original policy")
    elif ((geometry, reset) not in HOLDOUTS or (seconds, envs) != expected
          or (phase == "prepare" and (controller != "history_original" or scenario != "mixed"))):
        raise ValueError("changed predeclared evaluation matrix")
