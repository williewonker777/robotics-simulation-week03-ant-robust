"""CPU-only arm, initialization and no-overwrite checks for v13 training."""

import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from week03_ant.posture_study import ROOT, START, prepare_checkpoint, save_json

SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("posture_launcher_test", SCRIPTS / "posture_v13.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize("arm,weight", [("control", "0.0"), ("adaptive", "1.0")])
def test_arms_preserve_original_prior_and_support_guards(arm, weight):
    args = launcher.training_arguments([], arm)
    assert f"env.rewards.adaptive_posture.weight={weight}" in args
    assert "agent.algorithm.teacher_coef=0.02" in args
    assert "env.rewards.foothold_support.weight=0.0" in args
    assert "agent.policy.require_config_match=true" in args
    assert "Week03-Ant-Adaptive-Posture-Train-v13" in args


@pytest.mark.parametrize("args", [
    ["env.rewards.adaptive_posture.weight=0"], ["env.rewards.adaptive_posture.weight=nan"],
    ["env.rewards.adaptive_posture.weight=1", "env.rewards.adaptive_posture.weight=1"],
    ["agent.policy.prior_mode=free"], ["agent.algorithm.teacher_coef=0"], ["env.rewards.foothold_support.weight=1"],
    ["--task", "Isaac-Ant-v0"], ["--task"], ["--task=invalid"],
])
def test_arm_overrides_cannot_silently_change_treatment(args):
    with pytest.raises(ValueError):
        launcher.training_arguments(args, "adaptive")


def test_control_and_adaptive_args_differ_only_in_new_reward():
    control, adaptive = [launcher.training_arguments(["--num_envs", "4096"], a) for a in ("control", "adaptive")]
    differences = [(a, b) for a, b in zip(control, adaptive) if a != b]
    assert differences == [("env.rewards.adaptive_posture.weight=0.0", "env.rewards.adaptive_posture.weight=1.0")]


def test_training_initialization_checks_full_observation_and_rng():
    original = {"initial_state_sha256": {"observations": "a"}, "policy_state_sha256": {"actor": "b"},
                "rng_sha256": {"cuda": "c"}, "num_envs": 64, "dt": 1 / 60, "normalized_parameters": {"config": "x"}}
    launcher.check_initial(original, copy.deepcopy(original))
    for key in original:
        changed = copy.deepcopy(original)
        changed[key] = None
        with pytest.raises(ValueError, match=key):
            launcher.check_initial(changed, original)


def test_json_evidence_is_exclusive(tmp_path):
    path = tmp_path / "evidence.json"
    save_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        save_json(path, {"value": 2})
    assert '"value": 1' in path.read_text()


def test_finetune_initialization_retains_all_learned_depth_weights(tmp_path, monkeypatch):
    import week03_ant.posture_study as study

    # Only path serialization uses ROOT; original pinned model paths stay fixed.
    monkeypatch.setattr(study, "ROOT", tmp_path)
    destination = tmp_path / "init.pt"
    result = prepare_checkpoint(destination)
    saved = torch.load(destination, map_location="cpu", weights_only=False)
    source = torch.load(START, map_location="cpu", weights_only=False)["model_state_dict"]
    for name, value in saved["model_state_dict"].items():
        if name != "std":
            assert torch.equal(value, source[name]), name
    assert torch.equal(saved["model_state_dict"]["std"], torch.full((8,), .2))
    assert saved["optimizer_state_dict"]["state"] == {}
    assert saved["iter"] == 0 and result["all_non_std_tensors_identical"]
    with pytest.raises(FileExistsError):
        prepare_checkpoint(destination)


@pytest.mark.parametrize("mismatch", ["weights", "std", "teacher", "missing", "lr", "state", "class"])
def test_actual_training_initialization_rejects_incorrect_loaded_learner(mismatch):
    from week03_ant.posture_study import validate_training_start

    state = torch.load(START, map_location="cpu", weights_only=False)["model_state_dict"]
    state["std"] = torch.full_like(state["std"], .2)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1.e-4)
    validate_training_start(state, optimizer)
    if mismatch in ("weights", "teacher"):
        name = next(k for k in state if k.startswith("teacher." if mismatch == "teacher" else "actor."))
        state[name].flatten()[0] += 1.
    elif mismatch == "std":
        state["std"].fill_(1.)
    elif mismatch == "missing":
        state.pop("std")
    elif mismatch == "lr":
        optimizer.param_groups[0]["lr"] = 1.e-3
    elif mismatch == "state":
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    else:
        optimizer = torch.optim.SGD([parameter], lr=1.e-4)
    with pytest.raises(ValueError):
        validate_training_start(state, optimizer)
