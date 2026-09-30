"""CPU-only v16 warm-start and fixed-treatment contract tests."""

from __future__ import annotations

import importlib.util
import sys

import pytest
import torch

from week03_ant.contact_study import ROOT, V15, prepare_checkpoint, validate_training_start

sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("contact_launcher_test", ROOT / "scripts/contact_v16.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_arms_differ_only_by_contact_reward():
    control, slip = [launcher.training_arguments(["--num_envs", "4096"], arm) for arm in ("control", "slip")]
    assert [(a, b) for a, b in zip(control, slip) if a != b] == [
        ("env.rewards.contact_slip.weight=0.0", "env.rewards.contact_slip.weight=1.0"),
    ]
    for value in (
        "Week03-Ant-Contact-v16-Train-v0", "env.rewards.adaptive_posture.weight=1.0",
        "agent.policy.command_mode=conditioned",
        "agent.policy.require_config_match=true",
    ):
        assert value in control


@pytest.mark.parametrize("args", [
    ["env.rewards.contact_slip.weight=0"], ["env.rewards.contact_slip.weight=nan"],
    ["env.rewards.contact_slip.weight=1", "env.rewards.contact_slip.weight=1"],
    ["agent.policy.command_mode=masked"],
    ["agent.algorithm.teacher_coef=0"], ["--task", "Week03-Ant-Direction-v15-Train-v0"],
    ["--task"], ["--task=invalid"],
])
def test_treatment_or_task_substitution_rejected(args):
    with pytest.raises(ValueError):
        launcher.training_arguments(args, "slip")


def test_shared_v15_warmstart_resets_only_std(tmp_path, monkeypatch):
    import week03_ant.contact_study as study

    monkeypatch.setattr(study, "ROOT", tmp_path)
    checkpoint_path = tmp_path / "v16_init.pt"
    result = prepare_checkpoint(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = torch.load(V15, map_location="cpu", weights_only=False)["model_state_dict"]
    assert saved["model_state_dict"].keys() == source.keys()
    for name, value in source.items():
        expected = torch.full_like(value, .2) if name == "std" else value
        assert torch.equal(saved["model_state_dict"][name], expected), name
    assert saved["model_state_dict"]["actor.0.command_weight"].abs().sum() > 0
    assert saved["model_state_dict"]["critic.0.command_weight"].abs().sum() > 0
    assert saved["optimizer_state_dict"]["state"] == {} and saved["iter"] == 0
    assert result["learned_command_weights_preserved"] and result["optimizer_state_empty"]
    with pytest.raises(FileExistsError):
        prepare_checkpoint(checkpoint_path)


@pytest.mark.parametrize("mismatch", ["command", "std", "teacher", "mode", "schema", "missing", "lr", "state", "class"])
def test_start_validation_rejects_every_nonallowed_change(mismatch):
    state = torch.load(V15, map_location="cpu", weights_only=False)["model_state_dict"]
    state["std"] = torch.full_like(state["std"], .2)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1.e-4)
    validate_training_start(state, optimizer, "control")
    if mismatch == "command":
        state["actor.0.command_weight"].zero_()
    elif mismatch == "std":
        state["std"].fill_(1.)
    elif mismatch == "teacher":
        state[next(k for k in state if k.startswith("teacher."))].flatten()[0] += 1
    elif mismatch in ("mode", "schema"):
        state[f"command_{mismatch}_code"] += 1
    elif mismatch == "missing":
        state.pop("std")
    elif mismatch == "lr":
        optimizer.param_groups[0]["lr"] = .001
    elif mismatch == "state":
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    else:
        optimizer = torch.optim.SGD([parameter], lr=.0001)
    with pytest.raises(ValueError):
        validate_training_start(state, optimizer)
