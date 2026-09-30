"""CPU-only shared learned initialization and opt-in treatment contracts."""
import copy
import importlib.util
import sys

import pytest
import torch
from week03_ant.direction_study import ROOT, V14, prepare_checkpoint, validate_training_start

sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("direction_launcher_test", ROOT / "scripts/direction_v15.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_args_differ_only_by_additive_reward():
    control, stable = [launcher.training_arguments(["--num_envs", "4096"], a) for a in ("control", "stable")]
    assert [(a, b) for a, b in zip(control, stable) if a != b] == [
        ("env.rewards.directional_stability.weight=0.0", "env.rewards.directional_stability.weight=1.0")]
    for arg in ("agent.policy.command_mode=conditioned", "env.rewards.adaptive_posture.weight=1.0",
                "agent.algorithm.teacher_coef=0.02", "env.rewards.foothold_support.weight=0.0",
                "agent.policy.require_config_match=true", "Week03-Ant-Direction-v15-Train-v0"):
        assert arg in control


@pytest.mark.parametrize("args", [
    ["env.rewards.directional_stability.weight=0"], ["env.rewards.directional_stability.weight=nan"],
    ["env.rewards.directional_stability.weight=1", "env.rewards.directional_stability.weight=1"],
    ["env.rewards.adaptive_posture.weight=0"], ["agent.policy.command_mode=masked"],
    ["agent.algorithm.teacher_coef=0"], ["agent.policy.prior_mode=free"],
    ["--task", "Isaac-Ant-v0"], ["--task"], ["--task=invalid"],
])
def test_forbidden_treatment_overrides(args):
    with pytest.raises(ValueError):
        launcher.training_arguments(args, "stable")


def test_shared_91d_warmstart_preserves_every_learned_tensor(tmp_path, monkeypatch):
    import week03_ant.direction_study as study
    from week03_ant.command_policy import CommandPriorActorCritic

    monkeypatch.setattr(study, "ROOT", tmp_path)
    path = tmp_path / "init.pt"
    result = prepare_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = torch.load(V14, map_location="cpu", weights_only=False)["model_state_dict"]
    state = checkpoint["model_state_dict"]
    assert state.keys() == source.keys()
    for key in source:
        assert torch.equal(state[key], torch.full_like(source[key], .2) if key == "std" else source[key]), key
    assert state["actor.0.command_weight"].abs().sum() > 0
    assert state["critic.0.command_weight"].abs().sum() > 0
    assert checkpoint["iter"] == 0 and checkpoint["optimizer_state_dict"]["state"] == {}
    assert result["learned_command_weights_preserved"] and result["all_non_std_tensors_identical"]
    observations = torch.randn(7, 91)
    policy = CommandPriorActorCritic({"policy": observations}, {"policy": ["policy"], "critic": ["policy"]},
        8, command_mode="conditioned", prior_mode="anchored", input_mode="targets", require_config_match=True)
    policy.load_state_dict(state)
    rng = torch.get_rng_state().clone()
    parity = launcher.initial_policy_parity(policy, observations)
    assert all(parity[k] for k in ("actor", "critic", "teacher"))
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(FileExistsError):
        prepare_checkpoint(path)


@pytest.mark.parametrize("mismatch", ["command_actor", "command_critic", "std", "teacher", "mode", "schema", "missing", "lr", "state", "class"])
def test_loaded_start_rejects_any_nonallowed_change(mismatch):
    state = torch.load(V14, map_location="cpu", weights_only=False)["model_state_dict"]
    state["std"] = torch.full_like(state["std"], .2)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1.e-4)
    validate_training_start(state, optimizer, "control")
    validate_training_start(state, optimizer, "stable")
    if mismatch in ("command_actor", "command_critic"):
        state[f"{mismatch.removeprefix('command_')}.0.command_weight"].zero_()
    elif mismatch == "std":
        state["std"].fill_(1.)
    elif mismatch == "teacher":
        state[next(k for k in state if k.startswith('teacher.'))].flatten()[0] += 1
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


def test_full_initial_pairing_and_exclusive_evidence(tmp_path):
    initial = {"initial_state_sha256": {}, "initial_prefix_sha256": "prefix", "rng_sha256": {},
               "policy_state_sha256": {}, "num_envs": 4096, "dt": 1 / 60, "normalized_parameters": {}}
    launcher.check_initial(initial, copy.deepcopy(initial))
    for key in initial:
        changed = copy.deepcopy(initial)
        changed[key] = None
        with pytest.raises(ValueError, match=key):
            launcher.check_initial(changed, initial)
    path = tmp_path / "audit.json"
    launcher.save_json(path, initial)
    with pytest.raises(FileExistsError):
        launcher.save_json(path, {})
