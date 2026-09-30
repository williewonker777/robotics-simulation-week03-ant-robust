"""CPU-only proofs for the v17 launch, paired exposure, and training guards."""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

from week03_ant.lp_study_v17 import ROOT, V16_CONTROL, prepare_checkpoint, validate_training_start

sys.path.insert(0, str(ROOT / "scripts"))
_LAUNCHER = importlib.util.spec_from_file_location("v17_launcher_test", ROOT / "scripts/lp_v17.py")
launcher = importlib.util.module_from_spec(_LAUNCHER)
_LAUNCHER.loader.exec_module(launcher)
_RUNNER = importlib.util.spec_from_file_location("v17_runner_test", ROOT / "scripts/run_lp_study.py")
runner = importlib.util.module_from_spec(_RUNNER)
_RUNNER.loader.exec_module(runner)


def test_arms_differ_only_by_sampler_mode():
    fixed, lp = [launcher.training_arguments(["--num_envs", "4096"], arm) for arm in ("fixed", "lp")]
    assert [(a, b) for a, b in zip(fixed, lp) if a != b] == [
        ("env.events.reset_base.params.mode=fixed", "env.events.reset_base.params.mode=lp")]
    assert "env.rewards.contact_slip.weight=0.0" in fixed
    assert "agent.algorithm.teacher_coef=0.02" in fixed
    assert "env.events.reset_base.params.stage_size=1024" in fixed
    assert "Week03-Ant-LP-v17-Train-v0" in fixed


@pytest.mark.parametrize("args", [
    ["env.events.reset_base.params.mode=lp"],
    ["env.events.reset_base.params.stage_size=16"],
    ["env.rewards.contact_slip.weight=1.0"],
    ["env.rewards.directional_stability.weight=1.0"],
    ["env.rewards.adaptive_posture.weight=0.0"],
    ["agent.algorithm.teacher_coef=0.0"],
    ["agent.policy.command_mode=masked"],
    ["env.events.reset_base.func=other"],
    ["--task", "Week03-Ant-Contact-v16-Train-v0"],
    ["--task=invalid"],
])
def test_treatment_or_task_substitution_rejected(args):
    with pytest.raises(ValueError):
        launcher.training_arguments(args, "fixed")


def test_development_stage_size_is_explicit():
    args = launcher.training_arguments([], "lp", stage_size=16)
    assert "env.events.reset_base.params.stage_size=16" in args
    with pytest.raises(ValueError):
        launcher.training_arguments([], "lp", stage_size=32)


def test_direct_launcher_budget_guard():
    base = ["--seed", "49", "--resume", "--load_run", "v17_init",
            "--checkpoint", "model_0.pt", "env.scene.terrain.terrain_generator.seed=101"]
    launcher.validate_budget(base + ["--num_envs", "256", "--max_iterations", "300"],
                             development=True)
    launcher.validate_budget(base + ["--num_envs", "4096", "--max_iterations", "250"],
                             development=False)
    for bad in (["--num_envs", "4096", "--max_iterations", "250"],
                ["--num_envs", "256", "--max_iterations", "301"]):
        with pytest.raises(ValueError):
            launcher.validate_budget(base + bad, development=True)
    with pytest.raises(ValueError):
        launcher.validate_budget(base + ["--num_envs", "256", "--max_iterations", "300"],
                                 development=False)


def test_shared_v16_initialization_resets_only_std(tmp_path, monkeypatch):
    import week03_ant.lp_study_v17 as study

    monkeypatch.setattr(study, "ROOT", tmp_path)
    path = tmp_path / "model_0.pt"
    result = prepare_checkpoint(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    source = torch.load(V16_CONTROL, map_location="cpu", weights_only=False)["model_state_dict"]
    assert saved["model_state_dict"].keys() == source.keys()
    for name, value in source.items():
        expected = torch.full_like(value, .2) if name == "std" else value
        assert torch.equal(saved["model_state_dict"][name], expected), name
    assert saved["optimizer_state_dict"]["state"] == {} and saved["iter"] == 0
    assert result["learned_command_weights_preserved"] and result["optimizer_state_empty"]
    with pytest.raises(FileExistsError):
        prepare_checkpoint(path)


@pytest.mark.parametrize("mismatch", ["command", "std", "teacher", "mode", "lr", "state", "class"])
def test_initial_validation_rejects_changes(mismatch):
    state = torch.load(V16_CONTROL, map_location="cpu", weights_only=False)["model_state_dict"]
    state["std"] = torch.full_like(state["std"], .2)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1.e-4)
    validate_training_start(state, optimizer, "fixed")
    if mismatch == "command":
        state["actor.0.command_weight"].zero_()
    elif mismatch == "std":
        state["std"].fill_(1.)
    elif mismatch == "teacher":
        state[next(key for key in state if key.startswith("teacher."))].flatten()[0] += 1
    elif mismatch == "mode":
        state["command_mode_code"] += 1
    elif mismatch == "lr":
        optimizer.param_groups[0]["lr"] = 1.e-3
    elif mismatch == "state":
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    else:
        optimizer = torch.optim.SGD([parameter], lr=1.e-4)
    with pytest.raises(ValueError):
        validate_training_start(state, optimizer)


def test_lane_proxy_forwards_randomized_episode_lengths(monkeypatch):
    state = SimpleNamespace(lane=torch.tensor([0, 30, 34]))
    monkeypatch.setitem(sys.modules, "week03_ant.tasks.lanes",
                        SimpleNamespace(get_lane_state=lambda _: state))
    raw = SimpleNamespace(num_envs=3, device="cpu")
    vector = SimpleNamespace(episode_length_buf=torch.zeros(3, dtype=torch.long))
    vector.step = lambda actions: ("obs", "rew", "done", {})
    proxy = launcher.LaneOccupancyEnv(vector, raw)
    randomized = torch.tensor([12, 18, 4])
    proxy.episode_length_buf = randomized
    assert vector.episode_length_buf is randomized
    proxy.step(None)
    state.lane[:] = torch.tensor([25, 29, 30])
    proxy.step(None)
    summary = proxy.summary()
    assert summary["transition_steps"] == 6 and summary["policy_steps"] == 2
    assert summary["lane_transition_counts"][0] == 1
    assert summary["lane_transition_counts"][30] == 2
    assert summary["family_transition_counts"] == [1, 0, 0, 0, 0, 2, 3]


def test_sampling_validator_detects_budget_and_window_failure():
    sample = {"mode": "lp", "stage_size": 16, "seed": 49017,
              "invalid_nonfinite_count": 0, "policy_steps": 64,
              "transition_steps": 64 * 256, "lane_transition_counts": [64 * 256] + [0] * 34,
              "sampled_episode_counts": [1] + [0] * 34,
              "completed_counts": [1] + [0] * 34,
              "first_episode_exclusions": 1, "stage_updates": 0, "stages": []}
    runner.validate_sampling(sample, "lp", iterations=2, envs=256, stage_size=16, require_two=False)
    with pytest.raises(ValueError, match="windows"):
        runner.validate_sampling(sample, "lp", iterations=2, envs=256, stage_size=16, require_two=True)
    sample["transition_steps"] -= 1
    with pytest.raises(ValueError, match="budget"):
        runner.validate_sampling(sample, "lp", iterations=2, envs=256, stage_size=16, require_two=False)


def test_declared_command_budgets_and_seed(tmp_path):
    audit, sample = tmp_path / "audit.json", tmp_path / "sample.json"
    command = runner.train_command("lp", audit, sample, iterations=300, envs=256,
                                   development=True)
    assert command[0].endswith("run-python") and "--stage-size" in command
    assert "--seed" in command and command[command.index("--seed") + 1] == "49"
    assert "env.scene.terrain.terrain_generator.seed=101" in command
    with pytest.raises(ValueError):
        runner.train_command("lp", audit, sample, iterations=249, envs=4096)
