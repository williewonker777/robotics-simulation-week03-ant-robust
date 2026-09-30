"""CPU-only guards for the additive history-evaluation adapter."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_history_hybrid.py"
spec = importlib.util.spec_from_file_location("history_evaluator_under_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def arguments(tmp_path, controller="history", extra=()):
    return module.parse_args([
        "--controller", controller, "--geometry", "72", "--seed", "46",
        "--output", str(tmp_path / "new.json"), *extra,
    ])


@pytest.mark.parametrize("controller,mode", [
    ("history", "hybrid"), ("instant", "hybrid"), ("v5", "v5"), ("v10", "v10"),
])
def test_cli_routes_only_the_explicit_controller(tmp_path, controller, mode):
    args = arguments(tmp_path, controller)
    command = module.backend_argv(args, tmp_path / "backend.json")
    assert command[command.index("--mode") + 1] == mode
    assert command[command.index("--preset") + 1] == "cautious"
    assert command[command.index("--num_envs") + 1] == "175"
    assert str(tmp_path / "new.json") not in command


def test_stones_default_and_cli_validation(tmp_path):
    args = arguments(tmp_path, extra=("--scenario", "stones", "--seconds", "64"))
    assert args.num_envs == 10
    for extra in [("--num-envs", "0"), ("--geometry", "-1"),
                  ("--preset", "balanced"), ("--policy-seed", "44")]:
        with pytest.raises(SystemExit):
            arguments(tmp_path, extra=extra)


def test_dry_run_does_not_import_or_launch_simulator(tmp_path, capsys):
    before = set(sys.modules)
    module.main(["--geometry", "72", "--seed", "46", "--output",
                 str(tmp_path / "result.json"), "--dry-run"])
    assert "isaaclab.app" not in set(sys.modules) - before
    assert "hybrid" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_runtime_injections_restore_on_exception():
    old_factory, new_factory = object(), object()
    old_presets = {"cautious": object()}
    gate_module = SimpleNamespace(DepthPolicyGate=old_factory, PRESETS=old_presets)
    old_argv, old_path = sys.argv, sys.path.copy()
    with pytest.raises(RuntimeError):
        with module.patched_runtime(gate_module, new_factory, "history-config", ["new-argv"]):
            assert gate_module.DepthPolicyGate is new_factory
            assert gate_module.PRESETS == {"cautious": "history-config"}
            assert sys.argv == ["new-argv"]
            raise RuntimeError("simulator failure")
    assert gate_module.DepthPolicyGate is old_factory
    assert gate_module.PRESETS is old_presets
    assert sys.argv is old_argv
    assert sys.path == old_path


def test_existing_evidence_is_never_overwritten(tmp_path):
    path = tmp_path / "new.json"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        module.validate_paths(arguments(tmp_path))
    assert path.read_text() == "original"


def test_first_episode_events_include_terminal_step_but_not_auto_resets():
    class FakeGate:
        def __init__(self):
            self.target = torch.zeros(2, dtype=torch.bool)

        def step(self, *args, **kwargs):
            self.target.logical_not_()
            return {"target_v10": self.target.clone(), "switched": torch.ones(2, dtype=torch.bool),
                    "history_rough_fraction": torch.ones(2), "enter_span": torch.ones(2)}

        def reset(self, done_mask=None):
            self.target[done_mask] = False

    recorded = module.RecordedGate(FakeGate(), 2)
    recorded.step()
    recorded.reset(torch.tensor([True, False]))
    recorded.step()
    recorded.reset(torch.tensor([False, True]))
    recorded.step()
    assert [e["step"] for e in recorded.events[0]] == [1]
    assert [e["step"] for e in recorded.events[1]] == [1, 2]
    assert recorded.events[0][0]["history"]["history_rough_fraction"] == 1


def test_real_recorded_gate_supports_inference_step_and_ordinary_resets():
    from week03_ant.history_gate import HistoryDepthPolicyGate, HistoryGateConfig

    gate = module.RecordedGate(HistoryDepthPolicyGate(2, "cpu", 1 / 60, HistoryGateConfig()), 2)
    args = (torch.zeros(2, 825), torch.ones(2, 825, dtype=torch.bool),
            torch.zeros(2, 8), torch.ones(2, 8))
    with torch.inference_mode():
        gate.step(*args)
    gate.reset(torch.tensor([False, False]))
    gate.reset(torch.tensor([True, False]))
    assert gate.finished.tolist() == [True, False]
    with torch.inference_mode():
        gate.step(*args)
    gate.reset()
    assert gate.finished.all()


def test_completion_callback_runs_once_before_simulator_shutdown():
    from week03_ant.history_gate import HistoryDepthPolicyGate, HistoryGateConfig

    captured = []
    gate = module.RecordedGate(
        HistoryDepthPolicyGate(2, "cpu", 1 / 60, HistoryGateConfig()), 2,
        on_complete=lambda events: captured.append(events), max_steps=4,
    )
    args = (torch.zeros(2, 825), torch.ones(2, 825, dtype=torch.bool),
            torch.zeros(2, 8), torch.ones(2, 8))
    with torch.inference_mode():
        gate.step(*args)
    gate.reset(torch.tensor([True, False]))
    assert not captured
    with torch.inference_mode():
        gate.step(*args)
    gate.reset(torch.tensor([False, True]))
    assert captured == [[[], []]]
    with torch.inference_mode():
        gate.step(*args)
    gate.reset(torch.tensor([True, True]))
    assert len(captured) == 1


def test_zero_exit_without_backend_evidence_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "verify_originals", lambda: {"source_sha256": {}})
    monkeypatch.setattr(module, "sha", lambda path: "0" * 64)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(FileNotFoundError):
        module.main(["--geometry", "51", "--seed", "24", "--output", str(tmp_path / "final.json")])
    assert len(calls) == 1 and "--_worker" in calls[0]
    assert not (tmp_path / "final.json").exists()
