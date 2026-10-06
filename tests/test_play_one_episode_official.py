"""CPU-only regression checks for official checkpoint loading."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "play_one_episode_official.py"


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1", "cpu"])
def test_checkpoint_load_maps_storages_to_environment_device(device):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "runner"
        and node.func.attr == "load"
    ]
    assert len(calls) == 1
    runner = Mock()
    env = SimpleNamespace(unwrapped=SimpleNamespace(device=device))

    # Execute only the loader call, without importing Isaac Sim or requiring CUDA.
    code = compile(ast.Expression(body=calls[0]), str(SCRIPT), "eval")
    eval(
        code,
        {"__builtins__": {}},
        {"runner": runner, "env": env, "resume_path": "model_599.pt"},
    )

    runner.load.assert_called_once_with("model_599.pt", map_location=device)
