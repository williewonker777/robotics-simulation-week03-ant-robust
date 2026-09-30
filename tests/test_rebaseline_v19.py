"""The new study must not select models or expand its predeclared evaluation set."""

import ast
import pytest

from week03_ant.rebaseline_study_v19 import (
    CONTROLLERS, HOLDOUTS, ROOT, command_mode, model_key, models, policy_mode,
    validate_request,
)


@pytest.mark.parametrize("controller,key,access,mode", [
    ("v5", "original", None, "v5"),
    ("history_original", "original", None, "hybrid"),
    ("v16_control", "control", "conditioned", "v10"),
    ("history_control", "control", "conditioned", "hybrid"),
])
def test_fixed_controller_binding(controller, key, access, mode):
    assert model_key(controller) == key
    assert command_mode(controller) == access
    assert policy_mode(controller) == mode
    assert models()[key]["checkpoint"].endswith(".pt")


@pytest.mark.parametrize("function", [model_key, command_mode, policy_mode])
def test_unknown_controller_fails_closed(function):
    with pytest.raises(ValueError):
        function("latest")


@pytest.mark.parametrize("controller", CONTROLLERS)
@pytest.mark.parametrize("pair", HOLDOUTS)
@pytest.mark.parametrize("scenario,seconds,envs", [("mixed", 16, 175), ("stones", 64, 10)])
def test_declared_holdout_matrix(controller, pair, scenario, seconds, envs):
    validate_request(controller, *pair, scenario, seconds, envs, "holdout")


@pytest.mark.parametrize("case", [
    ("v5", 51, 24, "mixed", 16, 175, "holdout"),
    ("v5", 107, 68, "mixed", 16, 35, "smoke"),
    ("v5", 107, 68, "mixed", 16, 175, "prepare"),
    ("v5", 107, 68, "stones", 16, 10, "holdout"),
    ("v5", 107, 69, "mixed", 16, 175, "holdout"),
    ("v5", 107, 68, "mixed", 16, 176, "holdout"),
    ("history_original", 107, 68, "stones", 64, 10, "prepare"),
    ("v5", 51, 24, "mixed", 16, 35, "train"),
])
def test_unplanned_requests_rejected(case):
    with pytest.raises(ValueError):
        validate_request(*case)


def test_only_initialization_for_preparation():
    source = (ROOT / "scripts/evaluate_rebaseline_v19.py").read_text()
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    preparation = next(node for node in main.body if isinstance(node, ast.Try)).body
    index = next(i for i, node in enumerate(preparation)
                 if isinstance(node, ast.If) and ast.unparse(node.test) == "args.phase in ('prepare', 'prepare_development')")
    branch = preparation[index]
    assert any(isinstance(node, ast.Return) for node in branch.body)
    assert "scored_episodes=0" in ast.unparse(branch)
    assert not any(isinstance(node, ast.For) for node in preparation[:index])
    assert "policy_mode(args.controller)" in source


def test_development_cache_preparation_is_separate_and_unscored():
    validate_request("history_original", 51, 24, "mixed", 16, 35, "prepare_development")
    with pytest.raises(ValueError):
        validate_request("v5", 51, 24, "mixed", 16, 35, "prepare_development")


def test_legacy_rollout_unchanged():
    """Physics/telemetry rollout is literally unchanged from the frozen evaluator."""
    old = (ROOT / "scripts/evaluate_style_v18.py").read_text()
    new = (ROOT / "scripts/evaluate_rebaseline_v19.py").read_text()
    def rollout(source):
        return source[source.index("        for step in range("):source.index("        save_json(args.output, result)")]
    assert rollout(old) == rollout(new)
