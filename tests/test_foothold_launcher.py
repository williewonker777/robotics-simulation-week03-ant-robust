import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("foothold_launcher", Path(__file__).resolve().parents[1] / "scripts/foothold_v9.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("mode,weight", [("feet", "0.0"), ("targets", "0.0"), ("guided", "-1.0")])
def test_training_mode_sets_shaping(mode, weight):
    result = module.training_arguments([f"agent.policy.input_mode={mode}"])
    assert f"env.rewards.foothold_support.weight={weight}" in result
    assert "agent.policy.require_config_match=true" in result


@pytest.mark.parametrize("args", [
    ["agent.policy.input_mode=guided", "env.rewards.foothold_support.weight=0"],
    ["agent.policy.input_mode=feet", "env.rewards.foothold_support.weight=-1"],
    ["agent.policy.input_mode=targets", "env.rewards.foothold_support.weight=nan"],
    ["agent.policy.input_mode=unknown"],
    ["agent.policy.input_mode=feet", "agent.policy.input_mode=guided"],
    ["agent.policy.require_config_match=false"],
    ["+agent.policy.require_config_match=false"],
])
def test_invalid_training_contract_rejected(args):
    with pytest.raises(ValueError):
        module.training_arguments(args)
