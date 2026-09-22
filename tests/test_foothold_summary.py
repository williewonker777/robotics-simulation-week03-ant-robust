import importlib.util
from pathlib import Path
import sys

import pytest


scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("foothold_summary", scripts / "summarize_foothold.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

runner_spec = importlib.util.spec_from_file_location("foothold_runner", scripts / "run_foothold_experiment.py")
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


def evidence():
    groups = {
        name: dict(n=100, one=80, six=40, falls=10, world=0, flat_falls=1, flat_n=10)
        for name in module.GROUPS
    }
    groups["guided"]["one"] = 81
    seeds = {
        str(seed): {"guided": {"one": 81}, "targets": {"one": 80}, "feet": {"one": 79}}
        for seed in (42, 43, 44)
    }
    return groups, seeds


def test_balanced_guided_improvement_passes():
    assert module.promotion_gate(*evidence())["passed"]


@pytest.mark.parametrize("baseline", module.GROUPS[:-1])
@pytest.mark.parametrize("metric,bad", [("one", 79), ("six", 39), ("falls", 11)])
def test_any_pooled_regression_fails(baseline, metric, bad):
    groups, seeds = evidence()
    groups[baseline][metric] = 80 if metric == "one" else 40 if metric == "six" else 10
    groups["guided"][metric] = bad
    result = module.promotion_gate(groups, seeds)
    assert not result["passed"]
    assert not result["checks"][f"{metric}_vs_{baseline}"]


def test_flat_falls_and_world_exit_are_hard_gates():
    groups, seeds = evidence()
    groups["guided"]["flat_falls"] = 2
    assert not module.promotion_gate(groups, seeds)["passed"]
    groups, seeds = evidence()
    groups["guided"]["world"] = 1
    assert not module.promotion_gate(groups, seeds)["passed"]


def test_strict_seed_majority_requires_guided_greater_than_targets():
    groups, seeds = evidence()
    seeds["42"]["guided"]["one"] = 80
    seeds["43"]["guided"]["one"] = 80
    result = module.promotion_gate(groups, seeds)
    assert not result["passed"]
    assert not result["checks"]["guided_one_better_than_targets_majority"]


def test_rates_allow_distinct_frozen_reference_denominator():
    groups, seeds = evidence()
    for key in ("n", "one", "six", "falls", "flat_n", "flat_falls"):
        groups["frozen_v5"][key] *= 3
    assert module.promotion_gate(groups, seeds)["passed"]


def test_fixed_experiment_contract():
    assert runner.MODES == ("feet", "targets", "guided")
    assert runner.TRAIN_PAIRS == ((42, 51), (43, 58), (44, 59))
    assert runner.EVAL_PAIRS == ((64, 38), (65, 39))
    assert runner.ITERATIONS == 750 and runner.NUM_ENVS == 4096 and runner.STEPS_PER_ENV == 32


def test_saved_config_validation_accepts_only_mode_specific_reward(tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    (params / "agent.yaml").write_text("""
seed: 42
num_steps_per_env: 32
max_iterations: 750
experiment_name: week03_ant_foothold_v9
policy:
  class_name: FootholdActorCritic
  input_mode: guided
  require_config_match: true
  actor_obs_normalization: false
  critic_obs_normalization: false
  actor_hidden_dims: [400, 200, 100]
  critic_hidden_dims: [400, 200, 100]
algorithm:
  learning_rate: 0.0001
  schedule: fixed
  gamma: 0.995
  lam: 0.95
  entropy_coef: 0.002
  num_learning_epochs: 5
  num_mini_batches: 4
""")
    (params / "env.yaml").write_text("""
seed: 42
scene:
  num_envs: 4096
  terrain:
    terrain_generator:
      seed: 51
rewards:
  foothold_support:
    weight: -1.0
""")
    checks = runner.validate_training_config(params, "guided", 42, 51)
    assert all(checks.values())
    with pytest.raises(ValueError, match="guided reward"):
        runner.validate_training_config(params, "targets", 42, 51)


def test_v7_v8_source_map_is_current_and_v9_sources_are_covered():
    hashes = runner.source_hashes()
    required = {
        "src/week03_ant/foothold_math.py", "src/week03_ant/foothold_policy.py",
        "src/week03_ant/tasks/foothold_v9_cfg.py", "scripts/foothold_v9.py",
        "scripts/probe_foothold.py", "scripts/run_foothold_experiment.py",
        "scripts/summarize_foothold.py",
    }
    assert required <= hashes.keys()
