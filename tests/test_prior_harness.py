"""Lock v10 treatment selection and the predeclared benchmark verdicts."""

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("prior_v10")
summary = load("summarize_prior")
runner = load("run_prior_experiment")


@pytest.mark.parametrize("mode,coef", [("free", "0.0"), ("anchored", "0.02")])
def test_launcher_locks_treatment(mode, coef):
    args = launcher.training_arguments([f"agent.policy.prior_mode={mode}"])
    assert f"agent.algorithm.teacher_coef={coef}" in args
    assert "agent.policy.input_mode=targets" in args
    assert "agent.policy.require_config_match=true" in args
    assert "env.rewards.foothold_support.weight=0.0" in args


@pytest.mark.parametrize("args", [
    ["agent.policy.prior_mode=free", "agent.algorithm.teacher_coef=.02"],
    ["agent.policy.prior_mode=anchored", "agent.algorithm.teacher_coef=0"],
    ["agent.policy.prior_mode=bad"],
    ["agent.policy.prior_mode=free", "agent.policy.prior_mode=anchored"],
    ["agent.policy.input_mode=feet"],
    ["env.rewards.foothold_support.weight=-1"],
    ["env.rewards.foothold_support.weight=nan"],
    ["agent.policy.require_config_match=false"],
    ["+agent.policy.require_config_match=false"],
    ["agent.algorithm.class_name=PPO"],
    ["agent.policy.class_name=ActorCritic"],
])
def test_launcher_rejects_ambiguous_or_wrong_treatment(args):
    with pytest.raises(ValueError):
        launcher.training_arguments(args)


def evidence():
    groups = {name: dict(n=100, one=80, six=40, falls=10, lane=3, world=0, flat_falls=1, flat_n=10)
              for name in summary.GROUPS}
    groups["free"].update(falls=12, lane=6)
    seeds = {str(seed): {"free": dict(falls=4, lane=2), "anchored": dict(falls=3, lane=1)}
             for seed in (42, 43, 44)}
    return groups, seeds


def test_balanced_improvement_passes_both_gates():
    groups, seeds = evidence()
    assert summary.promotion_gate(groups, seeds)["passed"]
    assert summary.directional_gate(groups, seeds)["passed"]


@pytest.mark.parametrize("baseline", ("free", "frozen_v5"))
@pytest.mark.parametrize("metric,value", [("one", 79), ("six", 39), ("falls", 13), ("lane", 7)])
def test_any_comparator_regression_blocks_promotion(baseline, metric, value):
    groups, seeds = evidence()
    groups["anchored"][metric] = value
    gate = summary.promotion_gate(groups, seeds)
    assert not gate["passed"] and not gate["checks"][f"{metric}_vs_{baseline}"]


def test_directional_tradeoff_does_not_imply_promotion():
    groups, seeds = evidence()
    groups["free"]["six"] = 41
    assert summary.directional_gate(groups, seeds)["passed"]
    assert not summary.promotion_gate(groups, seeds)["passed"]


def test_directional_gate_requires_joint_safety_improvement_in_same_majority():
    groups, seeds = evidence()
    seeds["42"]["anchored"]["falls"] = 4
    seeds["43"]["anchored"]["lane"] = 2
    assert not summary.directional_gate(groups, seeds)["passed"]
    # Fall improvement still has a majority, so strict promotion can stand alone.
    assert summary.promotion_gate(groups, seeds)["passed"]


def test_reference_rates_not_replicated_counts():
    groups, seeds = evidence()
    for key in ("n", "one", "six", "falls", "lane", "flat_n", "flat_falls"):
        groups["frozen_v5"][key] *= 3
    assert summary.promotion_gate(groups, seeds)["passed"]
    assert summary.directional_gate(groups, seeds)["passed"]


def test_world_and_flat_fall_guards():
    groups, seeds = evidence()
    groups["anchored"]["world"] = 1
    assert not summary.promotion_gate(groups, seeds)["passed"]
    assert not summary.directional_gate(groups, seeds)["passed"]
    groups["anchored"].update(world=0, flat_falls=2)
    assert not summary.promotion_gate(groups, seeds)["passed"]


def test_predeclared_budget_and_unused_holdouts():
    assert runner.MODES == ("free", "anchored")
    assert runner.TRAIN_PAIRS == ((42, 51), (43, 58), (44, 59))
    assert runner.EVAL_PAIRS == ((66, 40), (67, 41))
    assert (runner.ITERATIONS, runner.NUM_ENVS, runner.STEPS_PER_ENV) == (750, 4096, 32)
    assert runner.upstream_provenance()["rsl_rl_version"] == "3.0.1"


def test_actual_archived_config_requires_only_declared_v10_differences(tmp_path):
    old = runner.ROOT / "artifacts/terrain_demo/foothold_v9/runs/v9_targets_seed42/params"
    agent = yaml.load((old / "agent.yaml").read_text(), Loader=yaml.BaseLoader)
    env = yaml.load((old / "env.yaml").read_text(), Loader=yaml.BaseLoader)
    agent["experiment_name"] = "week03_ant_prior_v10"
    agent["policy"].update(class_name="PriorActorCritic", input_mode="targets", prior_mode="anchored")
    agent["algorithm"].update(class_name="PriorPPO", teacher_coef="0.02")
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(agent))
    (tmp_path / "env.yaml").write_text(yaml.safe_dump(env))
    assert all(runner.validate_training_config(tmp_path, "anchored", 42, 51).values())
    with pytest.raises(ValueError, match="prior mode|teacher coefficient"):
        runner.validate_training_config(tmp_path, "free", 42, 51)
    env["rewards"]["foothold_support"]["weight"] = "-1.0"
    (tmp_path / "env.yaml").write_text(yaml.safe_dump(env))
    with pytest.raises(ValueError, match="support reward|unchanged rewards"):
        runner.validate_training_config(tmp_path, "anchored", 42, 51)


def test_horizon_aggregation_rechecks_terminal_and_censored_evidence():
    horizon = load("summarize_prior_horizon")
    data = dict(
        schema="week03_ant_prior_v10_horizon_v1", task="Week03-Ant-Prior-Lanes-Eval-v10",
        num_envs=10, policy_observation_dimensions=88,
        condition=dict(terrain_family="stepping_stones", terrain_level_index=4, terrain_difficulty=1.,
                       episode_length_seconds=64., max_steps=3840, distance_snapshot_seconds=16.,
                       one_tile_clearance_m=13.1, all_tiles_clearance_m=53.1, step_dt_seconds=1 / 60),
        scope=dict(promotion_metric=False, primary_175_env_assignment_reused=False),
        forward_distance=[54.] * 10, maximum_distance_m=[55.] * 10, episode_lengths=[3840] * 10,
        episode_terminated=[False] * 10, episode_out_of_lane=[False] * 10, episode_world_exit=[False] * 10,
        first_hit_one_seconds=[10.] * 10, first_hit_six_seconds=[63.] * 10, distance_at_16s=[15.] * 10,
        episode_full_64s_survival=[True] * 10, episode_strict_one_tile_success=[True] * 10,
        episode_strict_all_tiles_success=[True] * 10,
        aggregates=dict(strict_one_tile_successes=10, strict_all_tiles_successes=10,
                        posture_terminations=0, out_of_lane_exits=0, world_exits=0, full_64s_survivals=10),
    )
    assert horizon.aggregate(horizon.audit(data))["six"] == 10
    data["episode_terminated"][0] = True
    with pytest.raises(ValueError, match="strict success"):
        horizon.audit(data)
    data["episode_terminated"][0] = False
    data["distance_at_16s"][0] = None
    with pytest.raises(ValueError, match="censoring"):
        horizon.audit(data)


def test_horizon_requires_complete_primary_audit(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(runner, "ARTIFACT", tmp_path)
    with pytest.raises(ValueError, match="audit primary"):
        runner.evaluate("cpu", horizon=True)
    (tmp_path / "verification.json").write_text(json.dumps({"status": "pass", "total_primary_first_episodes": 175}))
    with pytest.raises(ValueError, match="complete primary audit"):
        runner.evaluate("cpu", horizon=True)
