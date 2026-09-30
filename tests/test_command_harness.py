"""The bounded run matrix and complete saved-config comparison are CPU-testable."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml
from week03_ant.command_study import ROOT, training_parameters, TRAINING_SOURCES, SOURCE_FILES

sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("command_study_runner_test", ROOT / "scripts/run_command_study.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_paired_training_changes_only_arm_audit_runname_and_expected_initial():
    a = runner.train_command("masked", "a")
    b = runner.train_command("conditioned", "b")
    assert [(x, y) for x, y in zip(a, b) if x != y] == [
        ("masked", "conditioned"), ("a", "b"), ("v14_paired_masked", "v14_paired_conditioned"), ("v14_init_masked", "v14_init_conditioned")]
    for flag, value in (("--num_envs", "4096"), ("--max_iterations", "250"),
                        ("--seed", "46"), ("--checkpoint", "model_0.pt")):
        assert a[a.index(flag)+1] == value
    assert "env.scene.terrain.terrain_generator.seed=85" in a and "--resume" in a


def test_matrix_commands_keep_horizons_separate():
    for scenario, seconds, count in (("mixed", "16", "175"), ("stones", "64", "10")):
        cmd = runner.eval_command("conditioned", 86, 58, scenario, "holdout", "evidence", "checkpoint")
        assert cmd[cmd.index("--seconds")+1] == seconds
        assert cmd[cmd.index("--num_envs")+1] == count
        assert cmd[cmd.index("--checkpoint")+1] == "checkpoint"


def config_files(tmp_path, arm):
    path = tmp_path / arm
    (path / "params").mkdir(parents=True)
    env = {"log_dir": str(path), "io_descriptors_output_dir": str(path),
           "rewards": {"adaptive_posture": {"weight": 1.0}, "progress": {"weight": 1.}},
           "observations": {"size": 91}, "seed": 46}
    agent = {"run_name": arm, "load_run": f"v14_init_{arm}", "load_checkpoint": "model_0.pt", "learning_rate": .0001, "policy": {"command_mode": arm, "class_name": "CommandPriorActorCritic", "require_config_match": True}}
    for name, data in (("env", env), ("agent", agent)):
        (path / "params" / f"{name}.yaml").write_text(yaml.dump(data))
    return path, env, agent


def test_saved_config_normalizes_only_explicit_allowed_differences(tmp_path):
    a, _, _ = config_files(tmp_path, "masked")
    b, env, _ = config_files(tmp_path, "conditioned")
    assert training_parameters(a, "masked")["normalized"] == training_parameters(b, "conditioned")["normalized"]
    env["observations"]["size"] = 60
    (b / "params/env.yaml").write_text(yaml.dump(env))
    assert training_parameters(a, "masked")["normalized"] != training_parameters(b, "conditioned")["normalized"]
    with pytest.raises(ValueError, match="mode/class"):
        training_parameters(a, "conditioned")


def test_source_inventory_is_unique_and_training_is_subset():
    assert len(SOURCE_FILES) == len(set(SOURCE_FILES)) == 17
    assert len(TRAINING_SOURCES) == 14
    assert set(TRAINING_SOURCES) < set(SOURCE_FILES)
    for path in TRAINING_SOURCES:
        assert (ROOT / path).is_file()


def test_evaluation_cache_manifest_is_pinned_between_phases(tmp_path):
    from week03_ant.command_study import evaluation_inputs, save_json, sha

    save_json(tmp_path / "frozen.json", {"frozen": True})
    save_json(tmp_path / "terrain_cache.json", {"cache": "original"})
    save_json(tmp_path / "evaluation_inputs.json", {
        "experiment_freeze_sha256": sha(tmp_path / "frozen.json"),
        "terrain_cache_manifest_sha256": sha(tmp_path / "terrain_cache.json")})
    evaluation_inputs(tmp_path)
    (tmp_path / "terrain_cache.json").write_text('{"cache":"changed"}')
    with pytest.raises(ValueError, match="input manifest changed"):
        evaluation_inputs(tmp_path)
