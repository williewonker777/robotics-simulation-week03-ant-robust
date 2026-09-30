"""v16 paired-budget, provenance, and saved-config guards without a simulator."""

from __future__ import annotations

import importlib.util
import sys

import pytest
import yaml

from week03_ant.contact_study import (
    ARMS, CONTROLLERS, HOLDOUTS, ROOT, TRAINING_GEOMETRY, TRAINING_SEED, TRAINING_SOURCES,
    init_path, training_parameters,
)

sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("contact_runner_test", ROOT / "scripts/run_contact_study.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_shared_init_and_paired_training_budget():
    control, slip = [runner.train_command(arm, f"{arm}_audit") for arm in ARMS]
    assert [(a, b) for a, b in zip(control, slip) if a != b] == [
        ("control", "slip"), ("control_audit", "slip_audit"),
        ("v16_paired_control", "v16_paired_slip"),
    ]
    assert init_path("control") == init_path("slip") == init_path()
    for flag, expected in (("--num_envs", "4096"), ("--max_iterations", "250"), ("--seed", str(TRAINING_SEED)),
                           ("--load_run", "v16_init"), ("--checkpoint", "model_0.pt")):
        assert control[control.index(flag) + 1] == expected
    assert f"env.scene.terrain.terrain_generator.seed={TRAINING_GEOMETRY}" in control


def test_controller_matrix_is_v16_specific():
    assert HOLDOUTS == ((99, 62), (100, 63))
    assert CONTROLLERS == ("v15_control", "control", "slip", "history_original", "history_control", "history_slip")


def _config_files(tmp_path, arm):
    path = tmp_path / arm
    (path / "params").mkdir(parents=True)
    env = {"log_dir": str(path), "io_descriptors_output_dir": str(path), "seed": 48,
           "rewards": {"adaptive_posture": {"weight": 1.0}, "contact_slip": {"weight": float(arm == "slip")},
                       "directional_stability": {"weight": 0.0}, "progress": {"weight": 1.0}}}
    agent = {"run_name": arm, "load_run": "v16_init", "load_checkpoint": "model_0.pt", "learning_rate": .0001,
             "policy": {"command_mode": "conditioned", "class_name": "CommandPriorActorCritic",
                        "require_config_match": True}}
    for name, value in (("env", env), ("agent", agent)):
        (path / "params" / f"{name}.yaml").write_text(yaml.dump(value))
    return path, env, agent


def test_saved_configs_normalize_only_contact_treatment(tmp_path):
    control, _, _ = _config_files(tmp_path, "control")
    slip, env, _ = _config_files(tmp_path, "slip")
    assert training_parameters(control, "control")["normalized"] == training_parameters(slip, "slip")["normalized"]
    env["rewards"]["directional_stability"]["weight"] = 1.
    (slip / "params/env.yaml").write_text(yaml.dump(env))
    with pytest.raises(ValueError, match="directional"):
        training_parameters(slip, "slip")


def test_training_manifest_keeps_task_and_plan_hashes():
    assert len(TRAINING_SOURCES) == len(set(TRAINING_SOURCES))
    assert "src/week03_ant/tasks/contact_v16.py" in TRAINING_SOURCES
    assert "outputs/contact_slip_v16_20260923/PLAN.md" in TRAINING_SOURCES


def test_post_action_sampler_masks_done_wrap_rows_and_aggregates_rollout(monkeypatch):
    import torch

    class Raw:
        num_envs = 4
        device = "cpu"

    class VectorEnv:
        def step(self, actions):
            return (actions, None, torch.tensor([False, True, False, False]), None)

    calls = []
    def metrics(env, term, mask):
        calls.append((env, term, mask.clone()))
        return {"sampled_rows": 2, "valid_rows": 2, "invalid_rows": 0,
                "reward": -.2, "bounded_cost": .2, "contact_fraction": .5, "no_contact_fraction": .5,
                "contacted_tip_speed": .3, "invalid_fraction": 0., "terrain_contact_fraction": .2,
                "flat_contact_fraction": .3, "per_foot_contact_fraction": [.1, .2, .3, .4],
                "per_target_per_foot_contact_fraction": [[.1, .2], [.3, .4], [.5, .6], [.7, .8]]}
    launcher_spec = importlib.util.spec_from_file_location("contact_launcher_sampler_test", ROOT / "scripts/contact_v16.py")
    launcher = importlib.util.module_from_spec(launcher_spec)
    launcher_spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "contact_geometry", metrics)
    sampler = launcher.ContactSamplingEnv(VectorEnv(), Raw(), wrap_count_getter=lambda: torch.zeros(4, dtype=torch.long))
    sampler._term = object()  # Unit test the passive proxy without Isaac Lab imports.
    wrap = iter((torch.tensor([False, False, True, False]), torch.zeros(4, dtype=torch.bool)))
    monkeypatch.setattr(sampler, "_wrapped_rows", lambda: next(wrap))
    sampler.step("a")
    sampler.step("b")
    summary = sampler.summary()
    assert len(calls) == 2 and summary["samples"] == 2
    assert summary["contact_fraction"] == .5
    assert summary["per_foot_contact_fraction"] == [.1, .2, .3, .4]
    assert summary["excluded_done_rows"] == 2 and summary["excluded_wrap_rows"] == 1
    assert calls[0][2].tolist() == [True, False, False, True]


def _complete_rollout(samples=64):
    return {
        "samples": samples, "sampled_rows": 128, "valid_rows": 128, "invalid_rows": 0,
        "invalid_fraction": 0., "contact_fraction": .25,
        "terrain_contact_fraction": .1, "flat_contact_fraction": .1,
        "per_foot_contact_fraction": [.1, .1, .1, .1],
        "per_target_per_foot_contact_fraction": [[.1, .1], [.1, .1], [.1, .1], [.1, .1]],
    }


def test_rollout_gate_requires_exact_post_action_count_and_target_foot_coverage():
    runner.validate_rollout(_complete_rollout(), 64, "preflight")
    with pytest.raises(ValueError, match="sample count"):
        runner.validate_rollout(_complete_rollout(63), 64, "preflight")
    missing = _complete_rollout()
    missing["per_target_per_foot_contact_fraction"][3][1] = 0.
    with pytest.raises(ValueError, match="every foot on both targets"):
        runner.validate_rollout(missing, 64, "preflight")


def test_full_source_manifest_includes_eval_harness():
    from week03_ant.contact_study import SOURCE_FILES

    assert "scripts/run_contact_eval.py" in SOURCE_FILES
    assert "tests/test_contact_eval.py" in SOURCE_FILES
