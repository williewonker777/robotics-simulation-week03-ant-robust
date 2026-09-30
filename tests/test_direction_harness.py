"""Predeclared paired budget, saved configuration, and evidence manifest guards."""
import importlib.util
import sys

import pytest
import yaml
from week03_ant.direction_study import ROOT, ARMS, HOLDOUTS, CONTROLLERS, TRAINING_SOURCES, SOURCE_FILES, training_parameters, init_path, command_mode

sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("direction_runner_test", ROOT / "scripts/run_direction_study.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_shared_init_and_paired_training_budget():
    a, b = [runner.train_command(arm, arm + '_audit') for arm in ARMS]
    assert [(x, y) for x, y in zip(a, b) if x != y] == [
        ("control", "stable"), ("control_audit", "stable_audit"), ("v15_paired_control", "v15_paired_stable")]
    assert init_path("control") == init_path("stable") == init_path()
    for flag, expected in (("--num_envs", "4096"), ("--max_iterations", "250"), ("--seed", "47"),
                           ("--load_run", "v15_init"), ("--checkpoint", "model_0.pt")):
        assert a[a.index(flag) + 1] == expected
    assert "env.scene.terrain.terrain_generator.seed=95" in a and "--resume" in a


def test_fresh_fixed_matrix_and_observation_modes():
    assert HOLDOUTS == ((96, 60), (97, 61)) and len(CONTROLLERS) == 6
    for controller in CONTROLLERS:
        assert command_mode(controller) == (None if controller == "history_original" else "conditioned")
    for scenario, seconds, count in (("mixed", "16", "175"), ("stones", "64", "10")):
        cmd = runner.eval_command("stable", 96, 60, scenario, "holdout", "evidence", "checkpoint")
        assert cmd[cmd.index("--seconds") + 1] == seconds
        assert cmd[cmd.index("--num_envs") + 1] == count


def config_files(tmp_path, arm):
    path = tmp_path / arm
    (path / "params").mkdir(parents=True)
    env = {"log_dir": str(path), "io_descriptors_output_dir": str(path),
           "rewards": {"adaptive_posture": {"weight": 1.0},
                       "directional_stability": {"weight": float(arm == 'stable')}, "progress": {"weight": 1.}},
           "observations": {"size": 91}, "seed": 47}
    agent = {"run_name": arm, "load_run": "v15_init", "load_checkpoint": "model_0.pt",
             "learning_rate": .0001, "policy": {"command_mode": "conditioned", "class_name": "CommandPriorActorCritic", "require_config_match": True}}
    for name, data in (("env", env), ("agent", agent)):
        (path / "params" / f"{name}.yaml").write_text(yaml.dump(data))
    return path, env, agent


def test_only_treatment_and_paths_are_normalized(tmp_path):
    a, _, _ = config_files(tmp_path, "control")
    b, env, _ = config_files(tmp_path, "stable")
    assert training_parameters(a, 'control')['normalized'] == training_parameters(b, 'stable')['normalized']
    env['observations']['size'] = 88
    (b / 'params/env.yaml').write_text(yaml.dump(env))
    assert training_parameters(a, 'control')['normalized'] != training_parameters(b, 'stable')['normalized']
    with pytest.raises(ValueError, match='reward arm'):
        training_parameters(a, 'stable')


@pytest.mark.parametrize('field,value', [('command_mode', 'masked'), ('class_name', 'PriorActorCritic'), ('require_config_match', False)])
def test_saved_policy_guards(tmp_path, field, value):
    path, _, agent = config_files(tmp_path, 'control')
    agent['policy'][field] = value
    (path / 'params/agent.yaml').write_text(yaml.dump(agent))
    with pytest.raises(ValueError, match='mode/class'):
        training_parameters(path, 'control')


def test_source_inventory_is_unique_complete_and_training_subset():
    assert len(SOURCE_FILES) == len(set(SOURCE_FILES)) == 17
    assert len(TRAINING_SOURCES) == 12
    assert set(TRAINING_SOURCES) < set(SOURCE_FILES)
    for name in TRAINING_SOURCES:
        assert (ROOT / name).is_file()


def test_inputs_pin_append_only_preholdout_ledger(tmp_path, monkeypatch):
    import week03_ant.direction_study as study
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    ledger = tmp_path / 'commands.jsonl'
    ledger.write_text('{"phase":"preparation"}\n')
    study.save_json(tmp_path / 'frozen.json', {'freeze': True})
    study.save_json(tmp_path / 'terrain_cache.json', {'cache': True})
    study.save_json(tmp_path / 'preholdout_ledger.json', {'path': 'commands.jsonl',
        'bytes': ledger.stat().st_size, 'lines': 1, 'sha256': study.sha(ledger)})
    study.save_json(tmp_path / 'evaluation_inputs.json', {
        'experiment_freeze_sha256': study.sha(tmp_path / 'frozen.json'),
        'terrain_cache_manifest_sha256': study.sha(tmp_path / 'terrain_cache.json'),
        'preholdout_ledger_sha256': study.sha(tmp_path / 'preholdout_ledger.json')})
    study.evaluation_inputs(tmp_path)
    with ledger.open('a') as stream:
        stream.write('{"phase":"evaluation"}\n')
    study.evaluation_inputs(tmp_path)
    ledger.write_text(ledger.read_text().replace('preparation', 'preparati0n'))
    with pytest.raises(ValueError, match='prefix changed'):
        study.evaluation_inputs(tmp_path)


def test_inputs_reject_substituted_ledger_path(tmp_path, monkeypatch):
    import week03_ant.direction_study as study
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    for name in ('frozen.json', 'terrain_cache.json'):
        study.save_json(tmp_path / name, {})
    study.save_json(tmp_path / 'preholdout_ledger.json', {'path': 'other.jsonl', 'bytes': 2, 'lines': 1, 'sha256': '0' * 64})
    study.save_json(tmp_path / 'evaluation_inputs.json', {
        'experiment_freeze_sha256': study.sha(tmp_path / 'frozen.json'),
        'terrain_cache_manifest_sha256': study.sha(tmp_path / 'terrain_cache.json'),
        'preholdout_ledger_sha256': study.sha(tmp_path / 'preholdout_ledger.json')})
    with pytest.raises(ValueError, match='ledger path'):
        study.evaluation_inputs(tmp_path)
