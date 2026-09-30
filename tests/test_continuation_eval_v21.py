"""CPU validation of v21 binding and unchanged frozen evaluation mechanics."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from week03_ant import continuation_study_v21 as study

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = load('v21_summary', 'summarize_continuation_v21.py')
runner = load('v21_eval_runner', 'run_continuation_eval_v21.py')


@pytest.fixture
def models(monkeypatch):
    manifest = json.loads((ROOT / 'artifacts/terrain_demo/contact_curriculum_v20/frozen.json').read_text())['models']
    manifest['no_cost'] = dict(manifest['parent'], transitions=study.TRANSITIONS, iteration=249)
    monkeypatch.setattr(study, 'models', lambda: manifest)
    monkeypatch.setattr(runner, 'models', lambda: manifest)
    return manifest


def fixture(controller):
    reference = {'no_cost': 'parent', 'history_no_cost': 'history_parent'}.get(controller, controller)
    data = json.loads((ROOT / f'outputs/contact_curriculum_v20_20260928/smoke_{reference}.json').read_text())
    data.update(schema=study.SCHEMA, controller=controller, evaluation_plan_sha256=study.sha(study.PLAN),
                policy_training_transitions=0 if study.model_key(controller) == 'parent' else study.TRANSITIONS)
    return data


@pytest.mark.parametrize('controller', study.CONTROLLERS)
def test_all_conditioned_bindings_audit_copies_without_mutation(models, controller):
    data = fixture(controller)
    before = deepcopy(data)
    assert len(summary.audit(data)) == 35
    assert data == before


@pytest.mark.parametrize('key,value', [('schema', 'v19'), ('task', 'wrong'), ('controller', 'v5'),
    ('command_mode', None), ('command_schema_version', True), ('new_training_transitions', study.TRANSITIONS),
    ('policy_training_transitions', 1), ('evaluation_plan_sha256', '0' * 64),
    ('teacher_tensor_identity_verified', False), ('checkpoint_sha256', '0' * 64)])
def test_rejects_binding_corruption(models, key, value):
    data = fixture('ramped')
    data[key] = value
    with pytest.raises(ValueError):
        summary.audit(data)


@pytest.mark.parametrize('key', ['episode_strict_one_tile_success', 'episode_strict_all_tiles_success'])
def test_raw_success_is_recomputed(models, key):
    data = fixture('parent')
    data[key][0] = not data[key][0]
    with pytest.raises(ValueError):
        summary.audit(data)


def test_preparation_never_counts_as_scoring(models):
    data = fixture('parent')
    data['phase'] = 'prepare_development'
    with pytest.raises(ValueError, match='preparation'):
        summary.audit(data)


def test_actor_and_history_parent_comparisons_at_all_reporting_levels(models):
    groups = {name: summary.audit(fixture(name)) for name in study.CONTROLLERS}
    section = summary._section(groups, True)
    keys = {f'{a}_vs_{b}' for a, b in (*study.COMPARISONS, *study.PARENT_COMPARISONS)}
    assert set(section['comparisons']) == keys
    assert len(keys) == 12
    assert not section['comparisons']['no_cost_vs_parent']['passed']
    assert section['family']['flat']['comparisons'] == {}
    assert all(set(item['comparisons']) == keys for name, item in section['family'].items() if name != 'flat')
    assert all(set(item['comparisons']) == keys for item in section['level'].values())
    assert all(set(item['comparisons']) == keys for name, item in section['family_level'].items() if not name.startswith('flat/'))


def test_runner_builds_declared_matrix(models):
    outputs = set()
    for scenario in ('mixed', 'stones'):
        for controller in study.CONTROLLERS:
            for geometry, reset in study.HOLDOUTS:
                path = Path(f'/tmp/{scenario}/{controller}__geometry{geometry}_reset{reset}.json')
                command = runner.eval_command(controller, geometry, reset, scenario, 'holdout', path)
                assert command[command.index('--num_envs') + 1] == ('175' if scenario == 'mixed' else '10')
                assert command[command.index('--seconds') + 1] == ('16' if scenario == 'mixed' else '64')
                assert command[command.index('--device') + 1] == 'cuda:1'
                outputs.add(str(path))
    assert len(outputs) == 32


def test_rollout_loop_ast_exactly_matches_frozen_v20():
    def loop(path):
        tree = ast.parse(path.read_text())
        return next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'step')
    old = loop(ROOT / 'scripts/evaluate_curriculum_v20.py')
    new = loop(ROOT / 'scripts/evaluate_continuation_v21.py')
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


def test_no_overwrite_or_silent_retry(models, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'WORK', tmp_path)
    output = tmp_path / 'raw.json'
    output.write_text('{}')
    with pytest.raises(FileExistsError):
        runner.run(['python', '--output', str(output)], 'test')


def test_nonfinal_checkpoint_and_budget_substitution_rejected(models):
    for key, value in [('iteration', 248), ('transitions', 32)]:
        old = models['ramped'][key]
        models['ramped'][key] = value
        with pytest.raises(ValueError, match='nonfinal'):
            summary.audit(fixture('ramped'))
        models['ramped'][key] = old


def test_missing_trained_models_blocks_development(monkeypatch):
    monkeypatch.setattr(runner, 'models', lambda: {'parent': {}})
    with pytest.raises(ValueError, match='complete trained model'):
        runner.verify_model_inventory()


def test_rejects_previous_holdout_maps(models):
    data = fixture('parent')
    data.update(phase='holdout', geometry_seed=111, reset_seed=71, num_envs=175)
    with pytest.raises(ValueError, match='evaluation matrix'):
        summary.audit(data)


def test_development_references_all_six_reused_controllers_only():
    tree = ast.parse((ROOT / 'scripts/run_continuation_eval_v21.py').read_text())
    development = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'development')
    references = next(node.value for node in development.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'references' for target in node.targets))
    values = eval(compile(ast.Expression(body=references), '<reference-map>', 'eval'),
                  {'ROOT': ROOT, 'CONTROLLERS': study.CONTROLLERS})
    assert set(values) == set(study.CONTROLLERS) - {'no_cost', 'history_no_cost'}
    assert len(values) == 6 and all(path.is_file() for path in values.values())


def test_declared_main_reused_and_scored_budgets():
    assert study.ARMS == ('no_cost',)
    assert len(study.CONTROLLERS) == 8
    assert len(study.HOLDOUTS) == 2
    assert study.TRANSITIONS == 32768000
    assert len(study.CONTROLLERS) * len(study.HOLDOUTS) * (175 + 10) == 2960
    assert len((*study.COMPARISONS, *study.PARENT_COMPARISONS)) == 12
