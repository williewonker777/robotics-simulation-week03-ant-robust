"""CPU validation of v22 binding and unchanged frozen evaluation mechanics."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from week03_ant import seed_study_v22 as study

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = load('v22_summary', 'summarize_seed_v22.py')
runner = load('v22_eval_runner', 'run_seed_eval_v22.py')


@pytest.fixture
def models(monkeypatch):
    parent = json.loads((study.V21_ART / 'frozen.json').read_text())['models']['parent']
    manifest = {'parent': parent, **{run: dict(parent, transitions=study.TRANSITIONS,
                iteration=249, training_seed=study.training_seed(run)) for run in study.RUNS}}
    monkeypatch.setattr(study, 'models', lambda: manifest)
    monkeypatch.setattr(runner, 'models', lambda: manifest)
    return manifest


def fixture(controller):
    reference = 'history_parent' if controller.startswith('history_') else 'parent'
    data = json.loads((ROOT / f'outputs/contact_continuation_v21_20260928/smoke_{reference}.json').read_text())
    key = study.model_key(controller)
    data.update(schema=study.SCHEMA, controller=controller, evaluation_plan_sha256=study.sha(study.PLAN),
                training_seed=None if key == 'parent' else study.training_seed(key),
                policy_training_transitions=0 if key == 'parent' else study.TRANSITIONS)
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
    data = fixture('seed53')
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
    keys = {f'{a}_vs_{b}' for a, b in study.COMPARISONS}
    assert set(section['comparisons']) == keys
    assert len(keys) == 12
    assert not section['comparisons']['seed51_vs_parent']['passed']
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


def test_rollout_loop_ast_exactly_matches_frozen_v21():
    def loop(path):
        tree = ast.parse(path.read_text())
        return next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'step')
    old = loop(ROOT / 'scripts/evaluate_continuation_v21.py')
    new = loop(ROOT / 'scripts/evaluate_seed_v22.py')
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


def test_no_overwrite_or_silent_retry(models, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'WORK', tmp_path)
    output = tmp_path / 'raw.json'
    output.write_text('{}')
    with pytest.raises(FileExistsError):
        runner.run(['python', '--output', str(output)], 'test')


def test_nonfinal_checkpoint_and_budget_substitution_rejected(models):
    for key, value in [('iteration', 248), ('transitions', 32)]:
        old = models['seed53'][key]
        models['seed53'][key] = value
        with pytest.raises(ValueError, match='nonfinal'):
            summary.audit(fixture('seed53'))
        models['seed53'][key] = old


def test_missing_trained_models_blocks_development(monkeypatch):
    monkeypatch.setattr(runner, 'models', lambda: {'parent': {}})
    with pytest.raises(ValueError, match='complete trained model'):
        runner.verify_model_inventory()


def test_rejects_previous_holdout_maps(models):
    data = fixture('parent')
    data.update(phase='holdout', geometry_seed=111, reset_seed=71, num_envs=175)
    with pytest.raises(ValueError, match='evaluation matrix'):
        summary.audit(data)


def test_development_references_four_exact_replays_only():
    tree = ast.parse((ROOT / 'scripts/run_seed_eval_v22.py').read_text())
    development = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'development')
    references = next(node.value for node in development.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'references' for target in node.targets))
    values = eval(compile(ast.Expression(body=references), '<reference-map>', 'eval'),
                  {'ROOT': ROOT, 'CONTROLLERS': study.CONTROLLERS})
    assert set(values) == {'parent', 'seed51', 'history_parent', 'history_seed51'}
    assert len(values) == 4 and all(path.is_file() for path in values.values())


def test_declared_main_reused_and_scored_budgets():
    assert study.TRAIN_SEEDS == (51, 52, 53)
    assert study.TOTAL_TRANSITIONS == 98304000
    assert study.DEVELOPMENT_TRANSITIONS == 835584
    assert len(study.CONTROLLERS) == 8
    assert len(study.HOLDOUTS) == 2
    assert study.TRANSITIONS == 32768000
    assert len(study.CONTROLLERS) * len(study.HOLDOUTS) * (175 + 10) == 2960
    assert len(study.COMPARISONS) == 12


@pytest.mark.parametrize('seed', [None, 52, True, '51'])
def test_raw_seed_identity_is_not_normalized(models, seed):
    raw = fixture('seed51')
    raw['training_seed'] = seed
    with pytest.raises(ValueError, match='seed'):
        summary.audit(raw)


def test_descriptive_ranges_keep_parent_once(models):
    groups = {name: summary.audit(fixture(name)) for name in study.CONTROLLERS}
    section = summary._section(groups, True)
    for report in (section, *section['family'].values(), *section['family_level'].values(), *section['level'].values()):
        for prefix, label in (('', 'actor'), ('history_', 'history')):
            ranges = report['seed_ranges'][label]
            assert ranges['training_seeds'] == [51, 52, 53]
            assert ranges['parent_counted_once'] is True
            assert ranges['rough_episodes_per_seed'] == report['groups'][prefix + 'parent']['n']
            for metric in ranges['metrics'].values():
                assert set(metric['values']) == set(study.RUNS)
                assert metric['range'] in (None, 0)


def test_seed_ranges_preserve_nonzero_spread_and_deltas(models):
    aggregates = {name: summary.aggregate(summary.audit(fixture(name))) for name in study.CONTROLLERS}
    for run, value in zip(study.RUNS, [1, 3, 2]):
        aggregates[run]['six'] = value
    actual = summary.seed_ranges(aggregates)['actor']['metrics']['six']
    assert (actual['minimum'], actual['maximum'], actual['range']) == (1, 3, 2)
    assert actual['parent_deltas']['seed52'] == 3 - aggregates['parent']['six']


@pytest.mark.parametrize('run', ['seed52', 'seed53'])
@pytest.mark.parametrize('corruption', ['initial', 'horizons', 'hash'])
def test_main_cannot_trust_pairing_flags(tmp_path, monkeypatch, run, corruption):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    directory = tmp_path / 'art'
    (directory / 'expected_main').mkdir(parents=True)
    expected = directory / 'expected_main' / f'{run}_initial.json'
    expected.write_text(json.dumps({'identity': run}))
    schedule = tmp_path / 'capacity_schedule.json'
    schedule.write_text(json.dumps({'episode_randomization': {'actual_sha256': run}}))
    frozen = {'expected_main_sha256': {str(expected.relative_to(tmp_path)): study.sha(expected)}}
    capacity = {'records': {run: {'schedule': {'path': schedule.name, 'sha256': study.sha(schedule)}}}}
    data = {'initial': {'identity': run}, 'schedule': {'episode_randomization': {'actual_sha256': run}}}
    def compare(left, right):
        if left != right:
            raise ValueError('initial differs')
    monkeypatch.setattr(study, 'check_initial', compare)
    summary.audit_same_seed_pairing(directory, run, data, frozen, capacity)
    if corruption == 'initial':
        data['initial']['identity'] = 'seed51'
    elif corruption == 'horizons':
        data['schedule']['episode_randomization']['actual_sha256'] = 'seed51'
    else:
        schedule.write_text('{}')
    with pytest.raises(ValueError):
        summary.audit_same_seed_pairing(directory, run, data, frozen, capacity)


@pytest.mark.parametrize('corruption', ['historical_source', 'historical_log', 'aliased_source'])
def test_equal_hash_cannot_replace_fresh_distinct_training_source(tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    fresh = tmp_path / 'logs' / 'rsl_rl' / study.EXPERIMENT
    old = tmp_path / 'logs' / 'rsl_rl' / 'week03_ant_contact_continuation_v21'
    checkpoints = {}
    for name, directory in [('seed51', fresh / 'seed51'), ('seed52', fresh / 'seed52'), ('old', old / 'no_cost')]:
        directory.mkdir(parents=True)
        checkpoint = directory / 'model_249.pt'
        checkpoint.write_bytes(b'identical checkpoint bytes')
        checkpoints[name] = checkpoint
    assert len({study.sha(path) for path in checkpoints.values()}) == 1
    def model(name):
        return {'source_checkpoint': str(checkpoints[name].relative_to(tmp_path)),
                'sha256': study.sha(checkpoints[name])}
    def initial(name):
        return {'log_dir': str(checkpoints[name].parent.relative_to(tmp_path))}
    sources = set()
    summary.audit_fresh_source(model('seed51'), initial('seed51'), sources)
    summary.audit_fresh_source(model('seed52'), initial('seed52'), sources)
    if corruption == 'historical_source':
        candidate, audit, prior = model('old'), initial('seed51'), set()
    elif corruption == 'historical_log':
        candidate, audit, prior = model('old'), initial('old'), set()
    else:
        candidate, audit, prior = model('seed51'), initial('seed51'), sources
    with pytest.raises(ValueError, match='fresh training source'):
        summary.audit_fresh_source(candidate, audit, prior)
