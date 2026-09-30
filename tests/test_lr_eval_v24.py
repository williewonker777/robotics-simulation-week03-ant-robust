"""CPU regressions for fixed budgets, physical audits and dependent-window accounting."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
from week03_ant import lr_study_v24 as study

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_lr_eval_v24 as runner
import summarize_lr_v24 as summary
from summarize_lr_v24 import seed_ranges
from summarize_lp_v17 import PARITY_FIELDS


from week03_ant import paired_horizon_study_v23 as historical


@pytest.fixture(autouse=True)
def model_registry(monkeypatch):
    models = historical.models()
    for seed in (51, 52, 53):
        models[f'low{seed}'] = dict(models[f'seed{seed}'],
            checkpoint=f'artifacts/terrain_demo/lr_continuation_v24/runs/low{seed}/model_249.pt',
            sha256=str(seed) * 32, learning_rate=1.e-5, lr_condition='low')
    monkeypatch.setattr(study, 'models', lambda: models)
    return models


def raw(controller='parent'):
    old_controller = controller.replace('low', 'seed')
    references = json.loads((historical.ART / 'development.json').read_text())['records']
    entry = next(r for r in references if r['controller'] == old_controller)
    data = json.loads((ROOT / entry['path']).read_text())['windows']['16']
    key = study.model_key(controller)
    # Historical high controllers are valid only on the new holdout matrix.
    high = key.startswith('seed')
    data.update(schema=study.SCHEMA, evaluation_plan_sha256=study.sha(study.PLAN), controller=controller,
                checkpoint=study.models()[key]['checkpoint'], checkpoint_sha256=study.models()[key]['sha256'],
                lr_condition='not_applicable' if key == 'parent' else ('low' if key.startswith('low') else 'high'),
                model_learning_rate=None if key == 'parent' else (1.e-5 if key.startswith('low') else 1.e-4))
    if high:
        # For 35-row synthetic provenance tests bypass only the request-population check.
        data['_test_high'] = True
    return data


@pytest.mark.parametrize('controller', study.DEVELOPMENT_CONTROLLERS)
def test_all_old_development_windows_audit_without_mutating(controller):
    data = raw(controller)
    before = deepcopy(data)
    assert len(summary.audit(data)) == 35
    assert data == before


def test_64_mixed_physical_auditor_not_old_stones_router():
    data = raw()
    data['condition'].update(seconds=64, max_steps=3840, snapshot_seconds=16)
    data['episode_full_horizon_survival'] = [False] * 35
    data['distance_at_snapshot_m'] = [None] * 35
    assert len(summary.audit(data)) == 35
    assert data['scenario'] == 'mixed' and data['condition']['seconds'] == 64


@pytest.mark.parametrize('field', PARITY_FIELDS)
def test_all_35_parity_fields_are_mandatory(field):
    a = {k: 0 for k in PARITY_FIELDS}
    b = deepcopy(a)
    b[field] = 1
    with pytest.raises(ValueError, match='35-field'):
        runner.parity(a, b)


def test_parity_field_inventory_frozen():
    assert len(PARITY_FIELDS) == 35
    assert 'condition' in PARITY_FIELDS and 'posture_telemetry' in PARITY_FIELDS
    assert len(study.COMPARISONS) == 18
    assert study.frozen_budget()['first_episodes'] == 4900
    assert study.frozen_budget()['window_observations'] == 9800


@pytest.mark.parametrize('field,value', [('scenario', 'stones'), ('new_training_transitions', 1), ('mode', 'v5'), ('teacher_tensor_identity_verified', False)])
def test_window_binding_corruption(field, value):
    data = raw()
    data[field] = value
    with pytest.raises(ValueError):
        summary.audit(data)


def test_strict_success_uses_final_not_maximum():
    data = raw()
    i = next(i for i, v in enumerate(data['episode_strict_one_tile_success']) if v)
    data['forward_distance'][i] = 0.
    with pytest.raises(ValueError, match='strict success'):
        summary.audit(data)


def row(six, **kwargs):
    return dict(family='rough', level=0, six=six, falls=0, lane=0, world=0, distance=60., first_six_hit=None, **kwargs)


def test_four_cells_and_overlapping_late_loss_not_hit_success():
    short = [row(0), row(0), row(1), row(1)]
    long = [row(0), row(1), row(0), row(1)]
    long[2].update(falls=1, lane=1, world=1, distance=52., first_six_hit=12.)
    long[0]['first_six_hit'] = 32.
    result = summary.duration_pairs(short, long)
    assert result['strict_six'] == dict(neither=1, late_gain=1, late_loss=1, both=1)
    assert set(result['late_loss_flags'].values()) == {1}
    assert result['first_six_hit_in_16_64'] == 1


def test_seed_ranges_do_not_triple_parent():
    metrics = dict(n=300, flat_n=50, one=1, six=2, falls=3, lane=4, world=0, flat_falls=0, flat_lane=0, flat_world=0, flat_mean_episode_speed=1.)
    aggregates = {c: dict(metrics) for c in study.CONTROLLERS}
    aggregates['seed52']['six'] = 5
    result = seed_ranges(aggregates)['actor_high']
    assert result['rough_episodes_per_seed'] == 300 and result['parent_counted_once']
    assert result['metrics']['six']['range'] == 3
    assert result['metrics']['six']['parent_deltas']['seed52'] == 3


def test_command_matrix_no_holdout_reference_or_stones():
    for controller in study.CONTROLLERS:
        command = runner.eval_command(controller, 119, 79, 'holdout', Path('raw.json'))
        assert '--no-prefixes' not in command
        assert command[command.index('--seconds') + 1] == '64'
        assert command[command.index('--num_envs') + 1] == '175'
    with pytest.raises(ValueError):
        runner.eval_command('parent', 51, 24, 'reference', Path('ref.json'))
    with pytest.raises(ValueError):
        study.validate_request('parent', 119, 79, 'mixed', 64, 175, 'holdout', instrumented=False)


def test_missing_window_and_passivity_fail_closed():
    with pytest.raises(ValueError):
        summary.audit_bundle(dict(schema=study.BUNDLE_SCHEMA, instrumented=True,
                                  physical_condition=dict(seconds=64, max_steps=3840, dt=1/60), windows={}))


def test_runner_exclusive_attempt_does_not_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'WORK', tmp_path)
    (tmp_path / 'retry.log').write_text('failure evidence')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('must not execute'))
    with pytest.raises(FileExistsError):
        runner.run(['python', '--output', str(tmp_path / 'result.json')], 'retry')


def bundle_fixture(monkeypatch):
    monkeypatch.setattr(summary, 'audit', lambda data: [dict(family='rough', level=0)])
    common = dict(controller='parent', geometry_seed=51, reset_seed=24, phase='smoke', num_envs=1, scenario='mixed',
                  training_seed=None, mode='v10', checkpoint='model', checkpoint_sha256='a'*64,
                  lr_condition='not_applicable', model_learning_rate=None,
                  initial_state_sha256={}, initial_prefix_sha256='a'*64, initial_rng_sha256={},
                  family_names=['rough'], difficulties=[.2], family_indices=[0], level_indices=[0],
                  episode_lengths=[7], episode_physical_done=[True], episode_physical_timeout=[False],
                  episode_window_censored=[False], episode_terminated=[True], episode_out_of_lane=[False], episode_world_exit=[False],
                  maximum_distance_m=[1.], forward_distance=[1.], episode_switch_steps=[[]], episode_switch_to_v10=[[]],
                  history_switch_events=[[]], first_hit_one_seconds=[None], first_hit_six_seconds=[None], captured_step=7,
                  capture_reason='all_first_episodes_finished')
    keys = ('root_state_sha256', 'joint_pos_sha256', 'joint_vel_sha256', 'episode_length_buf_sha256', 'observations_sha256',
            'cpu_rng_sha256', 'cuda_rng_sha256', 'policy_state_sha256', 'gate_state_sha256', 'full_first_episode_active_sha256')
    proof = dict(**{k: 'a'*64 for k in keys}, common_step_counter=7, policy_mode='v10', policy_training=False)
    return dict(**{k: common[k] for k in ('controller', 'geometry_seed', 'reset_seed', 'phase', 'num_envs', 'scenario', 'lr_condition', 'model_learning_rate', 'checkpoint', 'checkpoint_sha256')},
                schema=study.BUNDLE_SCHEMA, instrumented=True, physical_condition=dict(seconds=64, max_steps=3840, dt=1/60),
                physical_steps_executed=7, executed_first_episodes=1, window_observations=2,
                windows={str(w): dict(deepcopy(common), window_seconds=w, condition=dict(seconds=w)) for w in (16, 64)},
                snapshot_passivity={str(w): dict(before=deepcopy(proof), after=deepcopy(proof), unchanged=True) for w in (16, 64)})


def test_early_completed_windows_immutable(monkeypatch):
    data = bundle_fixture(monkeypatch)
    assert set(summary.audit_bundle(data)) == {'16', '64'}
    data['windows']['64']['forward_distance'][0] = 2.
    with pytest.raises(ValueError, match='completed first episode'):
        summary.audit_bundle(data)


@pytest.mark.parametrize('mutation', ('rng', 'missing', 'identity', 'counts', 'prefix', 'censor'))
def test_bundle_proof_and_prefix_corruption(monkeypatch, mutation):
    data = bundle_fixture(monkeypatch)
    if mutation == 'rng':
        data['snapshot_passivity']['16']['after']['cpu_rng_sha256'] = 'b'*64
    elif mutation == 'missing':
        del data['snapshot_passivity']['16']['before']['gate_state_sha256']
    elif mutation == 'identity':
        data['windows']['64']['reset_seed'] = 25
    elif mutation == 'counts':
        data['window_observations'] = 1
    elif mutation == 'prefix':
        data['windows']['16']['episode_switch_steps'] = [[1]]
    else:
        data['windows']['64']['episode_window_censored'] = [True]
    with pytest.raises(ValueError):
        summary.audit_bundle(data)


def test_entire_evaluator_ast_equivalent():
    assert runner.evaluator_equivalence()


@pytest.mark.parametrize('old,new', [
    ('gate.reset(done)', 'gate.reset(~done)'),
    ('sensor.update(0.0', 'sensor.update(0.1'),
    ('paired.mark_captured(seconds)', 'paired.mark_captured(64)'),
    ('step=step, rewards=rewards', 'step=step+1, rewards=rewards'),
    ('snapshot_passivity(before, after)', 'snapshot_passivity(after, after)'),
    ('1.e-5 if key.startswith("low")', '1.e-4 if key.startswith("low")'),
    ('from week03_ant.paired_horizon_v23', 'from week03_ant.paired_horizon_v24'),
])
def test_ast_rejects_physics_inference_tracker_and_lr_mutations(old, new):
    source = (ROOT / 'scripts/evaluate_lr_v24.py').read_text()
    assert old in source
    with pytest.raises(ValueError):
        runner.evaluator_equivalence(source.replace(old, new))


@pytest.mark.parametrize('controller', study.CONTROLLERS)
def test_exhaustive_controller_model_history_binding(controller, monkeypatch, model_registry):
    data = raw(controller)
    data.pop('_test_high', None)
    # Retain full real 35-row telemetry; isolate controller binding from request size.
    monkeypatch.setattr(study, 'validate_request', lambda *a, **k: None)
    rows = summary.audit(data)
    assert len(rows) == 35
    key = controller.removeprefix('history_')
    assert study.model_key(controller) == key
    assert study.policy_mode(controller) == ('hybrid' if controller.startswith('history_') else 'v10')
    assert study.command_mode(controller) == 'conditioned'
    assert data['checkpoint'] == model_registry[key]['checkpoint']
    assert data['checkpoint_sha256'] == model_registry[key]['sha256']
    if key != 'parent':
        other = ('seed' if key.startswith('low') else 'low') + key[-2:]
        data.update(checkpoint=model_registry[other]['checkpoint'], checkpoint_sha256=model_registry[other]['sha256'])
        with pytest.raises(ValueError, match='model/plan provenance'):
            summary.audit(data)


@pytest.mark.parametrize('field,value', [('lr_condition', 'high'), ('model_learning_rate', 1.e-4), ('checkpoint_sha256', 'b'*64)])
def test_low_arm_provenance_tampering(field, value):
    data = raw('low51')
    data[field] = value
    with pytest.raises(ValueError):
        summary.audit(data)


def test_exact_ordered_comparisons_and_seed_delta():
    expected = [(f'{p}low{s}', f'{p}seed{s}') for p in ('', 'history_') for s in (51,52,53)]
    expected += [(f'{p}{arm}{s}', f'{p}parent') for arm in ('low','seed') for p in ('','history_') for s in (51,52,53)]
    assert list(study.COMPARISONS) == expected
    metrics = dict(n=300, flat_n=50, one=1, six=2, falls=3, lane=4, world=0, flat_falls=0, flat_lane=0, flat_world=0, flat_mean_episode_speed=1.)
    groups = {c: dict(metrics) for c in study.CONTROLLERS}
    groups['low52']['six'] += 5
    assert set(summary.seed_ranges(groups)) == {'actor_high','actor_low','history_high','history_low'}
    assert summary.matched_seed_deltas(groups)['actor']['52']['six'] == 5
    assert len(summary.duration_sections([], [])) == 49


def test_development_order_and_request_contract():
    assert study.DEVELOPMENT_CONTROLLERS == ('parent','history_parent','low51','low52','low53','history_low51','history_low52','history_low53')
    for c in study.DEVELOPMENT_CONTROLLERS:
        study.validate_request(c,51,24,'mixed',64,35,'smoke')
    for c in ('seed51','history_seed53'):
        with pytest.raises(ValueError):
            study.validate_request(c,51,24,'mixed',64,35,'smoke')


def test_exact_ledger_order_budget_and_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ART', tmp_path)
    monkeypatch.setattr(runner, 'WORK', tmp_path)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'inside', lambda name: tmp_path / name)
    expected = [('parent',51,24,'prepare_development',tmp_path/'prepare_development.json','prepare_development')]
    expected += [(c,51,24,'smoke',tmp_path/f'smoke_{c}.json',f'smoke_{c}') for c in study.DEVELOPMENT_CONTROLLERS]
    expected += [('parent',g,r,'prepare',tmp_path/f'prepare_geometry{g}.json',f'prepare_geometry{g}') for g,r in study.HOLDOUTS]
    records = []
    def append(values):
        for c,g,r,phase,output,label in values:
            log=tmp_path/f'{label}.log'
            log.write_text('synthetic successful CPU fixture, not runtime evidence')
            records.append(dict(command=runner.eval_command(c,g,r,phase,output),label=label,returncode=0,
                                log=log.name,log_sha256=study.sha(log)))
        (tmp_path/'commands.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in records))
    append(expected)
    assert len(runner.verify_ledger()) == 11
    append([(c,g,r,'holdout',tmp_path/'evaluations'/f'{c}__geometry{g}_reset{r}.json',f'evaluations_{c}__geometry{g}_reset{r}') for c in study.CONTROLLERS for g,r in study.HOLDOUTS])
    assert len(runner.verify_ledger(complete=True)) == 39
    records[1],records[2]=records[2],records[1]
    append([])
    with pytest.raises(ValueError, match='order'):
        runner.verify_ledger(complete=True)
    records[1],records[2]=records[2],records[1]
    records.append(records[-1])
    append([])
    with pytest.raises(ValueError, match='budget'):
        runner.verify_ledger(complete=True)


def test_both_windows_use_full_mixed_gate(monkeypatch):
    called=[]
    original=summary.improvement
    def observe(a,b,*,primary):
        called.append(primary)
        return original(a,b,primary=primary)
    monkeypatch.setattr(summary,'improvement',observe)
    groups={c:summary.audit(raw(c)) for c in study.DEVELOPMENT_CONTROLLERS}
    for seed in (51,52,53):
        for prefix in ('','history_'):
            groups[f'{prefix}seed{seed}']=deepcopy(groups[f'{prefix}low{seed}'])
    for _window in (16,64):
        called.clear()
        result=summary._section(groups,True)
        assert len(result['comparisons']) == 18
        assert all(called[:18])
        assert all('flat_speed_not_lower' in gate['checks'] for gate in result['comparisons'].values())


def test_training_ledger_handoff_link_and_chronology(tmp_path, monkeypatch):
    import types
    training=[dict(finished_utc='2026-09-28T01:00:00+00:00')]*13
    monkeypatch.setitem(sys.modules,'run_lr_training_v24',types.SimpleNamespace(verify_training_ledger=lambda:training))
    monkeypatch.setattr(runner,'ART',tmp_path)
    path=tmp_path/'training_commands.jsonl'
    path.write_text('pinned successful training fixture')
    model=tmp_path/'trained_models.json'
    model.write_text(json.dumps(dict(training_commands_sha256=study.sha(path))))
    assert len(runner.verify_training_handoff()) == 13
    evalpath=tmp_path/'commands.jsonl'
    evalpath.write_text(json.dumps(dict(started_utc='2026-09-28T02:00:00+00:00'))+'\n')
    assert len(runner.verify_training_handoff()) == 13
    evalpath.write_text(json.dumps(dict(started_utc='2026-09-28T00:00:00+00:00'))+'\n')
    with pytest.raises(ValueError, match='before training'):
        runner.verify_training_handoff()
    evalpath.unlink()
    path.write_text('replaced')
    with pytest.raises(ValueError, match='link'):
        runner.verify_training_handoff()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        runner.verify_training_handoff()


@pytest.mark.parametrize('key', ('lr_condition', 'model_learning_rate'))
@pytest.mark.parametrize('position', ('before', 'after'))
def test_ast_rejects_duplicate_lr_keys_with_side_effects(key, position):
    source = (ROOT / 'scripts/evaluate_lr_v24.py').read_text()
    marker = f'"{key}":'
    if position == 'before':
        candidate = source.replace(marker, f'{marker} torch.manual_seed(999), {marker}', 1)
    else:
        # Insert after the valid common dictionary values, retaining valid Python.
        tail = '1.e-5 if key.startswith("low") else 1.e-4)}'
        assert tail in source
        candidate = source.replace(tail, tail[:-1] + f', {marker} torch.manual_seed(999)}}', 1)
    with pytest.raises(ValueError, match='LR provenance AST'):
        runner.evaluator_equivalence(candidate)


@pytest.mark.parametrize('key', ('lr_condition', 'model_learning_rate'))
def test_ast_rejects_missing_lr_key(key):
    source = (ROOT / 'scripts/evaluate_lr_v24.py').read_text()
    with pytest.raises(ValueError, match='LR provenance AST'):
        runner.evaluator_equivalence(source.replace(f'"{key}":', f'"missing_{key}":', 1))


@pytest.mark.parametrize('replacement', ('other_common = {', 'common = {}\n        common = {'))
def test_ast_rejects_missing_or_duplicate_common_dictionary(replacement):
    source = (ROOT / 'scripts/evaluate_lr_v24.py').read_text()
    assert 'common = {' in source
    with pytest.raises(ValueError, match='common raw provenance dictionary'):
        runner.evaluator_equivalence(source.replace('common = {', replacement, 1))
