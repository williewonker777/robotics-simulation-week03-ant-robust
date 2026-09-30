"""CPU regressions for fixed budgets, physical audits and dependent-window accounting."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
from week03_ant import paired_horizon_study_v23 as study

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_paired_horizon_v23 as runner
import summarize_paired_horizon_v23 as summary
from summarize_seed_v22 import seed_ranges
from summarize_lp_v17 import PARITY_FIELDS


def raw(controller='parent'):
    references = json.loads((study.prior.ART / 'development.json').read_text())['records']
    entry = next(r for r in references if r['controller'] == controller)
    data = json.loads((ROOT / entry['path']).read_text())
    data.update(schema=study.SCHEMA, evaluation_plan_sha256=study.sha(study.PLAN))
    return data


@pytest.mark.parametrize('controller', study.CONTROLLERS)
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
    assert len(study.COMPARISONS) == 12
    assert study.frozen_budget()['first_episodes'] == 2800
    assert study.frozen_budget()['window_observations'] == 5600


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
    result = seed_ranges(aggregates)['actor']
    assert result['rough_episodes_per_seed'] == 300 and result['parent_counted_once']
    assert result['metrics']['six']['range'] == 3
    assert result['metrics']['six']['parent_deltas']['seed52'] == 3


def test_command_matrix_no_holdout_reference_or_stones():
    for controller in study.CONTROLLERS:
        command = runner.eval_command(controller, 117, 77, 'holdout', Path('raw.json'))
        assert '--no-prefixes' not in command
        assert command[command.index('--seconds') + 1] == '64'
        assert command[command.index('--num_envs') + 1] == '175'
    assert '--no-prefixes' in runner.eval_command('parent', 51, 24, 'reference', Path('ref.json'))
    with pytest.raises(ValueError):
        study.validate_request('parent', 117, 77, 'mixed', 64, 175, 'holdout', instrumented=False)


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
    return dict(**{k: common[k] for k in ('controller', 'geometry_seed', 'reset_seed', 'phase', 'num_envs', 'scenario')},
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
