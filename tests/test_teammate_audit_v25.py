# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic adversarial evidence tests, not live training/evaluation evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v25_independent_audit', ROOT / 'scripts/audit_teammate_v25_raw.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
H = '1' * 64


def bundle(controller='v5', geometry=131, reset=111, n=175):
    b = dict(schema='week03_ant_teammate_v25_eval_v1', task='Week03-Ant-Contact-v16-Eval-v0',
        controller=controller, geometry_seed=geometry, reset_seed=reset, num_envs=n,
        seconds=64, dt=1 / 60, scenario='mixed', scored_episodes=n, new_training_transitions=0,
        family_names=list(audit.FAMILIES), difficulties=[.2, .4, .6, .8, 1.],
        family_indices=[f for f in range(7) for l in range(5) for _ in range(n // 35)],
        level_indices=[l for f in range(7) for l in range(5) for _ in range(n // 35)],
        checkpoint_sha256=audit.LEGACY_SHA.get(controller, H),
        checkpoint_sha256_after=audit.LEGACY_SHA.get(controller, H),
        v5_sha256=audit.TEACHER_SHA, teacher_tensor_identity_verified=True,
        mode='hybrid' if controller.startswith('history_') else 'v5' if controller == 'v5' else 'v10',
        command_mode=None if controller == 'v5' else 'conditioned',
        source_sha256={'scripts/evaluate_unseen_terrain.py': H}, source_sha256_after={'scripts/evaluate_unseen_terrain.py': H},
        initial_state_sha256=dict.fromkeys(('root_state', 'joint_pos', 'joint_vel', 'observations'), H),
        initial_prefix_sha256=H, initial_rng_sha256={'cpu': H, 'cuda': H}, terrain_config_sha256=H,
        started_utc='2026-10-01T01:00:00+00:00', finished_utc='2026-10-01T02:00:00+00:00',
        success_thresholds_m={'one_tile': 13.1, 'all_tiles': 53.1}, gate_config=deepcopy(audit.GATE_CONFIG), windows={},
        v25=dict(phase='holdout', model_phase='holdout', scoring_training_transitions=0,
                 substitutions=['CONTROLLERS', 'SCHEMA', 'resolve_model'], adapter_source_sha256={'new.py': H},
                 adapter_source_sha256_after={'new.py': H}, base_evaluator_sha256=H))
    for seconds in (16, 64):
        steps = seconds * 60
        w = dict(window_seconds=seconds, captured_step=steps, capture_reason='window_cutoff', routing={},
                 aggregates={'strict_all_tiles_successes': -999})
        for key in audit.BOOL_ARRAYS:
            w[key] = [key in ('episode_strict_one_tile_success', 'episode_strict_all_tiles_success',
                               'episode_full_horizon_survival') or key == 'episode_window_censored' and seconds == 16] * n
        for key in audit.NUMBER_ARRAYS:
            w[key] = [60.] * n
        for key in audit.NULLABLE_ARRAYS:
            w[key] = [1.] * n
        for key in audit.INT_ARRAYS:
            w[key] = [steps] * n
        route = w['routing']
        for key in audit.ROUTE_NUMBERS:
            route[key] = [0.] * n
        for key in audit.ROUTE_INTS:
            route[key] = [steps if key == 'episode_active_steps' else 0] * n
        for key in audit.ROUTE_NULLABLE:
            route[key] = [None] * n
        for key in audit.ROUTE_NESTED:
            route[key] = [[] for _ in range(n)]
        route['episode_fall_within_switch_window'] = [False] * n
        duty = 0. if controller == 'v5' or controller.startswith('history_') else 1.
        route['episode_v10_duty'] = [duty] * n
        route['episode_alpha_sum'] = [duty * steps] * n
        b['windows'][str(seconds)] = w
    return b


def matrix():
    records = [bundle(c, g, r) for c in audit.CONTROLLERS for g, r in audit.MAPS]
    frozen = dict(phase='holdout', created_utc='2026-10-01T00:00:00+00:00', v5_sha256=audit.TEACHER_SHA,
        matrix=[[c, g, r, 175] for c in audit.CONTROLLERS for g, r in audit.MAPS],
        source_sha256={'new.py': H}, base_source_sha256={'scripts/evaluate_unseen_terrain.py': H},
        models={c: dict(sha256=audit.LEGACY_SHA.get(c, H), mode='hybrid' if c.startswith('history_') else 'v5' if c == 'v5' else 'v10',
                       command_mode=None if c == 'v5' else 'conditioned') for c in audit.CONTROLLERS})
    return records, frozen


def test_independent_raw_counts_ignore_aggregates_and_fail_without_strict_gain():
    records, frozen = matrix()
    result = audit.audit_evaluation_records(records, frozen)
    assert result['physical_first_episodes'] == 8050
    assert result['dependent_window_observations'] == 16100
    assert result['paired_initial_conditions'] == 350
    assert not result['statistical_independence_claim']
    for seconds in ('16', '64'):
        assert result['windows'][seconds]['controllers']['v5']['total']['rough']['six'] == 300
        assert not any(c['pass'] for c in result['windows'][seconds]['contrasts'].values())


@pytest.mark.parametrize('field,value', [
    ('forward_distance', float('nan')), ('forward_distance', True), ('episode_steps', True),
    ('episode_steps', 0), ('episode_terminated', 1), ('episode_physical_timeout', True),
    ('episode_window_censored', False), ('episode_strict_all_tiles_success', False),
    ('first_hit_one_seconds', -1.), ('maximum_distance_m', 0.),
])
def test_malformed_or_false_tracker_values_rejected(field, value):
    b = bundle()
    b['windows']['16'][field][0] = value
    with pytest.raises(ValueError):
        audit.window_rows(b, 16)


@pytest.mark.parametrize('section,field', [('window', k) for k in audit.BOOL_ARRAYS + audit.NUMBER_ARRAYS + audit.INT_ARRAYS + audit.NULLABLE_ARRAYS]
                         + [('routing', k) for k in audit.ROUTE_NUMBERS + audit.ROUTE_INTS + audit.ROUTE_NULLABLE + audit.ROUTE_NESTED])
def test_every_declared_array_requires_exact_length(section, field):
    b = bundle()
    container = b['windows']['16'] if section == 'window' else b['windows']['16']['routing']
    container[field].pop()
    with pytest.raises(ValueError, match='array'):
        audit.window_rows(b, 16)


def test_float32_final_distance_not_maximum_or_earlier_hit():
    b = bundle()
    w = b['windows']['16']
    w['forward_distance'][0] = audit.f32(53.1)
    assert audit.window_rows(b, 16)[0]['six']
    w['forward_distance'][0] = 53.099
    w['episode_strict_all_tiles_success'][0] = False
    assert not audit.window_rows(b, 16)[0]['six']
    w['episode_out_of_lane'][0] = True
    w['episode_strict_one_tile_success'][0] = False
    assert not audit.window_rows(b, 16)[0]['one']


@pytest.mark.parametrize('field', audit.PAIR_FIELDS)
def test_pairing_substitution_rejected(field):
    records, frozen = matrix()
    if isinstance(records[1][field], dict):
        records[1][field][next(iter(records[1][field]))] = '2' * 64
    elif isinstance(records[1][field], list):
        records[1][field][0] = 'wrong'
    else:
        records[1][field] = '2' * 64
    with pytest.raises((ValueError, TypeError)):
        audit.validate_evaluations(records, frozen)


@pytest.mark.parametrize('change', ['checkpoint', 'source', 'controller', 'geometry', 'teacher', 'chronology', 'matrix', 'command'])
def test_substituted_provenance_rejected(change):
    records, frozen = matrix()
    b = records[0]
    if change == 'checkpoint': b['checkpoint_sha256'] = H
    if change == 'source': b['source_sha256_after']['extra'] = H
    if change == 'controller': b['controller'] = 'best'
    if change == 'geometry': b['geometry_seed'] = 130
    if change == 'teacher': b['v5_sha256'] = H
    if change == 'chronology': frozen['created_utc'] = '2026-10-02T00:00:00+00:00'
    if change == 'matrix': records.pop()
    if change == 'command': b['command_mode'] = 'conditioned'
    with pytest.raises(ValueError):
        audit.validate_evaluations(records, frozen)


def test_history_flat_drift_is_negative_result_not_invalid_experiment():
    records, frozen = matrix()
    b = next(x for x in records if x['controller'] == 'history_combined61')
    b['windows']['16']['forward_distance'][-1] = 59.
    result = audit.audit_evaluation_records(records, frozen)
    gate = result['windows']['16']['contrasts']['history:combined_vs_control']['seeds']['61']['gates']['131']
    assert not gate['checks']['flat_v5_identity']
    assert not gate['pass']


def test_retention_rejects_flat_lane_and_flat_speed_regressions():
    base = audit.window_rows(bundle('control61'), 16)
    candidate = deepcopy(base)
    candidate[0]['six'] = True
    base[0]['six'] = False
    assert audit.retention_gate(candidate, base)['pass']
    candidate[-1]['lane'] = True
    assert not audit.retention_gate(candidate, base)['pass']
    candidate[-1]['lane'] = False
    candidate[-1]['speed_m_s'] -= 1
    assert not audit.retention_gate(candidate, base)['pass']


def test_fewer_falls_alone_is_not_strict_success_improvement():
    base = audit.window_rows(bundle('control61'), 16)
    candidate = deepcopy(base)
    base[0]['fall'] = True
    assert not audit.retention_gate(candidate, base)['pass']


def test_json_duplicates_nonfinite_and_path_escape_rejected(tmp_path):
    for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
        p = tmp_path / 'bad.json'
        p.write_text(text)
        with pytest.raises(ValueError): audit.read(p)
    with pytest.raises(ValueError): audit.inside(tmp_path, '../escape')
    with pytest.raises(ValueError): audit.inside(tmp_path, '/absolute')


def schedule(arm='control', seed=61, updates=2, envs=256):
    runtime = dict(entropy_coef=.005 if arm == 'combined' else .002, teacher_coef=.02, learning_rate=1e-4,
        schedule='fixed', desired_kl=None, optimizer_groups=[dict(lr=1e-4, betas=[.9,.999], eps=1e-8, weight_decay=0, amsgrad=False)])
    n = updates * 32
    return dict(runtime_initial=runtime, runtime_final=deepcopy(runtime),
        runtime_updates=[dict(update=i, adam_steps=i*20, before=deepcopy(runtime), after=deepcopy(runtime)) for i in range(1,updates+1)],
        contact=dict(mode='no_cost', ramp_steps=4000, num_envs=envs, policy_steps=n,common_step_counters=list(range(1,n+1)),
            coefficients=[0.]*n,coefficient_sum=0,scaled_reward_mean=[0.]*n,scaled_reward_dt_sum=[0.]*n,realized_penalty_sum=0,
            raw_reward_mean=[0.]*n,step_dt=[1/60]*n,raw_reward_dt_sum=[0.]*n,valid_rows=[envs]*n),
        arm=arm, training_seed=seed, num_envs=envs, policy_steps=updates * 32, ppo_updates=updates,
        adam_steps=updates * 20, entropy_coef=.005 if arm == 'combined' else .002, teacher_coef=.02, learning_rate=1e-4,
        episode_randomization=dict(matches_raw_environment=True, assigned_sha256=H, actual_sha256=H, minimum=0, maximum=500, nonzero=250),
        recovery=[dict(step=i, enabled=arm != 'control', dt=1 / 60, raw_sum=-5., applied_sum=0. if arm == 'control' else -5.,
            abstained=0, clearance_risk=1., tilt_risk=1., action_delta_squared_sum=50., angular_xy_squared_sum=20.) for i in range(1, updates * 32 + 1)])


def test_training_schedule_recomputes_applied_formula_and_budget():
    for arm in audit.ARMS:
        result = audit.audit_schedule(schedule(arm), arm=arm, seed=61, envs=256, updates=2)
        assert result['transitions'] == 16384


@pytest.mark.parametrize('field,value', [('policy_steps', 63), ('ppo_updates', 3), ('adam_steps', 39),
    ('entropy_coef', .005), ('teacher_coef', 0), ('learning_rate', 1e-3), ('training_seed', 62)])
def test_training_counts_or_runtime_coefficients_substitution_rejected(field, value):
    data = schedule()
    data[field] = value
    with pytest.raises(ValueError): audit.audit_schedule(data, arm='control', seed=61, envs=256, updates=2)


@pytest.mark.parametrize('field,value', [('raw_sum', float('nan')), ('raw_sum', -6.), ('applied_sum', -5.),
    ('dt', 1.), ('enabled', True), ('abstained', 257), ('clearance_risk', -1.)])
def test_recovery_rate_semantics_and_control_zero_rejected(field, value):
    data = schedule()
    data['recovery'][0][field] = value
    with pytest.raises(ValueError): audit.audit_schedule(data, arm='control', seed=61, envs=256, updates=2)


def test_shadowed_random_horizon_rejected():
    data = schedule()
    data['episode_randomization']['actual_sha256'] = '2' * 64
    with pytest.raises(ValueError, match='horizons'):
        audit.audit_schedule(data, arm='control', seed=61, envs=256, updates=2)


def test_runtime_intermediate_coefficient_substitution_rejected():
    data = schedule()
    data['runtime_updates'][0]['after']['entropy_coef'] = .005
    with pytest.raises(ValueError, match='trace'):
        audit.audit_schedule(data, arm='control', seed=61, envs=256, updates=2)


def test_nonzero_contact_coefficient_rejected():
    data = schedule()
    data['contact']['coefficients'][0] = 1.
    with pytest.raises(ValueError, match='contact'):
        audit.audit_schedule(data, arm='control', seed=61, envs=256, updates=2)


@pytest.fixture(scope='module')
def model_fixture():
    import torch
    torch.set_num_threads(1)
    path = ROOT / 'artifacts/terrain_demo/contact_slip_v16/runs/control/model_249.pt'
    teacher_path = ROOT / 'artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt'
    parent = torch.load(path, map_location='cpu', weights_only=False)
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    initial = deepcopy(parent)
    initial['iter'] = 0
    initial['model_state_dict']['std'].fill_(.2)
    initial['optimizer_state_dict']['state'] = {}
    initial['optimizer_state_dict']['param_groups'][0]['lr'] = 1e-4
    final = deepcopy(parent)
    final['iter'] = 249
    final['optimizer_state_dict']['param_groups'] = deepcopy(initial['optimizer_state_dict']['param_groups'])
    for row in final['optimizer_state_dict']['state'].values():
        row['step'].fill_(5000)
    return initial, final, parent, teacher


def test_serialized_model_teacher_buffers_and_adam_steps_checked(model_fixture):
    result = audit.model_audit(*model_fixture)
    assert result['adam_steps'] == 5000
    assert result['teacher_and_command_buffers_unchanged']


@pytest.mark.parametrize('change', ['teacher', 'buffer', 'actor_shape', 'nan', 'optimizer_step', 'optimizer_coverage', 'std', 'fresh_adam', 'iteration'])
def test_model_or_optimizer_substitution_rejected(model_fixture, change):
    import torch
    initial, final, parent, teacher = deepcopy(model_fixture)
    if change == 'teacher': final['model_state_dict']['teacher.0.weight'][0, 0] += 1
    if change == 'buffer':
        key = next(k for k in final['model_state_dict'] if k not in audit.TRAINABLE and not k.startswith('teacher.'))
        final['model_state_dict'][key] = final['model_state_dict'][key] + 1
    if change == 'actor_shape': final['model_state_dict']['actor.0.weight'] = torch.zeros(1)
    if change == 'nan': final['model_state_dict']['std'][0] = float('nan')
    if change == 'optimizer_step': final['optimizer_state_dict']['state'][0]['step'].fill_(4999)
    if change == 'optimizer_coverage': final['optimizer_state_dict']['state'].pop(0)
    if change == 'std': initial['model_state_dict']['std'].fill_(.3)
    if change == 'fresh_adam': initial['optimizer_state_dict']['state'] = {0: {}}
    if change == 'iteration': final['iter'] = 248
    with pytest.raises(ValueError): audit.model_audit(initial, final, parent, teacher)


def test_switch_timestamp_is_start_of_policy_step_not_end():
    b = bundle('history_control')
    route = b['windows']['16']['routing']
    route['episode_switch_count'][0] = 1
    route['episode_switch_steps'][0] = [61]
    route['episode_switch_to_v10'][0] = [True]
    route['episode_switch_seconds'][0] = [1.]
    route['episode_first_switch_seconds'][0] = 1.
    route['episode_last_switch_seconds'][0] = 1.
    audit.window_rows(b, 16)
    route['episode_switch_seconds'][0] = [61 / 60]
    with pytest.raises(ValueError, match='switch seconds'):
        audit.window_rows(b, 16)


def test_early_ended_episode_cannot_be_rescored_after_reset():
    records, frozen = matrix()
    b = records[0]
    for seconds in (16, 64):
        w = b['windows'][str(seconds)]
        for key in ('episode_lengths', 'episode_steps'): w[key][0] = 60
        w['routing']['episode_active_steps'][0] = 60
        for key in ('episode_terminated', 'episode_physical_done'): w[key][0] = True
        for key in ('episode_full_horizon_survival', 'episode_strict_one_tile_success', 'episode_strict_all_tiles_success', 'episode_window_censored'):
            w[key][0] = False
    audit.validate_evaluations(records, frozen)
    b['windows']['64']['forward_distance'][0] += 1
    b['windows']['64']['maximum_distance_m'][0] += 1
    with pytest.raises(ValueError, match='early-ended'):
        audit.validate_evaluations(records, frozen)


def initial_record(arm='control', updates=2):
    return dict(arm=arm, training_seed=61, actual_seed=dict.fromkeys(('requested','saved_agent','saved_environment','live_environment'),61),
        num_envs=4096, observation_dimensions=[4096,91], dt=1/60, iteration=0, common_step_counter=0,
        optimizer_state_empty=True,initial_policy_parity=dict(actor=True,critic=True,teacher=True,device='cuda:1'),
        initial_state_sha256=dict.fromkeys(('root_state','joint_pos','joint_vel','observations'),H), initial_prefix_sha256=H,
        rng_sha256=dict(cpu=H,cuda=H),policy_state_sha256={'std':H},
        normalized_parameters=dict(env=dict(seed='61',scene=dict(terrain=dict(terrain_generator=dict(seed='130'))),
            rewards=dict(teammate_recovery=dict(func='week03_ant.tasks.teammate_v25_cfg:TeammateRecoveryReward',weight='1.0',params=dict(enabled=str(arm!='control').lower())))),
            agent=dict(seed='61',max_iterations=str(updates),num_steps_per_env='32',device='cuda:1',
                algorithm=dict(entropy_coef='.005' if arm=='combined' else '.002',teacher_coef='.02',learning_rate='.0001',schedule='fixed',num_learning_epochs='5',num_mini_batches='4'))))


def test_only_declared_treatments_and_own_capacity_budget_project():
    control, combined = initial_record(), initial_record('combined')
    audit.audit_initial(combined,arm='combined',seed=61,envs=4096,updates=2)
    assert audit.initial_pair(control,combined)
    assert audit.initial_pair(combined,initial_record('combined',250),capacity_to_main=True)


@pytest.mark.parametrize('key', ['initial_state_sha256','rng_sha256','policy_state_sha256','actual_seed'])
def test_initial_state_rng_checkpoint_and_live_seed_substitution_rejected(key):
    a,b=initial_record(),initial_record('combined')
    b[key][next(iter(b[key]))]='2'*64
    with pytest.raises(ValueError,match='pairing'):
        audit.initial_pair(a,b)


def test_unlisted_reward_or_optimizer_config_difference_rejected():
    a,b=initial_record(),initial_record('combined')
    b['normalized_parameters']['agent']['algorithm']['learning_rate']='.0002'
    with pytest.raises(ValueError): audit.initial_pair(a,b)
    with pytest.raises(ValueError): audit.audit_initial(b,arm='combined',seed=61,envs=4096,updates=2)


def test_all_controllers_mutated_gate_configuration_rejected():
    records,frozen=matrix()
    for b in records: b['gate_config']['history_seconds']=2.
    with pytest.raises(ValueError,match='gate configuration'):
        audit.validate_evaluations(records,frozen)
