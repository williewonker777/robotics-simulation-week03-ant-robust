"""Synthetic and adversarial checks independent of simulator/GPU evidence."""
import copy
import importlib.util
from pathlib import Path
import struct

import pytest

_SPEC = importlib.util.spec_from_file_location('hybrid_summary_under_test', Path(__file__).parents[1] / 'scripts/summarize_hybrid.py')
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def fixture(seconds=16, mode='hybrid', distance=54.0):
    length = seconds * 60
    return dict(
        schema='week03_ant_depth_switch_v11_v1', num_envs=1, mode=mode,
        condition=dict(seconds=seconds, max_steps=length, dt=1 / 60,
                       one_threshold=13.1, six_threshold=53.1, footprint_margin=1.1,
                       observations=88, v5_observations=60, actions=8,
                       snapshot_seconds=8 if seconds == 16 else 16),
        family_names=['flat', 'stones'], difficulties=[0.2, 1.0],
        family_indices=[1], level_indices=[1], forward_distance=[distance],
        episode_lengths=[length], episode_active_steps=[length], episode_terminated=[False],
        episode_out_of_lane=[False], episode_world_exit=[False],
        episode_strict_one_tile_success=[distance >= f32(13.1)],
        episode_strict_all_tiles_success=[distance >= f32(53.1)],
        episode_full_horizon_survival=[True], episode_switch_count=[0],
        episode_v10_duty=[float(mode == 'v10')], episode_alpha_sum=[float(mode == 'v10') * length],
        episode_switch_steps=[[]], episode_switch_to_v10=[[]], episode_uncertain_steps=[0],
        episode_fall_within_switch_window=[False], episode_max_action_jump_rms=[0.1],
        episode_mean_action_disagreement_rms=[0.3],
        first_hit_one_seconds=[1.0 if distance >= f32(13.1) else None],
        first_hit_six_seconds=[2.0 if distance >= f32(53.1) else None],
        distance_at_snapshot_m=[min(10.0, distance)], maximum_distance_m=[distance],
    )


def terminal(data, length, *, fall=True):
    data['episode_lengths'] = data['episode_active_steps'] = [length]
    data['episode_alpha_sum'] = [data['episode_v10_duty'][0] * length]
    data['episode_terminated'] = [fall]
    data['episode_full_horizon_survival'] = [False]
    if fall:
        data['episode_strict_one_tile_success'] = [False]
        data['episode_strict_all_tiles_success'] = [False]
    data['distance_at_snapshot_m'] = [None if length / 60 <= data['condition']['snapshot_seconds'] else 10.0]
    return data


def test_valid_primary_horizon_and_fixed_policy_endpoints():
    for seconds in (16, 64):
        for mode in ('hybrid', 'v5', 'v10'):
            row, = summary.audit(fixture(seconds, mode))
            assert (row['one'], row['six'], row['survival']) == (1, 1, 1)
            assert row['duty'] == float(mode == 'v10')


def test_float32_thresholds_and_first_episode_failure_override_distance():
    for threshold in (13.1, 53.1):
        exact = fixture(distance=f32(threshold))
        row, = summary.audit(exact)
        assert row['one' if threshold == 13.1 else 'six'] == 1
        below = fixture(distance=f32(threshold) - 1e-6)
        assert summary.audit(below)[0]['one' if threshold == 13.1 else 'six'] == 0
    for failure in ('episode_terminated', 'episode_out_of_lane', 'episode_world_exit'):
        data = fixture()
        data[failure] = [True]
        data['episode_strict_one_tile_success'] = data['episode_strict_all_tiles_success'] = [False]
        data['episode_full_horizon_survival'] = [failure != 'episode_terminated']
        assert summary.audit(data)[0]['six'] == 0
        data['episode_strict_all_tiles_success'] = [True]
        with pytest.raises(ValueError):
            summary.audit(data)


def test_flags_and_malformed_arrays_are_rejected():
    for key, bad in [('episode_terminated', [0]), ('episode_out_of_lane', [1]),
                     ('forward_distance', []), ('episode_lengths', [960.0]),
                     ('family_indices', [True]), ('episode_switch_to_v10', [None])]:
        data = fixture()
        data[key] = bad
        with pytest.raises(ValueError):
            summary.audit(data)


def test_counts_duty_and_diagnostics_have_strict_types():
    for key, bad in [('episode_active_steps', 960.0), ('episode_switch_count', False),
                     ('episode_v10_duty', False), ('episode_alpha_sum', False),
                     ('episode_uncertain_steps', True), ('episode_max_action_jump_rms', True),
                     ('episode_mean_action_disagreement_rms', False)]:
        data = fixture()
        data[key] = [bad]
        with pytest.raises(ValueError, match='.'):
            summary.audit(data)


def test_counts_duty_and_baseline_endpoint_consistency():
    for key, bad in [('episode_active_steps', 959), ('episode_v10_duty', 1.1),
                     ('episode_alpha_sum', 4.0), ('episode_uncertain_steps', 961),
                     ('episode_switch_count', 1), ('episode_mean_action_disagreement_rms', float('nan'))]:
        data = fixture()
        data[key] = [bad]
        with pytest.raises(ValueError):
            summary.audit(data)
    data = fixture(mode='v5')
    data['episode_v10_duty'], data['episode_alpha_sum'] = [0.5], [480.0]
    with pytest.raises(ValueError, match='endpoint'):
        summary.audit(data)


def test_switch_event_order_target_alternation_and_episode_bounds():
    for steps, targets in [([10, 10], [True, False]), ([20, 10], [True, False]),
                           ([0], [True]), ([961], [True]), ([True], [True]),
                           ([10], [False]), ([10], [1]), ([10, 20], [True, True])]:
        data = fixture()
        data['episode_switch_steps'] = [steps]
        data['episode_switch_to_v10'] = [targets]
        data['episode_switch_count'] = [len(steps)]
        with pytest.raises(ValueError):
            summary.audit(data)


def test_inclusive_switch_fall_window_uses_pre_action_switch_time():
    for switch, expected in ((270, False), (271, True), (300, True)):
        data = terminal(fixture(), 300)
        data['episode_switch_steps'], data['episode_switch_to_v10'] = [[switch]], [[True]]
        data['episode_switch_count'], data['episode_fall_within_switch_window'] = [1], [expected]
        assert summary.audit(data)[0]['near_switch_fall'] == int(expected)
        data['episode_fall_within_switch_window'] = [not expected]
        with pytest.raises(ValueError, match='switch-associated'):
            summary.audit(data)


def test_snapshot_terminal_at_instant_is_censored():
    for seconds in (16, 64):
        data = fixture(seconds)
        snapshot_step = data['condition']['snapshot_seconds'] * 60
        terminal(data, snapshot_step)
        assert summary.audit(data)[0]['steps'] == snapshot_step
        data['distance_at_snapshot_m'] = [10.0]
        with pytest.raises(ValueError, match='censor'):
            summary.audit(data)
        terminal(data, snapshot_step + 1)
        summary.audit(data)
        data['distance_at_snapshot_m'] = [None]
        with pytest.raises(ValueError, match='censor'):
            summary.audit(data)


def test_hit_times_must_agree_with_maximum_and_order():
    for key, value in [('first_hit_one_seconds', None), ('first_hit_six_seconds', None),
                       ('first_hit_one_seconds', 3.0), ('first_hit_six_seconds', 17.0)]:
        data = fixture()
        data[key] = [value]
        with pytest.raises(ValueError):
            summary.audit(data)
    data = fixture(distance=10.0)
    data['first_hit_one_seconds'] = [1.0]
    with pytest.raises(ValueError):
        summary.audit(data)


def test_aggregate_separates_flat_and_does_not_trust_saved_aggregate():
    terrain = fixture()
    terrain['aggregates'] = {'six': 9999}
    flat = fixture(mode='v10')
    flat['family_indices'] = [0]
    terminal(flat, 300)
    result = summary.aggregate(summary.audit(terrain) + summary.audit(flat))
    assert (result['n'], result['flat_n'], result['flat_falls']) == (1, 1, 1)
    assert (result['one'], result['six'], result['falls']) == (1, 1, 0)
    assert result['mean_v10_duty'] == 0 and result['flat_mean_v10_duty'] == 1
    assert summary.aggregate([])['mean_final_distance'] is None


def groups():
    v5 = dict(n=10, one=8, six=4, falls=2, lane=0, world=0, flat_n=2, flat_falls=1, flat_world=0)
    hybrid = dict(n=30, one=24, six=12, falls=6, lane=0, world=0, flat_n=6, flat_falls=3, flat_world=0)
    primary = dict(v5=v5, hybrid=hybrid, v10=copy.deepcopy(hybrid))
    horizon = dict(v5=dict(n=10, six=0, falls=2), hybrid=dict(n=30, six=1, falls=6))
    return primary, horizon


def test_promotion_compares_rates_with_different_denominators():
    primary, horizon = groups()
    assert summary.promotion(primary, horizon)['passed'] is True
    primary['hybrid']['one'] = 23  # still more absolute successes than v5
    result = summary.promotion(primary, horizon)
    assert not result['passed'] and not result['checks']['one_vs_v5']


def test_promotion_strict_long_horizon_improvement_and_each_safety_gate():
    for where, key, value, expected in [('primary', 'lane', 1, 'lane_vs_v5'),
                                       ('primary', 'falls', 7, 'falls_vs_v5'),
                                       ('primary', 'world', 1, 'world_zero'),
                                       ('primary', 'flat_falls', 4, 'flat_falls_vs_v5'),
                                       ('horizon', 'six', 0, '64s_stone_six_vs_v5'),
                                       ('horizon', 'falls', 7, '64s_stone_falls_vs_v5')]:
        primary, horizon = groups()
        (primary if where == 'primary' else horizon)['hybrid'][key] = value
        result = summary.promotion(primary, horizon)
        assert not result['passed'] and not result['checks'][expected]


def test_empty_promotion_denominator_rejected():
    primary, horizon = groups()
    primary['hybrid']['flat_n'] = 0
    with pytest.raises(ValueError, match='denominator'):
        summary.promotion(primary, horizon)


def test_flat_only_world_exit_is_counted_and_blocks_promotion():
    flat = fixture()
    flat['family_indices'] = [0]
    flat['episode_world_exit'] = [True]
    flat['episode_strict_one_tile_success'] = [False]
    flat['episode_strict_all_tiles_success'] = [False]
    result = summary.aggregate(summary.audit(fixture()) + summary.audit(flat))
    assert result['world'] == 0
    assert result['flat_world'] == 1
    primary, horizon = groups()
    primary['hybrid']['flat_world'] = 1
    gate = summary.promotion(primary, horizon)
    assert primary['hybrid']['world'] == 0
    assert not gate['passed']
    assert not gate['checks']['world_zero']
