import json

import pytest
import torch

from week03_ant.hybrid_telemetry import HybridTelemetry


def output(n=2, alpha=0.0, switched=False, command=0.0):
    return dict(actions=torch.full((n, 8), float(command)),
                alpha=torch.full((n,), float(alpha)), switched=torch.full((n,), switched),
                target_v10=torch.full((n,), alpha > 0), uncertain=torch.zeros(n, dtype=torch.bool),
                coverage=torch.ones(n), enter_span=torch.ones(n), enter_edge=torch.ones(n),
                retain_span=torch.ones(n), retain_edge=torch.ones(n))


def run_step(t, step, gate=None, done=None, terminated=None):
    n = t.num_envs
    active = ~t.finished.clone()
    t.before_step(step, active, output(n) if gate is None else gate, torch.zeros(n, 8), torch.ones(n, 8))
    t.after_step(step, active, torch.zeros(n, dtype=torch.bool) if done is None else torch.tensor(done),
                 torch.zeros(n, dtype=torch.bool) if terminated is None else torch.tensor(terminated))


def test_first_episode_autoreset_exclusion_and_exact_moments():
    t = HybridTelemetry(2, 'cpu', 0.1)
    g = output(alpha=0.5, switched=True, command=1)
    g['coverage'][:] = 0.7
    run_step(t, 1, g, [True, False], [True, False])
    g = output(alpha=1, command=3)
    run_step(t, 2, g, [False, False], [True, False])  # stale reset termination is ignored
    g = output(alpha=0, switched=True, command=2)
    run_step(t, 3, g, [True, True], [True, False])
    result = t.to_dict([1, 3])
    assert result['episode_switch_count'] == [1, 2]
    assert result['episode_switch_steps'] == [[1], [1, 3]]
    assert result['episode_switch_to_v10'] == [[True], [True, False]]
    assert result['episode_switch_seconds'] == [[0.0], [0.0, 0.2]]
    assert result['episode_v10_duty'] == [0.5, 0.5]
    assert result['episode_v10_target_steps'] == [1, 2]
    assert result['episode_mean_action_disagreement_rms'] == [1.0, 1.0]
    assert result['episode_max_action_jump_rms'] == [0.0, 2.0]
    assert result['episode_fall_within_switch_window'] == [True, False]
    assert result['episode_min_coverage'] == pytest.approx([0.7, 0.7])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('terminal_step,expected', [(1, True), (5, True), (6, False)])
def test_inclusive_fall_window(terminal_step, expected):
    t = HybridTelemetry(1, 'cpu', 0.1)
    for step in range(1, terminal_step + 1):
        run_step(t, step, output(1, switched=step == 1), [step == terminal_step], [step == terminal_step])
    assert t.to_dict([terminal_step])['episode_fall_within_switch_window'] == [expected]


@pytest.mark.parametrize('alpha', [0.0, 1.0])
def test_baseline_no_switch_and_first_action_not_jump(alpha):
    t = HybridTelemetry(1, 'cpu', 0.02)
    run_step(t, 1, output(1, alpha=alpha, command=99), [True], [True])
    r = t.to_dict([1])
    assert r['episode_v10_duty'] == [alpha]
    assert r['episode_switch_count'] == [0]
    assert r['episode_first_switch_seconds'] == [None]
    assert r['episode_max_action_jump_rms'] == [0]
    assert r['episode_fall_within_switch_window'] == [False]


def test_unknown_features_preserve_missingness_and_finite_observed_maxima():
    t = HybridTelemetry(2, 'cpu', 0.1)
    g = output()
    for key in t.feature_max:
        g[key][:] = float('nan')
    g['coverage'][:] = 0
    g['uncertain'][:] = True
    run_step(t, 1, g)
    r = t.to_dict([1, 1])
    assert r['episode_max_enter_span'] == [None, None]
    assert r['episode_uncertain_steps'] == [1, 1]
    g['enter_span'][1] = 2
    run_step(t, 2, g)
    assert t.to_dict([2, 2])['episode_max_enter_span'] == [None, 2]
    json.dumps(t.to_dict([2, 2]), allow_nan=False)


@pytest.mark.parametrize('key,value', [('actions', float('nan')), ('coverage', 1.1), ('alpha', -0.1),
                                       ('enter_span', float('inf')), ('enter_edge', -1)])
def test_invalid_values_do_not_partially_accumulate(key, value):
    t = HybridTelemetry(2, 'cpu', 0.1)
    g = output()
    g[key][:] = value
    with pytest.raises(ValueError):
        t.before_step(1, ~t.finished, g, torch.zeros(2, 8), torch.ones(2, 8))
    assert t.to_dict([0, 0])['episode_active_steps'] == [0, 0]


def test_step_pairing_shapes_types_active_and_lengths():
    t = HybridTelemetry(2, 'cpu', 0.1)
    with pytest.raises(ValueError):
        run_step(t, 2)
    with pytest.raises(RuntimeError):
        t.after_step(1, ~t.finished, t.finished, t.finished)
    for bad in [torch.ones(2), torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)]:
        with pytest.raises(ValueError):
            t.before_step(1, bad, output(), torch.zeros(2, 8), torch.ones(2, 8))
    t.before_step(1, ~t.finished, output(), torch.zeros(2, 8), torch.ones(2, 8))
    with pytest.raises(RuntimeError):
        t.to_dict([1, 1])
    with pytest.raises(RuntimeError):
        t.before_step(1, ~t.finished, output(), torch.zeros(2, 8), torch.ones(2, 8))
    with pytest.raises(ValueError):
        t.after_step(1, t.finished, t.finished, t.finished)
    with pytest.raises(ValueError):
        t.after_step(1, ~t.finished, t.finished, ~t.finished)
    t.after_step(1, ~t.finished, t.finished, t.finished)
    for lengths in [[1, 2], [1.0, 1.0], [True, True], [1]]:
        with pytest.raises(ValueError):
            t.to_dict(lengths)


@pytest.mark.parametrize('n,dt', [(0, 0.1), (True, 0.1), (1, 0), (1, float('nan')), (1, True)])
def test_invalid_constructor(n, dt):
    with pytest.raises(ValueError):
        HybridTelemetry(n, 'cpu', dt)


def test_latest_switch_refreshes_window_and_ended_rows_cannot_reactivate():
    t = HybridTelemetry(1, 'cpu', 0.1)
    for step in range(1, 11):
        run_step(t, step, output(1, switched=step in (1, 8)), [step == 10], [step == 10])
    assert t.to_dict([10])['episode_fall_within_switch_window'] == [True]
    with pytest.raises(ValueError):
        t.before_step(11, torch.ones(1, dtype=torch.bool), output(1), torch.zeros(1, 8), torch.ones(1, 8))
    run_step(t, 11, output(1, alpha=1, switched=True, command=100), [True], [True])
    r = t.to_dict([10])
    assert r['episode_switch_count'] == [2]
    assert r['episode_v10_duty'] == [0]


@pytest.mark.parametrize('key,bad', [('switched', torch.ones(2)),
                                    ('uncertain', torch.zeros(2, 1, dtype=torch.bool)),
                                    ('actions', torch.zeros(2, 7)),
                                    ('enter_span', torch.zeros(2, 1))])
def test_gate_shapes_and_boolean_types(key, bad):
    t = HybridTelemetry(2, 'cpu', 0.1)
    g = output()
    g[key] = bad
    with pytest.raises(ValueError):
        t.before_step(1, ~t.finished, g, torch.zeros(2, 8), torch.ones(2, 8))
