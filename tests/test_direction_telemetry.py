import copy
import math

import pytest
import torch

from week03_ant.direction_telemetry import DirectionTelemetry, audit_direction, aggregate_direction


def metrics():
    return dict(valid=torch.tensor([True, False]), lateral_velocity=torch.tensor([-2., float('nan')]),
                heading_error=torch.tensor([-.5, float('nan')]), yaw_error=torch.tensor([-.25, float('nan')]),
                reward=torch.tensor([-.7, float('nan')]))


def test_invalid_excluded_first_episode_only_and_no_mutation():
    meter = DirectionTelemetry(2, 'cpu')
    terms = metrics()
    saved = {k: v.clone() for k, v in terms.items()}
    meter.before_step(torch.tensor([True, True]), terms)
    meter.before_step(torch.tensor([False, True]), terms)
    for key in terms:
        torch.testing.assert_close(terms[key], saved[key], equal_nan=True)
    data = dict(direction_telemetry=meter.to_dict(torch.tensor([1, 2])))
    rows = [dict(steps=1), dict(steps=2)]
    audit_direction(data, rows)
    result = aggregate_direction(rows)
    assert result['valid_coverage'] == 1 / 3
    assert result['means']['absolute_lateral_velocity'] == 2
    assert result['invalid_steps'] == 2
    assert aggregate_direction(rows[1:])['means']['reward'] is None


@pytest.mark.parametrize('mutation', ['nan', 'reward_positive', 'reward_low', 'heading', 'valid_shape', 'active_type'])
def test_reject_invalid_valid_sample(mutation):
    meter = DirectionTelemetry(2, 'cpu')
    terms = metrics()
    active = torch.tensor([True, True])
    if mutation == 'nan':
        terms['lateral_velocity'][0] = float('nan')
    elif mutation == 'reward_positive':
        terms['reward'][0] = .1
    elif mutation == 'reward_low':
        terms['reward'][0] = -1.1
    elif mutation == 'heading':
        terms['heading_error'][0] = 4
    elif mutation == 'valid_shape':
        terms['valid'] = torch.tensor([True])
    else:
        active = active.long()
    with pytest.raises(ValueError):
        meter.before_step(active, terms)
    assert not meter.active_steps.any()


@pytest.mark.parametrize('key,value', [
    ('valid_steps', [2, 0]), ('valid_steps', [True, 0]), ('active_steps', [2, 1]),
    ('reward_sum', [.1, 0.]), ('reward_sum', [-1.1, 0.]),
    ('absolute_heading_error_sum', [math.pi + .01, 0.]),
    ('absolute_lateral_velocity_sum', [-.1, 0.]),
    ('absolute_yaw_error_sum', [float('nan'), 0.]),
    ('reward_sum', [-.7, -.1]), ('valid_steps', [1]),
])
def test_audit_rejects_bounds_coverage_and_nonfinite(key, value):
    meter = DirectionTelemetry(2, 'cpu')
    meter.before_step(torch.tensor([True, True]), metrics())
    data = dict(direction_telemetry=meter.to_dict(torch.tensor([1, 1])))
    data['direction_telemetry'][key] = value
    with pytest.raises(ValueError):
        audit_direction(data, [dict(steps=1), dict(steps=1)])


def test_duration_mismatch():
    with pytest.raises(ValueError):
        DirectionTelemetry(2, 'cpu').to_dict(torch.ones(2, dtype=torch.long))
