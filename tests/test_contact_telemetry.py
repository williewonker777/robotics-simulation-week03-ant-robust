"""First-episode contact reporting must not relabel missing/stale feet as success."""

import copy

import pytest
import torch

from week03_ant.contact_telemetry import ContactTelemetry, aggregate_contact, audit_contact


def metrics():
    return dict(valid=torch.tensor([True, True, False]),
                contact=torch.tensor([[True, False, False, False], [False] * 4, [False] * 4]),
                tip_speed=torch.tensor([[1.2, 0., 0., 0.], [0.] * 4, [0.] * 4]),
                contact_fraction=torch.tensor([.25, 0., 0.]),
                contacted_tip_speed=torch.tensor([1.2, 0., 0.]),
                bounded_cost=torch.tensor([.25, 0., 0.]))


def test_sample_mask_and_contacted_speed_cap():
    report = ContactTelemetry(3, 'cpu')
    report.before_step(torch.tensor([True] * 3), metrics(), sample_valid=torch.tensor([True, True, False]))
    report.before_step(torch.tensor([True] * 3), metrics(), sample_valid=torch.tensor([False, True, True]))
    raw = report.to_dict(torch.tensor([2] * 3))
    assert raw['valid_steps'] == [1, 2, 0]
    assert raw['no_contact_steps'] == [0, 2, 0]
    assert raw['contact_foot_samples'] == [1, 0, 0]
    assert raw['speed_capped_foot_samples'] == [1, 0, 0]
    rows = [{'steps': 2} for _ in range(3)]
    audit_contact({'contact_telemetry': raw}, rows)
    aggregated = aggregate_contact(rows)
    assert aggregated['no_contact_fraction'] == pytest.approx(2 / 3)
    assert aggregated['contacted_speed_cap_fraction'] == 1.
    assert aggregated['invalid_steps'] == 3


@pytest.mark.parametrize('key,value', [
    ('valid_steps', [3, 0, 0]), ('no_contact_steps', [2, 2, 0]),
    ('contact_foot_samples', [5, 0, 0]), ('speed_capped_foot_samples', [2, 0, 0]),
    ('contact_fraction_sum', [.3, 0., 0.]), ('bounded_cost_sum', [2., 0., 0.]),
])
def test_audit_rejects_inconsistent_counts_and_bounds(key, value):
    report = ContactTelemetry(3, 'cpu')
    report.before_step(torch.tensor([True] * 3), metrics())
    raw = report.to_dict(torch.tensor([1] * 3))
    corrupted = copy.deepcopy(raw)
    corrupted[key] = value
    with pytest.raises(ValueError):
        audit_contact({'contact_telemetry': corrupted}, [{'steps': 1} for _ in range(3)])


def test_inactive_invalid_and_finite_checks():
    report = ContactTelemetry(3, 'cpu')
    bad = metrics()
    bad['tip_speed'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        report.before_step(torch.tensor([True] * 3), bad)
    bad = metrics()
    bad['contact_fraction'][0] = .5
    report.before_step(torch.tensor([True] * 3), bad)
    with pytest.raises(ValueError, match='fraction'):
        audit_contact({'contact_telemetry': report.to_dict(torch.tensor([1] * 3))}, [{'steps': 1} for _ in range(3)])
