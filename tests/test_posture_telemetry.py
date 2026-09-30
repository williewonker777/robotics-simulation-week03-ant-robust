import pytest
import torch

from week03_ant.posture_math import terrain_posture_terms
from week03_ant.posture_telemetry import PostureTelemetry


def terms():
    return terrain_posture_terms(torch.full((2, 825), -.44), torch.ones(2, 825, dtype=torch.bool),
                                 torch.zeros(2, 4, 3), torch.ones(2, 4), torch.tensor([2., .5]), torch.ones(2))


def test_first_episode_only_and_denominator():
    meter = PostureTelemetry(2, "cpu")
    meter.before_step(torch.tensor([True, True]), terms())
    meter.before_step(torch.tensor([True, False]), terms())
    data = meter.to_dict(torch.tensor([2, 1]))
    assert data["active_steps"] == [2, 1]
    assert data["groups"]["clear"]["steps"] == [2, 1]
    assert data["groups"]["all_valid"]["low_speed_steps"] == [0, 1]
    assert data["groups"]["all_valid"]["foot_samples"] == [8, 4]
    assert data["groups"]["rough"]["steps"] == [0, 0]
    assert data["groups"]["clear"]["forward_speed_sum"] == [4., .5]
    with pytest.raises(ValueError, match="duration"):
        meter.to_dict(torch.tensor([2, 2]))


def test_unknown_is_not_clear_and_inactive_nan_cannot_leak():
    meter = PostureTelemetry(2, "cpu")
    sample = terms()
    sample["valid"][0] = False
    for key in ("body_clearance", "reward", "foot_clearance"):
        sample[key][1] = torch.nan
    meter.before_step(torch.tensor([True, False]), sample)
    data = meter.to_dict(torch.tensor([1, 0]))
    assert data["groups"]["all_valid"]["steps"] == [0, 0]
    assert data["groups"]["clear"]["steps"] == [0, 0]


def test_valid_nonfinite_is_rejected():
    sample = terms()
    sample["reward"][0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        PostureTelemetry(2, "cpu").before_step(torch.tensor([True, True]), sample)


def test_partition_thresholds():
    meter = PostureTelemetry(2, "cpu")
    sample = terms()
    sample["severity"] = torch.tensor([.1, .9])
    meter.before_step(torch.tensor([True, True]), sample)
    sample["severity"] = torch.tensor([.5, .5])
    meter.before_step(torch.tensor([True, True]), sample)
    data = meter.to_dict(torch.tensor([2, 2]))["groups"]
    assert data["clear"]["steps"] == [1, 0]
    assert data["rough"]["steps"] == [0, 1]
    assert data["intermediate"]["steps"] == [1, 1]
