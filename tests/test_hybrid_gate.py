import pytest
import torch

from week03_ant.foothold_math import foothold_grid
from week03_ant.hybrid_gate import DepthPolicyGate, GateConfig, PRESETS


def inputs(n=1):
    return torch.zeros(n, 825), torch.ones(n, 825, dtype=torch.bool), torch.zeros(n, 8), torch.ones(n, 8)


def gate(n=1, **kwargs):
    kwargs.setdefault("enter_steps", 1)
    return DepthPolicyGate(n, "cpu", .05, GateConfig(.18, .12, **kwargs))


def rough(h):
    h[:, foothold_grid().reshape(-1, 2)[:, 0] >= 1] = .4
    return h


def test_flat_step_gap_slope_and_offset():
    for kind in ("flat", "step", "gap", "slope"):
        args = inputs()
        x = foothold_grid().reshape(-1, 2)[:, 0]
        if kind == "step":
            rough(args[0])
        elif kind == "gap":
            args[0][:, (x >= .5) & (x <= 1.5)] = -.4
        elif kind == "slope":
            args[0][:] = x * .2
        out = gate().step(*args)
        assert out["target_v10"].item() == (kind != "flat")
        args[0].add_(7.)
        shifted = gate().step(*args)
        for key in ("enter_span", "enter_edge", "retain_span", "retain_edge"):
            torch.testing.assert_close(out[key], shifted[key], atol=1e-6, rtol=1e-5)


def test_unknown_holds_target_and_resets_clear_time():
    args = inputs()
    g = gate()
    rough(args[0])
    assert g.step(*args)["target_v10"].item()
    args[0].fill_(torch.nan)
    for _ in range(20):
        out = g.step(*args)
        assert out["target_v10"].item() and out["uncertain"].item()
        assert not out["switched"].item()
        assert out["coverage"].item() == 0
    g.reset()
    assert not g.step(*args)["target_v10"].item()


def test_hysteresis_dwell_clear_and_fade():
    args = inputs()
    g = gate()
    rough(args[0])
    first = g.step(*args)
    assert first["switched"].item()
    assert first["alpha"].item() == pytest.approx(1 / 3)
    for _ in range(2):
        out = g.step(*args)
    assert torch.equal(out["actions"], args[3])
    # Hysteresis band retains v10 even after dwell elapses.
    args[0].mul_(.35)
    for _ in range(12):
        assert g.step(*args)["target_v10"].item()
    args[0].zero_()
    for _ in range(5):
        assert g.step(*args)["target_v10"].item()
    out = g.step(*args)
    assert out["switched"].item() and not out["target_v10"].item()
    rough(args[0])
    for _ in range(9):
        assert not g.step(*args)["target_v10"].item()
    assert g.step(*args)["target_v10"].item()


def test_initial_entry_but_minimum_dwell_before_exit():
    args = inputs()
    g = gate()
    rough(args[0])
    g.step(*args)
    args[0].zero_()
    for _ in range(9):
        assert g.step(*args)["target_v10"].item()
    assert not g.step(*args)["target_v10"].item()


def test_clear_must_be_continuous():
    args = inputs()
    g = gate(min_dwell_seconds=.05)
    rough(args[0])
    g.step(*args)
    args[0].zero_()
    for _ in range(5):
        g.step(*args)
    args[1].zero_()
    g.step(*args)
    args[1].fill_(True)
    for _ in range(5):
        assert g.step(*args)["target_v10"].item()
    assert not g.step(*args)["target_v10"].item()


def test_roi_boundaries_and_rear_retention():
    args = inputs()
    g = gate()
    assert g._enter_roi.sum().item() == 21 * 21
    assert g._retain_roi.sum().item() == 33 * 23
    x = foothold_grid().reshape(-1, 2)[:, 0]
    args[0][:, x < -.2] = .4
    out = g.step(*args)
    assert out["enter_span"].item() == 0
    assert out["retain_span"].item() == pytest.approx(.4)
    assert not out["target_v10"].item()
    rough(args[0])
    g.step(*args)
    args[0][:, x >= 0] = 0
    for _ in range(20):
        assert g.step(*args)["target_v10"].item()


@pytest.mark.parametrize("mode,index,alpha", [("v5", 2, 0), ("v10", 3, 1)])
def test_fixed_modes_are_exact(mode, index, alpha):
    args = inputs(2)
    args[2].normal_()
    args[3].normal_()
    out = gate(2).step(*args, mode=mode)
    assert torch.equal(out["actions"], args[index])
    assert torch.all(out["alpha"] == alpha)
    assert not out["switched"].any()


def test_batch_reset_and_output_snapshots():
    args = inputs(2)
    g = gate(2)
    rough(args[0])
    first = g.step(*args)
    g.reset(torch.tensor([True, False]))
    assert g.alpha[0] == 0 and g.alpha[1] > 0
    assert first["target_v10"].all()  # saved telemetry cannot be mutated by reset
    out = g.step(*args)
    assert out["switched"].tolist() == [True, False]
    assert out["alpha"][1] > out["alpha"][0]


@pytest.mark.parametrize("field,value", [("span_enter", 0), ("edge_enter", float("nan")),
    ("exit_ratio", 1), ("min_coverage", 1.1), ("blend_seconds", 0),
    ("clear_seconds", -1), ("min_dwell_seconds", float("inf")), ("span_enter", True)])
def test_invalid_configs(field, value):
    values = dict(span_enter=.18, edge_enter=.12)
    values[field] = value
    with pytest.raises(ValueError):
        GateConfig(**values)


def test_invalid_inputs():
    g = gate()
    for index, replacement in [(0, torch.zeros(1, 824)), (1, torch.ones(1, 825)),
        (2, torch.zeros(1, 7)), (3, torch.full((1, 8), torch.inf))]:
        args = list(inputs())
        args[index] = replacement
        with pytest.raises(ValueError):
            g.step(*args)
    with pytest.raises(ValueError):
        g.reset(torch.ones(1))
    with pytest.raises(ValueError):
        g.step(*inputs(), mode="unknown")
    for dt in (0, float("nan")):
        with pytest.raises(ValueError):
            DepthPolicyGate(1, "cpu", dt, PRESETS["balanced"])


def test_coverage_boundary():
    args = inputs()
    g = gate()
    indices = g._enter_roi.flatten().nonzero().flatten()
    rough(args[0])
    args[1][:, indices[:45]] = False  # 396/441 < .9
    assert g.step(*args)["uncertain"].item()
    args[1].fill_(True)
    assert not g.step(*args)["uncertain"].item()


def test_two_frame_entry_and_unknown_confirmation_reset():
    args = inputs()
    rough(args[0])
    g = gate(enter_steps=2)
    assert not g.step(*args)["target_v10"].item()
    args[1].zero_()
    assert g.step(*args)["uncertain"].item()
    args[1].fill_(True)
    assert not g.step(*args)["target_v10"].item()
    assert g.step(*args)["target_v10"].item()
    assert PRESETS["balanced"].enter_steps == 2
    assert PRESETS["balanced"].min_coverage == .9


def test_edge_coverage_can_reject_good_ray_coverage():
    args = inputs()
    rough(args[0])
    g = gate()
    # Isolated missing rays spoil four neighbors each: rays >90%, edges <90%.
    mask = args[1].reshape(1, 25, 33)
    mask[:, 4:21:3, 14:32:3] = False
    out = g.step(*args)
    assert out["enter_coverage"].item() >= .9
    assert out["enter_edge_coverage"].item() < .9
    assert out["uncertain"].item()
    assert not out["target_v10"].item()


@pytest.mark.parametrize("value", [0, 1.5, True])
def test_invalid_enter_steps(value):
    with pytest.raises(ValueError):
        GateConfig(.18, .12, enter_steps=value)


def test_inference_step_then_reset_outside_context_matches_plain_calls():
    args = inputs(2)
    rough(args[0])
    mixed, plain = gate(2), gate(2)
    for mode in ("hybrid", "v10", "v5", "hybrid"):
        with torch.inference_mode():
            actual = mixed.step(*args, mode=mode)
        expected = plain.step(*args, mode=mode)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        done = torch.tensor([True, False])
        mixed.reset(done)
        plain.reset(done)
        actual = mixed.step(*args, mode=mode)
        expected = plain.step(*args, mode=mode)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    mixed.reset()
    assert not mixed.target_v10.any()
    assert not mixed.alpha.any()
