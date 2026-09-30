"""CPU invariants for the predeclared v15 planar reward."""

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from week03_ant.direction_math import DirectionConfig, direction_metrics


def inputs(n=1):
    return (torch.tensor([[1., 0.]], dtype=torch.float64).repeat(n, 1),
            torch.tensor([[1., 0., 0.]], dtype=torch.float64).repeat(n, 1),
            torch.zeros(n, 3, dtype=torch.float64), torch.zeros(n, 3, dtype=torch.float64))


def test_aligned_zero_and_vertical_unpenalized():
    args = inputs(3)
    args[2][:] = torch.tensor([[0., 0., 0.], [100., 0., 50.], [-20., 0., -40.]])
    out = direction_metrics(*args)
    assert out['valid'].all()
    assert out['reward'].eq(0).all()


def test_exact_lateral_and_corrective_yaw():
    args = inputs(3)
    args[0][:] = torch.tensor([math.cos(.5), math.sin(.5)])
    args[3][:, 2] = torch.tensor([.5, 0., -.5])
    out = direction_metrics(*args)
    torch.testing.assert_close(out['target_yaw_rate'], torch.full((3,), .5, dtype=torch.float64))
    assert out['reward'][0] > out['reward'][1] > out['reward'][2]
    args = inputs()
    args[2][0, 1] = 2.
    assert direction_metrics(*args)['reward'].item() == pytest.approx(-.75 * (1 - math.exp(-12)))


def test_wrapped_heading_and_yaw_clipping():
    args = inputs(2)
    args[0][:] = torch.tensor([math.cos(-3.), math.sin(-3.)])
    args[1][0, :2] = torch.tensor([math.cos(3.), math.sin(3.)])
    out = direction_metrics(*args)
    assert out['heading_error'][0] == pytest.approx(2 * math.pi - 6)
    assert out['target_yaw_rate'][1] == -1


def test_rotation_and_mirror_invariance():
    generator = torch.Generator().manual_seed(47)
    args = [torch.randn(100, k, generator=generator, dtype=torch.float64) for k in (2, 3, 3, 3)]
    baseline = direction_metrics(*args)
    angle = 1.234
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]], dtype=torch.float64)
    rotated = [value.clone() for value in args]
    for value in rotated[:3]:
        value[:, :2] = value[:, :2] @ rotation.T
    result = direction_metrics(*rotated)
    for key in baseline:
        torch.testing.assert_close(result[key], baseline[key])
    mirrored = [value.clone() for value in args]
    for value in mirrored[:3]:
        value[:, 1] *= -1
    mirrored[3][:, 2] *= -1  # angular velocity is an axial vector
    result = direction_metrics(*mirrored)
    for key in baseline:
        torch.testing.assert_close(result[key], baseline[key] if key in ('reward', 'valid') else -baseline[key])


@pytest.mark.parametrize('slot', range(4))
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_all_nonfinite_rows_abstain(slot, bad):
    args = inputs(2)
    args[slot][0, -1] = bad
    out = direction_metrics(*args)
    assert out['valid'].tolist() == [False, True]
    for value in out.values():
        assert value[0] == 0
        assert torch.isfinite(value).all()


def test_zero_direction_projected_threshold_and_normalization():
    args = inputs(5)
    args[0][0] = 0
    args[1][1] = torch.tensor([.099, 0., 1.])
    args[1][2] = torch.tensor([.1, 0., 1.])
    args[0][3] *= 1e300
    args[0][4] *= 1e-300
    out = direction_metrics(*args)
    assert out['valid'].tolist() == [False, False, True, True, True]
    assert out['reward'].eq(0).all()


def test_bounds_stateless_and_no_input_mutation_or_rng():
    args = inputs(1000)
    args[2][:, 1] = torch.linspace(-1000, 1000, 1000)
    args[3][:, 2] = torch.linspace(1000, -1000, 1000)
    before = [value.clone() for value in args]
    rng = torch.get_rng_state().clone()
    out = direction_metrics(*args)
    assert out['valid'].all()
    assert ((out['reward'] >= -1) & (out['reward'] <= 0)).all()
    for key, value in direction_metrics(*args).items():
        assert torch.equal(value, out[key])
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(args, before))


def test_shape_dtype_validation_and_empty_batch():
    args = list(inputs())
    args[1] = torch.zeros(1, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        direction_metrics(*args)
    args = list(inputs())
    args[0] = args[0].long()
    with pytest.raises(ValueError):
        direction_metrics(*args)
    args = list(inputs())
    args[2] = args[2].float()
    with pytest.raises(ValueError):
        direction_metrics(*args)
    assert direction_metrics(*inputs(0))['reward'].shape == (0,)


@pytest.mark.parametrize('kwargs', [dict(lateral_scale=0), dict(yaw_weight=-1),
                                    dict(lateral_weight=.5), dict(heading_gain=float('nan')),
                                    dict(max_yaw_rate=0), dict(min_forward_xy=0)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        DirectionConfig(**kwargs)


def test_frozen_config():
    with pytest.raises(FrozenInstanceError):
        DirectionConfig().yaw_scale = 2
