"""CPU reward semantics, independent of Isaac Sim."""
import pytest
import torch
from week03_ant.contact_math import contact_slip_metrics


def inputs(n=1):
    return torch.zeros(n, 4, 3), torch.zeros(n, 4, 3)


def test_no_contact_stationary_contact_and_vertical_motion():
    f, v = inputs()
    v[..., :2] = 20.
    out = contact_slip_metrics(f, v)
    assert out['reward'].item() == 0 and out['no_contact'].item()
    f[..., 2] = 3.
    v.zero_(); v[..., 2] = 5.
    out = contact_slip_metrics(f, v)
    assert out['reward'].item() == 0 and not out['no_contact'].item()
    assert out['contact_fraction'].item() == 1 and out['contacted_tip_speed'].item() == 0


def test_threshold_four_foot_denominator_and_saturation():
    f, v = inputs()
    f[:, :, 2] = torch.tensor([2., 2.001, 3., 4.])
    v[:, :, 0] = torch.tensor([99., .5, 1., 20.])
    out = contact_slip_metrics(f, v)
    torch.testing.assert_close(out['reward'], torch.tensor([-(.25 + 1 + 1) / 4]))
    assert out['contact'].tolist() == [[False, True, True, True]]
    assert out['contact_fraction'].item() == .75
    torch.testing.assert_close(out['contacted_tip_speed'], torch.tensor([21.5 / 3]))


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('which', [0, 1])
def test_invalid_rows_abstain_not_airborne(invalid, which):
    f, v = inputs(2); f[..., 2] = 4; v[..., 0] = .5
    (f, v)[which][0, 0, 0] = invalid
    out = contact_slip_metrics(f, v)
    assert out['valid'].tolist() == [False, True]
    assert not out['no_contact'][0] and out['reward'][0] == 0
    for value in out.values():
        assert torch.isfinite(value).all()
    assert out['reward'][1] == -.25


def test_rotation_mirror_bounds_and_extreme_finite():
    rng = torch.Generator().manual_seed(12)
    f = torch.randn(40, 4, 3, generator=rng) * 4
    v = torch.randn(40, 4, 3, generator=rng)
    base = contact_slip_metrics(f, v)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    mirror = torch.diag(torch.tensor([-1., 1., 1.]))
    for transform in (rotation, mirror):
        out = contact_slip_metrics(f @ transform, v @ transform)
        for key in base:
            torch.testing.assert_close(base[key], out[key])
    assert ((base['reward'] >= -1) & (base['reward'] <= 0)).all()
    f[0] = torch.finfo(f.dtype).max
    out = contact_slip_metrics(f, v)
    assert not out['valid'][0]
    assert all(torch.isfinite(value).all() for value in out.values())


def test_input_guards():
    f, v = inputs()
    for ff, vv in ((f[:, :3], v), (f, v[:, :3]), (f.long(), v), (f, v.double())):
        with pytest.raises(ValueError):
            contact_slip_metrics(ff, vv)
