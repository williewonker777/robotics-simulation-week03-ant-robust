import pytest
import torch
from week03_ant.command_math import command_features


def geometry():
    return dict(target_height=torch.tensor([.44, .58]), body_clearance=torch.tensor([.44, .58]),
                valid=torch.ones(2, dtype=torch.bool), motion_valid=torch.ones(2, dtype=torch.bool))


def test_scaling_and_clear_is_not_unknown():
    values = geometry()
    torch.testing.assert_close(command_features(values), torch.tensor([[0., 0., 1.], [1., 1., 1.]]))
    values['valid'][0] = False
    assert command_features(values)[0].eq(0).all()


@pytest.mark.parametrize('key', ['valid', 'motion_valid', 'target_height', 'body_clearance'])
def test_nonfinite_abstains(key):
    values = geometry()
    values[key] = values[key].float()
    values[key][0] = float('nan')
    out = command_features(values)
    assert torch.isfinite(out).all() and out[0].eq(0).all()


def test_clipping_motion_and_shape():
    values = geometry()
    values['body_clearance'] = torch.tensor([-100., 100.])
    assert command_features(values)[:, 1].tolist() == [-8., 8.]
    values['motion_valid'][1] = False
    assert command_features(values)[1].eq(0).all()
    values['valid'] = torch.ones(2, 1)
    with pytest.raises(ValueError):
        command_features(values)
