"""CPU dimensional, boundary, invalid-input and rate-integration contracts."""
import pytest
import torch
from week03_ant.teammate_recovery_v25 import recovery_terms, applied_rate


def inputs(n=3):
    return [torch.full((n,),.44),torch.full((n,),.44),torch.ones(n,dtype=torch.bool),
            torch.ones(n),torch.zeros(n,8),torch.zeros(n,8),torch.zeros(n,3)]


def test_flat_target_and_exact_control():
    terms=recovery_terms(*inputs())
    assert torch.equal(terms['rate'],torch.zeros(3))
    args=inputs(); args[0][:]=.2
    terms=recovery_terms(*args)
    assert torch.equal(terms['rate'],torch.full((3,),-2.))
    assert torch.equal(applied_rate(terms,False),torch.zeros(3))
    assert torch.equal(applied_rate(terms,True),terms['rate'])


def test_warning_monotonicity_and_cap():
    args=inputs(); args[0][:]=torch.tensor([.48,.395,.31]); args[1][:]=.58
    terms=recovery_terms(*args)
    assert torch.allclose(terms['clearance_risk'],torch.tensor([0.,.25,1.]))
    args=inputs(); args[3][:]=torch.tensor([.93,.715,.5])
    assert torch.allclose(recovery_terms(*args)['tilt_risk'],torch.tensor([0.,.25,1.]))


def test_actions_are_sum_not_mean_derivative_and_angular_body_xy_only():
    args=inputs(); args[4][:]=1.; args[6][:]=torch.tensor([2.,3.,100.])
    rate=recovery_terms(*args)['rate']
    assert torch.allclose(rate,torch.full((3,),-.01*8-.025*13))
    assert torch.allclose(rate/60,torch.full((3,),(-.01*8-.025*13)/60))


def test_invalid_scan_abstains_only_clearance():
    args=inputs(); args[2][:]=False; args[0][:]=float('nan'); args[1][:]=float('nan'); args[3][:]=.5
    terms=recovery_terms(*args)
    assert terms['clearance_abstained'].all()
    assert torch.equal(terms['rate'],torch.full((3,),-2.))


@pytest.mark.parametrize('index',[0,1,3,4,5,6])
def test_nonfinite_valid_physical_state_rejected(index):
    args=inputs(); args[index].flatten()[0]=float('nan')
    with pytest.raises(ValueError): recovery_terms(*args)


@pytest.mark.parametrize('index',[0,1,2,3,4,5,6])
def test_malformed_shape(index):
    args=inputs(); args[index]=args[index][:2]
    with pytest.raises(ValueError): recovery_terms(*args)


def test_invalid_target_and_flag():
    args=inputs(); args[1][:]=.31
    with pytest.raises(ValueError): recovery_terms(*args)
    with pytest.raises(ValueError): applied_rate({'rate':torch.zeros(1)},1)
