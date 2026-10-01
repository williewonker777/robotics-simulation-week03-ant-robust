"""Paired proof and exact capacity projection negative tests."""
import copy
import sys
import pytest
from week03_ant.teammate_study_v25 import ROOT, init_path, RUNS, CONTROLLERS, model_entry
sys.path.insert(0,str(ROOT/'scripts'))
from run_teammate_training_v25 import project_main_initial, paired_initial, train_command


def initial(arm='control'):
    return dict(arm=arm,training_seed=61,actual_seed={'live':61},num_envs=4096,dt=1/60,
        normalized_parameters={'agent':{'seed':'61','max_iterations':'2','algorithm':{'entropy_coef':'.005' if arm=='combined' else '.002'}},
            'env':{'seed':'61','rewards':{'teammate_recovery':{'params':{'enabled':str(arm!='control').lower()}}}}},
        observation_dimensions=[4096,91],initial_state_sha256={'x':'a'},initial_prefix_sha256='prefix',policy_state_sha256={'x':'b'},
        rng_sha256={'x':'c'},initial_policy_parity={'actor':True},optimizer_state_empty=True,iteration=0,common_step_counter=0)


def test_projection_only_maxiterations():
    source=initial(); projected=project_main_initial(source)
    assert source['normalized_parameters']['agent']['max_iterations']=='2'
    projected['normalized_parameters']['agent']['max_iterations']='2'
    assert source==projected
    source['num_envs']=256
    with pytest.raises(ValueError):project_main_initial(source)


@pytest.mark.parametrize('arm',['recovery','combined'])
def test_paired_allowed_treatments_only(arm):
    paired_initial(initial(),initial(arm))
    bad=initial(arm); bad['normalized_parameters']['env']['seed']='62'
    with pytest.raises(ValueError):paired_initial(initial(),bad)
    bad=initial(arm); bad['rng_sha256']['x']='wrong'
    with pytest.raises(ValueError):paired_initial(initial(),bad)


@pytest.mark.parametrize('run,envs,iters,device',[('control62',256,2,'cuda:1'),('control61',4096,251,'cuda:1'),
    ('control61',4096,250,'cuda:0'),('seed51',4096,250,'cuda:1')])
def test_budget_fail_closed(run,envs,iters,device):
    with pytest.raises(ValueError):train_command(run,'i','s',envs=envs,iterations=iters,device=device)


def test_common_seed_initial_and_matrix():
    assert len(RUNS)==9 and len(CONTROLLERS)==23
    assert init_path(61).name=='model_0.pt'
    with pytest.raises(ValueError):init_path(True)
    with pytest.raises(ValueError):model_entry('control62',phase='development')
