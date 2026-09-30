"""CPU contract proofs for v20 paired source/budget/evidence gates."""
import copy
import importlib.util
import sys

import pytest

from week03_ant.curriculum_study_v20 import ROOT, TRAINING_SOURCES, TRANSITIONS
from week03_ant.contact_curriculum_v20 import coefficient

sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('curriculum_runner_test', ROOT / 'scripts/run_curriculum_training_v20.py')
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('envs,iters,ramp', [(256, 2, 32), (4096, 2, 4000), (4096, 250, 4000)])
def test_budget_commands(envs, iters, ramp):
    for arm in runner.ARMS:
        command = runner.train_command(arm, 'initial', 'schedule', envs=envs, iterations=iters)
        import contact_curriculum_v20 as launcher
        index = command.index('--headless')
        forwarded = command[index:]
        forwarded = launcher.training_arguments(forwarded, arm, ramp_steps=ramp)
        launcher.validate_budget(forwarded, ramp)
        assert 'agent.device=cuda:1' in forwarded
        assert command[command.index('--ramp-steps') + 1] == str(ramp)
        assert command[command.index('--seed') + 1] == '51'
    assert TRANSITIONS == 32768000


@pytest.mark.parametrize('arm,envs,iters,device', [('other',4096,250,'cuda:1'), ('ramped',256,250,'cuda:1'),
                                                  ('immediate',4096,249,'cuda:1'), ('ramped',4096,250,'cuda:0')])
def test_undeclared_budgets(arm, envs, iters, device):
    with pytest.raises(ValueError):
        runner.train_command(arm, 'a', 'b', envs=envs, iterations=iters, device=device)


def _proof():
    envs, steps, ramp = 256, 64, 32
    factors = [coefficient('ramped', t, ramp) for t in range(1, steps + 1)]
    initial = dict(arm='ramped',num_envs=envs,iteration=0,common_step_counter=0,contact_slip_weight=1.,
                   optimizer_state_empty=True,observation_dimensions=[envs,91],
                   initial_policy_parity=dict(actor=True,critic=True,teacher=True,device='cuda:1'))
    schedule = dict(arm='ramped',mode='ramped',num_envs=envs,ramp_steps=ramp,policy_steps=steps,
                    common_step_counters=list(range(1, steps+1)),coefficients=factors,coefficient_sum=sum(factors),
                    raw_reward_mean=[-.5]*steps,scaled_reward_mean=[-.5*f for f in factors],
                    valid_rows=[envs]*steps,step_dt=[1/60]*steps,
                    raw_reward_dt_sum=[-.5*envs/60]*steps,
                    scaled_reward_dt_sum=[-.5*f*envs/60 for f in factors],
                    realized_penalty_sum=sum(.5*f*envs/60 for f in factors),
                    episode_randomization=dict(matches_raw_environment=True, assigned_sha256='a', actual_sha256='a',
                                               minimum=0,maximum=959,nonzero=255))
    rollout = dict(arm='ramped',passive_post_action=True,rollout=dict(samples=steps,sampled_rows=1000,
                   valid_rows=1000,invalid_rows=0,invalid_fraction=0.,contact_fraction=.25,
                   terrain_contact_fraction=.1,flat_contact_fraction=.1,per_foot_contact_fraction=[.1]*4,
                   per_target_per_foot_contact_fraction=[[.1,.1]]*4))
    return initial, schedule, rollout


def test_complete_actual_reward_and_horizon_evidence():
    runner.validate_evidence(*_proof(), 'ramped',256,2,32)


@pytest.mark.parametrize('section,key,value',[(0,'common_step_counter',1),(0,'optimizer_state_empty',False),
    (0,'contact_slip_weight',0.),(0,'observation_dimensions',[256,88]),(1,'arm','immediate'),
    (1,'valid_rows',[250]*64),(2,'passive_post_action',False)])
def test_evidence_corruption_rejected(section,key,value):
    values = _proof()
    values[section][key] = value
    with pytest.raises(ValueError):
        runner.validate_evidence(*values, 'ramped',256,2,32)


def test_shadowed_or_unpaired_actual_horizons_rejected():
    initial, schedule, rollout = _proof()
    other = copy.deepcopy(schedule)
    other['episode_randomization']['actual_sha256'] = 'b'
    with pytest.raises(ValueError, match='shadowed'):
        runner.validate_evidence(initial,other,rollout,'ramped',256,2,32)
    with pytest.raises(ValueError, match='horizons differ'):
        runner.paired_randomization(schedule,other)


def test_manifest_contains_new_task_plan_and_all_training_proof_sources():
    assert len(TRAINING_SOURCES) == len(set(TRAINING_SOURCES))
    for name in ('src/week03_ant/tasks/contact_curriculum_v20_cfg.py',
                 'scripts/run_curriculum_training_v20.py', 'tests/test_curriculum_harness_v20.py',
                 'docs/experiment_plans/contact_curriculum_v20.md'):
        assert name in TRAINING_SOURCES


def test_no_implicit_log_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(runner,'WORK',tmp_path)
    (tmp_path / 'used.log').write_text('preserve')
    with pytest.raises(FileExistsError):
        runner.run(['false'],'used')
    assert (tmp_path / 'used.log').read_text() == 'preserve'


def test_retry_requires_hashed_infrastructure_reason(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(runner, 'ART', tmp_path)
    runner.validate_retry('preflight', 1)
    with pytest.raises(FileNotFoundError):
        runner.validate_retry('preflight', 2)
    path = tmp_path / 'preflight_attempt02_infrastructure_reason.json'
    path.write_text(json.dumps(dict(phase='preflight', attempt=2, reason='bad metric wants another try',
                                    infrastructure_only=False, prior_evidence_sha256={'old': 'hash'})))
    with pytest.raises(ValueError, match='infrastructure reason'):
        runner.validate_retry('preflight', 2)
    calls = []
    monkeypatch.setattr(runner, 'verify_hashes', calls.append)
    path.write_text(json.dumps(dict(phase='preflight', attempt=2, reason='Cache infrastructure repaired before freeze',
                                    infrastructure_only=True, prior_evidence_sha256={'old': 'hash'})))
    runner.validate_retry('preflight', 2)
    assert calls == [{'old': 'hash'}]
