"""CPU contract proofs for v21 paired source/budget/evidence gates."""
import copy
import importlib.util
import sys

import pytest

from week03_ant.continuation_study_v21 import ROOT, TRAINING_SOURCES, TRANSITIONS
from week03_ant.contact_continuation_v21 import coefficient

sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('continuation_runner_test', ROOT / 'scripts/run_continuation_training_v21.py')
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('envs,iters,ramp', [(256, 2, 32), (4096, 2, 4000), (4096, 250, 4000)])
def test_budget_commands(envs, iters, ramp):
    for arm in (runner.DEV_ARMS if iters == 2 else runner.ARMS):
        command = runner.train_command(arm, 'initial', 'schedule', envs=envs, iterations=iters)
        import contact_continuation_v21 as launcher
        index = command.index('--headless')
        forwarded = command[index:]
        forwarded = launcher.training_arguments(forwarded, arm, ramp_steps=ramp)
        launcher.validate_budget(forwarded, ramp)
        assert 'agent.device=cuda:1' in forwarded
        assert command[command.index('--ramp-steps') + 1] == str(ramp)
        assert command[command.index('--seed') + 1] == '51'
    assert TRANSITIONS == 32768000


@pytest.mark.parametrize('arm,envs,iters,device', [('other',4096,250,'cuda:1'), ('no_cost',256,250,'cuda:1'),
                                                  ('immediate',4096,250,'cuda:1'), ('no_cost',4096,250,'cuda:0')])
def test_undeclared_budgets(arm, envs, iters, device):
    with pytest.raises(ValueError):
        runner.train_command(arm, 'a', 'b', envs=envs, iterations=iters, device=device)


def _proof():
    envs, steps, ramp = 256, 64, 32
    factors = [coefficient('no_cost', t, ramp) for t in range(1, steps + 1)]
    initial = dict(arm='no_cost',num_envs=envs,iteration=0,common_step_counter=0,contact_slip_weight=1.,
                   optimizer_state_empty=True,observation_dimensions=[envs,91],
                   initial_policy_parity=dict(actor=True,critic=True,teacher=True,device='cuda:1'))
    schedule = dict(arm='no_cost',mode='no_cost',num_envs=envs,ramp_steps=ramp,policy_steps=steps,
                    common_step_counters=list(range(1, steps+1)),coefficients=factors,coefficient_sum=sum(factors),
                    raw_reward_mean=[-.5]*steps,scaled_reward_mean=[-.5*f for f in factors],
                    valid_rows=[envs]*steps,step_dt=[1/60]*steps,
                    raw_reward_dt_sum=[-.5*envs/60]*steps,
                    scaled_reward_dt_sum=[-.5*f*envs/60 for f in factors],
                    realized_penalty_sum=sum(.5*f*envs/60 for f in factors),
                    episode_randomization=dict(matches_raw_environment=True, assigned_sha256='a', actual_sha256='a',
                                               minimum=0,maximum=959,nonzero=255))
    rollout = dict(arm='no_cost',passive_post_action=True,rollout=dict(samples=steps,sampled_rows=1000,
                   valid_rows=1000,invalid_rows=0,invalid_fraction=0.,contact_fraction=.25,
                   terrain_contact_fraction=.1,flat_contact_fraction=.1,per_foot_contact_fraction=[.1]*4,
                   per_target_per_foot_contact_fraction=[[.1,.1]]*4))
    return initial, schedule, rollout


def test_complete_actual_reward_and_horizon_evidence():
    runner.validate_evidence(*_proof(), 'no_cost',256,2,32)


@pytest.mark.parametrize('section,key,value',[(0,'common_step_counter',1),(0,'optimizer_state_empty',False),
    (0,'contact_slip_weight',0.),(0,'observation_dimensions',[256,88]),(1,'arm','immediate'),
    (1,'valid_rows',[250]*64),(2,'passive_post_action',False)])
def test_evidence_corruption_rejected(section,key,value):
    values = _proof()
    values[section][key] = value
    with pytest.raises(ValueError):
        runner.validate_evidence(*values, 'no_cost',256,2,32)


def test_shadowed_or_unpaired_actual_horizons_rejected():
    initial, schedule, rollout = _proof()
    other = copy.deepcopy(schedule)
    other['episode_randomization']['actual_sha256'] = 'b'
    with pytest.raises(ValueError, match='shadowed'):
        runner.validate_evidence(initial,other,rollout,'no_cost',256,2,32)
    with pytest.raises(ValueError, match='horizons differ'):
        runner.paired_randomization(schedule,other)


def test_manifest_contains_new_task_plan_and_all_training_proof_sources():
    assert len(TRAINING_SOURCES) == len(set(TRAINING_SOURCES))
    for name in ('src/week03_ant/tasks/contact_continuation_v21_cfg.py',
                 'scripts/run_continuation_training_v21.py', 'tests/test_continuation_harness_v21.py',
                 'docs/experiment_plans/contact_continuation_v21.md'):
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


def test_exact_checkpoint_comparison_detects_policy_optimizer_and_metadata_changes():
    import torch
    reference = {'model_state_dict': {'actor': torch.tensor([1., 2.])},
                 'optimizer_state_dict': {'state': {0: {'exp_avg': torch.tensor([.1]), 'step': torch.tensor(40.)}},
                                          'param_groups': [{'lr': 1.e-4, 'params': [0]}]},
                 'iter': 1, 'infos': None}
    runner.exact_nested(reference, copy.deepcopy(reference))
    mutations = []
    actor = copy.deepcopy(reference)
    actor['model_state_dict']['actor'][0] += 1.e-6
    mutations.append(actor)
    adam = copy.deepcopy(reference)
    adam['optimizer_state_dict']['state'][0]['exp_avg'] += 1.e-7
    mutations.append(adam)
    metadata = copy.deepcopy(reference)
    metadata['optimizer_state_dict']['param_groups'][0]['lr'] = 2.e-4
    mutations.append(metadata)
    for changed in mutations:
        with pytest.raises(ValueError):
            runner.exact_nested(changed, reference)


def _saved_config(tmp_path):
    import yaml  # type: ignore[import-untyped]
    old = runner.read(runner.V20_ART / 'training/immediate_initial.json')
    logdir = ROOT / old['log_dir']
    new = tmp_path / 'new_run'
    (new / 'params').mkdir(parents=True)
    values = {name: yaml.load((logdir / 'params' / f'{name}.yaml').read_text(), Loader=yaml.BaseLoader)
              for name in ('env', 'agent')}
    env, agent = values['env'], values['agent']
    env['log_dir'] = env['io_descriptors_output_dir'] = str(new)
    env['rewards']['contact_slip']['func'] = 'week03_ant.tasks.contact_continuation_v21_cfg:ContinuationContactSlipReward'
    env['rewards']['contact_slip']['params']['mode'] = 'no_cost'
    agent['experiment_name'] = 'week03_ant_contact_continuation_v21'
    agent['load_run'] = 'v21_init'
    agent['run_name'] = 'v21_paired_no_cost'
    for name, value in values.items():
        (new / 'params' / f'{name}.yaml').write_text(yaml.dump(value))
    return new, values, old


def test_canonical_yaml_exactly_matches_historical_except_allowlisted_identity(tmp_path):
    from week03_ant.continuation_study_v21 import training_parameters
    new, _, old = _saved_config(tmp_path)
    assert training_parameters(new, 'no_cost')['normalized'] == old['normalized_parameters']


@pytest.mark.parametrize('section,keys,value', [
    ('env', ('rewards','contact_slip','func'), 'other:Reward'),
    ('agent', ('experiment_name',), 'wrong_experiment'),
    ('agent', ('load_run',), 'v20_init'),
    ('agent', ('algorithm','teacher_coef'), '0.03'),
])
def test_identity_or_treatment_substitution_not_hidden(tmp_path, section, keys, value):
    import yaml  # type: ignore[import-untyped]
    new, values, _ = _saved_config(tmp_path)
    selected = values[section]
    for key in keys[:-1]:
        selected = selected[key]
    selected[keys[-1]] = value
    (new / 'params' / f'{section}.yaml').write_text(yaml.dump(values[section]))
    with pytest.raises(ValueError):
        runner.training_parameters(new, 'no_cost')


def test_unrelated_reward_change_fails_initial_pair_not_normalized_away(tmp_path):
    import yaml  # type: ignore[import-untyped]
    new, values, old = _saved_config(tmp_path)
    values['env']['rewards']['progress']['weight'] = '2.0'
    (new / 'params/env.yaml').write_text(yaml.dump(values['env']))
    changed = copy.deepcopy(old)
    changed['normalized_parameters'] = runner.training_parameters(new, 'no_cost')['normalized']
    with pytest.raises(ValueError, match='normalized_parameters'):
        runner.check_initial(changed, old)


def test_positive_replay_checks_applied_cost_before_checkpoint(monkeypatch):
    _, schedule, rollout = _proof()
    old = copy.deepcopy(schedule)
    old['raw_reward_mean'][0] = -.25
    monkeypatch.setattr(runner, 'read', lambda _: old)
    with pytest.raises(ValueError, match='reward-time trajectory'):
        runner.validate_positive_replay({'schedule':schedule,'rollout':rollout}, {'schedule':{'path':'old'}})


def test_scalar_replay_excludes_only_timing_not_learning():
    assert runner.TIMING_SCALARS == {'Perf/total_fps','Perf/collection time','Perf/learning_time',
                                     'Train/mean_reward/time','Train/mean_episode_length/time'}
    assert not any(name.startswith(('Loss/','Episode_','Policy/')) for name in runner.TIMING_SCALARS)
