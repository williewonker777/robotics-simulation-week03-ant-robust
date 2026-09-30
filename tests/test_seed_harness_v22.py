"""CPU contract proofs for v22 paired source/budget/evidence gates."""
import copy
import importlib.util
import sys

import pytest

from week03_ant.seed_study_v22 import ROOT, TRAINING_SOURCES, TRANSITIONS
from week03_ant.contact_continuation_v21 import coefficient

sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('continuation_runner_test', ROOT / 'scripts/run_seed_training_v22.py')
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('envs,iters,ramp', [(256, 2, 32), (4096, 2, 4000), (4096, 250, 4000)])
def test_budget_commands(envs, iters, ramp):
    for seed in runner.TRAIN_SEEDS:
        command = runner.train_command(seed, 'initial', 'schedule', envs=envs, iterations=iters)
        assert command[command.index('--training-seed') + 1] == str(seed)
        assert command[command.index('--seed') + 1] == str(seed)
        assert command[command.index('--arm') + 1] == 'no_cost'
        assert command[command.index('--load_run') + 1] == 'v22_init'
        assert command[command.index('--ramp-steps') + 1] == str(ramp)
    assert TRANSITIONS == 32768000
    assert runner.TOTAL_TRANSITIONS == 98304000
    assert runner.DEVELOPMENT_TRANSITIONS == 835584


@pytest.mark.parametrize('seed,envs,iters,device', [(50,4096,250,'cuda:1'), ('51',4096,250,'cuda:1'),
    (True,4096,250,'cuda:1'), (51,256,250,'cuda:1'), (52,4096,251,'cuda:1'), (53,4096,250,'cuda:0')])
def test_undeclared_budgets(seed, envs, iters, device):
    with pytest.raises(ValueError):
        runner.train_command(seed, 'a', 'b', envs=envs, iterations=iters, device=device)


def _proof():
    envs, steps, ramp = 256, 64, 32
    factors = [coefficient('no_cost', t, ramp) for t in range(1, steps + 1)]
    initial = dict(training_seed=51, actual_seed=dict.fromkeys(('requested','saved_agent','saved_environment','live_environment'),51),
                   normalized_parameters={'env':{'seed':'51'},'agent':{'seed':'51'}},arm='no_cost',num_envs=envs,iteration=0,common_step_counter=0,contact_slip_weight=1.,
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
    runner.validate_evidence(*_proof(), 51,256,2,32)


@pytest.mark.parametrize('section,key,value',[(0,'common_step_counter',1),(0,'optimizer_state_empty',False),
    (0,'contact_slip_weight',0.),(0,'observation_dimensions',[256,88]),(1,'arm','immediate'),
    (1,'valid_rows',[250]*64),(2,'passive_post_action',False)])
def test_evidence_corruption_rejected(section,key,value):
    values = _proof()
    values[section][key] = value
    with pytest.raises(ValueError):
        runner.validate_evidence(*values, 51,256,2,32)


def test_shadowed_or_unpaired_actual_horizons_rejected():
    initial, schedule, rollout = _proof()
    other = copy.deepcopy(schedule)
    other['episode_randomization']['actual_sha256'] = 'b'
    with pytest.raises(ValueError, match='shadowed'):
        runner.validate_evidence(initial,other,rollout,51,256,2,32)
    with pytest.raises(ValueError, match='horizons differ'):
        runner.paired_randomization(schedule,other)


def test_manifest_contains_new_task_plan_and_all_training_proof_sources():
    assert len(TRAINING_SOURCES) == len(set(TRAINING_SOURCES))
    for name in ('src/week03_ant/tasks/seed_continuation_v22_cfg.py',
                 'scripts/run_seed_training_v22.py', 'tests/test_seed_harness_v22.py',
                 'docs/experiment_plans/seed_continuation_v22.md'):
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




def _capacity(seed=51):
    path = runner.V21_ART / 'capacity.json'
    entry = runner.read(path)['records']['no_cost']['initial']
    value = runner.read(ROOT / entry['path'])
    for name in ('env', 'agent'):
        value['normalized_parameters'][name]['seed'] = str(seed)
    value['training_seed'] = seed
    value['actual_seed'] = dict.fromkeys(('requested','saved_agent','saved_environment','live_environment'),seed)
    return value


@pytest.mark.parametrize('seed',[51,52,53])
def test_capacity_projection_changes_only_iteration_field(seed):
    original = _capacity(seed)
    expected = copy.deepcopy(original)
    expected['normalized_parameters']['agent']['max_iterations'] = '250'
    assert runner.project_main_initial(original,seed) == expected
    assert original['normalized_parameters']['agent']['max_iterations'] == '2'


@pytest.mark.parametrize('kind',['seed','envs','iteration','live_seed'])
def test_corrupt_capacity_or_actual_seed_fails(kind):
    value = _capacity(52)
    if kind == 'seed':
        value['normalized_parameters']['env']['seed'] = '51'
    elif kind == 'envs':
        value['num_envs'] = 256
    elif kind == 'iteration':
        value['normalized_parameters']['agent']['max_iterations'] = '250'
    else:
        initial,schedule,rollout = _proof()
        initial['actual_seed']['live_environment'] = 52
        with pytest.raises(ValueError, match='live training seed'):
            runner.validate_evidence(initial,schedule,rollout,51,256,2,32)
        return
    with pytest.raises(ValueError):
        runner.project_main_initial(value,52)


def test_cross_seed_common_initial_does_not_claim_physics_rng_equal():
    first, second = _capacity(), _capacity(52)
    second['rng_sha256'] = {'different':'deliberate'}
    second['initial_state_sha256'] = {'different':'deliberate'}
    runner.common_initial(second,first,52)
    second['normalized_parameters']['agent']['algorithm']['teacher_coef'] = '0.03'
    with pytest.raises(ValueError,match='nonseed'):
        runner.common_initial(second,first,52)


def test_cross_seed_policy_change_fails():
    first,second = _capacity(),_capacity(53)
    second['policy_state_sha256']['std'] = 'changed'
    with pytest.raises(ValueError,match='common initialization'):
        runner.common_initial(second,first,53)


def test_frozen_budget_is_ordered_and_counts_only_fresh_training():
    budget = runner.frozen_budget()
    assert budget['seeds'] == [51,52,53]
    assert budget['runs'] == ['seed51','seed52','seed53']
    assert budget['total_training_transitions'] == 98304000
    assert budget['development_training_transitions'] == 835584
    assert budget['selection'] == 'fresh final model249 only'


def test_missing_full_replay_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ART',tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.verify_full_replay()


def test_positive_replay_checks_applied_cost_before_checkpoint(monkeypatch):
    _,schedule,rollout = _proof()
    old=copy.deepcopy(schedule)
    old['raw_reward_mean'][0] = -.25
    monkeypatch.setattr(runner,'read',lambda _:old)
    with pytest.raises(ValueError,match='reward-time trajectory'):
        runner.validate_positive_replay({'schedule':schedule,'rollout':rollout},{'schedule':{'path':'old'}},249)


def test_scalar_replay_excludes_only_five_timing_tags():
    assert runner.TIMING_SCALARS == {'Perf/total_fps','Perf/collection time','Perf/learning_time',
                                    'Train/mean_reward/time','Train/mean_episode_length/time'}


@pytest.mark.parametrize('kind',['historical','alias','wrong_iteration','init'])
def test_fresh_final_rejects_historical_fallback_and_alias(tmp_path,monkeypatch,kind):
    from week03_ant.seed_study_v22 import EXPERIMENT
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    final=tmp_path/'logs/rsl_rl'/EXPERIMENT/'fresh'/'model_249.pt'
    previous=[]
    if kind=='historical':
        final=tmp_path/'historical/model_249.pt'
    elif kind=='alias':
        previous=[final]
    elif kind=='wrong_iteration':
        final=final.with_name('model_248.pt')
    else:
        final=final.parent.parent/'v22_init/model_249.pt'
    with pytest.raises(ValueError,match='fallback or checkpoint alias'):
        runner.validate_fresh_final(final,previous)


def test_fresh_final_validates_new_checkpoint_not_historical(tmp_path,monkeypatch):
    from week03_ant.seed_study_v22 import EXPERIMENT
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    final=tmp_path/'logs/rsl_rl'/EXPERIMENT/'fresh/model_249.pt'
    called=[]
    monkeypatch.setattr(runner,'checkpoint_finite',lambda path,iteration:called.append((path,iteration)))
    runner.validate_fresh_final(final)
    assert called==[(final,249)]


def test_full_replay_bad_gate_rejected_before_raw_access(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(runner,'ART',tmp_path)
    (tmp_path/'full_replay.json').write_text(json.dumps({'passed':False,'seed':51}))
    with pytest.raises(ValueError,match='full replay gate'):
        runner.verify_full_replay()


def test_frozen_budget_mutation_fails_before_evidence_access(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(runner,'ART',tmp_path)
    monkeypatch.setattr(runner,'source_hashes',lambda **_: {})
    monkeypatch.setattr(runner,'legacy_hashes',lambda: {})
    frozen=dict(source_sha256={},all_source_sha256={},legacy={},source_checkpoint_sha256=runner.PARENT_SHA,
                **runner.frozen_budget())
    frozen['total_training_transitions']=runner.TRANSITIONS
    (tmp_path/'training_frozen.json').write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match='budget changed'):
        runner.verify_training_freeze()


def _saved_config(tmp_path,seed):
    import yaml  # type: ignore[import-untyped]
    from week03_ant.seed_study_v22 import EXPERIMENT
    old=runner.read(runner.V21_ART/'training/no_cost_initial.json')
    logdir=ROOT/old['log_dir']
    new=tmp_path/'new_run'
    (new/'params').mkdir(parents=True)
    values={name:yaml.load((logdir/'params'/f'{name}.yaml').read_text(),Loader=yaml.BaseLoader)
            for name in ('env','agent')}
    values['env']['log_dir']=values['env']['io_descriptors_output_dir']=str(new)
    values['agent']['experiment_name']=EXPERIMENT
    values['agent']['load_run']='v22_init'
    values['agent']['run_name']=f'v22_main_seed{seed}'
    for name,value in values.items():
        value['seed']=str(seed)
        (new/'params'/f'{name}.yaml').write_text(yaml.dump(value))
    return new,values,old


@pytest.mark.parametrize('seed',[51,52,53])
def test_actual_saved_yaml_keeps_seed_and_only_projects_identity(tmp_path,seed):
    new,_,old=_saved_config(tmp_path,seed)
    expected=copy.deepcopy(old['normalized_parameters'])
    for name in ('env','agent'):
        expected[name]['seed']=str(seed)
    assert runner.training_parameters(new,seed)['normalized']==expected


@pytest.mark.parametrize('section,keys,value',[
    ('env',('seed',),'50'),('agent',('seed',),'53'),
    ('env',('scene','terrain','terrain_generator','seed'),'111'),
    ('agent',('load_run',),'v21_init'),
    ('agent',('algorithm','teacher_coef'),'0.03')])
def test_saved_yaml_wrong_seed_or_configuration_rejected(tmp_path,section,keys,value):
    import yaml  # type: ignore[import-untyped]
    new,values,_=_saved_config(tmp_path,52)
    selected=values[section]
    for key in keys[:-1]:
        selected=selected[key]
    selected[keys[-1]]=value
    (new/'params'/f'{section}.yaml').write_text(yaml.dump(values[section]))
    with pytest.raises(ValueError):
        runner.training_parameters(new,52)


def test_unrelated_reward_not_normalized_away(tmp_path):
    import yaml  # type: ignore[import-untyped]
    new,values,old=_saved_config(tmp_path,51)
    values['env']['rewards']['progress']['weight']='2.0'
    (new/'params/env.yaml').write_text(yaml.dump(values['env']))
    changed=copy.deepcopy(old)
    changed['normalized_parameters']=runner.training_parameters(new,51)['normalized']
    with pytest.raises(ValueError,match='normalized_parameters'):
        runner.check_initial(changed,old)
