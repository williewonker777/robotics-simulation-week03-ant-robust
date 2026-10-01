"""Closed-world bridge and binding checks without simulator imports."""
import sys
import pytest
from week03_ant.teammate_study_v25 import ROOT, RUNS, run_identity, entropy, TOTAL_TRANSITIONS, DEVELOPMENT_TRANSITIONS
sys.path.insert(0,str(ROOT/'scripts'))
from teammate_v25 import training_arguments
from run_teammate_training_v25 import train_command


@pytest.mark.parametrize('run',RUNS)
@pytest.mark.parametrize('envs,iterations',[(4096,2),(4096,250)])
def test_exact_budget(run,envs,iterations):
    arm,seed=run_identity(run)
    command=train_command(run,'i','s',envs=envs,iterations=iterations)
    forwarded=command[command.index('--headless'):]
    result=training_arguments(forwarded,arm,training_seed=seed)
    assert 'agent.algorithm.entropy_coef='+str(entropy(arm)) in result
    assert '--task' in result
    assert result[result.index('--seed')+1]==str(seed)
    assert 'env.scene.terrain.terrain_generator.seed=130' in result
    assert TOTAL_TRANSITIONS==294912000
    assert DEVELOPMENT_TRANSITIONS==2408448


@pytest.mark.parametrize('override',['agent.seed=99','env.seed=99','agent.algorithm.entropy_coef=0.1',
    '+env.rewards.foo.weight=1','--unknown','env.rewards.contact_slip.params.mode=ramp'])
def test_override_rejected(override):
    command=train_command('control61','i','s')
    with pytest.raises(ValueError):
        training_arguments(command[command.index('--headless'):]+[override],'control',training_seed=61)


def test_duplicates_and_wrong_initializer():
    command=train_command('control61','i','s'); args=command[command.index('--headless'):]
    with pytest.raises(ValueError): training_arguments(args+['--seed','61'],'control',training_seed=61)
    args[args.index('--load_run')+1]='v22_init'
    with pytest.raises(ValueError): training_arguments(args,'control',training_seed=61)


def test_source_contract_no_observation_action_noise():
    text=(ROOT/'src/week03_ant/tasks/teammate_v25_cfg.py').read_text()
    assert 'env.action_manager.action, env.action_manager.prev_action' in text
    assert 'state.root_ang_vel_b' in text
    assert "candidate['rewards'].pop('teammate_recovery')" in text
    assert 'return rate' in text


def _actual_method(path, class_name, method_name, namespace):
    """Execute the real source method without importing simulator-only modules."""
    import ast
    tree=ast.parse(path.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==class_name)
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==method_name)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace[method_name]


def test_actual_contact_schedule_producer_round_trip():
    from types import SimpleNamespace
    from teammate_v25 import contact_schedule
    from week03_ant.contact_continuation_v21 import audit_schedule
    path=ROOT/'src/week03_ant/tasks/contact_continuation_v21_cfg.py'
    method=_actual_method(path,'ContinuationContactSlipReward','schedule',{})
    # Actual producer method receives representative already-sampled reward history.
    reward=SimpleNamespace(history=[(step,0.,-.5,0.,4,1/60,-2/60,0.) for step in range(1,65)])
    reward.schedule=lambda mode,ramp_steps:method(reward,mode,ramp_steps)
    with pytest.raises(ValueError,match='missing environment count'):
        audit_schedule(reward.schedule('no_cost',4000),mode='no_cost',policy_steps=64,ramp_steps=4000)
    record=contact_schedule(reward,4)
    assert record['num_envs']==4
    assert audit_schedule(record,mode='no_cost',policy_steps=64,ramp_steps=4000)


def test_installed_reward_manager_integrates_actual_rate_once_and_preserves_history():
    import importlib.util
    from types import SimpleNamespace
    import torch
    from week03_ant.teammate_recovery_v25 import recovery_terms, applied_rate
    sdk=next(iter(importlib.util.find_spec('isaaclab').submodule_search_locations))
    from pathlib import Path
    path=Path(sdk)/'managers/reward_manager.py'
    compute=_actual_method(path,'RewardManager','compute',{'torch':torch})
    reset=_actual_method(path,'RewardManager','reset',{'torch':torch})
    class Term:
        def __init__(self): self.history=[]
        def __call__(self,env,enabled=False):
            self.history.append(env.common_step_counter)
            terms=recovery_terms(torch.full((2,),.31),torch.full((2,),.44),torch.ones(2,dtype=torch.bool),
                torch.ones(2),torch.ones(2,8),torch.zeros(2,8),torch.zeros(2,3))
            return applied_rate(terms,enabled)
        def reset(self,env_ids=None): pass
    for enabled in (False,True):
        term=Term(); cfg=SimpleNamespace(func=term,weight=1.,params={'enabled':enabled})
        env=SimpleNamespace(common_step_counter=0,max_episode_length_s=16.)
        manager=SimpleNamespace(_reward_buf=torch.zeros(2),_term_names=['new'],_term_cfgs=[cfg],
            _class_term_cfgs=[cfg],_env=env,_episode_sums={'new':torch.zeros(2)},_step_reward=torch.zeros(2,1))
        reset(manager)
        assert term.history==[]  # Actual reset does not invoke reward __call__.
        for step in range(1,65):
            env.common_step_counter=step
            result=compute(manager,1/60)
            expected=(-2-.08)/60 if enabled else 0.
            assert torch.allclose(result,torch.full((2,),expected))
            reset(manager)
        assert term.history==list(range(1,65))


def test_parser_main_requires_own_existing_expected_before_startup(tmp_path,monkeypatch,capsys):
    import teammate_v25 as bridge
    monkeypatch.setattr(bridge,'ART',tmp_path/'artifacts')
    canonical=bridge.ART/'expected_main/control61_initial.json'
    canonical.parent.mkdir(parents=True)
    args=train_command('control61',tmp_path/'initial.json',tmp_path/'schedule.json')[2:]+['--dry-run']
    with pytest.raises(ValueError,match='exact frozen expected'):
        bridge.main(args)
    with pytest.raises(ValueError,match='exact frozen expected'):
        bridge.main(args+['--expected-initial',str(canonical.with_name('recovery61_initial.json'))])
    with pytest.raises(FileNotFoundError):
        bridge.main(args+['--expected-initial',str(canonical)])
    canonical.write_text('{}')
    bridge.main(args+['--expected-initial',str(canonical)])
    assert '--max_iterations' in capsys.readouterr().out
    alias=tmp_path/'alias.json'; alias.symlink_to(canonical)
    with pytest.raises(ValueError,match='exact frozen expected'):
        bridge.main(args+['--expected-initial',str(alias)])


@pytest.mark.parametrize('envs',[256,4096])
def test_parser_development_rejects_expected_main_before_startup(tmp_path,envs):
    import teammate_v25 as bridge
    args=train_command('control61',tmp_path/'initial.json',tmp_path/'schedule.json',
                       envs=envs,iterations=2)[2:]+['--dry-run']
    bridge.main(args)
    with pytest.raises(ValueError,match='development cannot substitute'):
        bridge.main(args+['--expected-initial',str(tmp_path/'anything.json')])
