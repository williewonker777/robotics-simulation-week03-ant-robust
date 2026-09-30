"""CPU proofs for passive windows, terminal precedence and frozen loop ordering."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from week03_ant.horizon import HorizonEpisodeTracker
from week03_ant.paired_horizon_v23 import PairedHorizonTracker, immutable_snapshot, snapshot_passivity


ROOT = Path(__file__).resolve().parents[1]


def evidence(n, *, distance=60., last=-999., done=None, terminated=None):
    return dict(rewards=torch.ones(n), dones=torch.tensor(done or [False]*n),
                reset_terminated=torch.tensor(terminated or [False]*n),
                current_distance=torch.full((n,),distance), last_distance=torch.full((n,),last),
                current_out_of_lane=torch.zeros(n,dtype=torch.bool), last_out_of_lane=torch.zeros(n,dtype=torch.bool),
                current_world_exit=torch.zeros(n,dtype=torch.bool), last_world_exit=torch.zeros(n,dtype=torch.bool))


@pytest.fixture(scope='module')
def trajectory():
    tracker=PairedHorizonTracker(3,'cpu')
    old=HorizonEpisodeTracker(3,'cpu',dt=1/60,max_steps=960,snapshot_seconds=8)
    snapshots={}
    rng=torch.get_rng_state().clone()
    for step in range(1,971):
        value=evidence(3,distance=60. if step<=960 else 40.)
        if step==10:
            value['dones'][0]=value['reset_terminated'][0]=True
            value['last_distance'][0]=7.
            value['current_distance'][0]=9999.
        if step==960:
            value['dones'][1]=value['reset_terminated'][1]=True
            value['last_distance'][1]=55.
            value['last_out_of_lane'][1]=True
            value['current_distance'][1]=-999.
        if step==970:
            value['dones'][2]=True
            value['last_distance'][2]=40.
            value['current_distance'][2]=9999.
        saved={key:val.clone() for key,val in value.items()}
        if step<=960:
            old_value={key:val.clone() for key,val in value.items()}
            if step==960:
                virtual=~old_value['dones']
                for terminal,current in (('last_distance','current_distance'),('last_out_of_lane','current_out_of_lane'),('last_world_exit','current_world_exit')):
                    old_value[terminal]=torch.where(virtual,old_value[current],old_value[terminal])
                old_value['dones']=torch.ones(3,dtype=torch.bool)
            old.update(step=step,**old_value)
        tracker.update(step=step,**value)
        assert all(torch.equal(value[key],val) for key,val in saved.items())
        for seconds in tracker.ready_windows():
            snapshots[seconds]=tracker.window_data(seconds)
            tracker.mark_captured(seconds)
    assert torch.equal(rng,torch.get_rng_state())
    return tracker,snapshots,old.to_dict()


def test_prefix_exact_legacy_tracker_fields(trajectory):
    _,windows,legacy=trajectory
    legacy['distance_at_snapshot_m']=legacy.pop('distance_at_16s')
    legacy['episode_full_horizon_survival']=legacy.pop('episode_full_64s_survival')
    legacy['aggregates']['full_horizon_survivals']=legacy['aggregates'].pop('full_64s_survivals')
    assert all(windows[16][key]==value for key,value in legacy.items())


def test_virtual_cutoff_realdone_collision_uses_terminal_not_reset(trajectory):
    _,windows,_=trajectory
    short=windows[16]
    assert short['forward_distance']==[7.,55.,60.]
    assert short['episode_terminated']==[True,True,False]
    assert short['episode_out_of_lane']==[False,True,False]
    assert short['episode_physical_done']==[True,True,False]
    assert short['episode_window_censored']==[False,False,True]
    assert short['episode_lengths']==[10,960,960]


def test_full_active_episode_survives_prefix_then_final_not_max(trajectory):
    tracker,windows,_=trajectory
    assert tracker.full.complete
    assert windows[64]['episode_lengths']==[10,960,970]
    assert windows[16]['episode_strict_all_tiles_success'][2]
    assert not windows[64]['episode_strict_all_tiles_success'][2]
    assert windows[64]['maximum_distance_m'][2]==60.
    assert windows[64]['forward_distance'][2]==40.
    assert windows[64]['episode_window_censored']==[False]*3
    assert windows[64]['episode_physical_timeout']==[False,False,True]
    assert windows[64]['capture_reason']=='all_first_episodes_finished'
    assert windows[16]['capture_reason']=='window_cutoff'


def test_auxiliary_snapshots_have_distinct_correct_times(trajectory):
    _,windows,_=trajectory
    assert windows[16]['distance_at_snapshot_m']==[None,60.,60.]
    assert windows[64]['distance_at_snapshot_m']==[None,None,60.]


def test_early_all_complete_records_both_windows_without_survivors():
    tracker=PairedHorizonTracker(2,'cpu')
    value=evidence(2,done=[True,True],terminated=[True,True],last=5.)
    tracker.update(step=1,**value)
    assert tracker.ready_windows()==(16,64)
    for seconds in tracker.ready_windows():
        data=tracker.window_data(seconds)
        assert data['captured_step']==1
        assert data['capture_reason']=='all_first_episodes_finished'
        assert data['episode_lengths']==[1,1]
        assert data['episode_physical_done']==[True,True]
        assert data['episode_window_censored']==[False,False]
        assert data['episode_full_horizon_survival']==[False,False]


def test_reference_disables_prefix_and_keeps_full_horizon():
    tracker=PairedHorizonTracker(1,'cpu',instrumented=False)
    assert set(tracker.trackers)=={64}
    tracker.update(step=1,**evidence(1,done=[True],terminated=[True],last=2.))
    assert tracker.ready_windows()==(64,)
    with pytest.raises(ValueError):
        tracker.window_data(16)


@pytest.mark.parametrize('instrumented',[None,1,'true'])
def test_instrumented_flag_is_explicit_boolean(instrumented):
    with pytest.raises(ValueError):
        PairedHorizonTracker(1,'cpu',instrumented=instrumented)


def test_unfinished_duplicate_and_nonsequential_capture_rejected():
    tracker=PairedHorizonTracker(1,'cpu')
    with pytest.raises(ValueError):
        tracker.window_data(16)
    with pytest.raises(ValueError):
        tracker.update(step=2,**evidence(1))
    with pytest.raises(ValueError):
        tracker.mark_captured(16)
    tracker.update(step=1,**evidence(1,done=[True],terminated=[True],last=3.))
    tracker.mark_captured(16)
    with pytest.raises(ValueError):
        tracker.window_data(16)
    with pytest.raises(ValueError):
        tracker.update(step=2,**evidence(1))


def test_termination_without_done_rejected():
    tracker=PairedHorizonTracker(1,'cpu')
    with pytest.raises(ValueError,match='termination without'):
        tracker.update(step=1,**evidence(1,terminated=[True]))


def test_snapshot_detaches_nested_events_and_telemetry_without_rng():
    state=torch.get_rng_state().clone()
    live={'events':[[{'step':960,'history':{'count':2}}]],'telemetry':{'values':[1.,2.]}}
    frozen=immutable_snapshot(live)
    live['events'][0][0]['history']['count']=3
    live['events'][0].append({'step':961})
    live['telemetry']['values'][0]=10.
    assert frozen=={'events':[[{'step':960,'history':{'count':2}}]],'telemetry':{'values':[1.,2.]}}
    assert torch.equal(state,torch.get_rng_state())
    with pytest.raises(ValueError):
        immutable_snapshot({'nonfinite':float('nan')})


@pytest.mark.parametrize('field',['root','joint','episode_counter','observations','cpu_rng','cuda_rng','policy_mode','history'])
def test_snapshot_passivity_detects_each_mutable_state_category(field):
    before={name:'a' for name in ('root','joint','episode_counter','observations','cpu_rng','cuda_rng','policy_mode','history')}
    proof=snapshot_passivity(before,deepcopy(before))
    assert proof['unchanged']
    before[field]='changed'
    assert proof['before'][field]=='a'
    with pytest.raises(ValueError,match='snapshot changed'):
        snapshot_passivity(before,proof['before'])


def _loop(path):
    tree=ast.parse(path.read_text())
    return next(node for node in ast.walk(tree) if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='step')


def test_frozen_action_sensor_env_step_body_ast_unchanged():
    old=_loop(ROOT/'scripts/evaluate_seed_v22.py')
    new=_loop(ROOT/'scripts/evaluate_paired_horizon_v23.py')
    body=lambda node:next(item for item in node.body if isinstance(item,ast.With))
    assert ast.dump(body(old),include_attributes=False)==ast.dump(body(new),include_attributes=False)


def test_only_real_done_reaches_gate_reset_and_full_tracker_controls_active():
    loop=_loop(ROOT/'scripts/evaluate_paired_horizon_v23.py')
    resets=[node for node in ast.walk(loop) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and node.func.attr=='reset']
    assert len(resets)==1
    assert ast.unparse(resets[0])=='gate.reset(done)'
    assert ast.unparse(loop.body[0])=='active = (~tracker.finished).clone()'
    source=(ROOT/'scripts/evaluate_paired_horizon_v23.py').read_text()
    assert 'tracker = paired.full' in source
    assert source.index('routing.after_step(step, active, done, terminated)') < source.index('gate.reset(done)') < source.index('for seconds in paired.ready_windows():')


def test_full_horizon_requires_real_physical_completion_not_synthetic_done():
    tracker=PairedHorizonTracker(1,'cpu',instrumented=False)
    for step in range(1,3841):
        tracker.update(step=step,**evidence(1))
    assert not tracker.full.complete
    assert tracker.ready_windows()==()
    with pytest.raises(ValueError):
        tracker.update(step=3841,**evidence(1))


def _production_snapshot_functions():
    import hashlib
    tree=ast.parse((ROOT/'scripts/evaluate_paired_horizon_v23.py').read_text())
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('state_value','physical_state')]
    def tensor_sha(value):
        return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
    namespace={'torch':torch,'tensor_sha':tensor_sha}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'snapshot_functions','exec'),namespace)
    return namespace


def test_actual_gate_tree_is_hashable_and_every_mutable_buffer_is_visible():
    import json
    from types import SimpleNamespace
    from week03_ant.history_gate import HistoryDepthPolicyGate,HistoryGateConfig
    state_value=_production_snapshot_functions()['state_value']
    gate=SimpleNamespace(gate=HistoryDepthPolicyGate(2,'cpu',1/60,HistoryGateConfig()),events=[[],[]],finished=None)
    before=state_value(gate)
    json.dumps(before,allow_nan=False)  # NaN ring entries are hashed as tensors, not JSON floats.
    for key,value in vars(gate.gate).items():
        if torch.is_tensor(value):
            assert 'tensor_sha256' in before['gate'][key]
    gate.gate._cursor+=1
    assert state_value(gate)!=before
    gate.gate._cursor-=1
    gate.events[0].append({'step':960})
    assert state_value(gate)!=before


def test_production_snapshot_fingerprint_preserves_rng_and_detects_history_and_counters(monkeypatch):
    from types import SimpleNamespace
    from week03_ant.history_gate import HistoryDepthPolicyGate,HistoryGateConfig
    fingerprint=_production_snapshot_functions()['physical_state']
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda device:torch.get_rng_state().clone())
    robot=SimpleNamespace(root_state_w=torch.zeros(2,13),joint_pos=torch.zeros(2,8),joint_vel=torch.zeros(2,8))
    raw=SimpleNamespace(device='cpu',episode_length_buf=torch.tensor([960,2]),common_step_counter=960)
    observations={'policy':torch.zeros(2,91)}
    policy=torch.nn.Linear(91,8).eval()
    gate=SimpleNamespace(gate=HistoryDepthPolicyGate(2,'cpu',1/60,HistoryGateConfig()),events=[[],[]],finished=torch.zeros(2,dtype=torch.bool))
    tracker=SimpleNamespace(finished=torch.tensor([False,True]))
    state=torch.get_rng_state().clone()
    before=fingerprint(raw,robot,observations,gate,policy,tracker,'hybrid')
    snapshot_passivity(before,fingerprint(raw,robot,observations,gate,policy,tracker,'hybrid'))
    assert torch.equal(state,torch.get_rng_state())
    assert set(before)=={'root_state_sha256','joint_pos_sha256','joint_vel_sha256','episode_length_buf_sha256',
        'common_step_counter','observations_sha256','cpu_rng_sha256','cuda_rng_sha256','policy_mode',
        'policy_training','policy_state_sha256','gate_state_sha256','full_first_episode_active_sha256'}
    raw.episode_length_buf[0]+=1
    with pytest.raises(ValueError):
        snapshot_passivity(before,fingerprint(raw,robot,observations,gate,policy,tracker,'hybrid'))
    raw.episode_length_buf[0]-=1
    gate.gate.history_recorded[0,0]=True
    with pytest.raises(ValueError):
        snapshot_passivity(before,fingerprint(raw,robot,observations,gate,policy,tracker,'hybrid'))
