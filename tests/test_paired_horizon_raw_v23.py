"""Synthetic corruption tests for the independent, stdlib-only paired-window audit."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('independent_v23',ROOT/'scripts/audit_paired_horizon_v23_raw.py')
assert SPEC is not None and SPEC.loader is not None
verifier=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def raw(seconds=16,controller='parent',development=False):
    n=35 if development else 175
    key=controller.removeprefix('history_')
    data={key:[0]*n for key in verifier.FLAT_FIELDS}
    data.update(schema='week03_ant_paired_horizon_v23_window_v1',task='Week03-Ant-Contact-v16-Eval-v0',
        phase='smoke' if development else 'holdout',controller=controller,geometry_seed=51 if development else 117,
        reset_seed=24 if development else 77,scenario='mixed',num_envs=n,command_mode='conditioned',command_schema_version=1,
        mode='hybrid' if controller.startswith('history_') else 'v10',teacher_tensor_identity_verified=True,
        default_config_parity=True,new_training_transitions=0,policy_training_transitions=0 if key=='parent' else 32768000,
        training_seed=None if key=='parent' else int(key[4:]),checkpoint=f'{key}.pt',checkpoint_sha256='a'*64,
        family_names=list(verifier.FAMILIES),difficulties=[.2,.4,.6,.8,1.],
        initial_state_sha256={k:'a'*64 for k in ('root_state','joint_pos','joint_vel','observations')},
        initial_prefix_sha256='b'*64,initial_rng_sha256={'cpu':'c'*64,'cuda':'d'*64},
        condition=dict(seconds=seconds,max_steps=seconds*60,dt=1/60,snapshot_seconds=8 if seconds==16 else 16,
            one_threshold=13.1,six_threshold=53.1,footprint_margin=1.1,observations=91,expert_observations=91,v5_observations=60,actions=8),
        window_seconds=seconds,captured_step=seconds*60,capture_reason='window_cutoff',
        episode_physical_done=[seconds==64]*n,episode_physical_timeout=[seconds==64]*n,
        episode_window_censored=[seconds==16]*n)
    repeats=1 if development else 5
    data['family_indices']=[i//(5*repeats) for i in range(n)]
    data['level_indices']=[(i//repeats)%5 for i in range(n)]
    for field in ('episode_terminated','episode_out_of_lane','episode_world_exit'):
        data[field]=[False]*n
    for field in ('episode_strict_one_tile_success','episode_strict_all_tiles_success','episode_full_horizon_survival'):
        data[field]=[True]*n
    for field in ('forward_distance','maximum_distance_m'):
        data[field]=[60. if seconds==16 else 100.]*n
    for field in ('episode_lengths','episode_active_steps'):
        data[field]=[seconds*60]*n
    for field in ('episode_switch_steps','episode_switch_to_v10','history_switch_events'):
        data[field]=[[] for _ in range(n)]
    data['first_hit_one_seconds']=[1.]*n
    data['first_hit_six_seconds']=[2.]*n
    data['distance_at_snapshot_m']=[20.]*n
    data['contact_telemetry']={'active_steps':[seconds*60]*n,'valid_steps':[seconds*60-1]*n,'bounded_cost_sum':[1.]*n}
    data['posture_telemetry']={'active_steps':[seconds*60]*n,'groups':{'all_valid':{'steps':[seconds*60]*n,'reward_sum':[1.]*n}}}
    return data


def bundle(controller='parent',development=False,instrumented=True):
    windows={str(seconds):raw(seconds,controller,development) for seconds in ((16,64) if instrumented else (64,))}
    common=windows['64']
    if not instrumented:
        common['phase']='reference'
    value={key:common[key] for key in ('controller','geometry_seed','reset_seed','phase','num_envs','scenario')}
    value.update(schema='week03_ant_paired_horizon_v23_bundle_v1',instrumented=instrumented,
        physical_condition=dict(seconds=64,max_steps=3840,dt=1/60),windows=windows,
        physical_steps_executed=3840,executed_first_episodes=common['num_envs'],window_observations=common['num_envs']*len(windows))
    proofs={}
    for window,data in windows.items():
        state={key:'a'*64 for key in verifier.PASSIVITY_HASH_KEYS}
        state.update(common_step_counter=int(window)*60,policy_mode=data['mode'],policy_training=False)
        proofs[window]=dict(before=deepcopy(state),after=deepcopy(state),unchanged=True)
    value['snapshot_passivity']=proofs
    return value


def test_imports_only_stdlib_and_fixed_definition_inventory():
    tree=ast.parse((ROOT/'scripts/audit_paired_horizon_v23_raw.py').read_text())
    allowed={'__future__','argparse','collections','datetime','hashlib','json','math','pathlib','re','statistics','struct'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            assert {alias.name.split('.')[0] for alias in node.names}<=allowed
        elif isinstance(node,ast.ImportFrom):
            assert node.module and node.module.split('.')[0] in allowed
    assert len(verifier.SOURCE_FILES)==8
    assert len(verifier.PARITY_FIELDS)==35
    assert len(set(verifier.PAIRS))==12
    assert 'scripts/audit_paired_horizon_v23_raw.py' not in verifier.SOURCE_FILES


@pytest.mark.parametrize('controller',verifier.CONTROLLERS)
def test_complete_paired_bundle_and_same_episode_budget(controller):
    result=verifier.bundle_rows(bundle(controller))
    assert set(result)=={'16','64'}
    assert len(result['16'])==len(result['64'])==175
    assert verifier.counts(result['64'])['n']==150
    paired=verifier.duration_sections(result['16'],result['64'])
    assert paired['all']['n']==175
    assert paired['all']['strict_six']['both']==175
    assert paired['family_level/stepping_stones/4']['n']==5


@pytest.mark.parametrize('key,value',[('episode_strict_all_tiles_success',False),('episode_terminated',True),
    ('episode_world_exit',True),('forward_distance',0.),('maximum_distance_m',1.),('episode_lengths',1)])
def test_strict_raw_mutations_fail(key,value):
    data=bundle()
    data['windows']['16'][key][0]=value
    with pytest.raises(ValueError):
        verifier.bundle_rows(data)


@pytest.mark.parametrize('kind',['missing16','missing64','disabledholdout','wrongduration','newseed','newmap','reorderedrows','count'])
def test_bundle_identity_inventory_and_count_corruption(kind):
    data=bundle()
    if kind.startswith('missing'):
        del data['windows'][kind.removeprefix('missing')]
    elif kind=='disabledholdout':
        data=bundle(development=False,instrumented=False)
    elif kind=='wrongduration':
        data['windows']['64']['condition']['seconds']=16
    elif kind=='newseed':
        data['windows']['64']['training_seed']=51
    elif kind=='newmap':
        data['windows']['64']['geometry_seed']=118
    elif kind=='reorderedrows':
        data['windows']['64']['family_indices'][0],data['windows']['64']['family_indices'][25]=1,0
    else:
        data['window_observations']=175
    with pytest.raises(ValueError):
        verifier.bundle_rows(data)


@pytest.mark.parametrize('key',verifier.PASSIVITY_HASH_KEYS)
def test_all_passivity_hash_categories_required_and_equal(key):
    data=bundle()
    data['snapshot_passivity']['16']['after'][key]='b'*64
    with pytest.raises(ValueError,match='passivity'):
        verifier.bundle_rows(data)
    del data['snapshot_passivity']['16']['before'][key]
    del data['snapshot_passivity']['16']['after'][key]
    with pytest.raises(ValueError,match='passivity'):
        verifier.bundle_rows(data)


@pytest.mark.parametrize('window,key', [('16','episode_physical_done'),('16','episode_window_censored'),
    ('64','episode_physical_timeout'),('64','episode_window_censored'),('64','episode_physical_done')])
def test_physical_timeout_is_not_fall_or_virtual_reset(window,key):
    data=bundle()
    data['windows'][window][key][0]=not data['windows'][window][key][0]
    with pytest.raises(ValueError):
        verifier.bundle_rows(data)


def test_reference_mode_is_exact_two_development_controllers():
    for controller in ('parent','history_parent'):
        assert set(verifier.bundle_rows(bundle(controller,True,False),development=True))=={'64'}
    with pytest.raises(ValueError):
        verifier.bundle_rows(bundle('seed51',True,False),development=True)


def test_early_complete_and_reset_contamination():
    data=bundle()
    for raw in data['windows'].values():
        n=raw['num_envs']
        raw.update(captured_step=20,capture_reason='all_first_episodes_finished',
            episode_physical_done=[True]*n,episode_physical_timeout=[False]*n,episode_window_censored=[False]*n,
            episode_terminated=[True]*n,episode_lengths=[20]*n,episode_active_steps=[20]*n,
            episode_strict_one_tile_success=[False]*n,episode_strict_all_tiles_success=[False]*n,
            episode_full_horizon_survival=[False]*n,first_hit_one_seconds=[.1]*n,first_hit_six_seconds=[.2]*n,
            forward_distance=[60.]*n,maximum_distance_m=[60.]*n,distance_at_snapshot_m=[None]*n)
        raw['contact_telemetry']={'active_steps':[20]*n,'valid_steps':[19]*n,'bounded_cost_sum':[1.]*n}
        raw['posture_telemetry']={'active_steps':[20]*n,'groups':{'all_valid':{'steps':[20]*n,'reward_sum':[1.]*n}}}
    data['physical_steps_executed']=20
    verifier.bundle_rows(data)
    data['windows']['64']['forward_distance'][0]=59.
    with pytest.raises(ValueError,match='completed first episode'):
        verifier.bundle_rows(data)


def test_routing_event_prefix_mutation():
    data=bundle()
    for window in ('16','64'):
        raw=data['windows'][window]
        raw['episode_switch_count'][0]=1
        raw['episode_switch_steps'][0]=[2]
        raw['episode_switch_to_v10'][0]=[True]
        raw['history_switch_events'][0]=[{'step':2,'target_v10':True,'history':{'x':1}}]
    verifier.bundle_rows(data)
    data['windows']['64']['history_switch_events'][0][0]['history']['x']=2
    with pytest.raises(ValueError,match='prefix'):
        verifier.bundle_rows(data)


def test_four_cells_overlap_and_threshold_boundary():
    row=dict(family='rough',level=0,six=0,falls=0,lane=0,world=0,distance=60.,first_six_hit=None)
    short=[dict(row,six=v) for v in (0,0,1,1)]
    long=[dict(row,six=v) for v in (0,1,0,1)]
    long[2].update(falls=1,lane=1,world=1,distance=52.,first_six_hit=12.)
    long[0]['first_six_hit']=32.
    result=verifier.duration_pairs(short,long)
    assert result['strict_six']==dict(neither=1,late_gain=1,late_loss=1,both=1)
    assert set(result['late_loss_flags'].values())=={1}
    assert result['first_six_hit_in_16_64']==1
    long[2]['distance']=53.099998474121094
    assert verifier.duration_pairs(short,long)['late_loss_flags']['final_distance_below_53_1']==0


def test_all_sections_gates_ranges_and_parent_once():
    groups={name:verifier.bundle_rows(bundle(name))['16'] for name in verifier.CONTROLLERS}
    section=verifier.computed_section(groups,True)
    verifier.compare_section(groups,section,True)
    assert len(section['comparisons'])==12
    assert section['seed_ranges']['actor']['parent_counted_once'] is True
    for kind in ('range','parent','subgroupgate'):
        changed=deepcopy(section)
        if kind=='range':
            changed['seed_ranges']['actor']['metrics']['six']['range']=1
        elif kind=='parent':
            changed['groups']['parent']['n']*=3
        else:
            changed['family_level']['stairs/level3']['comparisons']['seed51_vs_parent']['passed']=True
        with pytest.raises(ValueError):
            verifier.compare_section(groups,changed,True)


@pytest.mark.parametrize('field',verifier.PARITY_FIELDS)
def test_all_35_physical_parity_fields_fixed(field):
    old={key:[] for key in verifier.PARITY_FIELDS}
    changed=deepcopy(old)
    changed[field]=['bad']
    with pytest.raises(ValueError,match='35-field'):
        verifier.parity(changed,old)


def test_incomplete_input_writes_no_audit(tmp_path):
    with pytest.raises(FileNotFoundError):
        verifier.audit(root=tmp_path,directory=tmp_path)
    assert not (tmp_path/'independent_raw_audit.json').exists()


def test_nonfinite_duplicate_json_and_escaping_path(tmp_path):
    path=tmp_path/'bad.json'
    for text in ('{"a":NaN}','{"a":1,"a":2}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            verifier.read(path)
    with pytest.raises(ValueError):
        verifier.inside(tmp_path,'../escape')


def test_29command_order_prefix_no_scored_retries_or_disabled_prefixes(tmp_path):
    directory=tmp_path/'artifacts/terrain_demo/paired_horizon_v23'
    directory.mkdir(parents=True)
    work=tmp_path/'outputs/paired_horizon_v23_20260928'
    work.mkdir(parents=True)
    for name in ('evaluation_inputs.json','terrain_cache.json'):
        (directory/name).write_text('{}')
    specs=[('parent',51,24,'prepare_development',work/'prepare_development.json','prepare_development')]
    for phase,controllers in (('reference',('parent','history_parent')),('smoke',verifier.CONTROLLERS)):
        specs.extend((c,51,24,phase,work/f'{phase}_{c}.json',f'{phase}_{c}') for c in controllers)
    specs.extend(('parent',g,r,'prepare',work/f'prepare_geometry{g}.json',f'prepare_geometry{g}') for g,r in verifier.MAPS)
    specs.extend((c,g,r,'holdout',directory/'evaluations'/f'{c}__geometry{g}_reset{r}.json',f'evaluations_{c}__geometry{g}_reset{r}') for c in verifier.CONTROLLERS for g,r in verifier.MAPS)
    entries=[]
    rawfiles,devfiles={},{}
    for c,g,r,phase,output,label in specs:
        command=[str(tmp_path.parent/'run-python'),'scripts/evaluate_paired_horizon_v23.py','--headless','--device','cuda:1',
            '--controller',c,'--geometry',str(g),'--seed',str(r),'--scenario','mixed','--seconds','64',
            '--num_envs','35' if phase in ('prepare_development','reference','smoke') else '175','--phase',phase,'--output',str(output)]
        if phase=='reference':
            command.append('--no-prefixes')
        log=work/f'{label}.log'
        log.write_text('successful')
        record=dict(command=command,label=label,returncode=0,log=str(log.relative_to(tmp_path)),log_sha256=verifier.sha(log),
                    started_utc='2026-09-28T00:00:00+00:00',finished_utc='2026-09-28T00:00:00+00:00')
        if phase=='holdout':
            rawfiles[output.resolve()]={}
            record.update(evaluation_inputs_sha256=verifier.sha(directory/'evaluation_inputs.json'),
                          terrain_cache_manifest_sha256=verifier.sha(directory/'terrain_cache.json'))
        elif phase.startswith('prepare'):
            output.write_text('{"scored_episodes":0}')
        else:
            devfiles[output.resolve()]={}
        entries.append(record)
    path=directory/'commands.jsonl'
    prefix=('\n'.join(json.dumps(record) for record in entries[:13])+'\n').encode()
    path.write_bytes(prefix)
    pin=dict(path=str(path.relative_to(tmp_path)),bytes=len(prefix),lines=13,sha256=verifier.sha(path))
    (directory/'preholdout_ledger.json').write_text(json.dumps(pin))
    def write(records):
        path.write_text('\n'.join(json.dumps(record) for record in records)+'\n')
    write(entries)
    assert verifier.command_ledger(tmp_path,directory,rawfiles,devfiles)==29
    for kind in ('missing','duplicate','order','disabled','hash','prefix'):
        changed=deepcopy(entries)
        if kind=='missing':
            changed.pop()
        elif kind=='duplicate':
            changed[-1]=deepcopy(changed[-2])
        elif kind=='order':
            changed[-1],changed[-2]=changed[-2],changed[-1]
        elif kind=='disabled':
            changed[-1]['command'].append('--no-prefixes')
        elif kind=='hash':
            changed[-1]['log_sha256']='0'*64
        else:
            changed[0]['label']='alteredprefix'
        write(changed)
        with pytest.raises(ValueError):
            verifier.command_ledger(tmp_path,directory,rawfiles,devfiles)


def test_development_control_and_paired_group_swaps_fail_closed():
    reference=bundle('parent',development=True,instrumented=False)
    paired=bundle('parent',development=True,instrumented=True)
    verifier.development_group('controls',reference)
    verifier.development_group('records',paired)
    for group,raw in (('controls',paired),('records',reference)):
        with pytest.raises(ValueError,match='group substituted'):
            verifier.development_group(group,raw)
    disguised=deepcopy(reference)
    disguised['phase']='smoke'
    with pytest.raises(ValueError,match='group substituted'):
        verifier.development_group('controls',disguised)


def test_impossible_one_after_six_in_both_windows_fails():
    data=bundle()
    for raw in data['windows'].values():
        raw['first_hit_one_seconds'][0]=3.
        raw['first_hit_six_seconds'][0]=2.
    with pytest.raises(ValueError,match='one-tile hit follows'):
        verifier.bundle_rows(data)


@pytest.mark.parametrize('window,role',[('16','primary'),('64','secondary descriptive')])
def test_window_report_roles_cannot_be_swapped(window,role):
    verifier.report_role(window,{'role':role})
    for wrong in ({},{'role':'secondary descriptive' if window=='16' else 'primary'}):
        with pytest.raises(ValueError,match='window role'):
            verifier.report_role(window,wrong)
