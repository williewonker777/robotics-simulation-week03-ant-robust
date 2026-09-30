"""Synthetic mutation tests for the post-experiment, stdlib-only v22 verifier."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_v22', ROOT / 'scripts/audit_seed_v22_raw.py')
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def raw(controller='parent', scenario='mixed'):
    seconds, n = (16, 175) if scenario == 'mixed' else (64, 10)
    data = {key: [0] * n for key in verifier.FLAT_FIELDS}
    data.update(training_seed=None if controller.removeprefix('history_')=='parent' else int(controller.removeprefix('history_')[4:]),schema='week03_ant_seed_continuation_v22_v1', task='Week03-Ant-Contact-v16-Eval-v0',
                phase='holdout', controller=controller, geometry_seed=115, reset_seed=75, scenario=scenario, num_envs=n,
                command_mode='conditioned', command_schema_version=1,
                mode='hybrid' if controller.startswith('history_') else 'v10',
                teacher_tensor_identity_verified=True, default_config_parity=True, new_training_transitions=0,
                policy_training_transitions=0 if controller.removeprefix('history_') == 'parent' else verifier.TRANSITIONS,
                family_names=list(verifier.FAMILIES), difficulties=[.2, .4, .6, .8, 1.],
                initial_state_sha256={k: 'a' * 64 for k in ('root_state', 'joint_pos', 'joint_vel', 'observations')},
                initial_prefix_sha256='b' * 64, initial_rng_sha256={'cpu': 'c' * 64, 'cuda': 'd' * 64},
                condition=dict(seconds=seconds, max_steps=seconds * 60, dt=1 / 60, snapshot_seconds=8 if seconds == 16 else 16,
                               one_threshold=13.1, six_threshold=53.1, footprint_margin=1.1, observations=91,
                               expert_observations=91, v5_observations=60, actions=8))
    data['family_indices'] = [i // 25 for i in range(n)] if scenario == 'mixed' else [5] * n
    data['level_indices'] = [(i // 5) % 5 for i in range(n)] if scenario == 'mixed' else [4] * n
    for key in ('episode_terminated', 'episode_out_of_lane', 'episode_world_exit'):
        data[key] = [False] * n
    for key in ('episode_strict_one_tile_success', 'episode_strict_all_tiles_success', 'episode_full_horizon_survival'):
        data[key] = [True] * n
    for key in ('forward_distance', 'maximum_distance_m'):
        data[key] = [60.] * n
    for key in ('episode_lengths', 'episode_active_steps'):
        data[key] = [seconds * 60] * n
    data['first_hit_one_seconds'], data['first_hit_six_seconds'] = [1.] * n, [2.] * n
    return data


def rows(data):
    return verifier.raw_rows(data, controller=data['controller'], geometry=115, reset=75, scenario=data['scenario'])


def schedule(arm='no_cost', *, envs=4096, steps=8000, ramp_steps=4000):
    factors = [0. if arm == 'no_cost' else 1. if arm == 'immediate' else min(i / (ramp_steps - 1), 1.) for i in range(steps)]
    scaled = [-.5 * factor for factor in factors]
    totals = [v * envs / 60 for v in scaled]
    return dict(arm=arm, mode=arm, num_envs=envs, ramp_steps=ramp_steps, policy_steps=steps,
                common_step_counters=list(range(1, steps + 1)), coefficients=factors,
                raw_reward_mean=[-.5] * steps, scaled_reward_mean=scaled, valid_rows=[envs] * steps,
                step_dt=[1 / 60] * steps, raw_reward_dt_sum=[-.5 * envs / 60] * steps,
                scaled_reward_dt_sum=totals, coefficient_sum=sum(factors), realized_penalty_sum=-sum(totals),
                episode_randomization=dict(matches_raw_environment=True, assigned_sha256='e' * 64,
                    actual_sha256='e' * 64, nonzero=envs - 1, minimum=0, maximum=959))


def test_standalone_imports_only_standard_library():
    tree = ast.parse((ROOT / 'scripts/audit_seed_v22_raw.py').read_text())
    allowed = {'__future__', 'argparse', 'collections', 'datetime', 'hashlib', 'json', 'math', 'pathlib', 're', 'statistics', 'struct'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {a.name.split('.')[0] for a in node.names} <= allowed
        if isinstance(node, ast.ImportFrom):
            assert node.module is not None and node.module.split('.')[0] in allowed


@pytest.mark.parametrize('scenario,n,flat', [('mixed', 150, 25), ('stones', 10, 0)])
def test_synthetic_strict_counts(scenario, n, flat):
    result = verifier.counts(rows(raw(scenario=scenario)))
    assert result['n'] == result['one'] == result['six'] == n
    assert result['flat_n'] == flat
    assert result['falls'] == result['lane'] == result['world'] == 0
    assert result['completed_first_episodes'] == n + flat


@pytest.mark.parametrize('field,value', [('episode_strict_one_tile_success', False),
    ('episode_strict_all_tiles_success', False), ('episode_terminated', True), ('episode_world_exit', True),
    ('episode_out_of_lane', True), ('episode_lengths', 1), ('family_indices', True),
    ('first_hit_six_seconds', None), ('forward_distance', float('nan'))])
def test_first_episode_mutations_fail(field, value):
    data = raw()
    data[field][0] = value
    with pytest.raises(ValueError):
        rows(data)


def test_float32_threshold_and_safe_first_episode_semantics():
    data = raw()
    distance = 53.099998474121094  # float32(53.1), exactly the tensor comparison threshold.
    data['forward_distance'][0] = data['maximum_distance_m'][0] = distance
    assert rows(data)[0]['six'] == 1
    data['episode_out_of_lane'][0] = True
    data['episode_strict_one_tile_success'][0] = data['episode_strict_all_tiles_success'][0] = False
    assert rows(data)[0]['six'] == 0


def test_gate_requires_strict_rough_improvement_and_nonlower_flat_speed():
    ref = rows(raw())
    ref[0]['six'] = 0
    candidate = deepcopy(ref)
    candidate[0]['six'] = 1
    assert verifier.judgment(candidate, ref, True)['passed']
    assert verifier.judgment(candidate, ref, True)['paired_six']['gains'] == 1
    assert not verifier.judgment(ref, ref, True)['passed']
    candidate[-1]['speed'] -= .001
    assert not verifier.judgment(candidate, ref, True)['passed']
    candidate[-1]['speed'] += .001
    candidate[0]['falls'] = 1
    assert not verifier.judgment(candidate, ref, True)['passed']


@pytest.mark.parametrize('arm,mass', [('no_cost', 0)])
def test_training_schedule_independent_full_budget(arm, mass):
    result = verifier.schedule(schedule(arm), arm)
    assert result['coefficient_step_mass'] == mass
    assert result['realized_penalty_sum'] == pytest.approx(.5 * 4096 / 60 * mass)


@pytest.mark.parametrize('field,value', [('common_step_counters', 1), ('coefficients', .9),
    ('raw_reward_mean', -2.), ('scaled_reward_mean', -.3), ('valid_rows', 0),
    ('step_dt', 1.), ('raw_reward_dt_sum', -99.), ('scaled_reward_dt_sum', -99.)])
def test_training_sequence_mutations_fail(field, value):
    data = schedule()
    data[field][5000] = value
    with pytest.raises(ValueError):
        verifier.schedule(data, 'no_cost')


def test_report_aggregate_mutation_fails():
    data = rows(raw())
    recorded = verifier.counts(data)
    recorded['six'] -= 1
    with pytest.raises(ValueError, match='aggregate'):
        verifier.compare_group(data, recorded)


def test_missing_inputs_produce_no_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        verifier.audit(root=tmp_path, directory=tmp_path, cache_root=tmp_path / 'cache')
    assert not (tmp_path / 'independent_raw_audit.json').exists()


def test_duplicate_and_nonfinite_json_rejected(tmp_path):
    path = tmp_path / 'data.json'
    for content in ('{"a": 1, "a": 2}', '{"a": NaN}'):
        path.write_text(content)
        with pytest.raises(ValueError):
            verifier.read(path)


def test_digest_and_path_mutations_fail(tmp_path):
    path = tmp_path / 'evidence.json'
    path.write_text('{}')
    verifier.hashes(tmp_path, {'evidence.json': verifier.sha(path)})
    with pytest.raises(ValueError):
        verifier.hashes(tmp_path, {'evidence.json': '0' * 64})
    with pytest.raises(ValueError):
        verifier.inside(tmp_path, '../outside')


def test_ledger_duplicate_attempt_log_mutation_and_prefix_drift(tmp_path):
    directory = tmp_path / 'art'
    directory.mkdir()
    (directory / 'evaluation_inputs.json').write_text('{}')
    raw_path = directory / 'result.json'
    data = raw()
    data['checkpoint'] = 'model.pt'
    command = ['python', '--phase', 'holdout', '--output', str(raw_path), '--controller', 'parent',
               '--geometry', '115', '--seed', '75', '--scenario', 'mixed', '--num_envs', '175',
               '--seconds', '16', '--checkpoint', str(tmp_path / 'model.pt'), '--device', 'cuda:1']
    log = tmp_path / 'log.txt'
    log.write_text('exit zero')
    rec = dict(command=command, returncode=0, log='log.txt', log_sha256=verifier.sha(log),
               evaluation_inputs_sha256=verifier.sha(directory / 'evaluation_inputs.json'),
               terrain_cache_manifest_sha256='a' * 64)
    prefix = json.dumps({'command': ['prepare']}).encode() + b'\n'
    path = directory / 'commands.jsonl'
    path.write_bytes(prefix)
    pin = dict(path='art/commands.jsonl', bytes=len(prefix), lines=1, sha256=verifier.sha(path))
    (directory / 'preholdout_ledger.json').write_text(json.dumps(pin))
    inputs = {'terrain_cache_manifest_sha256': 'a' * 64}
    for records, valid in (([rec], True), ([rec, rec], False), ([dict(rec, returncode=1)], False),
                           ([dict(rec, log_sha256='0' * 64)], False)):
        path.write_bytes(prefix + b''.join(json.dumps(r).encode() + b'\n' for r in records))
        if valid:
            assert verifier.ledger(tmp_path, directory, inputs, {raw_path: data}) == 1
        else:
            with pytest.raises(ValueError):
                verifier.ledger(tmp_path, directory, inputs, {raw_path: data})
    path.write_bytes(b'X' + path.read_bytes()[1:])
    with pytest.raises(ValueError, match='prefix'):
        verifier.ledger(tmp_path, directory, inputs, {raw_path: data})


def synthetic_section(groups, primary, breakdown=True):
    return verifier.computed_section(groups,primary,breakdown)


def test_all_six_gates_and_family_level_report_mutations():
    groups = {name: rows(raw(name)) for name in verifier.CONTROLLERS}
    section = synthetic_section(groups, True)
    verifier.compare_section(groups, section, True)
    assert len(section['comparisons']) == 12
    corrupt = deepcopy(section)
    corrupt['family']['stairs']['comparisons']['seed53_vs_seed52']['passed'] = True
    with pytest.raises(ValueError, match='gates'):
        verifier.compare_section(groups, corrupt, True)
    corrupt = deepcopy(section)
    corrupt['family_level']['waves/level3']['groups']['parent']['six'] -= 1
    with pytest.raises(ValueError, match='aggregate|seed ranges'):
        verifier.compare_section(groups, corrupt, True)
    corrupt = deepcopy(section)
    corrupt['level']['0']['comparisons']['seed53_vs_seed52']['paired_six']['gains'] += 1
    with pytest.raises(ValueError, match='gates'):
        verifier.compare_section(groups, corrupt, True)


def test_cache_bytes_and_missing_tiles_are_independently_verified(tmp_path):
    cache = dict(geometries=[110], tiles_per_geometry=240, tiles=[])
    for i in range(240):
        key = f'{i:032x}'
        folder = tmp_path / key
        folder.mkdir()
        digests = {}
        for name in ('cfg.yaml', 'mesh.obj', 'origin.csv'):
            path = folder / name
            path.write_text(str(i))
            digests[name] = verifier.sha(path)
        cache['tiles'].append(dict(cache_key=key, sha256=digests))
    verifier.cache_hashes(cache, tmp_path)
    (tmp_path / f'{239:032x}' / 'mesh.obj').write_text('changed mesh')
    with pytest.raises(ValueError, match='terrain bytes'):
        verifier.cache_hashes(cache, tmp_path)
    cache['tiles'].pop()
    with pytest.raises(ValueError, match='inventory'):
        verifier.cache_hashes(cache, tmp_path)


@pytest.mark.parametrize('key', ['scaled_reward_mean', 'scaled_reward_dt_sum', 'realized_penalty_sum'])
def test_no_cost_rejects_nonzero_even_below_old_floating_tolerance(key):
    data = schedule('no_cost')
    if key == 'realized_penalty_sum':
        data[key] = 1.e-12
    else:
        data[key][0] = -1.e-12
    with pytest.raises(ValueError, match='exactly zero'):
        verifier.schedule(data, 'no_cost')


def test_stalled_and_zero_contact_episodes_stay_in_denominators():
    data = raw('seed51')
    data['contact_telemetry'] = {'no_contact_steps': [960] * 175}
    data['posture_telemetry'] = {'low_speed_steps': [960] * 175}
    for key in ('forward_distance', 'maximum_distance_m'):
        data[key] = [0.] * 175
    for key in ('first_hit_one_seconds', 'first_hit_six_seconds'):
        data[key] = [None] * 175
    for key in ('episode_strict_one_tile_success', 'episode_strict_all_tiles_success'):
        data[key] = [False] * 175
    result = verifier.counts(rows(data))
    assert result['n'] == 150 and result['flat_n'] == 25
    assert result['completed_first_episodes'] == 175
    assert result['one'] == result['six'] == 0


def test_computed_scores_do_not_require_a_report():
    groups = {name: rows(raw(name)) for name in verifier.CONTROLLERS}
    result = verifier.computed_section(groups, True)
    assert len(result['comparisons']) == 12
    verifier.compare_section(groups, result, True)
    assert verifier.DEVELOPMENT_TRANSITIONS == 835584
    assert len(verifier.CONTROLLERS) * len(verifier.MAPS) * (175 + 10) == 2960


def test_legacy_parity_accepts_exact_fixed_inventory_and_evidence():
    evidence = {field: ['same'] for field in verifier.PARITY_FIELDS}
    verifier.legacy_parity({'equal_fields': list(verifier.PARITY_FIELDS)}, evidence, deepcopy(evidence))
    assert {'initial_state_sha256', 'initial_prefix_sha256', 'initial_rng_sha256', 'condition',
            'family_names', 'difficulties', 'family_indices', 'level_indices', 'aggregates',
            'contact_telemetry', 'posture_telemetry'} <= set(verifier.PARITY_FIELDS)


@pytest.mark.parametrize('fields', [None, [], list(verifier.PARITY_FIELDS[:-1]),
    list(verifier.PARITY_FIELDS[1:]), [*verifier.PARITY_FIELDS, 'extra'],
    [*verifier.PARITY_FIELDS, verifier.PARITY_FIELDS[0]]])
def test_legacy_parity_rejects_empty_truncated_or_changed_inventory(fields):
    evidence = {field: ['same'] for field in verifier.PARITY_FIELDS}
    record = {} if fields is None else {'equal_fields': fields}
    with pytest.raises(ValueError, match='field inventory'):
        verifier.legacy_parity(record, evidence, deepcopy(evidence))


@pytest.mark.parametrize('field', ['initial_state_sha256', 'initial_prefix_sha256', 'initial_rng_sha256',
                                  'condition', 'family_indices', 'aggregates', 'contact_telemetry', 'posture_telemetry'])
def test_legacy_parity_compares_every_required_evidence_category(field):
    evidence = {key: ['same'] for key in verifier.PARITY_FIELDS}
    current = deepcopy(evidence)
    current[field] = ['changed']
    with pytest.raises(ValueError, match='legacy parity differs'):
        verifier.legacy_parity({'equal_fields': list(verifier.PARITY_FIELDS)}, evidence, current)
    del current[field]
    with pytest.raises(ValueError, match='missing a required field'):
        verifier.legacy_parity({'equal_fields': list(verifier.PARITY_FIELDS)}, evidence, current)


@pytest.mark.parametrize('seed',[51,52,53])
def test_raw_seed_identity(seed):
    data=raw(f'seed{seed}')
    rows(data)
    data['training_seed']=52 if seed==51 else 51
    with pytest.raises(ValueError,match='training seed'):
        rows(data)


@pytest.mark.parametrize('kind',['parent_counted_once','range','delta','denominator'])
def test_range_parent_duplication_and_delta_mutations(kind):
    groups={name:rows(raw(name)) for name in verifier.CONTROLLERS}
    section=verifier.computed_section(groups,True)
    if kind=='parent_counted_once':
        section['seed_ranges']['actor']['parent_counted_once']=False
    elif kind=='range':
        section['seed_ranges']['actor']['metrics']['six']['range']=1
    elif kind=='delta':
        section['seed_ranges']['actor']['metrics']['six']['parent_deltas']['seed52']=1
    else:
        section['groups']['parent']['n']*=3
    with pytest.raises(ValueError):
        verifier.compare_section(groups,section,True)


def initial(seed):
    return dict(training_seed=seed,actual_seed=dict.fromkeys(('requested','saved_agent','saved_environment','live_environment'),seed),
                normalized_parameters={'agent':{'seed':str(seed),'max_iterations':'2'},'env':{'seed':str(seed)}},
                num_envs=4096,dt=1/60,initial_state_sha256={k:'a'*64 for k in ('root_state','joint_pos','joint_vel','observations')},
                initial_prefix_sha256='b'*64,policy_state_sha256={'std':'c'*64},rng_sha256={'cpu':'d'*64,'cuda':'e'*64})


@pytest.mark.parametrize('seed',[51,52,53])
def test_same_seed_capacity_projection_and_horizons(seed):
    capacity=initial(seed)
    expected=verifier.project_capacity(capacity,seed)
    assert capacity['normalized_parameters']['agent']['max_iterations']=='2'
    assert expected['normalized_parameters']['agent']['max_iterations']=='250'
    clock=schedule()
    verifier.same_seed_main(deepcopy(expected),expected,capacity,clock,clock,seed)
    for field in ('rng_sha256','initial_state_sha256','policy_state_sha256'):
        bad=deepcopy(expected)
        bad[field]={**bad[field],'changed':'f'*64}
        with pytest.raises(ValueError):
            verifier.same_seed_main(bad,expected,capacity,clock,clock,seed)
    wrong=deepcopy(expected)
    wrong['training_seed']=53 if seed!=53 else 52
    with pytest.raises(ValueError):
        verifier.same_seed_main(expected,wrong,capacity,clock,clock,seed)
    wrongclock=deepcopy(clock)
    wrongclock['episode_randomization']['actual_sha256']='f'*64
    with pytest.raises(ValueError,match='horizons'):
        verifier.same_seed_main(expected,expected,capacity,clock,wrongclock,seed)


@pytest.mark.parametrize('seed',[52,53])
def test_cross_seed_capacity_substitution_fails(seed):
    expected=verifier.project_capacity(initial(seed),seed)
    with pytest.raises(ValueError,match='seed identity'):
        verifier.same_seed_main(expected,expected,initial(51),schedule(),schedule(),seed)


def test_full_replay_gate_fail_closed(tmp_path):
    path=tmp_path/'full_replay.json'
    for value in ({'passed':False,'seed':51},{'passed':True,'seed':52}):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError,match='seed51 full replay gate'):
            verifier.full_replay(tmp_path,tmp_path,{}, {'records':{}},{})


def test_fixed_protocol_counts_and_no_pseudoreplicated_parent():
    assert verifier.TOTAL_TRANSITIONS==98304000
    assert verifier.DEVELOPMENT_TRANSITIONS==835584
    assert verifier.TOTAL_TRANSITIONS+verifier.DEVELOPMENT_TRANSITIONS==99139584
    assert len(verifier.PARITY_FIELDS)==35
    assert len(set(verifier.PAIRS))==12
    assert verifier.CONTROLLERS.count('parent')==verifier.CONTROLLERS.count('history_parent')==1


def test_complete_command_budget_rejects_missing_duplicate_and_training_override(tmp_path):
    directory = tmp_path / 'art'
    directory.mkdir()
    commands = []
    def output(name):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
        return {'path': name, 'sha256': verifier.sha(path)}
    def command(args):
        log = output(f'log/{len(commands)}.txt')
        record = dict(command=['python', *args, '--device', 'cuda:1'], returncode=0,
                      log=log['path'], log_sha256=log['sha256'])
        commands.append(record)
        return log['path']
    raw_paths = {}
    for i in range(32):
        ref = output(f'raw/{i}.json')
        path = tmp_path / ref['path']
        raw_paths[path] = {}
        command(['--output', str(path)])
    extra = []
    for i in range(12):
        ref = output(f'extra/{i}.json')
        extra.append(ref)
        command(['--output', str(tmp_path / ref['path'])])
    (directory / 'development.json').write_text(json.dumps({'preparation': extra[0], 'records': extra[1:9]}))
    (directory / 'terrain_cache.json').write_text(json.dumps({'preparation': extra[9:11]}))
    (directory / 'training_cache.json').write_text(json.dumps({'preparation': extra[11]['path'], 'preparation_sha256': extra[11]['sha256']}))
    for phase, envs, iterations, ramp, arms in [('preflight', 256, 2, 32, verifier.RUNS),
             ('capacity', 4096, 2, 4000, verifier.RUNS), ('training_validation', 4096, 250, 4000, verifier.RUNS)]:
        records = {}
        for arm in arms:
            initial, sched = output(f'{phase}/{arm}_initial.json'), output(f'{phase}/{arm}_schedule.json')
            log = command(['--audit-output', str(tmp_path / initial['path']), '--schedule-output', str(tmp_path / sched['path']),
                '--arm', 'no_cost', '--training-seed', arm[4:], '--num_envs', str(envs), '--max_iterations', str(iterations), '--ramp-steps', str(ramp),
                '--seed', arm[4:], '--load_run', 'v22_init', '--checkpoint', 'model_0.pt'])
            records[arm] = {'initial': initial, 'schedule': sched, 'learning_log': {'text_log': log}}
        (directory / (phase + '.json')).write_text(json.dumps({'records': records}))
    ledger = directory / 'commands.jsonl'
    def write(records):
        ledger.write_text(''.join(json.dumps(record) + '\n' for record in records))
    write(commands)
    assert verifier.command_inventory(tmp_path, directory, raw_paths) == 53
    write(commands[:-1])
    with pytest.raises(ValueError, match='GPU commands'):
        verifier.command_inventory(tmp_path, directory, raw_paths)
    corrupt = deepcopy(commands)
    corrupt[-1]['command'][corrupt[-1]['command'].index('--seed') + 1] = '52'
    write(corrupt)
    with pytest.raises(ValueError, match='training GPU budget'):
        verifier.command_inventory(tmp_path, directory, raw_paths)
    corrupt = deepcopy(commands)
    corrupt[-1] = deepcopy(corrupt[-2])
    write(corrupt)
    with pytest.raises(ValueError, match='repeated'):
        verifier.command_inventory(tmp_path, directory, raw_paths)




def test_exact_replay_json_and_scalar_inventory_mutations(tmp_path):
    def write(name,value):
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
        return {'path':name,'sha256':verifier.sha(path)}
    tags=['Loss/value_function','Loss/surrogate','Loss/prior_loss','Policy/mean_noise_std']+[f'metric{i}' for i in range(29)]
    timing=['Perf/total_fps','Perf/collection time','Perf/learning_time','Train/mean_reward/time','Train/mean_episode_length/time']
    records=[]
    checkpoints=[]
    for folder in ('old','new'):
        value=initial(51)
        value['log_dir']=folder
        record={key:write(f'{folder}/{key}.json',raw) for key,raw in (
            ('initial',value),('schedule',schedule()),('rollout',{'rollout':{'samples':8000,'contact':.5}}))}
        record['learning_log']={'scalar_counts':{tag:250 for tag in tags+timing}}
        records.append(record)
        checkpoints.append(write(f'{folder}/model_249.pt',{'not_deserialized':'bytes'}))
    old,new=records
    reference={key:old[key] for key in ('initial','schedule','rollout')}
    reference['checkpoint']=checkpoints[0]
    proof=dict(exact_model_optimizer_infos=True,exact_reward_contact=True,historical_checkpoint=checkpoints[0],
               new_checkpoint=checkpoints[1]['path'],new_checkpoint_sha256=checkpoints[1]['sha256'],
               scalars={'exact_tags':tags,'timing_only_exclusions':timing})
    verifier.replay_evidence(tmp_path,proof,new,reference,old,249)
    broken=deepcopy(proof)
    broken['scalars']['exact_tags'].pop()
    with pytest.raises(ValueError,match='scalar replay'):
        verifier.replay_evidence(tmp_path,broken,new,reference,old,249)
    broken=deepcopy(proof)
    broken['exact_model_optimizer_infos']=False
    with pytest.raises(ValueError,match='replay gate'):
        verifier.replay_evidence(tmp_path,broken,new,reference,old,249)
    changed=schedule()
    changed['raw_reward_mean'][0]=-.25
    new['schedule']=write('new/changed_schedule.json',changed)
    with pytest.raises(ValueError,match='reward/contact'):
        verifier.replay_evidence(tmp_path,proof,new,reference,old,249)


@pytest.mark.parametrize('same_container',[True,False])
def test_seed51_endpoint_binding_allows_different_serialization(same_container):
    historical={'sha256':'a'*64}
    model={'sha256':('a' if same_container else 'b')*64,'source_checkpoint':'fresh/model_249.pt'}
    reference={'checkpoint':{'path':'old/model_249.pt','sha256':historical['sha256']}}
    replay=dict(new_checkpoint_sha256=model['sha256'],new_checkpoint=model['source_checkpoint'],
                historical_checkpoint=deepcopy(reference['checkpoint']),exact_model_optimizer_infos=True)
    assert verifier.seed51_checkpoint_binding(model,historical,reference,replay) is same_container
    for target,key,value in [('model','sha256','c'*64),('model','source_checkpoint','old/model_249.pt'),
                             ('historical','sha256','c'*64),('replay','exact_model_optimizer_infos',False),
                             ('replay','new_checkpoint_sha256','c'*64)]:
        changed={'model':deepcopy(model),'historical':deepcopy(historical),'reference':deepcopy(reference),'replay':deepcopy(replay)}
        changed[target][key]=value
        with pytest.raises(ValueError,match='endpoint proof binding'):
            verifier.seed51_checkpoint_binding(changed['model'],changed['historical'],changed['reference'],changed['replay'])
