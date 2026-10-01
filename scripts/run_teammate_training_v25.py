"""Serial v25 initialization/development/freeze/main/validation, exclusive evidence."""
from __future__ import annotations
import argparse
import copy
import fcntl
import importlib.metadata
import math
from pathlib import Path
import shutil
import subprocess
import time
import json
from week03_ant.teammate_study_v25 import (
    ROOT, ART, WORK, PLAN, EXPERIMENT, ARMS, SEEDS, RUNS, ENVS, ITERATIONS, STEPS,
    TRAIN_GEOMETRY, TRANSITIONS, TOTAL_TRANSITIONS, DEVELOPMENT_TRANSITIONS, PARENT_SHA,
    source_hashes, run_identity, entropy, init_path, prepare_checkpoint, read, save_json,
    sha, verify_hashes, utc, training_parameters, validate_teacher,
)
from week03_ant.posture_study import cache_snapshot
from week03_ant.command_study import validate_learning_log
from week03_ant.contact_continuation_v21 import audit_schedule
from teammate_v25 import compare_initial
from run_contact_study import validate_rollout

PYTHON = str(ROOT.parent/'run-python')
CACHE = Path('/tmp/isaaclab/terrains')


def train_command(run, initial, schedule, expected=None, *, envs=ENVS, iterations=ITERATIONS, device='cuda:1'):
    arm,seed=run_identity(run)
    if (device != 'cuda:1' or (envs,iterations) not in ((256,2),(4096,2),(4096,250))
            or (envs==256 and seed!=61)):
        raise ValueError('undeclared device/budget/seed')
    phase='preflight' if envs==256 else 'capacity' if iterations==2 else 'main'
    command=[PYTHON,'scripts/teammate_v25.py','--arm',arm,'--training-seed',str(seed),
        '--audit-output',str(initial),'--schedule-output',str(schedule),'--headless','--device',device,
        '--num_envs',str(envs),'--max_iterations',str(iterations),'--seed',str(seed),
        '--run_name',f'v25_{phase}_{run}','--resume','--load_run',f'v25_init_seed{seed}',
        '--checkpoint','model_0.pt']
    if expected is not None:
        command+=['--expected-initial',str(expected)]
    return command


def run_command(command,label):
    log=WORK/f'{label}.log'
    if log.exists():
        raise FileExistsError(log)
    started,begin=utc(),time.monotonic()
    with log.open('x') as stream:
        process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    record=dict(label=label,command=command,started_utc=started,finished_utc=utc(),
        seconds=time.monotonic()-begin,returncode=process.returncode,
        log=str(log.relative_to(ROOT)),log_sha256=sha(log))
    with (ART/'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    if process.returncode:
        raise RuntimeError('command failed; evidence retained: '+str(log))
    return record


def initialize():
    if (ART/'initial_shared.json').exists():
        raise FileExistsError('initialization exists')
    novelty=read(WORK/'geometry_novelty.json')
    if (novelty.get('status') != 'PASS' or novelty.get('errors') != []
        or novelty.get('new_training_geometry') != 130 or novelty.get('new_holdout_pairs') != [[131,111],[132,112]]):
        raise ValueError('missing/invalid geometry novelty inventory')
    save_json(ART/'geometry_novelty.json',novelty)
    original=ROOT/'artifacts/PUBLICATION_SHA256SUMS'
    copied=ART/'original_publication.sha256'
    with copied.open('xb') as stream:
        stream.write(original.read_bytes())
    verify_original()
    runtime={name:importlib.metadata.version(name) for name in ('torch','rsl-rl-lib','isaaclab')}
    import inspect
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.algorithms import PPO
    import torch
    runtime_sources={str(Path(inspect.getsourcefile(obj)).resolve()):sha(inspect.getsourcefile(obj))
                     for obj in (OnPolicyRunner,PPO,torch.optim.Adam)}
    files={str(path.relative_to(ROOT)):sha(path) for path in (PLAN,)}
    models={str(seed):prepare_checkpoint(seed) for seed in SEEDS}
    save_json(ART/'initial_shared.json',dict(created_utc=utc(),models=models,runtime=runtime,runtime_source_sha256=runtime_sources,
        plan_sha256=files[str(PLAN.relative_to(ROOT))],parent_sha256=PARENT_SHA,
        original_inventory_sha256=sha(copied),geometry_novelty_sha256=sha(ART/'geometry_novelty.json')))


def verify_original():
    entries={line.split('  ',1)[1]:line.split('  ',1)[0]
             for line in (ART/'original_publication.sha256').read_text().splitlines()}
    verify_hashes(entries)


def prepare_cache():
    output=WORK/'prepare_training_geometry130.json'
    from week03_ant.teammate_study_v25 import START
    run_command([PYTHON,'scripts/evaluate_contact_v16.py','--headless','--device','cuda:1',
        '--controller','history_original','--geometry','130','--seed','61','--phase','prepare',
        '--scenario','mixed','--seconds','16','--num_envs','35','--output',str(output),
        '--checkpoint',str(START)],'prepare_training_geometry130')
    if read(output).get('scored_episodes')!=0:
        raise ValueError('cache preparation scored episodes')
    cache=cache_snapshot(CACHE,[TRAIN_GEOMETRY])
    if not cache:
        raise ValueError('missing prepared terrain cache')
    save_json(ART/'training_cache.json',dict(created_utc=utc(),preparation=str(output.relative_to(ROOT)),
        preparation_sha256=sha(output),scored_episodes=0,cache=cache))


def assert_cache():
    expected=read(ART/'training_cache.json')
    verify_hashes({expected['preparation']:expected['preparation_sha256']})
    if cache_snapshot(CACHE,[TRAIN_GEOMETRY])!=expected['cache']:
        raise ValueError('training terrain cache changed')


def validate_evidence(initial,schedule,rollout,run,envs,iterations):
    arm,seed=run_identity(run)
    if (initial['arm']!=arm or initial['training_seed']!=seed
        or initial['actual_seed']!=dict.fromkeys(('requested','saved_agent','saved_environment','live_environment'),seed)
        or initial['num_envs']!=envs or initial['observation_dimensions']!=[envs,91]
        or initial['iteration']!=0 or initial['common_step_counter']!=0
        or initial['optimizer_state_empty'] is not True or abs(initial['dt']-1/60)>1e-12
        or initial['initial_policy_parity']!=dict(actor=True,critic=True,teacher=True,device='cuda:1')):
        raise ValueError('initial identity/budget/parity differs')
    if (schedule['arm']!=arm or schedule['training_seed']!=seed or schedule['num_envs']!=envs
        or schedule['policy_steps']!=iterations*32 or schedule['ppo_updates']!=iterations
        or schedule['adam_steps']!=iterations*20 or schedule['entropy_coef']!=entropy(arm)
        or schedule['teacher_coef']!=.02 or schedule['learning_rate']!=1e-4):
        raise ValueError('actual training counts/coefficients differ')
    trace=schedule['runtime_updates']
    baseline=schedule['runtime_initial']
    if (len(trace)!=iterations or schedule['runtime_final']!=baseline
        or baseline['entropy_coef']!=entropy(arm) or baseline['teacher_coef']!=.02
        or baseline['schedule']!='fixed' or baseline['desired_kl'] is not None
        or baseline['learning_rate']!=1e-4):
        raise ValueError('runtime coefficient trace differs')
    for index,row in enumerate(trace,1):
        if row != dict(update=index,adam_steps=index*20,before=baseline,after=baseline):
            raise ValueError('PPO runtime trace changes')
    rows=schedule['recovery']
    if len(rows)!=iterations*32:
        raise ValueError('missing reward-time diagnostic rows')
    for step,row in enumerate(rows,1):
        if (row['step']!=step or row['enabled'] is not (arm!='control') or abs(row['dt']-1/60)>1e-12
            or not all(math.isfinite(row[key]) for key in ('raw_sum','applied_sum','clearance_risk',
                'tilt_risk','action_delta_squared_sum','angular_xy_squared_sum'))
            or not 0<=row['abstained']<=envs or row['raw_sum']>0
            or row['applied_sum']!=(0 if arm=='control' else row['raw_sum'])):
            raise ValueError('reward diagnostics/treatment changed')
        reconstructed=-2*row['clearance_risk']-2*row['tilt_risk']-.01*row['action_delta_squared_sum']-.025*row['angular_xy_squared_sum']
        if not math.isclose(reconstructed,row['raw_sum'],rel_tol=2e-6,abs_tol=1e-5):
            raise ValueError('recovery components do not reconstruct rate')
    audit_schedule(schedule['contact'],mode='no_cost',policy_steps=iterations*32,ramp_steps=4000)
    proof=schedule['episode_randomization']
    if (not proof or proof['matches_raw_environment'] is not True
        or proof['assigned_sha256']!=proof['actual_sha256'] or not 0<proof['nonzero']<=envs
        or not 0<=proof['minimum']<proof['maximum']):
        raise ValueError('missing actual random horizon proof')
    if rollout['arm']!=arm or rollout['passive_post_action'] is not True:
        raise ValueError('rollout identity differs')
    validate_rollout(rollout['rollout'],iterations*32,'v25')


def paired_initial(left,right):
    """Within-seed pair, preserving seeds and every non-treatment config value."""
    a,b=copy.deepcopy(left),copy.deepcopy(right)
    if a['training_seed']!=b['training_seed']:
        raise ValueError('cannot pair different seeds')
    for item in (a,b):
        item['arm']='<paired>'
        params=item['normalized_parameters']
        params['env']['rewards']['teammate_recovery']['params']['enabled']='<paired>'
        params['agent']['algorithm']['entropy_coef']='<paired>'
    compare_initial(a,b)


def project_main_initial(initial):
    if initial['num_envs']!=4096 or initial['normalized_parameters']['agent']['max_iterations']!='2':
        raise ValueError('projection requires own 4096x2 capacity')
    result=copy.deepcopy(initial)
    result['normalized_parameters']['agent']['max_iterations']='250'
    return result


def paths(initial,schedule):
    return dict(initial=initial,schedule=schedule,rollout=initial.with_name(initial.stem+'_contact_rollout.json'))


def inspect_run(evidence,run,envs,iterations,log):
    data={key:read(path) for key,path in evidence.items()}
    validate_evidence(data['initial'],data['schedule'],data['rollout'],run,envs,iterations)
    arm,seed=run_identity(run)
    directory=ROOT/data['initial']['log_dir']
    saved=training_parameters(directory,arm,seed)
    if saved['normalized']!=data['initial']['normalized_parameters'] or saved['sha256']!=data['initial']['saved_parameter_sha256']:
        raise ValueError('saved configuration changed')
    record={key:dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for key,path in evidence.items()}
    record['learning_log']=validate_learning_log(directory,log,iterations,envs*iterations*32)
    return data,record


def record_data(record,run,envs,iterations):
    verify_hashes({entry['path']:entry['sha256'] for key,entry in record.items() if key!='learning_log'})
    logs=record['learning_log']
    verify_hashes({logs['text_log']:logs['text_log_sha256'],**logs['event_files_sha256']})
    data,computed=inspect_run({key:ROOT/record[key]['path'] for key in ('initial','schedule','rollout')},
        run,envs,iterations,ROOT/logs['text_log'])
    if computed!=record:
        raise ValueError('record differs from raw evidence')
    return data


def verify_development(proof):
    phase=proof['phase']
    runs=RUNS[:3] if phase=='preflight' else RUNS if phase=='capacity' else ()
    if (proof['passed'] is not True or proof['source_sha256']!=source_hashes()
        or set(proof['records'])!=set(runs) or not runs):
        raise ValueError('development completion/source inventory differs')
    envs=256 if phase=='preflight' else 4096
    paired={}
    for name in runs:
        data=record_data(proof['records'][name],name,envs,2)
        _,seed=run_identity(name)
        if seed in paired:
            reference=paired[seed]
            paired_initial(data['initial'],reference['initial'])
            if data['schedule']['episode_randomization']!=reference['schedule']['episode_randomization']:
                raise ValueError('paired actual randomized horizons differ')
        else:
            paired[seed]=data


def development(phase):
    if phase not in ('preflight','capacity') or (ART/'training_frozen.json').exists():
        raise ValueError('development phase closed')
    if (ART/f'{phase}.json').exists():
        raise FileExistsError('development exists')
    if phase=='capacity':
        verify_development(read(ART/'preflight.json'))
    sources=source_hashes()
    runs=RUNS[:3] if phase=='preflight' else RUNS
    envs=256 if phase=='preflight' else 4096
    records={}
    for name in runs:
        assert_cache()
        initial,schedule=WORK/f'{phase}_{name}_initial.json',WORK/f'{phase}_{name}_schedule.json'
        run_command(train_command(name,initial,schedule,envs=envs,iterations=2),f'{phase}_{name}')
        _,records[name]=inspect_run(paths(initial,schedule),name,envs,2,WORK/f'{phase}_{name}.log')
    proof=dict(created_utc=utc(),phase=phase,passed=True,source_sha256=sources,records=records)
    verify_development(proof)
    assert_cache(); verify_original()
    save_json(ART/f'{phase}.json',proof)


def budget():
    return dict(seeds=list(SEEDS),runs=list(RUNS),geometry=130,envs=4096,iterations=250,steps=32,
        transitions_per_run=TRANSITIONS,total_training_transitions=TOTAL_TRANSITIONS,
        development_training_transitions=DEVELOPMENT_TRANSITIONS)



def verify_scored_development():
    proof=read(ART/'development.json')
    if (proof.get('schema')!='week03_ant_teammate_v25_development_v1'
        or proof.get('verdict')!='PASS' or proof.get('physical_first_episodes')!=280
        or proof.get('dependent_window_observations')!=560
        or proof.get('exact_per_environment_parity') is not True
        or not proof.get('independent_raw_audit')):
        raise ValueError('scored development parity/audit did not pass')
    verify_hashes(proof['evidence_sha256'])
    frozen=read(ROOT/proof['freeze_path'])
    if (sha(ROOT/proof['freeze_path'])!=proof['freeze_sha256']
        or frozen['source_sha256']!=source_hashes() or frozen['phase']!='development'):
        raise ValueError('scored development source/model freeze changed')
    from run_teammate_eval_v25 import verify
    verify(frozen)
    return proof


def freeze():
    verify_scored_development()
    for phase in ('preflight','capacity'):
        verify_development(read(ART/f'{phase}.json'))
    expected={}
    capacity=read(ART/'capacity.json')
    for name in RUNS:
        path=ART/'expected_main'/f'{name}_initial.json'
        save_json(path,project_main_initial(read(ROOT/capacity['records'][name]['initial']['path'])))
        expected[str(path.relative_to(ROOT))]=sha(path)
    evidence=('initial_shared.json','training_cache.json','preflight.json','capacity.json','geometry_novelty.json','original_publication.sha256','development.json')
    verify_original(); assert_cache()
    save_json(ART/'training_frozen.json',dict(created_utc=utc(),source_sha256=source_hashes(True),
        all_source_sha256=source_hashes(),expected_main_sha256=expected,
        artifacts_sha256={str((ART/name).relative_to(ROOT)):sha(ART/name) for name in evidence},
        initializers=read(ART/'initial_shared.json')['models'],parent_sha256=PARENT_SHA,**budget()))


def verify_training_freeze():
    frozen=read(ART/'training_frozen.json')
    if (frozen['source_sha256']!=source_hashes(True) or frozen['all_source_sha256']!=source_hashes()
        or frozen['parent_sha256']!=PARENT_SHA or any(frozen[key]!=value for key,value in budget().items())):
        raise ValueError('scientific freeze changed')
    verify_hashes(frozen['artifacts_sha256']); verify_hashes(frozen['expected_main_sha256'])
    initial_manifest=read(ART/'initial_shared.json')
    for name,version in initial_manifest['runtime'].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError('runtime dependency version changed')
    for path,digest in initial_manifest['runtime_source_sha256'].items():
        if sha(path)!=digest:
            raise ValueError('runtime dependency source changed')
    if frozen['initializers']!=read(ART/'initial_shared.json')['models']:
        raise ValueError('initializer manifest changed')
    for item in frozen['initializers'].values():
        verify_hashes({item['checkpoint']:item['sha256']})
    for name in RUNS:
        capacity=read(ART/'capacity.json')['records'][name]
        expected=read(ART/'expected_main'/f'{name}_initial.json')
        if expected!=project_main_initial(read(ROOT/capacity['initial']['path'])):
            raise ValueError('own capacity projection changed')
    verify_original(); assert_cache(); verify_scored_development()
    return frozen


def checkpoint_finite(path,iteration):
    import torch
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data['iter']!=iteration or not all(torch.isfinite(value).all() for value in data['model_state_dict'].values()):
        raise ValueError('invalid checkpoint iteration/tensors')
    validate_teacher(data['model_state_dict'])
    if not (data['model_state_dict']['std']>0).all() or not data['optimizer_state_dict']['state']:
        raise ValueError('invalid std/empty trained optimizer')
    for entry in data['optimizer_state_dict']['state'].values():
        for value in entry.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError('nonfinite optimizer')
        if float(entry['step'])!=(iteration+1)*20:
            raise ValueError('saved Adam step count changed')
    return data


def train():
    if (ART/'trained_models.json').exists():
        raise FileExistsError('all finals already exist')
    records,models={},{}
    for name in RUNS:
        verify_training_freeze()
        arm,seed=run_identity(name)
        initial,schedule=ART/'training'/f'{name}_initial.json',ART/'training'/f'{name}_schedule.json'
        run_command(train_command(name,initial,schedule,ART/'expected_main'/f'{name}_initial.json'),f'main_{name}')
        data,records[name]=inspect_run(paths(initial,schedule),name,4096,250,WORK/f'main_{name}.log')
        source=ROOT/data['initial']['log_dir']/'model_249.pt'
        if source.is_symlink() or not source.resolve().is_relative_to((ROOT/'logs/rsl_rl'/EXPERIMENT).resolve()):
            raise ValueError('checkpoint outside fresh training namespace')
        checkpoint_finite(source,249)
        destination=ART/'runs'/name/'model_249.pt'
        destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as reader,destination.open('xb') as writer:
            shutil.copyfileobj(reader,writer)
        shutil.copytree(source.parent/'params',destination.parent/'params')
        models[name]=dict(checkpoint=str(destination.relative_to(ROOT)),sha256=sha(destination),
            iteration=249,training_seed=seed,arm=arm,transitions=TRANSITIONS,
            initial_audit_sha256=records[name]['initial']['sha256'],schedule_sha256=records[name]['schedule']['sha256'])
    save_json(ART/'training_validation.json',dict(created_utc=utc(),records=records,passed=True,
        training_freeze_sha256=sha(ART/'training_frozen.json'),**budget()))
    save_json(ART/'trained_models.json',dict(created_utc=utc(),models=models,
        training_freeze_sha256=sha(ART/'training_frozen.json'),
        training_validation_sha256=sha(ART/'training_validation.json'),**budget()))
    validate()


def validate():
    verify_training_freeze()
    trained,validation=read(ART/'trained_models.json'),read(ART/'training_validation.json')
    if (set(trained['models'])!=set(RUNS) or set(validation['records'])!=set(RUNS)
        or trained['training_validation_sha256']!=sha(ART/'training_validation.json')
        or trained['training_freeze_sha256']!=sha(ART/'training_frozen.json')):
        raise ValueError('incomplete/changed final manifest')
    from week03_ant.teammate_study_v25 import model_entry
    for name in RUNS:
        entry=model_entry(name)
        checkpoint_finite(ROOT/entry['checkpoint'],249)
        data=record_data(validation['records'][name],name,4096,250)
        compare_initial(data['initial'],read(ART/'expected_main'/f'{name}_initial.json'))
        capacity=read(ROOT/read(ART/'capacity.json')['records'][name]['schedule']['path'])
        if data['schedule']['episode_randomization']!=capacity['episode_randomization']:
            raise ValueError('main random horizons differ from own capacity')
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('init','cache','preflight','capacity','freeze','train','validate'))
    args=parser.parse_args(argv)
    ART.mkdir(parents=True,exist_ok=True); WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.operation in ('preflight','capacity'):
            development(args.operation)
        else:
            {'init':initialize,'cache':prepare_cache,'freeze':freeze,'train':train,'validate':validate}[args.operation]()

if __name__=='__main__':
    main()
