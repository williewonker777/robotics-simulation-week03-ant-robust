"""Serial, fail-closed v21 paired development and fixed final training."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from week03_ant.command_study import validate_learning_log
from week03_ant.contact_continuation_v21 import audit_schedule
from week03_ant.continuation_study_v21 import (
    ARMS, DEV_ARMS, INIT_SOURCE, INIT_SHA, V20_ART, ART, ENVS, ITERATIONS, ROOT, START, TRAIN_GEOMETRY, TRAIN_SEED,
    TRANSITIONS, PARENT_SHA, RAMP_STEPS, WORK, check_initial, init_path,
    legacy_hashes, prepare_checkpoint, save_json, sha, source_hashes,
    training_parameters, utc, validate_teacher, verify_hashes,
)
from week03_ant.posture_study import cache_snapshot
from run_contact_study import validate_rollout

PYTHON = str(ROOT.parent / 'run-python')
CACHE = Path('/tmp/isaaclab/terrains')


def read(path):
    return json.loads(Path(path).read_text())


def run(command, label):
    log = WORK / f'{label}.log'
    flags = ('--output', '--audit-output', '--schedule-output')
    outputs = [Path(command[command.index(flag) + 1]) for flag in flags if flag in command]
    if log.exists() or any(path.exists() for path in outputs):
        raise FileExistsError('no implicit retries or overwrites: ' + label)
    started, begin = utc(), time.monotonic()
    with log.open('x') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(label=label, command=command, started_utc=started, finished_utc=utc(),
                  seconds=time.monotonic() - begin, returncode=result.returncode,
                  log=str(log.relative_to(ROOT)), log_sha256=sha(log))
    with (ART / 'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(f"[v21] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode or any(not path.is_file() for path in outputs):
        print('\n'.join(log.read_text().splitlines()[-40:]), flush=True)
        raise RuntimeError('command failed; preserve evidence: ' + label)
    return record


def train_command(arm, initial, schedule, expected=None, *, envs=ENVS,
                  iterations=ITERATIONS, device='cuda:1'):
    if arm not in DEV_ARMS or (envs, iterations) not in ((256, 2), (4096, 2), (ENVS, ITERATIONS)):
        raise ValueError('undeclared v21 training arm/budget')
    if arm == 'immediate' and iterations != 2:
        raise ValueError('immediate is development-only')
    if device != 'cuda:1':
        raise ValueError('v21 is assigned GPU1 only')
    phase = 'preflight' if envs == 256 else 'capacity' if iterations == 2 else 'paired'
    ramp = 32 if phase == 'preflight' else RAMP_STEPS
    command = [PYTHON, 'scripts/contact_continuation_v21.py', 'train', '--arm', arm,
               '--audit-output', str(initial), '--schedule-output', str(schedule),
               '--ramp-steps', str(ramp), '--headless', '--device', device,
               '--num_envs', str(envs), '--max_iterations', str(iterations),
               '--seed', str(TRAIN_SEED), '--run_name', f'v21_{phase}_{arm}',
               '--resume', '--load_run', 'v21_init', '--checkpoint', 'model_0.pt',
               f'env.scene.terrain.terrain_generator.seed={TRAIN_GEOMETRY}',
               f'agent.device={device}']
    if expected is not None:
        command += ['--expected-initial', str(expected)]
    return command


def assert_cache(expected):
    if cache_snapshot(CACHE, [TRAIN_GEOMETRY]) != expected:
        raise ValueError('v21 training terrain cache changed')


def checkpoint_finite(path, iteration):
    import torch

    data = torch.load(path, map_location='cpu', weights_only=False)
    if (data['iter'] != iteration or not all(torch.isfinite(v).all() for v in data['model_state_dict'].values())
            or not bool((data['model_state_dict']['std'] > 0).all())):
        raise ValueError('invalid historical development checkpoint')
    validate_teacher(data['model_state_dict'])
    if not data['optimizer_state_dict']['state']:
        raise ValueError('historical learned optimizer state is empty')
    for group in data['optimizer_state_dict']['state'].values():
        for value in group.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError('historical optimizer is nonfinite')
    return data


def exact_nested(left, right, context='checkpoint'):
    """Require exact tensor values and metadata, not serialized container bytes."""
    import torch

    if torch.is_tensor(left):
        if (not torch.is_tensor(right) or left.dtype != right.dtype or left.shape != right.shape
                or not torch.equal(left, right)):
            raise ValueError('positive-control tensor differs: ' + context)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError('positive-control keys differ: ' + context)
        for key in left:
            exact_nested(left[key], right[key], context + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError('positive-control sequence differs: ' + context)
        for i, (a, b) in enumerate(zip(left, right)):
            exact_nested(a, b, context + '/' + str(i))
    elif type(left) is not type(right) or left != right:
        raise ValueError('positive-control scalar differs: ' + context)


def pin_historical_references():
    legacy_hashes()
    inventory, stages = {}, {}
    def pin(path):
        name = str(path.relative_to(ROOT))
        value = sha(path)
        inventory[name] = value
        return {'path': name, 'sha256': value}
    if sha(INIT_SOURCE) != INIT_SHA:
        raise ValueError('historical shared init changed')
    pin(INIT_SOURCE)
    for name in ('frozen.json', 'training_frozen.json', 'training_validation.json',
                 'trained_models.json', 'training_cache.json', 'initial_shared.json'):
        pin(V20_ART / name)
    for phase in ('preflight', 'capacity', 'main'):
        manifest = V20_ART / ('training_validation.json' if phase == 'main' else phase + '.json')
        pin(manifest)
        prior = read(manifest)['records']['immediate']
        stage = {}
        for name in ('initial', 'schedule', 'rollout'):
            entry = prior[name]
            verify_hashes({entry['path']: entry['sha256']})
            stage[name] = pin(ROOT / entry['path'])
        logs = prior['learning_log']
        verify_hashes({logs['text_log']: logs['text_log_sha256'], **logs['event_files_sha256']})
        pin(ROOT / logs['text_log'])
        for name in logs['event_files_sha256']:
            pin(ROOT / name)
        initial = read(ROOT / stage['initial']['path'])
        checkpoint = ROOT / initial['log_dir'] / ('model_249.pt' if phase == 'main' else 'model_1.pt')
        checkpoint_finite(checkpoint, 249 if phase == 'main' else 1)
        stage['checkpoint'] = pin(checkpoint)
        stages[phase] = stage
    save_json(ART / 'historical_references.json', dict(created_utc=utc(), stages=stages, sha256=inventory,
              disclosure='Historical model1 hashes are newly pinned now; they were not v20-era frozen checkpoint hashes.'))


def verify_references():
    references = read(ART / 'historical_references.json')
    if set(references['stages']) != {'preflight', 'capacity', 'main'}:
        raise ValueError('historical reference stage inventory differs')
    verify_hashes(references['sha256'])
    for stage in references['stages'].values():
        for item in stage.values():
            if references['sha256'].get(item['path']) != item['sha256']:
                raise ValueError('historical reference linkage changed')
    return references


TIMING_SCALARS = {'Perf/total_fps', 'Perf/collection time', 'Perf/learning_time',
                  'Train/mean_reward/time', 'Train/mean_episode_length/time'}


def exact_learning_scalars(new_directory, old_directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    left = EventAccumulator(str(new_directory), size_guidance={'scalars': 0}).Reload()
    right = EventAccumulator(str(old_directory), size_guidance={'scalars': 0}).Reload()
    a, b = set(left.Tags()['scalars']), set(right.Tags()['scalars'])
    if a != b or not TIMING_SCALARS <= a:
        raise ValueError('positive-control scalar inventory differs')
    for tag in sorted(a - TIMING_SCALARS):
        if [(v.step, v.value) for v in left.Scalars(tag)] != [(v.step, v.value) for v in right.Scalars(tag)]:
            raise ValueError('positive-control learning scalar differs: ' + tag)
    return {'exact_tags': sorted(a - TIMING_SCALARS), 'timing_only_exclusions': sorted(TIMING_SCALARS)}


def validate_positive_replay(data, reference):
    if data['schedule'] != read(ROOT / reference['schedule']['path']):
        raise ValueError('positive-control reward-time trajectory differs')
    old_rollout = read(ROOT / reference['rollout']['path'])
    if data['rollout']['rollout'] != old_rollout['rollout']:
        raise ValueError('positive-control passive contacts differ')
    logdir = ROOT / data['initial']['log_dir']
    final = logdir / 'model_1.pt'
    old_checkpoint = ROOT / reference['checkpoint']['path']
    exact_nested(checkpoint_finite(final, 1), checkpoint_finite(old_checkpoint, 1))
    old_initial = read(ROOT / reference['initial']['path'])
    scalar_proof = exact_learning_scalars(logdir, ROOT / old_initial['log_dir'])
    return dict(exact_model_optimizer_infos=True, exact_reward_contact=True,
                new_checkpoint=str(final.relative_to(ROOT)), new_checkpoint_sha256=sha(final),
                historical_checkpoint=reference['checkpoint'], scalars=scalar_proof)


def prepare():
    if (ART / 'initial_shared.json').exists():
        raise FileExistsError('v21 initialization already prepared')
    pin_historical_references()
    save_json(ART / 'initial_shared.json', prepare_checkpoint(init_path()))


def prepare_cache(device):
    verify_references()
    initial = read(ART / 'initial_shared.json')
    verify_hashes({initial['checkpoint']: initial['sha256']})
    output = WORK / 'prepare_training_geometry110.json'
    run([PYTHON, 'scripts/evaluate_contact_v16.py', '--headless', '--device', device,
         '--controller', 'history_original', '--geometry', str(TRAIN_GEOMETRY),
         '--seed', str(TRAIN_SEED), '--phase', 'prepare', '--scenario', 'mixed',
         '--seconds', '16', '--num_envs', '35', '--output', str(output),
         '--checkpoint', str(START)], 'prepare_training_geometry110')
    if read(output).get('scored_episodes') != 0:
        raise ValueError('training cache preparation scored episodes')
    if cache_snapshot(CACHE, [TRAIN_GEOMETRY]) != read(V20_ART / 'training_cache.json')['cache']:
        raise ValueError('training cache no longer equals historical v20')
    save_json(ART / 'training_cache.json', dict(created_utc=utc(), development_only=True,
              preparation=str(output.relative_to(ROOT)), preparation_sha256=sha(output),
              cache=cache_snapshot(CACHE, [TRAIN_GEOMETRY])))


def validate_evidence(initial, schedule, rollout, arm, envs, iterations, ramp_steps):
    audit_schedule(schedule, mode=arm, policy_steps=iterations * 32, ramp_steps=ramp_steps)
    if schedule['arm'] != arm or schedule['num_envs'] != envs:
        raise ValueError('reward evidence treatment/budget differs')
    if any(valid / envs < .99 for valid in schedule['valid_rows']):
        raise ValueError('reward-time invalid fraction exceeds one percent')
    proof = schedule['episode_randomization']
    if (not proof['matches_raw_environment'] or proof['assigned_sha256'] != proof['actual_sha256']
            or not 0 < proof['nonzero'] <= envs or proof['minimum'] < 0
            or proof['maximum'] <= proof['minimum']):
        raise ValueError('actual randomized horizons missing or shadowed')
    if (initial['arm'] != arm or initial['num_envs'] != envs or initial['iteration'] != 0
            or initial['common_step_counter'] != 0 or initial['contact_slip_weight'] != 1.
            or not initial['optimizer_state_empty'] or initial['observation_dimensions'] != [envs, 91]
            or any(initial['initial_policy_parity'].get(key) is not True for key in ('actor', 'critic', 'teacher'))
            or initial['initial_policy_parity']['device'] != 'cuda:1'):
        raise ValueError('initialization proof differs')
    if rollout['arm'] != arm or rollout.get('passive_post_action') is not True:
        raise ValueError('post-action contact provenance differs')
    validate_rollout(rollout['rollout'], iterations * 32, 'v21')


def paired_randomization(left, right):
    if left['episode_randomization'] != right['episode_randomization']:
        raise ValueError('paired actual episode horizons differ')


def evidence_paths(initial, schedule):
    return {'initial': initial, 'schedule': schedule,
            'rollout': initial.with_name(initial.stem + '_contact_rollout.json')}


def inspect_run(paths, arm, envs, iterations, ramp, log):
    data = {name: read(path) for name, path in paths.items()}
    validate_evidence(data['initial'], data['schedule'], data['rollout'], arm, envs, iterations, ramp)
    logdir = ROOT / data['initial']['log_dir']
    saved = training_parameters(logdir, arm, ramp)
    if (saved['normalized'] != data['initial']['normalized_parameters']
            or saved['sha256'] != data['initial']['saved_parameter_sha256']):
        raise ValueError('saved YAML changed after initialization')
    learning = validate_learning_log(logdir, log, iterations, envs * iterations * 32)
    record = {name: {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
              for name, path in paths.items()}
    record.update(learning_log=learning, coefficient_step_mass=data['schedule']['coefficient_sum'],
                  realized_penalty_sum=data['schedule']['realized_penalty_sum'])
    return data, record


def validate_retry(phase, attempt):
    """A new development attempt needs preserved failure evidence and a reason."""
    if attempt == 1:
        return
    path = ART / f'{phase}_attempt{attempt:02d}_infrastructure_reason.json'
    reason = read(path)
    if (reason.get('phase') != phase or reason.get('attempt') != attempt
            or reason.get('infrastructure_only') is not True
            or not isinstance(reason.get('reason'), str) or len(reason['reason'].strip()) < 20
            or not reason.get('prior_evidence_sha256')):
        raise ValueError('new development attempt requires a recorded infrastructure reason')
    verify_hashes(reason['prior_evidence_sha256'])
    if any(name.startswith('artifacts/terrain_demo/contact_continuation_v21/evaluations/')
           or name.startswith('artifacts/terrain_demo/contact_continuation_v21/horizon/')
           for name in reason['prior_evidence_sha256']):
        raise ValueError('scored evidence is not a development retry authority')


def paired_development(device, phase, attempt=1):
    if phase not in ('preflight', 'capacity') or attempt < 1:
        raise ValueError('invalid development phase/attempt')
    if (ART / f'{phase}.json').exists():
        raise FileExistsError('development proof already exists')
    validate_retry(phase, attempt)
    references = verify_references()
    old = references['stages'][phase]
    expected_initial = ROOT / old['initial']['path']
    before, legacy = source_hashes(training=True), legacy_hashes()
    cache = read(ART / 'training_cache.json')['cache']
    assert_cache(cache)
    if phase == 'capacity':
        previous = read(ART / 'preflight.json')
        verify_development(previous)
    envs, iterations, ramp = (256, 2, 32) if phase == 'preflight' else (4096, 2, RAMP_STEPS)
    records, replay = {}, None
    for arm in DEV_ARMS:
        verify_references()
        assert_cache(cache)
        label = f'{phase}_attempt{attempt:02d}_{arm}'
        initial, schedule = WORK / f'{label}_initial.json', WORK / f'{label}_schedule.json'
        command = train_command(arm, initial, schedule, expected_initial,
                                envs=envs, iterations=iterations, device=device)
        run(command, label)
        assert_cache(cache)
        data, records[arm] = inspect_run(evidence_paths(initial, schedule), arm, envs, iterations,
                                         ramp, WORK / f'{label}.log')
        check_initial(data['initial'], read(expected_initial))
        paired_randomization(data['schedule'], read(ROOT / old['schedule']['path']))
        if arm == 'immediate':
            replay = validate_positive_replay(data, old)
    if source_hashes(training=True) != before or legacy_hashes() != legacy:
        raise ValueError('source changed during development')
    verify_references()
    save_json(ART / f'{phase}.json', dict(created_utc=utc(), phase=phase, passed=True, development_only=True,
              source_sha256=before, legacy=legacy, terrain_cache=cache,
              historical_initial_pairing=True, actual_random_horizons_paired=True,
              positive_control_exact_replay=replay, records=records,
              historical_references_sha256=sha(ART / 'historical_references.json')))


def verify_development(proof):
    if (proof['passed'] is not True or proof['source_sha256'] != source_hashes(training=True)
            or set(proof['records']) != set(DEV_ARMS)
            or proof.get('phase') not in ('preflight', 'capacity')
            or proof.get('historical_initial_pairing') is not True
            or proof.get('actual_random_horizons_paired') is not True
            or proof['legacy'] != legacy_hashes()):
        raise ValueError('development sources or completion differs')
    if proof['historical_references_sha256'] != sha(ART / 'historical_references.json'):
        raise ValueError('development reference pin changed')
    reference = verify_references()['stages'][proof['phase']]
    envs, iterations, ramp = (256, 2, 32) if proof['phase'] == 'preflight' else (4096, 2, RAMP_STEPS)
    replay = proof['positive_control_exact_replay']
    if replay.get('exact_model_optimizer_infos') is not True or replay.get('exact_reward_contact') is not True:
        raise ValueError('positive-control replay missing')
    verify_hashes({replay['new_checkpoint']: replay['new_checkpoint_sha256']})
    for arm, record in proof['records'].items():
        for name in ('initial', 'schedule', 'rollout'):
            item = record[name]
            verify_hashes({item['path']: item['sha256']})
        logs = record['learning_log']
        verify_hashes({logs['text_log']: logs['text_log_sha256'], **logs['event_files_sha256']})
        paths = {name: ROOT / record[name]['path'] for name in ('initial', 'schedule', 'rollout')}
        data, recomputed = inspect_run(paths, arm, envs, iterations, ramp, ROOT / logs['text_log'])
        if recomputed != record:
            raise ValueError('development evidence summary differs from raw data')
        check_initial(data['initial'], read(ROOT / reference['initial']['path']))
        paired_randomization(data['schedule'], read(ROOT / reference['schedule']['path']))
        if arm == 'immediate' and validate_positive_replay(data, reference) != replay:
            raise ValueError('positive-control replay differs from raw evidence')


def freeze_train():
    proofs = [read(ART / f'{phase}.json') for phase in ('preflight', 'capacity')]
    for proof in proofs:
        verify_development(proof)
    if proofs[0]['terrain_cache'] != proofs[1]['terrain_cache']:
        raise ValueError('development terrain mismatch')
    assert_cache(proofs[1]['terrain_cache'])
    verify_references()
    initial = read(ART / 'initial_shared.json')
    verify_hashes({initial['checkpoint']: initial['sha256']})
    save_json(ART / 'training_frozen.json', dict(
        frozen_at=utc(), source_sha256=source_hashes(training=True), legacy=legacy_hashes(),
        starting_checkpoint=str(init_path().relative_to(ROOT)), starting_checkpoint_sha256=sha(init_path()),
        source_checkpoint_sha256=PARENT_SHA, seed=TRAIN_SEED, geometry=TRAIN_GEOMETRY,
        envs=ENVS, iterations_per_arm=ITERATIONS, steps_per_env=32, ramp_steps=RAMP_STEPS,
        transitions_per_arm=TRANSITIONS, arms=list(ARMS), selection='final model249 only',
        preflight_sha256=sha(ART / 'preflight.json'), capacity_sha256=sha(ART / 'capacity.json'),
        initial_shared_sha256=sha(ART / 'initial_shared.json'),
        training_cache_sha256=sha(ART / 'training_cache.json'),
        historical_references_sha256=sha(ART / 'historical_references.json'),
        terrain_cache=proofs[1]['terrain_cache']))


def verify_training_freeze():
    frozen = read(ART / 'training_frozen.json')
    if (frozen['source_sha256'] != source_hashes(training=True) or frozen['legacy'] != legacy_hashes()
            or frozen['source_checkpoint_sha256'] != PARENT_SHA):
        raise ValueError('training frozen sources/ancestry changed')
    verify_hashes({frozen['starting_checkpoint']: frozen['starting_checkpoint_sha256']})
    for key, filename in (('preflight_sha256', 'preflight.json'), ('capacity_sha256', 'capacity.json'),
                          ('initial_shared_sha256', 'initial_shared.json'), ('training_cache_sha256', 'training_cache.json'),
                          ('historical_references_sha256', 'historical_references.json')):
        if frozen[key] != sha(ART / filename):
            raise ValueError('training frozen evidence changed: ' + filename)
    for phase in ('preflight', 'capacity'):
        verify_development(read(ART / f'{phase}.json'))
    assert_cache(frozen['terrain_cache'])
    return frozen


def train(device):
    import torch

    if (ART / 'trained_models.json').exists():
        raise FileExistsError('paired v21 training already complete')
    verify_training_freeze()
    references = verify_references()['stages']['main']
    first_path = ROOT / references['initial']['path']
    first_data = {'initial': read(first_path), 'schedule': read(ROOT / references['schedule']['path'])}
    models, records = {}, {}
    for arm in ARMS:
        verify_training_freeze()
        initial = ART / 'training' / f'{arm}_initial.json'
        schedule = ART / 'training' / f'{arm}_schedule.json'
        initial.parent.mkdir(parents=True, exist_ok=True)
        run(train_command(arm, initial, schedule, first_path, device=device), f'train_{arm}')
        verify_training_freeze()
        data, records[arm] = inspect_run(evidence_paths(initial, schedule), arm, ENVS, ITERATIONS,
                                         RAMP_STEPS, WORK / f'train_{arm}.log')
        if first_data is not None:
            check_initial(data['initial'], first_data['initial'])
            paired_randomization(data['schedule'], first_data['schedule'])
        first_path, first_data = first_path or initial, first_data or data
        logdir = ROOT / data['initial']['log_dir']
        final = logdir / 'model_249.pt'
        state = torch.load(final, map_location='cpu', weights_only=False)
        if (state.get('iter') != 249 or not all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
                or not bool((state['model_state_dict']['std'] > 0).all())):
            raise ValueError('nonfinite/invalid final model')
        validate_teacher(state['model_state_dict'])
        if not state['optimizer_state_dict']['state']:
            raise ValueError('final Adam state empty')
        for values in state['optimizer_state_dict']['state'].values():
            for value in values.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError('nonfinite Adam state')
        destination = ART / 'runs' / arm / 'model_249.pt'
        destination.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(final, destination)
        for name in ('env.yaml', 'agent.yaml'):
            shutil.copyfile(logdir / 'params' / name, destination.parent / name)
        if sha(destination) != sha(final):
            raise ValueError('copied final model differs')
        models[arm] = dict(checkpoint=str(destination.relative_to(ROOT)), sha256=sha(destination),
                           source_checkpoint=str(final.relative_to(ROOT)), iteration=249,
                           transitions=TRANSITIONS, initial_audit_sha256=sha(initial), schedule_sha256=sha(schedule))
    save_json(ART / 'training_validation.json', dict(created_utc=utc(), records=records,
              historical_initial_pairing=True, actual_random_horizons_paired=True,
              historical_references_sha256=sha(ART / 'historical_references.json')))
    save_json(ART / 'trained_models.json', dict(completed_utc=utc(), models=models,
              total_training_transitions=TRANSITIONS, training_freeze_sha256=sha(ART / 'training_frozen.json'),
              training_validation_sha256=sha(ART / 'training_validation.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'prepare_cache', 'preflight', 'capacity', 'freeze_train', 'train'))
    parser.add_argument('--device', choices=('cuda:1',), default='cuda:1')
    parser.add_argument('--development-attempt', type=int, default=1)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'outputs/hybrid_v11_20260922/gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'prepare_cache': lambda: prepare_cache(args.device),
         'preflight': lambda: paired_development(args.device, 'preflight', args.development_attempt),
         'capacity': lambda: paired_development(args.device, 'capacity', args.development_attempt),
         'freeze_train': freeze_train, 'train': lambda: train(args.device)}[args.phase]()


if __name__ == '__main__':
    main()
