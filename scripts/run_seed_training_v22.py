"""Serial, fail-closed v22 paired development and fixed final training."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from week03_ant.command_study import validate_learning_log
from week03_ant.contact_continuation_v21 import audit_schedule
from week03_ant.seed_study_v22 import (
    RUNS, TRAIN_SEEDS, TOTAL_TRANSITIONS, DEVELOPMENT_TRANSITIONS, training_seed, INIT_SOURCE, INIT_SHA, V21_ART, ART, ENVS, ITERATIONS, ROOT, START, TRAIN_GEOMETRY, TRAIN_SEED,
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
    print(f"[v22] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode or any(not path.is_file() for path in outputs):
        print('\n'.join(log.read_text().splitlines()[-40:]), flush=True)
        raise RuntimeError('command failed; preserve evidence: ' + label)
    return record


def train_command(seed, initial, schedule, expected=None, *, envs=ENVS,
                  iterations=ITERATIONS, device='cuda:1'):
    if type(seed) is not int or seed not in TRAIN_SEEDS or (envs, iterations) not in ((256, 2), (4096, 2), (ENVS, ITERATIONS)):
        raise ValueError('undeclared v22 training seed/budget')
    if device != 'cuda:1':
        raise ValueError('v22 is assigned GPU1 only')
    phase = 'preflight' if envs == 256 else 'capacity' if iterations == 2 else 'main'
    ramp = 32 if phase == 'preflight' else RAMP_STEPS
    command = [PYTHON, 'scripts/seed_continuation_v22.py', 'train', '--arm', 'no_cost',
               '--training-seed', str(seed), '--audit-output', str(initial), '--schedule-output', str(schedule),
               '--ramp-steps', str(ramp), '--headless', '--device', device,
               '--num_envs', str(envs), '--max_iterations', str(iterations),
               '--seed', str(seed), '--run_name', f'v22_{phase}_seed{seed}',
               '--resume', '--load_run', 'v22_init', '--checkpoint', 'model_0.pt',
               f'env.scene.terrain.terrain_generator.seed={TRAIN_GEOMETRY}',
               f'agent.device={device}']
    if expected is not None:
        command += ['--expected-initial', str(expected)]
    return command


def assert_cache(expected):
    if cache_snapshot(CACHE, [TRAIN_GEOMETRY]) != expected:
        raise ValueError('v22 training terrain cache changed')


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
        pin(V21_ART / name)
    for phase in ('preflight', 'capacity', 'main'):
        manifest = V21_ART / ('training_validation.json' if phase == 'main' else phase + '.json')
        pin(manifest)
        prior = read(manifest)['records']['no_cost']
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
        if phase == 'main' and sha(checkpoint) != read(V21_ART / 'trained_models.json')['models']['no_cost']['sha256']:
            raise ValueError('historical final differs from frozen v21 model')
        stage['checkpoint'] = pin(checkpoint)
        stages[phase] = stage
    save_json(ART / 'historical_references.json', dict(created_utc=utc(), stages=stages, sha256=inventory,
              disclosure='Historical model1 hashes are newly pinned now; they were not v21-era frozen checkpoint hashes.'))


def verify_references():
    references = read(ART / 'historical_references.json')
    if set(references['stages']) != {'preflight', 'capacity', 'main'}:
        raise ValueError('historical reference stage inventory differs')
    verify_hashes(references['sha256'])
    for stage in references['stages'].values():
        if set(stage) != {'initial', 'schedule', 'rollout', 'checkpoint'}:
            raise ValueError('historical reference evidence inventory differs')
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
    if a != b or not TIMING_SCALARS <= a or len(a - TIMING_SCALARS) != 33:
        raise ValueError('positive-control scalar inventory differs')
    for tag in sorted(a - TIMING_SCALARS):
        if [(v.step, v.value) for v in left.Scalars(tag)] != [(v.step, v.value) for v in right.Scalars(tag)]:
            raise ValueError('positive-control learning scalar differs: ' + tag)
    return {'exact_tags': sorted(a - TIMING_SCALARS), 'timing_only_exclusions': sorted(TIMING_SCALARS)}


def validate_positive_replay(data, reference, iteration=1):
    if data['schedule'] != read(ROOT / reference['schedule']['path']):
        raise ValueError('positive-control reward-time trajectory differs')
    old_rollout = read(ROOT / reference['rollout']['path'])
    if data['rollout']['rollout'] != old_rollout['rollout']:
        raise ValueError('positive-control passive contacts differ')
    logdir = ROOT / data['initial']['log_dir']
    final = logdir / f'model_{iteration}.pt'
    old_checkpoint = ROOT / reference['checkpoint']['path']
    exact_nested(checkpoint_finite(final, iteration), checkpoint_finite(old_checkpoint, iteration))
    old_initial = read(ROOT / reference['initial']['path'])
    scalar_proof = exact_learning_scalars(logdir, ROOT / old_initial['log_dir'])
    return dict(exact_model_optimizer_infos=True, exact_reward_contact=True,
                new_checkpoint=str(final.relative_to(ROOT)), new_checkpoint_sha256=sha(final),
                historical_checkpoint=reference['checkpoint'], scalars=scalar_proof)


def prepare():
    if (ART / 'initial_shared.json').exists():
        raise FileExistsError('v22 initialization already prepared')
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
    if cache_snapshot(CACHE, [TRAIN_GEOMETRY]) != read(V21_ART / 'training_cache.json')['cache']:
        raise ValueError('training cache no longer equals historical v21')
    save_json(ART / 'training_cache.json', dict(created_utc=utc(), development_only=True,
              preparation=str(output.relative_to(ROOT)), preparation_sha256=sha(output),
              cache=cache_snapshot(CACHE, [TRAIN_GEOMETRY])))


def validate_evidence(initial, schedule, rollout, seed, envs, iterations, ramp_steps):
    arm = 'no_cost'
    validate_seed(initial, seed)
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
    validate_rollout(rollout['rollout'], iterations * 32, 'v22')


def paired_randomization(left, right):
    if left['episode_randomization'] != right['episode_randomization']:
        raise ValueError('paired actual episode horizons differ')


def evidence_paths(initial, schedule):
    return {'initial': initial, 'schedule': schedule,
            'rollout': initial.with_name(initial.stem + '_contact_rollout.json')}


def inspect_run(paths, seed, envs, iterations, ramp, log):
    data = {name: read(path) for name, path in paths.items()}
    validate_evidence(data['initial'], data['schedule'], data['rollout'], seed, envs, iterations, ramp)
    logdir = ROOT / data['initial']['log_dir']
    saved = training_parameters(logdir, seed, ramp)
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
    if any(name.startswith('artifacts/terrain_demo/seed_continuation_v22/evaluations/')
           or name.startswith('artifacts/terrain_demo/seed_continuation_v22/horizon/')
           for name in reason['prior_evidence_sha256']):
        raise ValueError('scored evidence is not a development retry authority')


def validate_seed(initial, seed, *, historical=False):
    if type(seed) is not int or seed not in TRAIN_SEEDS:
        raise ValueError('undeclared training seed')
    if not historical:
        actual = initial.get('actual_seed', {})
        if (type(initial.get('training_seed')) is not int or initial['training_seed'] != seed
                or actual != dict.fromkeys(('requested', 'saved_agent', 'saved_environment', 'live_environment'), seed)
                or any(type(value) is not int for value in actual.values())):
            raise ValueError('live training seed proof differs')
    parameters = initial['normalized_parameters']
    if any(parameters[name]['seed'] != str(seed) for name in ('env', 'agent')):
        raise ValueError('simulator/agent seed mismatch')


def common_initial(initial, reference, seed):
    """Compare only explicitly common fields; never equate cross-seed physics/RNG."""
    validate_seed(initial, seed)
    validate_seed(reference, 51, historical=True)
    left, right = copy.deepcopy(initial['normalized_parameters']), reference['normalized_parameters']
    for name in ('env', 'agent'):
        left[name]['seed'] = '51'
    if left != right:
        raise ValueError('nonseed configuration differs')
    for key in ('policy_state_sha256', 'initial_policy_parity', 'optimizer_state_empty',
                'iteration', 'common_step_counter', 'contact_slip_weight', 'observation_dimensions'):
        if initial[key] != reference[key]:
            raise ValueError('common initialization differs: ' + key)


def project_main_initial(initial, seed):
    validate_seed(initial, seed)
    if initial['num_envs'] != ENVS or initial['normalized_parameters']['agent']['max_iterations'] != '2':
        raise ValueError('main projection requires same-seed 4096-env two-iteration capacity')
    result = copy.deepcopy(initial)
    result['normalized_parameters']['agent']['max_iterations'] = '250'
    return result


def paired_development(device, phase, attempt=1):
    if phase not in ('preflight', 'capacity') or attempt < 1:
        raise ValueError('invalid development phase/attempt')
    if (ART / 'training_frozen.json').exists() or (ART / f'{phase}.json').exists():
        raise FileExistsError('development proof or training freeze already exists')
    validate_retry(phase, attempt)
    old = verify_references()['stages'][phase]
    expected_initial = ROOT / old['initial']['path']
    before, legacy = source_hashes(), legacy_hashes()
    cache = read(ART / 'training_cache.json')['cache']
    assert_cache(cache)
    if phase == 'capacity':
        verify_development(read(ART / 'preflight.json'))
    envs, iterations, ramp = (256, 2, 32) if phase == 'preflight' else (4096, 2, RAMP_STEPS)
    records, replay = {}, None
    for name in RUNS:
        seed = training_seed(name)
        verify_references()
        assert_cache(cache)
        label = f'{phase}_attempt{attempt:02d}_{name}'
        initial, schedule = WORK / f'{label}_initial.json', WORK / f'{label}_schedule.json'
        run(train_command(seed, initial, schedule, expected_initial if seed == 51 else None,
                          envs=envs, iterations=iterations, device=device), label)
        assert_cache(cache)
        data, records[name] = inspect_run(evidence_paths(initial, schedule), seed, envs, iterations,
                                          ramp, WORK / f'{label}.log')
        common_initial(data['initial'], read(expected_initial), seed)
        if seed == 51:
            check_initial(data['initial'], read(expected_initial))
            paired_randomization(data['schedule'], read(ROOT / old['schedule']['path']))
            replay = validate_positive_replay(data, old)
    if source_hashes() != before or legacy_hashes() != legacy:
        raise ValueError('source changed during development')
    save_json(ART / f'{phase}.json', dict(created_utc=utc(), phase=phase, passed=True, development_only=True,
              source_sha256=before, legacy=legacy, terrain_cache=cache,
              historical_seed51_initial_pairing=True, common_nonseed_initialization=True,
              positive_control_exact_replay=replay, records=records,
              historical_references_sha256=sha(ART / 'historical_references.json')))


def record_data(record, seed, envs, iterations, ramp):
    for name in ('initial', 'schedule', 'rollout'):
        item = record[name]
        verify_hashes({item['path']: item['sha256']})
    logs = record['learning_log']
    verify_hashes({logs['text_log']: logs['text_log_sha256'], **logs['event_files_sha256']})
    paths = {name: ROOT / record[name]['path'] for name in ('initial', 'schedule', 'rollout')}
    data, recomputed = inspect_run(paths, seed, envs, iterations, ramp, ROOT / logs['text_log'])
    if recomputed != record:
        raise ValueError('evidence summary differs from raw data')
    return data


def verify_development(proof):
    if (proof['passed'] is not True or proof['source_sha256'] != source_hashes()
            or set(proof['records']) != set(RUNS) or proof.get('phase') not in ('preflight', 'capacity')
            or proof.get('historical_seed51_initial_pairing') is not True
            or proof.get('common_nonseed_initialization') is not True or proof['legacy'] != legacy_hashes()):
        raise ValueError('development sources or completion differs')
    if proof['historical_references_sha256'] != sha(ART / 'historical_references.json'):
        raise ValueError('development reference pin changed')
    reference = verify_references()['stages'][proof['phase']]
    envs, iterations, ramp = (256, 2, 32) if proof['phase'] == 'preflight' else (4096, 2, RAMP_STEPS)
    for name in RUNS:
        seed = training_seed(name)
        data = record_data(proof['records'][name], seed, envs, iterations, ramp)
        common_initial(data['initial'], read(ROOT / reference['initial']['path']), seed)
        if seed == 51:
            check_initial(data['initial'], read(ROOT / reference['initial']['path']))
            paired_randomization(data['schedule'], read(ROOT / reference['schedule']['path']))
            if validate_positive_replay(data, reference) != proof['positive_control_exact_replay']:
                raise ValueError('positive-control replay differs from raw evidence')


def expected_main_path(name):
    training_seed(name)
    return ART / 'expected_main' / f'{name}_initial.json'


def frozen_budget():
    return dict(seeds=list(TRAIN_SEEDS), runs=list(RUNS), geometry=TRAIN_GEOMETRY, envs=ENVS,
                iterations_per_run=ITERATIONS, steps_per_env=32, ramp_steps=RAMP_STEPS,
                transitions_per_run=TRANSITIONS, total_training_transitions=TOTAL_TRANSITIONS,
                development_training_transitions=DEVELOPMENT_TRANSITIONS,
                selection='fresh final model249 only')


def freeze_train():
    proofs = [read(ART / f'{phase}.json') for phase in ('preflight', 'capacity')]
    for proof in proofs:
        verify_development(proof)
    if proofs[0]['terrain_cache'] != proofs[1]['terrain_cache']:
        raise ValueError('development terrain mismatch')
    assert_cache(proofs[1]['terrain_cache'])
    references = verify_references()
    expected = {}
    for name in RUNS:
        initial = read(ROOT / proofs[1]['records'][name]['initial']['path'])
        projected = project_main_initial(initial, training_seed(name))
        if name == 'seed51':
            check_initial(projected, read(ROOT / references['stages']['main']['initial']['path']))
        path = expected_main_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, projected)
        expected[str(path.relative_to(ROOT))] = sha(path)
    initial = read(ART / 'initial_shared.json')
    verify_hashes({initial['checkpoint']: initial['sha256']})
    save_json(ART / 'training_frozen.json', dict(
        frozen_at=utc(), source_sha256=source_hashes(training=True), all_source_sha256=source_hashes(), legacy=legacy_hashes(), **frozen_budget(),
        starting_checkpoint=str(init_path().relative_to(ROOT)), starting_checkpoint_sha256=sha(init_path()),
        source_checkpoint_sha256=PARENT_SHA, expected_main_sha256=expected,
        preflight_sha256=sha(ART / 'preflight.json'), capacity_sha256=sha(ART / 'capacity.json'),
        initial_shared_sha256=sha(ART / 'initial_shared.json'), training_cache_sha256=sha(ART / 'training_cache.json'),
        historical_references_sha256=sha(ART / 'historical_references.json'), terrain_cache=proofs[1]['terrain_cache']))


def verify_training_freeze():
    frozen = read(ART / 'training_frozen.json')
    if (frozen['source_sha256'] != source_hashes(training=True) or frozen['all_source_sha256'] != source_hashes() or frozen['legacy'] != legacy_hashes()
            or frozen['source_checkpoint_sha256'] != PARENT_SHA
            or any(frozen.get(key) != value for key, value in frozen_budget().items())
            or frozen['starting_checkpoint'] != str(init_path().relative_to(ROOT))
            or frozen['starting_checkpoint_sha256'] != INIT_SHA):
        raise ValueError('training frozen sources/ancestry/budget changed')
    verify_hashes({frozen['starting_checkpoint']: frozen['starting_checkpoint_sha256']})
    for key, filename in (('preflight_sha256', 'preflight.json'), ('capacity_sha256', 'capacity.json'),
                          ('initial_shared_sha256', 'initial_shared.json'), ('training_cache_sha256', 'training_cache.json'),
                          ('historical_references_sha256', 'historical_references.json')):
        if frozen[key] != sha(ART / filename):
            raise ValueError('training frozen evidence changed: ' + filename)
    for phase in ('preflight', 'capacity'):
        verify_development(read(ART / f'{phase}.json'))
    if set(frozen['expected_main_sha256']) != {str(expected_main_path(name).relative_to(ROOT)) for name in RUNS}:
        raise ValueError('expected main inventory changed')
    verify_hashes(frozen['expected_main_sha256'])
    capacity = read(ART / 'capacity.json')
    for name in RUNS:
        raw = read(ROOT / capacity['records'][name]['initial']['path'])
        if read(expected_main_path(name)) != project_main_initial(raw, training_seed(name)):
            raise ValueError('capacity to main projection changed')
    assert_cache(frozen['terrain_cache'])
    return frozen


def verify_full_replay():
    proof = read(ART / 'full_replay.json')
    if (proof.get('passed') is not True or proof.get('seed') != 51
            or proof['training_freeze_sha256'] != sha(ART / 'training_frozen.json')
            or proof['historical_references_sha256'] != sha(ART / 'historical_references.json')):
        raise ValueError('full replay gate missing or changed')
    reference = verify_references()['stages']['main']
    data = record_data(proof['record'], 51, ENVS, ITERATIONS, RAMP_STEPS)
    check_initial(data['initial'], read(ROOT / reference['initial']['path']))
    paired_randomization(data['schedule'], read(ROOT / reference['schedule']['path']))
    if validate_positive_replay(data, reference, 249) != proof['replay']:
        raise ValueError('full replay recomputation differs')
    return proof


def validate_fresh_final(final, previous=()):
    """Never substitute a historical path or alias another fresh run."""
    from week03_ant.seed_study_v22 import EXPERIMENT

    resolved = final.resolve()
    namespace = ROOT / 'logs' / 'rsl_rl' / EXPERIMENT
    if (not resolved.is_relative_to(namespace.resolve()) or final.is_symlink()
            or final.name != 'model_249.pt' or resolved in {Path(path).resolve() for path in previous}
            or final.parent.name == 'v22_init'):
        raise ValueError('historical fallback or checkpoint alias forbidden')
    return checkpoint_finite(final, 249)


def train(device):
    if (ART / 'trained_models.json').exists():
        raise FileExistsError('v22 training already complete')
    models, records = {}, {}
    for name in RUNS:
        seed = training_seed(name)
        verify_training_freeze()
        if seed != 51:
            verify_full_replay()
        initial = ART / 'training' / f'{name}_initial.json'
        schedule = ART / 'training' / f'{name}_schedule.json'
        initial.parent.mkdir(parents=True, exist_ok=True)
        run(train_command(seed, initial, schedule, expected_main_path(name), device=device), f'train_{name}')
        verify_training_freeze()
        data, records[name] = inspect_run(evidence_paths(initial, schedule), seed, ENVS, ITERATIONS,
                                          RAMP_STEPS, WORK / f'train_{name}.log')
        check_initial(data['initial'], read(expected_main_path(name)))
        capacity = read(ART / 'capacity.json')['records'][name]
        paired_randomization(data['schedule'], read(ROOT / capacity['schedule']['path']))
        if seed == 51:
            reference = verify_references()['stages']['main']
            check_initial(data['initial'], read(ROOT / reference['initial']['path']))
            paired_randomization(data['schedule'], read(ROOT / reference['schedule']['path']))
            replay = validate_positive_replay(data, reference, 249)
            save_json(ART / 'full_replay.json', dict(created_utc=utc(), passed=True, seed=51,
                      record=records[name], replay=replay, training_freeze_sha256=sha(ART / 'training_frozen.json'),
                      historical_references_sha256=sha(ART / 'historical_references.json')))
            verify_full_replay()
        logdir = ROOT / data['initial']['log_dir']
        final = logdir / 'model_249.pt'
        validate_fresh_final(final, [ROOT / item['source_checkpoint'] for item in models.values()])
        destination = ART / 'runs' / name / 'model_249.pt'
        destination.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(final, destination)
        for filename in ('env.yaml', 'agent.yaml'):
            shutil.copyfile(logdir / 'params' / filename, destination.parent / filename)
        if sha(destination) != sha(final):
            raise ValueError('copied final model differs')
        models[name] = dict(checkpoint=str(destination.relative_to(ROOT)), sha256=sha(destination),
                           source_checkpoint=str(final.relative_to(ROOT)), iteration=249, training_seed=seed,
                           transitions=TRANSITIONS, initial_audit_sha256=sha(initial), schedule_sha256=sha(schedule))
    if len({item['source_checkpoint'] for item in models.values()}) != len(RUNS):
        raise ValueError('fresh final checkpoints must not alias')
    save_json(ART / 'training_validation.json', dict(created_utc=utc(), records=records,
              same_seed_capacity_initial_pairing=True, actual_random_horizons_paired=True,
              historical_references_sha256=sha(ART / 'historical_references.json'),
              full_replay_sha256=sha(ART / 'full_replay.json')))
    save_json(ART / 'trained_models.json', dict(completed_utc=utc(), models=models,
              total_training_transitions=TOTAL_TRANSITIONS, training_freeze_sha256=sha(ART / 'training_frozen.json'),
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
