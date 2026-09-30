"""Serial, immutable high replay and paired low-LR continuation harness."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from week03_ant import lr_study_v24 as study
from week03_ant import seed_study_v22 as historical
from week03_ant import paired_horizon_study_v23 as v23
from week03_ant.lr_study_v24 import (
    ROOT, ART, WORK, TRAIN_SEEDS, RUNS, ENVS, ITERATIONS, RAMP_STEPS, TRANSITIONS,
    TRAIN_GEOMETRY, START, PARENT_SHA, INIT_SOURCE,
    TOTAL_TRANSITIONS, DEVELOPMENT_TRANSITIONS, NEW_TRANSITIONS,
    source_hashes, legacy_hashes, learning_rate, init_path, prepare_checkpoint,
    save_json, sha, utc, verify_hashes, training_parameters, check_initial, read,
)
from week03_ant.posture_study import cache_snapshot
from week03_ant.command_study import validate_learning_log
from run_seed_training_v22 import (checkpoint_finite, validate_positive_replay,
    paired_randomization, validate_evidence, project_main_initial)

PYTHON = str(ROOT.parent / 'run-python')
CACHE = Path('/tmp/isaaclab/terrains')


def run(command, label):
    log = WORK / f'{label}.log'
    outputs = [Path(command[command.index(flag) + 1]) for flag in ('--output', '--audit-output', '--schedule-output') if flag in command]
    if log.exists() or any(path.exists() for path in outputs):
        raise FileExistsError('no retries/overwrites: ' + label)
    started, begin = utc(), time.monotonic()
    with log.open('x') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(label=label, command=command, started_utc=started, finished_utc=utc(),
                  seconds=time.monotonic() - begin, returncode=result.returncode,
                  log=str(log.relative_to(ROOT)), log_sha256=sha(log))
    with (ART / 'training_commands.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(f"[v24] {label}: exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
    if result.returncode or any(not path.is_file() for path in outputs):
        raise RuntimeError('command failed; preserve evidence: ' + label)
    return record


def train_command(seed, initial, schedule, expected=None, *, condition='low', envs=ENVS,
                  iterations=ITERATIONS, device='cuda:1'):
    learning_rate(condition)
    if (type(seed) is not int or seed not in TRAIN_SEEDS or device != 'cuda:1'
            or (envs, iterations) not in ((256, 2), (4096, 2), (4096, 250))
            or (envs == 256 and seed != 51) or (iterations == 250 and condition == 'high' and seed != 51)):
        raise ValueError('undeclared training condition/seed/budget')
    phase = 'preflight' if envs == 256 else 'capacity' if iterations == 2 else 'main'
    command = [PYTHON, 'scripts/lr_continuation_v24.py', 'train', '--arm', 'no_cost',
        '--lr-condition', condition, '--training-seed', str(seed), '--audit-output', str(initial),
        '--schedule-output', str(schedule), '--ramp-steps', str(32 if envs == 256 else RAMP_STEPS),
        '--headless', '--device', device, '--num_envs', str(envs), '--max_iterations', str(iterations),
        '--seed', str(seed), '--run_name', f'v24_{phase}_{condition}{seed}', '--resume',
        '--load_run', f'v24_{condition}_init', '--checkpoint', 'model_0.pt',
        f'env.scene.terrain.terrain_generator.seed={TRAIN_GEOMETRY}', f'agent.device={device}']
    if expected is not None:
        command += ['--expected-initial', str(expected)]
    return command


def assert_cache(expected):
    if cache_snapshot(CACHE, [TRAIN_GEOMETRY]) != expected:
        raise ValueError('training cache changed')


def pin_historical_references():
    legacy_hashes()
    inventory, stages = {}, {}
    def pin(path):
        name = str(path.relative_to(ROOT))
        inventory[name] = sha(path)
        return {'path': name, 'sha256': inventory[name]}
    pin(INIT_SOURCE)
    for directory, names in ((historical.ART, ('frozen.json', 'training_frozen.json', 'training_validation.json',
        'trained_models.json', 'training_cache.json', 'initial_shared.json', 'preflight.json', 'capacity.json')),
        (v23.ART, ('frozen.json', 'development.json', 'final_validation.json'))):
        for name in names:
            pin(directory / name)
    for phase in ('preflight', 'capacity', 'main'):
        records = read(historical.ART / ('training_validation.json' if phase == 'main' else phase + '.json'))['records']
        stages[phase] = {}
        for seed in TRAIN_SEEDS:
            record = records[f'seed{seed}']
            stage = {name: pin(ROOT / record[name]['path']) for name in ('initial', 'schedule', 'rollout')}
            logs = record['learning_log']
            pin(ROOT / logs['text_log'])
            for name in logs['event_files_sha256']:
                pin(ROOT / name)
            initial = read(ROOT / stage['initial']['path'])
            logdir = ROOT / initial['log_dir']
            for name in ('env.yaml', 'agent.yaml'):
                pin(logdir / 'params' / name)
            stage['checkpoint'] = pin(logdir / ('model_249.pt' if phase == 'main' else 'model_1.pt'))
            stages[phase][f'seed{seed}'] = stage
    evaluation_records = read(v23.ART / 'development.json')['records']
    for row in evaluation_records:
        pin(ROOT / row['path'])
    save_json(ART / 'historical_references.json', dict(created_utc=utc(), stages=stages,
        evaluation_records=evaluation_records, v23_frozen_sha256=sha(v23.ART / 'frozen.json'), sha256=inventory))


def verify_references():
    result = study.verify_references()
    if set(result['stages']) != {'preflight', 'capacity', 'main'}:
        raise ValueError('reference stage inventory changed')
    for stages in result['stages'].values():
        if set(stages) != {'seed51', 'seed52', 'seed53'}:
            raise ValueError('reference seed inventory changed')
        for stage in stages.values():
            if set(stage) != {'initial', 'schedule', 'rollout', 'checkpoint'}:
                raise ValueError('reference evidence inventory changed')
            for item in stage.values():
                if result['sha256'].get(item['path']) != item['sha256']:
                    raise ValueError('reference linkage changed')
    if result['evaluation_records'] != read(v23.ART / 'development.json')['records']:
        raise ValueError('evaluation references changed')
    return result


def prepare():
    if (ART / 'initial_shared.json').exists():
        raise FileExistsError('initialization already prepared')
    pin_historical_references()
    records = {condition: prepare_checkpoint(init_path(condition), condition) for condition in ('high', 'low')}
    save_json(ART / 'initial_shared.json', dict(created_utc=utc(), initializers=records))


def prepare_cache(device):
    verify_references()
    for row in read(ART / 'initial_shared.json')['initializers'].values():
        verify_hashes({row['checkpoint']: row['sha256']})
    output = WORK / 'prepare_training_geometry110.json'
    run([PYTHON, 'scripts/evaluate_contact_v16.py', '--headless', '--device', device,
        '--controller', 'history_original', '--geometry', '110', '--seed', '51', '--phase', 'prepare',
        '--scenario', 'mixed', '--seconds', '16', '--num_envs', '35', '--output', str(output),
        '--checkpoint', str(START)], 'prepare_training_geometry110')
    cache = cache_snapshot(CACHE, [TRAIN_GEOMETRY])
    if read(output).get('scored_episodes') != 0 or cache != read(historical.ART / 'training_cache.json')['cache']:
        raise ValueError('cache preparation changed historical terrain or scored')
    save_json(ART / 'training_cache.json', dict(created_utc=utc(), preparation=str(output.relative_to(ROOT)),
              preparation_sha256=sha(output), cache=cache))


def evidence_paths(initial, schedule):
    return dict(initial=initial, schedule=schedule, rollout=initial.with_name(initial.stem + '_contact_rollout.json'),
                lr=initial.with_name(initial.stem + '_lr.json'))


def validate_lr(proof, condition, iterations):
    rate = learning_rate(condition)
    if (proof['passed'] is not True or proof['lr_condition'] != condition or proof['learning_rate'] != rate
            or proof['expected_updates'] != iterations or proof['update_count'] != iterations
            or proof['adam_step_count'] != iterations * 20 or proof['teacher_unchanged'] is not True
            or len(proof['updates']) != iterations or len(proof['adam_steps']) != iterations * 20
            or proof['initial_optimizer_groups'] != proof['final_optimizer_groups']):
        raise ValueError('LR proof incomplete or changed')
    expected = proof['initial']
    names, trainable = expected['parameter_names'], expected['trainable_parameter_names']
    groups = proof['initial_optimizer_groups']
    identifiers = [index for group in groups for index in group['params']]
    if (not names or len(set(names)) != len(names) or not set(trainable) <= set(names)
            or identifiers != list(range(len(names))) or len(groups) != 1
            or any(group['lr'] != rate for group in groups)
            or 'std' not in trainable or not all(any(name.startswith(prefix) for name in trainable)
                                               for prefix in ('actor.', 'critic.'))):
        raise ValueError('LR parameter coverage/options proof differs')
    if (expected['saved_learning_rate'] != rate or expected['algorithm_learning_rate'] != rate
            or not expected['group_learning_rates'] or any(v != rate for v in expected['group_learning_rates'])
            or expected['schedule'] != 'fixed' or expected['desired_kl'] is not None
            or proof['final'] != expected):
        raise ValueError('LR observation differs')
    for kind, records in (('updates', proof['updates']), ('adam_steps', proof['adam_steps'])):
        for index, row in enumerate(records):
            if row['index'] != index or row['before'] != expected or row['after'] != expected:
                raise ValueError('LR per-event observation differs')
            if kind == 'updates' and (row['step_start'], row['step_end']) != (index * 20, (index + 1) * 20):
                raise ValueError('PPO/Adam grouping differs')
            if kind == 'adam_steps' and row['update_index'] != index // 20:
                raise ValueError('Adam update assignment differs')


def inspect_run(paths, seed, envs, iterations, ramp, log, condition):
    data = {name: read(path) for name, path in paths.items()}
    validate_evidence(data['initial'], data['schedule'], data['rollout'], seed, envs, iterations, ramp)
    if data['initial']['lr_condition'] != condition or data['initial']['learning_rate'] != learning_rate(condition):
        raise ValueError('initial LR identity differs')
    validate_lr(data['lr'], condition, iterations)
    logdir = ROOT / data['initial']['log_dir']
    saved = training_parameters(logdir, seed, ramp, condition)
    if saved['normalized'] != data['initial']['normalized_parameters'] or saved['sha256'] != data['initial']['saved_parameter_sha256']:
        raise ValueError('saved YAML changed')
    learning = validate_learning_log(logdir, log, iterations, envs * iterations * 32)
    record = {name: dict(path=str(path.relative_to(ROOT)), sha256=sha(path)) for name, path in paths.items()}
    record.update(learning_log=learning, coefficient_step_mass=data['schedule']['coefficient_sum'],
                  realized_penalty_sum=data['schedule']['realized_penalty_sum'])
    return data, record


def paired_initial(high_initial, low_initial, seed):
    """Same-seed pre-learning equality; LR is the only comparison projection."""
    from run_seed_training_v22 import validate_seed
    validate_seed(high_initial, seed)
    validate_seed(low_initial, seed)
    if (high_initial['lr_condition'], low_initial['lr_condition']) != ('high', 'low'):
        raise ValueError('paired arm identity differs')
    a, b = copy.deepcopy(high_initial), copy.deepcopy(low_initial)
    for value, expected in ((a, 1.e-4), (b, 1.e-5)):
        if float(value['normalized_parameters']['agent']['algorithm']['learning_rate']) != expected or value['learning_rate'] != expected:
            raise ValueError('paired LR differs')
    b['normalized_parameters']['agent']['algorithm']['learning_rate'] = a['normalized_parameters']['agent']['algorithm']['learning_rate']
    check_initial(a, b)
    for key in ('initial_policy_parity', 'optimizer_state_empty', 'iteration', 'common_step_counter', 'contact_slip_weight', 'actual_seed'):
        if a[key] != b[key]:
            raise ValueError('paired pre-learning field differs: ' + key)


def record_data(record, seed, envs, iterations, ramp, condition):
    for key in ('initial', 'schedule', 'rollout', 'lr'):
        verify_hashes({record[key]['path']: record[key]['sha256']})
    logs = record['learning_log']
    verify_hashes({logs['text_log']: logs['text_log_sha256'], **logs['event_files_sha256']})
    paths = {key: ROOT / record[key]['path'] for key in ('initial', 'schedule', 'rollout', 'lr')}
    data, recomputed = inspect_run(paths, seed, envs, iterations, ramp, ROOT / logs['text_log'], condition)
    if recomputed != record:
        raise ValueError('raw evidence summary differs')
    return data


def paired_development(device, phase):
    if phase not in ('preflight', 'capacity'):
        raise ValueError('unknown phase')
    if (ART / 'training_frozen.json').exists() or (ART / f'{phase}.json').exists():
        raise FileExistsError('development already exists or frozen')
    references = verify_references()['stages'][phase]
    before, legacy = source_hashes(), legacy_hashes()
    cache = read(ART / 'training_cache.json')['cache']
    assert_cache(cache)
    if phase == 'capacity':
        verify_development(read(ART / 'preflight.json'))
    envs, ramp = (256, 32) if phase == 'preflight' else (4096, 4000)
    records, replays = {}, {}
    for seed in ((51,) if phase == 'preflight' else TRAIN_SEEDS):
        pair = {}
        for condition in ('high', 'low'):
            verify_references()
            label = f'{phase}_{condition}{seed}'
            initial, schedule = WORK / f'{label}_initial.json', WORK / f'{label}_schedule.json'
            reference = references[f'seed{seed}']
            expected = ROOT / reference['initial']['path'] if condition == 'high' else None
            run(train_command(seed, initial, schedule, expected, condition=condition, envs=envs, iterations=2, device=device), label)
            assert_cache(cache)
            data, records[f'{condition}{seed}'] = inspect_run(evidence_paths(initial, schedule), seed, envs, 2, ramp, WORK / f'{label}.log', condition)
            pair[condition] = data
            if condition == 'high':
                check_initial(data['initial'], read(expected))
                paired_randomization(data['schedule'], read(ROOT / reference['schedule']['path']))
                replays[f'high{seed}'] = validate_positive_replay(data, reference)
        paired_initial(pair['high']['initial'], pair['low']['initial'], seed)
        paired_randomization(pair['high']['schedule'], pair['low']['schedule'])
    if source_hashes() != before or legacy_hashes() != legacy:
        raise ValueError('source changed during development')
    save_json(ART / f'{phase}.json', dict(created_utc=utc(), phase=phase, passed=True,
        source_sha256=before, legacy=legacy, terrain_cache=cache, records=records,
        positive_control_exact_replay=replays, same_seed_prelearning_pairing=True,
        historical_references_sha256=sha(ART / 'historical_references.json')))


def verify_development(proof):
    phase = proof['phase']
    if phase not in ('preflight', 'capacity'):
        raise ValueError('unknown development phase')
    seeds = (51,) if phase == 'preflight' else TRAIN_SEEDS
    if (proof['passed'] is not True or proof['source_sha256'] != source_hashes() or proof['legacy'] != legacy_hashes()
            or set(proof['records']) != {f'{arm}{seed}' for seed in seeds for arm in ('high', 'low')}
            or proof['historical_references_sha256'] != sha(ART / 'historical_references.json')
            or proof['same_seed_prelearning_pairing'] is not True):
        raise ValueError('development identity differs')
    references = verify_references()['stages'][phase]
    envs, ramp = (256, 32) if phase == 'preflight' else (4096, 4000)
    for seed in seeds:
        pair = {arm: record_data(proof['records'][f'{arm}{seed}'], seed, envs, 2, ramp, arm) for arm in ('high', 'low')}
        paired_initial(pair['high']['initial'], pair['low']['initial'], seed)
        paired_randomization(pair['high']['schedule'], pair['low']['schedule'])
        reference = references[f'seed{seed}']
        check_initial(pair['high']['initial'], read(ROOT / reference['initial']['path']))
        paired_randomization(pair['high']['schedule'], read(ROOT / reference['schedule']['path']))
        if validate_positive_replay(pair['high'], reference) != proof['positive_control_exact_replay'][f'high{seed}']:
            raise ValueError('development replay differs')


def expected_main_path(name):
    if name not in ('high51', *RUNS):
        raise ValueError('undeclared main run')
    return ART / 'expected_main' / f'{name}_initial.json'


def frozen_budget():
    return dict(seeds=list(TRAIN_SEEDS), runs=list(RUNS), geometry=110, envs=4096,
        iterations_per_run=250, steps_per_env=32, ramp_steps=4000, transitions_per_run=TRANSITIONS,
        total_training_transitions=TOTAL_TRANSITIONS, diagnostic_replay_transitions=TRANSITIONS,
        development_training_transitions=DEVELOPMENT_TRANSITIONS, new_training_transitions=NEW_TRANSITIONS,
        selection='fresh low final model249 only; historical high controls unchanged')


def freeze_train():
    proofs = [read(ART / f'{phase}.json') for phase in ('preflight', 'capacity')]
    for proof in proofs:
        verify_development(proof)
    if proofs[0]['terrain_cache'] != proofs[1]['terrain_cache']:
        raise ValueError('development cache differs')
    assert_cache(proofs[1]['terrain_cache'])
    expected = {}
    for name in ('high51', *RUNS):
        seed = int(name[-2:])
        raw = read(ROOT / proofs[1]['records'][name]['initial']['path'])
        projected = project_main_initial(raw, seed)
        if name == 'high51':
            check_initial(projected, read(ROOT / verify_references()['stages']['main']['seed51']['initial']['path']))
        path = expected_main_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, projected)
        expected[str(path.relative_to(ROOT))] = sha(path)
    initializers = read(ART / 'initial_shared.json')['initializers']
    verify_hashes({row['checkpoint']: row['sha256'] for row in initializers.values()})
    links = {name.removesuffix('.json') + '_sha256': sha(ART / name) for name in
             ('preflight.json', 'capacity.json', 'initial_shared.json', 'training_cache.json', 'historical_references.json')}
    save_json(ART / 'training_frozen.json', dict(frozen_at=utc(), source_sha256=source_hashes(training=True),
        all_source_sha256=source_hashes(), legacy=legacy_hashes(), **frozen_budget(), **links,
        source_checkpoint_sha256=PARENT_SHA, initializers=initializers, expected_main_sha256=expected,
        terrain_cache=proofs[1]['terrain_cache']))


def verify_training_freeze():
    frozen = read(ART / 'training_frozen.json')
    if (frozen['source_sha256'] != source_hashes(training=True) or frozen['all_source_sha256'] != source_hashes()
            or frozen['legacy'] != legacy_hashes() or frozen['source_checkpoint_sha256'] != PARENT_SHA
            or any(frozen.get(key) != value for key, value in frozen_budget().items())):
        raise ValueError('training freeze changed')
    for name in ('preflight', 'capacity', 'initial_shared', 'training_cache', 'historical_references'):
        if frozen[name + '_sha256'] != sha(ART / (name + '.json')):
            raise ValueError('frozen evidence changed: ' + name)
    verify_hashes({row['checkpoint']: row['sha256'] for row in frozen['initializers'].values()})
    for phase in ('preflight', 'capacity'):
        verify_development(read(ART / f'{phase}.json'))
    if set(frozen['expected_main_sha256']) != {str(expected_main_path(name).relative_to(ROOT)) for name in ('high51', *RUNS)}:
        raise ValueError('main expectations inventory differs')
    verify_hashes(frozen['expected_main_sha256'])
    capacity = read(ART / 'capacity.json')
    for name in ('high51', *RUNS):
        raw = read(ROOT / capacity['records'][name]['initial']['path'])
        if read(expected_main_path(name)) != project_main_initial(raw, int(name[-2:])):
            raise ValueError('main projection changed')
    assert_cache(frozen['terrain_cache'])
    return frozen


def verify_full_replay():
    proof = read(ART / 'full_replay.json')
    if (proof['passed'] is not True or proof['seed'] != 51 or proof['lr_condition'] != 'high'
            or proof['training_freeze_sha256'] != sha(ART / 'training_frozen.json')
            or proof['historical_references_sha256'] != sha(ART / 'historical_references.json')):
        raise ValueError('full replay identity changed')
    data = record_data(proof['record'], 51, 4096, 250, 4000, 'high')
    reference = verify_references()['stages']['main']['seed51']
    check_initial(data['initial'], read(ROOT / reference['initial']['path']))
    paired_randomization(data['schedule'], read(ROOT / reference['schedule']['path']))
    if validate_positive_replay(data, reference, 249) != proof['replay']:
        raise ValueError('full replay differs')
    return proof


def train(device):
    if (ART / 'trained_models.json').exists():
        raise FileExistsError('training already complete')
    models, records = {}, {}
    for name in ('high51', *RUNS):
        seed, condition = int(name[-2:]), 'high' if name == 'high51' else 'low'
        verify_training_freeze()
        if condition == 'low':
            verify_full_replay()
        initial, schedule = ART / 'training' / f'{name}_initial.json', ART / 'training' / f'{name}_schedule.json'
        initial.parent.mkdir(parents=True, exist_ok=True)
        run(train_command(seed, initial, schedule, expected_main_path(name), condition=condition, device=device), f'train_{name}')
        verify_training_freeze()
        data, record = inspect_run(evidence_paths(initial, schedule), seed, 4096, 250, 4000, WORK / f'train_{name}.log', condition)
        check_initial(data['initial'], read(expected_main_path(name)))
        paired_randomization(data['schedule'], read(ROOT / read(ART / 'capacity.json')['records'][name]['schedule']['path']))
        if condition == 'high':
            reference = verify_references()['stages']['main']['seed51']
            check_initial(data['initial'], read(ROOT / reference['initial']['path']))
            paired_randomization(data['schedule'], read(ROOT / reference['schedule']['path']))
            replay = validate_positive_replay(data, reference, 249)
            save_json(ART / 'full_replay.json', dict(created_utc=utc(), passed=True, seed=51, lr_condition='high',
                record=record, replay=replay, training_freeze_sha256=sha(ART / 'training_frozen.json'),
                historical_references_sha256=sha(ART / 'historical_references.json')))
            verify_full_replay()
            continue
        records[name] = record
        logdir = ROOT / data['initial']['log_dir']
        final = logdir / 'model_249.pt'
        if (final.is_symlink() or not final.resolve().is_relative_to((ROOT / 'logs/rsl_rl' / study.EXPERIMENT).resolve())
                or final.parent.name.endswith('_init') or str(final.relative_to(ROOT)) in {m['source_checkpoint'] for m in models.values()}):
            raise ValueError('fresh checkpoint namespace/alias differs')
        checkpoint = checkpoint_finite(final, 249)
        if any(group['lr'] != 1.e-5 for group in checkpoint['optimizer_state_dict']['param_groups']):
            raise ValueError('final Adam LR differs')
        destination = ART / 'runs' / name / 'model_249.pt'
        destination.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(final, destination)
        for filename in ('env.yaml', 'agent.yaml'):
            shutil.copyfile(logdir / 'params' / filename, destination.parent / filename)
        if sha(destination) != sha(final):
            raise ValueError('checkpoint copy differs')
        models[name] = dict(checkpoint=str(destination.relative_to(ROOT)), sha256=sha(destination),
            source_checkpoint=str(final.relative_to(ROOT)), iteration=249, training_seed=seed,
            transitions=TRANSITIONS, learning_rate=1.e-5, lr_condition='low',
            initial_audit_sha256=sha(initial), schedule_sha256=sha(schedule), lr_audit_sha256=record['lr']['sha256'])
    verify_training_ledger()
    save_json(ART / 'training_validation.json', dict(created_utc=utc(), records=records,
        same_seed_capacity_initial_pairing=True, actual_random_horizons_paired=True,
        historical_references_sha256=sha(ART / 'historical_references.json'), full_replay_sha256=sha(ART / 'full_replay.json')))
    save_json(ART / 'trained_models.json', dict(completed_utc=utc(), models=models,
        total_training_transitions=TOTAL_TRANSITIONS, new_training_transitions=NEW_TRANSITIONS,
        training_freeze_sha256=sha(ART / 'training_frozen.json'), training_validation_sha256=sha(ART / 'training_validation.json'),
        training_commands_sha256=sha(ART / 'training_commands.jsonl')))


def verify_training_ledger():
    records = [json.loads(line) for line in (ART / 'training_commands.jsonl').read_text().splitlines()]
    labels = ['prepare_training_geometry110', 'preflight_high51', 'preflight_low51',
              *[f'capacity_{arm}{seed}' for seed in TRAIN_SEEDS for arm in ('high', 'low')],
              'train_high51', 'train_low51', 'train_low52', 'train_low53']
    if [row['label'] for row in records] != labels:
        raise ValueError('training ledger count/order differs')
    for index, row in enumerate(records):
        if row['returncode'] != 0 or row['started_utc'] > row['finished_utc']:
            raise ValueError('failed or invalid training command')
        verify_hashes({row['log']: row['log_sha256']})
        command = row['command']
        if index:
            label = row['label']
            phase, arm_seed = label.split('_', 1)
            condition = 'high' if arm_seed.startswith('high') else 'low'
            seed = int(arm_seed[-2:])
            initial = Path(command[command.index('--audit-output') + 1])
            schedule = Path(command[command.index('--schedule-output') + 1])
            expected = Path(command[command.index('--expected-initial') + 1]) if '--expected-initial' in command else None
            wanted = train_command(seed, initial, schedule, expected, condition=condition,
                                   envs=256 if phase == 'preflight' else 4096,
                                   iterations=250 if phase == 'train' else 2)
            if command != wanted:
                raise ValueError('training ledger executed configuration differs')
        elif (command[0:2] != [PYTHON, 'scripts/evaluate_contact_v16.py']
              or command[command.index('--phase') + 1] != 'prepare'
              or command[command.index('--geometry') + 1] != '110'):
            raise ValueError('training ledger preparation differs')
        if index and records[index - 1]['finished_utc'] > row['started_utc']:
            raise ValueError('training jobs overlap')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'prepare_cache', 'preflight', 'capacity', 'freeze_train', 'train'))
    parser.add_argument('--device', choices=('cuda:1',), default='cuda:1')
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'outputs/hybrid_v11_20260922/gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'prepare_cache': lambda: prepare_cache(args.device),
         'preflight': lambda: paired_development(args.device, 'preflight'),
         'capacity': lambda: paired_development(args.device, 'capacity'),
         'freeze_train': freeze_train, 'train': lambda: train(args.device)}[args.phase]()


if __name__ == '__main__':
    main()
