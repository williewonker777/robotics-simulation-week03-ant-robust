"""Synthetic mutation tests for the post-experiment, stdlib-only v21 verifier."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_v21', ROOT / 'scripts/audit_continuation_v21_raw.py')
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def raw(controller='parent', scenario='mixed'):
    seconds, n = (16, 175) if scenario == 'mixed' else (64, 10)
    data = {key: [0] * n for key in verifier.FLAT_FIELDS}
    data.update(schema='week03_ant_contact_continuation_v21_v1', task='Week03-Ant-Contact-v16-Eval-v0',
                phase='holdout', controller=controller, geometry_seed=113, reset_seed=73, scenario=scenario, num_envs=n,
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
    return verifier.raw_rows(data, controller=data['controller'], geometry=113, reset=73, scenario=data['scenario'])


def schedule(arm='ramped', *, envs=4096, steps=8000, ramp_steps=4000):
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
    tree = ast.parse((ROOT / 'scripts/audit_continuation_v21_raw.py').read_text())
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


@pytest.mark.parametrize('arm,mass', [('immediate', 8000), ('ramped', 6000), ('no_cost', 0)])
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
        verifier.schedule(data, 'ramped')


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
               '--geometry', '113', '--seed', '73', '--scenario', 'mixed', '--num_envs', '175',
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
    section = dict(groups={name: verifier.counts(value) for name, value in groups.items()},
        comparisons={f'{a}_vs_{b}': verifier.judgment(groups[a], groups[b], primary) for a, b in verifier.PAIRS})
    if not breakdown:
        return section
    section.update(family={}, family_level={}, level={})
    families = sorted({r['family'] for values in groups.values() for r in values})
    levels = sorted({r['level'] for values in groups.values() for r in values})
    for family in families:
        selected = {name: [r for r in value if r['family'] == family] for name, value in groups.items()}
        make = (lambda values: dict(groups={k: verifier.counts(v) for k, v in values.items()}, comparisons={})) if family == 'flat' else (
            lambda values: synthetic_section(values, False, False))
        section['family'][family] = make(selected)
        for level in levels:
            subset = {name: [r for r in value if r['level'] == level] for name, value in selected.items()}
            if all(subset.values()):
                section['family_level'][f'{family}/level{level}'] = make(subset)
    for level in levels:
        subset = {name: [r for r in value if r['level'] == level] for name, value in groups.items()}
        section['level'][str(level)] = synthetic_section(subset, primary, False)
    return section


def test_all_six_gates_and_family_level_report_mutations():
    groups = {name: rows(raw(name)) for name in verifier.CONTROLLERS}
    section = synthetic_section(groups, True)
    verifier.compare_section(groups, section, True)
    assert len(section['comparisons']) == 12
    corrupt = deepcopy(section)
    corrupt['family']['stairs']['comparisons']['ramped_vs_immediate']['passed'] = True
    with pytest.raises(ValueError, match='gates'):
        verifier.compare_section(groups, corrupt, True)
    corrupt = deepcopy(section)
    corrupt['family_level']['waves/level3']['groups']['parent']['six'] -= 1
    with pytest.raises(ValueError, match='aggregate'):
        verifier.compare_section(groups, corrupt, True)
    corrupt = deepcopy(section)
    corrupt['level']['0']['comparisons']['ramped_vs_immediate']['paired_six']['gains'] += 1
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
    data = raw('no_cost')
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
    assert verifier.DEVELOPMENT_TRANSITIONS == 557056
    assert len(verifier.CONTROLLERS) * len(verifier.MAPS) * (175 + 10) == 2960


def historical_fixture(root):
    old_dir, directory = root / 'artifacts/terrain_demo/contact_curriculum_v20', root / 'v21'
    old_dir.mkdir(parents=True)
    directory.mkdir()
    tags = ['Loss/value_function', 'Loss/surrogate', 'Loss/prior_loss', 'Policy/mean_noise_std']
    timing = ['Perf/total_fps', 'Perf/collection time', 'Perf/learning_time',
              'Train/mean_reward/time', 'Train/mean_episode_length/time']
    source = {'source.py': 'a' * 64}
    cache = {'pinned': True}
    pins = {}

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path.relative_to(root)), sha256=verifier.sha(path))

    def run(prefix, arm, envs, ramp, created):
        logdir = root / prefix
        env_record = write(logdir / 'params/env.yaml', {})
        agent_record = write(logdir / 'params/agent.yaml', {})
        initial = dict(arm=arm, created_utc=created, num_envs=envs, iteration=0, common_step_counter=0,
            optimizer_state_empty=True, contact_slip_weight=1., observation_dimensions=[envs, 91], dt=1 / 60,
            initial_policy_parity=dict(actor=True, critic=True, teacher=True), normalized_parameters={},
            initial_state_sha256={k: 'b' * 64 for k in ('root_state', 'joint_pos', 'joint_vel', 'observations')},
            initial_prefix_sha256='c' * 64, policy_state_sha256={'actor': 'd' * 64},
            rng_sha256={'cpu': 'e' * 64, 'cuda': 'f' * 64}, log_dir=prefix,
            saved_parameter_sha256={'env': env_record['sha256'], 'agent': agent_record['sha256']})
        initial_ref = write(logdir / 'initial.json', initial)
        schedule_raw = schedule(arm, envs=envs, steps=64, ramp_steps=ramp)
        schedule_ref = write(logdir / 'schedule.json', schedule_raw)
        rollout_ref = write(logdir / 'rollout.json', dict(arm=arm, passive_post_action=True, rollout={'samples': 64, 'contacts': .5}))
        checkpoint = write(logdir / 'model_1.pt', {'mock_tensor': 1})
        log = logdir / 'learn.log'
        log.write_text(''.join(f'Learning iteration {i}/2\nTotal timesteps: {(i + 1) * envs * 32}\n' for i in range(2)))
        event = write(logdir / 'events.raw', {})
        record = dict(initial=initial_ref, schedule=schedule_ref, rollout=rollout_ref,
            coefficient_step_mass=schedule_raw['coefficient_sum'], realized_penalty_sum=schedule_raw['realized_penalty_sum'],
            learning_log=dict(text_log=str(log.relative_to(root)), text_log_sha256=verifier.sha(log),
                event_files_sha256={event['path']: event['sha256']}, iterations_logged=2, transitions_logged=envs * 64,
                all_scalars_finite=True, scalar_counts={name: 2 for name in tags + timing}))
        return record, checkpoint

    stages, proofs = {}, {}
    for phase, envs, ramp in [('preflight', 256, 32), ('capacity', 4096, 4000)]:
        old, old_ckpt = run('old/' + phase, 'immediate', envs, ramp, '2026-09-28T00:00:00+00:00')
        immediate, new_ckpt = run('new/' + phase + '_immediate', 'immediate', envs, ramp, '2026-09-28T02:00:00+00:00')
        no_cost, _ = run('new/' + phase + '_no_cost', 'no_cost', envs, ramp, '2026-09-28T02:01:00+00:00')
        write(old_dir / (phase + '.json'), {'records': {'immediate': old}})
        stages[phase] = {**{k: old[k] for k in ('initial', 'schedule', 'rollout')}, 'checkpoint': old_ckpt}
        for entry in stages[phase].values():
            pins[entry['path']] = entry['sha256']
        proofs[phase] = dict(phase=phase, passed=True, development_only=True, historical_initial_pairing=True,
            actual_random_horizons_paired=True, source_sha256=source, terrain_cache=cache,
            records=dict(immediate=immediate, no_cost=no_cost), positive_control_exact_replay=dict(
                exact_model_optimizer_infos=True, exact_reward_contact=True, new_checkpoint=new_ckpt['path'],
                new_checkpoint_sha256=new_ckpt['sha256'], historical_checkpoint=old_ckpt,
                scalars=dict(exact_tags=tags, timing_only_exclusions=timing)))
    stages['main'] = deepcopy(stages['capacity'])  # Main references are separately checked by training().
    reference = write(directory / 'historical_references.json', dict(created_utc='2026-09-28T01:00:00+00:00',
        disclosure='newly pinned historical model1', stages=stages, sha256=pins))
    for phase, proof in proofs.items():
        proof['historical_references_sha256'] = reference['sha256']
        write(directory / (phase + '.json'), proof)
    return directory, dict(historical_references_sha256=reference['sha256'], source_sha256=source, terrain_cache=cache)


def test_historical_positive_control_raw_replay_and_corruption(tmp_path):
    directory, frozen = historical_fixture(tmp_path)
    verifier.historical_development(tmp_path, directory, frozen)
    proof_path = directory / 'preflight.json'
    proof = verifier.read(proof_path)
    proof['positive_control_exact_replay']['exact_model_optimizer_infos'] = False
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='checkpoint proof'):
        verifier.historical_development(tmp_path, directory, frozen)


def test_historical_horizon_corruption_is_not_accepted(tmp_path):
    directory, frozen = historical_fixture(tmp_path)
    proof_path = directory / 'capacity.json'
    proof = verifier.read(proof_path)
    record = proof['records']['no_cost']['schedule']
    path = tmp_path / record['path']
    data = verifier.read(path)
    data['episode_randomization']['assigned_sha256'] = data['episode_randomization']['actual_sha256'] = '9' * 64
    path.write_text(json.dumps(data))
    record['sha256'] = verifier.sha(path)
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='randomized horizons'):
        verifier.historical_development(tmp_path, directory, frozen)


def test_pin_after_replay_is_rejected(tmp_path):
    directory, frozen = historical_fixture(tmp_path)
    path = directory / 'historical_references.json'
    reference = verifier.read(path)
    reference['created_utc'] = '2026-09-28T03:00:00+00:00'
    path.write_text(json.dumps(reference))
    frozen['historical_references_sha256'] = verifier.sha(path)
    for phase in ('preflight', 'capacity'):
        target = directory / (phase + '.json')
        proof = verifier.read(target)
        proof['historical_references_sha256'] = frozen['historical_references_sha256']
        target.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='pinned before'):
        verifier.historical_development(tmp_path, directory, frozen)


def test_development_rows_are_35_per_controller_and_not_scored_holdouts():
    data = raw('no_cost')
    data.update(phase='smoke', geometry_seed=51, reset_seed=24, num_envs=35)
    for key, value in list(data.items()):
        if isinstance(value, list) and len(value) == 175:
            data[key] = value[:35]
    data['family_indices'] = [i // 5 for i in range(35)]
    data['level_indices'] = [i % 5 for i in range(35)]
    result = verifier.raw_rows(data, controller='no_cost', geometry=51, reset=24, scenario='mixed', development=True)
    assert len(result) == 35 and len(result) * 8 == 280
    with pytest.raises(ValueError):
        verifier.raw_rows(data, controller='no_cost', geometry=51, reset=24, scenario='mixed')


def test_cli_report_comparison_is_optional_and_output_exclusive(tmp_path, monkeypatch):
    import sys
    observed = []
    def fake_audit(**kwargs):
        observed.append(kwargs['compare_report'])
        return dict(passed=True, raw_files=32, first_episodes=2960, report_comparison=kwargs['compare_report'], verifier_sha256='a' * 64)
    monkeypatch.setattr(verifier, 'audit', fake_audit)
    monkeypatch.setattr(sys, 'argv', ['audit', '--directory', str(tmp_path), '--skip-summary'])
    verifier.main()
    assert observed == [False]
    assert verifier.read(tmp_path / 'independent_raw_audit.json')['report_comparison'] is False
    with pytest.raises(ValueError, match='replace'):
        verifier.main()
    assert observed == [False]


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
    for phase, envs, iterations, ramp, arms in [('preflight', 256, 2, 32, ('immediate', 'no_cost')),
             ('capacity', 4096, 2, 4000, ('immediate', 'no_cost')), ('training_validation', 4096, 250, 4000, ('no_cost',))]:
        records = {}
        for arm in arms:
            initial, sched = output(f'{phase}/{arm}_initial.json'), output(f'{phase}/{arm}_schedule.json')
            log = command(['--audit-output', str(tmp_path / initial['path']), '--schedule-output', str(tmp_path / sched['path']),
                '--arm', arm, '--num_envs', str(envs), '--max_iterations', str(iterations), '--ramp-steps', str(ramp),
                '--seed', '51', '--load_run', 'v21_init', '--checkpoint', 'model_0.pt'])
            records[arm] = {'initial': initial, 'schedule': sched, 'learning_log': {'text_log': log}}
        (directory / (phase + '.json')).write_text(json.dumps({'records': records}))
    ledger = directory / 'commands.jsonl'
    def write(records):
        ledger.write_text(''.join(json.dumps(record) + '\n' for record in records))
    write(commands)
    assert verifier.command_inventory(tmp_path, directory, raw_paths) == 49
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
