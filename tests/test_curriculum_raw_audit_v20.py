"""Synthetic mutation tests for the post-experiment, stdlib-only v20 verifier."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_v20', ROOT / 'scripts/audit_curriculum_v20_raw.py')
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def raw(controller='parent', scenario='mixed'):
    seconds, n = (16, 175) if scenario == 'mixed' else (64, 10)
    data = {key: [0] * n for key in verifier.FLAT_FIELDS}
    data.update(schema='week03_ant_contact_curriculum_v20_v1', task='Week03-Ant-Contact-v16-Eval-v0',
                phase='holdout', controller=controller, geometry_seed=111, reset_seed=71, scenario=scenario, num_envs=n,
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
    return verifier.raw_rows(data, controller=data['controller'], geometry=111, reset=71, scenario=data['scenario'])


def schedule(arm='ramped'):
    factors = [1. if arm == 'immediate' else min(i / 3999, 1.) for i in range(8000)]
    scaled = [-.5 * factor for factor in factors]
    totals = [v * 4096 / 60 for v in scaled]
    return dict(arm=arm, mode=arm, num_envs=4096, ramp_steps=4000, policy_steps=8000,
                common_step_counters=list(range(1, 8001)), coefficients=factors,
                raw_reward_mean=[-.5] * 8000, scaled_reward_mean=scaled, valid_rows=[4096] * 8000,
                step_dt=[1 / 60] * 8000, raw_reward_dt_sum=[-.5 * 4096 / 60] * 8000,
                scaled_reward_dt_sum=totals, coefficient_sum=sum(factors), realized_penalty_sum=-sum(totals),
                episode_randomization=dict(matches_raw_environment=True, assigned_sha256='e' * 64,
                    actual_sha256='e' * 64, nonzero=4090, minimum=0, maximum=959))


def test_standalone_imports_only_standard_library():
    tree = ast.parse((ROOT / 'scripts/audit_curriculum_v20_raw.py').read_text())
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


@pytest.mark.parametrize('arm,mass', [('immediate', 8000), ('ramped', 6000)])
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
               '--geometry', '111', '--seed', '71', '--scenario', 'mixed', '--num_envs', '175',
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
    assert len(section['comparisons']) == 6
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
