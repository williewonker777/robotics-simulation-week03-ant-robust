"""Independent frozen-policy v22 raw audit and descriptive, paired comparisons."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from week03_ant import seed_study_v22 as study
from week03_ant.contact_telemetry import audit_contact
from summarize_contact_v16 import (
    CONTACT_CONFIG, FAMILIES, FLAT_IDENTITY_FIELDS, HISTORY, POSTURE,
    _flat_identity, _paired,
)

from week03_ant.posture_study import cache_snapshot
from summarize_rebaseline_v19 import _finite, aggregate, improvement, paired_six, audit_ledger

ART = study.ART
PROJECT = {name: 'history_control' if name.startswith('history_') else 'control'
           for name in study.CONTROLLERS}


def audit(data):
    """Audit original metadata, then project a COPY to frozen physical auditors."""
    _finite(data)
    controller = data.get('controller')
    if (data.get('schema') != study.SCHEMA or data.get('task') != study.TASK
            or controller not in study.CONTROLLERS):
        raise ValueError('invalid v22 schema/task/controller')
    condition = data.get('condition', {})
    study.validate_request(controller, data.get('geometry_seed'), data.get('reset_seed'),
                           data.get('scenario'), condition.get('seconds'),
                           data.get('num_envs'), data.get('phase'))
    if data.get('phase') in ('prepare', 'prepare_development'):
        raise ValueError('preparation is not scored evidence')
    expected = dict(observations=91, expert_observations=91,
                    v5_observations=60, actions=8)
    if (any(type(condition.get(k)) is not int or condition[k] != v for k, v in expected.items())
            or data.get('command_mode') != study.command_mode(controller)
            or type(data.get('command_schema_version')) is not int
            or data['command_schema_version'] != 1
            or data.get('mode') != study.policy_mode(controller)
            or data.get('contact_config') != CONTACT_CONFIG
            or data.get('default_config_parity') is not True
            or type(data.get('new_training_transitions')) is not int
            or data['new_training_transitions'] != 0):
        raise ValueError('changed v22 policy/observation/training contract')
    model = study.models()[study.model_key(controller)]
    if (data.get('checkpoint') != model['checkpoint'] or data.get('checkpoint_sha256') != model['sha256']
            or data.get('evaluation_plan_sha256') != study.sha(study.PLAN)):
        raise ValueError('changed v22 model/plan linkage')
    if study.model_key(controller) != 'parent' and (model.get('transitions') != study.TRANSITIONS or model.get('iteration') != 249
            or model.get('training_seed') != study.training_seed(study.model_key(controller))):
        raise ValueError('nonfinal or incomplete trained checkpoint')
    expected_seed = None if study.model_key(controller) == 'parent' else study.training_seed(study.model_key(controller))
    if data.get('training_seed') != expected_seed or (expected_seed is not None and type(data['training_seed']) is not int):
        raise ValueError('raw training seed identity differs')
    expected_transitions = 0 if study.model_key(controller) == 'parent' else study.TRANSITIONS
    if (type(data.get('policy_training_transitions')) is not int
            or data['policy_training_transitions'] != expected_transitions):
        raise ValueError('evaluation/model training accounting differs')
    HISTORY._hashes(data.get('initial_state_sha256'), HISTORY.INITIAL_KEYS)
    HISTORY._hashes({'prefix': data.get('initial_prefix_sha256')})
    HISTORY._hashes(data.get('initial_rng_sha256'), ('cpu', 'cuda'))
    if data['phase'] == 'holdout':
        HISTORY._hashes({key: data.get(key) for key in (
            'experiment_freeze_sha256', 'evaluation_inputs_sha256', 'terrain_cache_manifest_sha256')})
    projected = deepcopy(data)
    projected.update(schema=POSTURE.SCHEMA, task=POSTURE.TASK,
                     controller=PROJECT[controller], phase='smoke')
    projected['condition']['observations'] = 88
    rows = POSTURE.audit(projected)
    audit_contact(data, rows)
    expected_counts = ({(family, level): data['num_envs'] // 35
                        for family in FAMILIES for level in range(5)}
                       if data['scenario'] == 'mixed' else {('stepping_stones', 4): 10})
    if Counter((row['family'], row['level']) for row in rows) != expected_counts:
        raise ValueError('changed v22 family/level inventory')
    return rows


def _comparisons(groups, primary):
    aggregates = {name: aggregate(rows) for name, rows in groups.items()}
    return {f'{candidate}_vs_{reference}': dict(
                **improvement(aggregates[candidate], aggregates[reference], primary=primary),
                paired_six=paired_six(groups[candidate], groups[reference]))
            for candidate, reference in study.COMPARISONS}


def seed_ranges(aggregates):
    """Three per-seed observations, not a pooled pseudo-replicated parent trial."""
    result = {}
    metrics = ('one', 'six', 'falls', 'lane', 'world', 'flat_falls', 'flat_lane',
               'flat_world', 'flat_mean_episode_speed')
    for prefix in ('', 'history_'):
        parent = aggregates[prefix + 'parent']
        seeds = {run: aggregates[prefix + run] for run in study.RUNS}
        if any((row['n'], row['flat_n']) != (parent['n'], parent['flat_n']) for row in seeds.values()):
            raise ValueError('unequal per-seed denominators')
        values = {}
        for metric in metrics:
            observations = {run: row[metric] for run, row in seeds.items()}
            present = [value for value in observations.values() if value is not None]
            if present and len(present) != len(study.RUNS):
                raise ValueError('partially missing seed metric')
            values[metric] = dict(values=observations, parent=parent[metric],
                minimum=min(present) if present else None,
                maximum=max(present) if present else None,
                range=max(present) - min(present) if present else None,
                parent_deltas={run: value - parent[metric] if value is not None and parent[metric] is not None else None
                               for run, value in observations.items()})
        result[prefix.rstrip('_') or 'actor'] = dict(training_seeds=list(study.TRAIN_SEEDS),
            rough_episodes_per_seed=parent['n'], flat_episodes_per_seed=parent['flat_n'],
            parent_counted_once=True, metrics=values)
    return result


def _section(groups, primary):
    result = dict(groups={name: aggregate(rows) for name, rows in groups.items()},
                  comparisons=_comparisons(groups, primary))
    families = sorted({r['family'] for rows in groups.values() for r in rows})
    levels = sorted({r['level'] for rows in groups.values() for r in rows})
    result['family'] = {}
    result['family_level'] = {}
    result['level'] = {}
    for family in families:
        subset = {name: [r for r in rows if r['family'] == family] for name, rows in groups.items()}
        result['family'][family] = dict(groups={name: aggregate(rows) for name, rows in subset.items()},
            comparisons=_comparisons(subset, False) if family != 'flat' else {},
            gate_scope='rough-only descriptive' if family != 'flat' else 'flat safety/speed reported; no rough gate')
        for level in levels:
            selected = {name: [r for r in rows if r['level'] == level] for name, rows in subset.items()}
            if not all(selected.values()):
                continue
            result['family_level'][f'{family}/level{level}'] = dict(
                groups={name: aggregate(rows) for name, rows in selected.items()},
                comparisons=_comparisons(selected, False) if family != 'flat' else {})
    for level in levels:
        subset = {name: [r for r in rows if r['level'] == level] for name, rows in groups.items()}
        result['level'][str(level)] = dict(groups={name: aggregate(rows) for name, rows in subset.items()},
                                          comparisons=_comparisons(subset, primary))
    for section in (result, *result['family'].values(), *result['family_level'].values(), *result['level'].values()):
        section['seed_ranges'] = seed_ranges(section['groups'])
    return result


def _read(path):
    return json.loads(Path(path).read_text())


def _inside(name):
    path = (study.ROOT / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(study.ROOT.resolve()):
        raise ValueError('ledger path escapes repository')
    return path


def audit_same_seed_pairing(directory, run, data, training_frozen, capacity):
    """Reopen hash-pinned same-seed initial and real horizon evidence."""
    expected_path = directory / 'expected_main' / f'{run}_initial.json'
    if training_frozen['expected_main_sha256'].get(str(expected_path.relative_to(study.ROOT))) != study.sha(expected_path):
        raise ValueError('same-seed expected initialization changed')
    study.check_initial(data['initial'], _read(expected_path))
    capacity_schedule = capacity['records'][run]['schedule']
    capacity_path = _inside(capacity_schedule['path'])
    if study.sha(capacity_path) != capacity_schedule['sha256']:
        raise ValueError('same-seed capacity schedule changed')
    if data['schedule']['episode_randomization'] != _read(capacity_path)['episode_randomization']:
        raise ValueError('same-seed main/capacity actual horizons differ')


def audit_fresh_source(model, initial, sources):
    """Require a distinct final checkpoint in this run's fresh v22 log directory."""
    source = _inside(model['source_checkpoint'])
    expected = (_inside(initial['log_dir']) / 'model_249.pt').resolve()
    namespace = (study.ROOT / 'logs' / 'rsl_rl' / study.EXPERIMENT).resolve()
    if source != expected or not source.is_relative_to(namespace) or source in sources:
        raise ValueError('fresh training source checkpoint binding or distinctness differs')
    sources.add(source)


def audit_training(directory=ART):
    """Reopen all three fresh runs; capacity proofs pair only the SAME training seed."""
    from run_seed_training_v22 import verify_training_freeze, inspect_run, verify_references, verify_full_replay

    directory = Path(directory)
    training_frozen = verify_training_freeze()
    capacity = _read(directory / 'capacity.json')
    references = verify_references()
    replay = verify_full_replay()
    trained = _read(directory / 'trained_models.json')
    validation = _read(directory / 'training_validation.json')
    if (trained.get('total_training_transitions') != study.TOTAL_TRANSITIONS
            or trained.get('training_freeze_sha256') != study.sha(directory / 'training_frozen.json')
            or trained.get('training_validation_sha256') != study.sha(directory / 'training_validation.json')
            or set(trained.get('models', {})) != set(study.RUNS)
            or set(validation.get('records', {})) != set(study.RUNS)
            or validation.get('same_seed_capacity_initial_pairing') is not True
            or validation.get('actual_random_horizons_paired') is not True
            or validation.get('historical_references_sha256') != study.sha(directory / 'historical_references.json')
            or validation.get('full_replay_sha256') != study.sha(directory / 'full_replay.json')):
        raise ValueError('seed-study training manifests/budgets/paired proofs differ')
    results = {}
    sources = set()
    for run in study.RUNS:
        seed = study.training_seed(run)
        record, model = validation['records'][run], trained['models'][run]
        paths = {name: _inside(record[name]['path']) for name in ('initial', 'schedule', 'rollout')}
        if any(study.sha(path) != record[name]['sha256'] for name, path in paths.items()):
            raise ValueError('training raw evidence changed')
        if (model.get('iteration') != 249 or model.get('transitions') != study.TRANSITIONS
                or type(model.get('training_seed')) is not int or model['training_seed'] != seed
                or model.get('initial_audit_sha256') != study.sha(paths['initial'])
                or model.get('schedule_sha256') != study.sha(paths['schedule'])
                or study.sha(_inside(model['checkpoint'])) != model['sha256']
                or study.sha(_inside(model['source_checkpoint'])) != model['sha256']
                or not _inside(model['checkpoint']).is_relative_to((directory / 'runs' / run).resolve())):
            raise ValueError('fresh final per-seed checkpoint provenance differs')
        data, rebuilt = inspect_run(paths, seed, study.ENVS, study.ITERATIONS, study.RAMP_STEPS,
                                    _inside(record['learning_log']['text_log']))
        if rebuilt != record:
            raise ValueError('seed training scalar/schedule audit changed')
        audit_fresh_source(model, data['initial'], sources)
        audit_same_seed_pairing(directory, run, data, training_frozen, capacity)
        if data['initial'].get('training_seed') != seed or data['initial'].get('actual_seed') != {
                'requested': seed, 'saved_agent': seed, 'saved_environment': seed, 'live_environment': seed}:
            raise ValueError('requested/saved/live seed proof differs')
        if (data['initial']['normalized_parameters']['agent']['seed'] != str(seed)
                or data['initial']['normalized_parameters']['env']['seed'] != str(seed)
                or record['coefficient_step_mass'] != 0. or record['realized_penalty_sum'] != 0.):
            raise ValueError('seed normalized away or nonzero contact penalty')
        results[run] = dict(training_seed=seed, coefficient_step_mass=0., realized_penalty_sum=0.)
    # Calls are required even though their detailed evidence is represented by hashes.
    if references is None or replay is None:
        raise ValueError('historical/full replay verifier returned no evidence')
    return results


def summarize(directory=ART):
    directory = Path(directory)
    frozen = study.verify_frozen()
    if frozen.get('reused_training_transitions') != 0:
        raise ValueError('frozen reused training accounting differs')
    training = audit_training(directory)
    if study.sha(directory / 'frozen.json') != study.sha(ART / 'frozen.json'):
        raise ValueError('summary directory freeze differs')
    inputs = study.evaluation_inputs(directory)
    provenance = {key: study.sha(directory / filename) for key, filename in (
        ('experiment_freeze_sha256', 'frozen.json'), ('evaluation_inputs_sha256', 'evaluation_inputs.json'),
        ('terrain_cache_manifest_sha256', 'terrain_cache.json'))}
    cache = _read(directory / 'terrain_cache.json')
    if cache['cache'] != cache_snapshot(Path('/tmp/isaaclab/terrains'), [g for g, _ in study.HOLDOUTS]):
        raise ValueError('pinned terrain cache bytes changed')
    for record in cache['preparation']:
        if study.sha(_inside(record['path'])) != record['sha256']:
            raise ValueError('terrain preparation evidence changed')
    # The input helper checks the manifest and pinned pre-scoring ledger prefix.
    provenance.update({key: frozen[key] for key in ('training_freeze_sha256',
                       'training_validation_sha256', 'trained_models_sha256', 'development_sha256', 'full_replay_sha256')})
    raw_provenance = {key: provenance[key] for key in ('experiment_freeze_sha256',
                      'evaluation_inputs_sha256', 'terrain_cache_manifest_sha256')}
    result = dict(files=[], provenance=provenance, new_training_transitions=study.TOTAL_TRANSITIONS, reused_training_transitions=0,
                  evaluation_training_transitions=0, training=training)
    for scenario, folder in (('mixed', 'evaluations'), ('stones', 'horizon')):
        expected = {f'{c}__geometry{g}_reset{r}.json' for c in study.CONTROLLERS for g, r in study.HOLDOUTS}
        if {p.name for p in (directory / folder).glob('*.json')} != expected:
            raise ValueError('missing or unexpected v22 raw file matrix')
        groups, maps, initial, flats = defaultdict(list), defaultdict(dict), {}, {}
        for controller in study.CONTROLLERS:
            for geometry, reset in study.HOLDOUTS:
                path = directory / folder / f'{controller}__geometry{geometry}_reset{reset}.json'
                data = _read(path)
                if ((data.get('controller'), data.get('geometry_seed'), data.get('reset_seed'),
                     data.get('scenario'), data.get('phase')) != (controller, geometry, reset, scenario, 'holdout')
                        or any(data.get(key) != value for key, value in raw_provenance.items())):
                    raise ValueError('raw condition/input linkage differs')
                if not HISTORY._timestamp(inputs['created_utc']) < HISTORY._timestamp(data['started_utc']) <= HISTORY._timestamp(data['finished_utc']):
                    raise ValueError('score chronology precedes frozen inputs')
                rows = audit(data)
                _paired(initial, (geometry, reset), data)
                if scenario == 'mixed' and controller.startswith('history_'):
                    _flat_identity(flats, (geometry, reset), data, rows)
                groups[controller].extend(rows)
                maps[f'geometry{geometry}_reset{reset}'][controller] = rows
                result['files'].append(dict(path=str(path.relative_to(directory)), sha256=study.sha(path), episodes=len(rows)))
        section = _section(groups, scenario == 'mixed')
        section['per_map'] = {name: _section(rows, scenario == 'mixed') for name, rows in maps.items()}
        if scenario == 'mixed':
            section['history_flat_identity'] = dict(verified=True, maps=len(flats), fields=list(FLAT_IDENTITY_FIELDS))
        result[folder] = section
    if len(result['files']) != 32 or sum(f['episodes'] for f in result['files']) != 2960:
        raise ValueError('incomplete v22 32-file/2960-episode audit')
    result['provenance']['ledger'] = audit_ledger(directory, result['files'], inputs)
    result['interpretation'] = (
        'Three fresh continuation training seeds, conditional on one fixed parent and training terrain; '
        'no new network initialization or reward tuning. Fresh seed51 exactly replays historical v21 '
        'recorded evidence and endpoint; it is not an independent fourth realization. Parent is evaluated '
        'once per map/horizon and is never pooled three times. Per-seed extrema/ranges are descriptive, '
        'not significance, confidence intervals or general algorithm robustness from three seeds. '
        'All per-map failures remain visible; 64s secondary cannot override 16s primary. No automatic promotion.')
    return result


def markdown(result):
    lines = ['# v22 no-cost continuation: training-seed sensitivity', '', result['interpretation'], '']
    for folder, title in (('evaluations', 'Primary: 16s mixed'), ('horizon', 'Secondary: 64s stones')):
        lines += [f'## {title}', '']
        section = result[folder]
        for label, group in [('Pooled', section), *section['per_map'].items()]:
            lines += [f'### {label}', '', '| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|']
            for controller in study.CONTROLLERS:
                g = group['groups'][controller]
                speed = g['flat_mean_episode_speed']
                lines.append(f"| {controller} | {g['one']}/{g['n']} | {g['six']}/{g['n']} | {g['falls']} | {g['lane']} | {g['world'] + g['flat_world']} | {g['flat_falls']}/{g['flat_lane']} | {'—' if speed is None else f'{speed:.4f}'} |")
            for name, gate in group['comparisons'].items():
                pairs = gate['paired_six']
                failures = ', '.join(k for k, passed in gate['checks'].items() if not passed) or 'none'
                lines += ['', f"{name}: **{'PASS' if gate['passed'] else 'FAIL'}**; paired six gains/losses {pairs['gains']}/{pairs['losses']}; failed checks: {failures}."]
            for policy, ranges in group['seed_ranges'].items():
                for metric in ('one', 'six', 'falls', 'lane', 'flat_mean_episode_speed'):
                    value = ranges['metrics'][metric]
                    lines.append(f"{policy} {metric}: three-seed min/max {value['minimum']}/{value['maximum']}, range {value['range']}; parent {value['parent']} (counted once).")
            lines.append('')
    lines += ['See summary.json for family/level, contact/posture telemetry, paired outcomes and raw-file provenance.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=ART)
    print(markdown(summarize(parser.parse_args().directory)))
