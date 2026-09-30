"""Independent frozen-policy v20 raw audit and descriptive, paired comparisons."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from week03_ant import curriculum_study_v20 as study
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
        raise ValueError('invalid v20 schema/task/controller')
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
        raise ValueError('changed v20 policy/observation/training contract')
    model = study.models()[study.model_key(controller)]
    if (data.get('checkpoint') != model['checkpoint'] or data.get('checkpoint_sha256') != model['sha256']
            or data.get('evaluation_plan_sha256') != study.sha(study.PLAN)):
        raise ValueError('changed v20 model/plan linkage')
    if study.model_key(controller) != 'parent' and (model.get('transitions') != study.TRANSITIONS or model.get('iteration') != 249):
        raise ValueError('nonfinal or incomplete trained checkpoint')
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
        raise ValueError('changed v20 family/level inventory')
    return rows


def _comparisons(groups, primary):
    aggregates = {name: aggregate(rows) for name, rows in groups.items()}
    return {f'{candidate}_vs_{reference}': dict(
                **improvement(aggregates[candidate], aggregates[reference], primary=primary),
                paired_six=paired_six(groups[candidate], groups[reference]))
            for candidate, reference in (*study.COMPARISONS, *study.PARENT_COMPARISONS)}


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
    return result


def _read(path):
    return json.loads(Path(path).read_text())


def _inside(name):
    path = (study.ROOT / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(study.ROOT.resolve()):
        raise ValueError('ledger path escapes repository')
    return path


def audit_training(directory=ART):
    """Reopen full-budget schedule, initialization, saved YAML and scalar evidence."""
    from run_curriculum_training_v20 import verify_training_freeze, inspect_run, paired_randomization

    directory = Path(directory)
    verify_training_freeze()
    trained = _read(directory / 'trained_models.json')
    validation = _read(directory / 'training_validation.json')
    if (trained.get('total_training_transitions') != 2 * study.TRANSITIONS
            or trained.get('training_freeze_sha256') != study.sha(directory / 'training_frozen.json')
            or trained.get('training_validation_sha256') != study.sha(directory / 'training_validation.json')
            or set(trained.get('models', {})) != set(study.ARMS)
            or set(validation.get('records', {})) != set(study.ARMS)
            or validation.get('paired_initialization') is not True
            or validation.get('actual_random_horizons_paired') is not True):
        raise ValueError('training manifests/budget/pairing differ')
    first, results = None, {}
    for arm in study.ARMS:
        record, model = validation['records'][arm], trained['models'][arm]
        paths = {name: _inside(record[name]['path']) for name in ('initial', 'schedule', 'rollout')}
        if any(study.sha(path) != record[name]['sha256'] for name, path in paths.items()):
            raise ValueError('training raw evidence changed')
        if (model.get('iteration') != 249 or model.get('transitions') != study.TRANSITIONS
                or model.get('initial_audit_sha256') != study.sha(paths['initial'])
                or model.get('schedule_sha256') != study.sha(paths['schedule'])
                or study.sha(_inside(model['checkpoint'])) != model['sha256']
                or study.sha(_inside(model['source_checkpoint'])) != model['sha256']):
            raise ValueError('trained checkpoint/full-budget linkage changed')
        data, rebuilt = inspect_run(paths, arm, study.ENVS, study.ITERATIONS, study.RAMP_STEPS,
                                    _inside(record['learning_log']['text_log']))
        if rebuilt != record:
            raise ValueError('training scalar/schedule audit changed')
        if first is not None:
            study.check_initial(data['initial'], first['initial'])
            paired_randomization(data['schedule'], first['schedule'])
        first = first or data
        results[arm] = dict(coefficient_step_mass=record['coefficient_step_mass'],
                            realized_penalty_sum=record['realized_penalty_sum'])
    return results


def summarize(directory=ART):
    directory = Path(directory)
    frozen = study.verify_frozen()
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
                       'training_validation_sha256', 'trained_models_sha256', 'development_sha256')})
    raw_provenance = {key: provenance[key] for key in ('experiment_freeze_sha256',
                      'evaluation_inputs_sha256', 'terrain_cache_manifest_sha256')}
    result = dict(files=[], provenance=provenance, new_training_transitions=2 * study.TRANSITIONS,
                  evaluation_training_transitions=0, training=training)
    for scenario, folder in (('mixed', 'evaluations'), ('stones', 'horizon')):
        expected = {f'{c}__geometry{g}_reset{r}.json' for c in study.CONTROLLERS for g, r in study.HOLDOUTS}
        if {p.name for p in (directory / folder).glob('*.json')} != expected:
            raise ValueError('missing or unexpected v20 raw file matrix')
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
    if len(result['files']) != 24 or sum(f['episodes'] for f in result['files']) != 2220:
        raise ValueError('incomplete v20 24-file/2220-episode audit')
    result['provenance']['ledger'] = audit_ledger(directory, result['files'], inputs)
    result['interpretation'] = (
        'Descriptive single-seed paired reward-curriculum experiment on two maps, not statistical superiority. '
        'The arms differ in coefficient timing AND coefficient-step mass; realized penalty depends on visited states. '
        'Both arms preserve the same raw contact reward, parent, sensors and inference. '
        'Comparisons against the frozen parent are required: beating a degraded immediate arm is insufficient. '
        'The separate 64s secondary cannot override the 16s primary. No automatic promotion or physical-robot safety claim.')
    return result


def markdown(result):
    lines = ['# v20 contact-cost curriculum: immediate versus ramped', '', result['interpretation'], '']
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
            lines.append('')
    lines += ['See summary.json for family/level, contact/posture telemetry, paired outcomes and raw-file provenance.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=ART)
    print(markdown(summarize(parser.parse_args().directory)))
