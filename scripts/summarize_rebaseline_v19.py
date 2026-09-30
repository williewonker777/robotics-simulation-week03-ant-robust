"""Independent frozen-policy v19 raw audit and descriptive, paired comparisons."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

from week03_ant import rebaseline_study_v19 as study
from week03_ant.contact_telemetry import audit_contact
from summarize_contact_v16 import (
    CONTACT_CONFIG, FAMILIES, FLAT_IDENTITY_FIELDS, HISTORY, POSTURE,
    _flat_identity, _paired, aggregate as _aggregate,
)

ART = study.ART
PROJECT = dict(v5='v5', history_original='history_original',
               v16_control='control', history_control='history_control')


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite raw evidence')
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def audit(data):
    """Audit original metadata, then project a COPY to frozen physical auditors."""
    _finite(data)
    controller = data.get('controller')
    if (data.get('schema') != study.SCHEMA or data.get('task') != study.TASK
            or controller not in study.CONTROLLERS):
        raise ValueError('invalid v19 schema/task/controller')
    condition = data.get('condition', {})
    study.validate_request(controller, data.get('geometry_seed'), data.get('reset_seed'),
                           data.get('scenario'), condition.get('seconds'),
                           data.get('num_envs'), data.get('phase'))
    if data.get('phase') in ('prepare', 'prepare_development'):
        raise ValueError('preparation is not scored evidence')
    expected = dict(observations=91, expert_observations=88 if study.model_key(controller) == 'original' else 91,
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
        raise ValueError('changed v19 policy/observation/training contract')
    model = study.models()[study.model_key(controller)]
    if (data.get('checkpoint') != model['checkpoint'] or data.get('checkpoint_sha256') != model['sha256']
            or data.get('evaluation_plan_sha256') != study.sha(study.PLAN)):
        raise ValueError('changed v19 model/plan linkage')
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
        raise ValueError('changed v19 family/level inventory')
    return rows


def aggregate(rows):
    result = _aggregate(rows)
    result["completed_first_episodes"] = len(rows)
    return result


def improvement(candidate, reference, *, primary=True):
    if candidate['n'] != reference['n'] or candidate['n'] <= 0:
        raise ValueError('unpaired rough comparison counts')
    checks = {f'{key}_not_lower': candidate[key] >= reference[key] for key in ('one', 'six')}
    checks.update({f'{key}_not_higher': candidate[key] <= reference[key] for key in ('falls', 'lane')})
    checks['world_zero_including_flat'] = candidate['world'] + candidate['flat_world'] == 0
    checks['strict_rough_improvement'] = (any(candidate[k] > reference[k] for k in ('one', 'six'))
                                          or any(candidate[k] < reference[k] for k in ('falls', 'lane')))
    if primary:
        if candidate['flat_n'] != reference['flat_n'] or candidate['flat_n'] <= 0:
            raise ValueError('unpaired flat comparison counts')
        checks.update({f'{key}_not_higher': candidate[key] <= reference[key] for key in ('flat_falls', 'flat_lane')})
        speeds = (candidate['flat_mean_episode_speed'], reference['flat_mean_episode_speed'])
        if any(v is None or not math.isfinite(v) for v in speeds):
            raise ValueError('missing finite flat speed')
        checks['flat_speed_not_lower'] = speeds[0] >= speeds[1]
    return dict(passed=all(checks.values()), checks=checks)


def paired_six(candidate, reference):
    if len(candidate) != len(reference) or any(
            (a['family'], a['level']) != (b['family'], b['level']) for a, b in zip(candidate, reference)):
        raise ValueError('paired episode ordering differs')
    pairs = [(a['six'], b['six']) for a, b in zip(candidate, reference) if a['family'] != 'flat']
    return dict(n=len(pairs), gains=sum(a > b for a, b in pairs), losses=sum(a < b for a, b in pairs),
                both_success=sum(a == b == 1 for a, b in pairs), both_failure=sum(a == b == 0 for a, b in pairs))


def _section(groups, primary):
    result = {'groups': {name: aggregate(rows) for name, rows in groups.items()}}
    result['comparisons'] = {
        f'{candidate}_vs_{reference}': dict(
            **improvement(result['groups'][candidate], result['groups'][reference], primary=primary),
            paired_six=paired_six(groups[candidate], groups[reference]))
        for candidate, reference in study.COMPARISONS}
    result['family_level'] = {
        name: {f'{family}/level{level}': aggregate([r for r in rows if (r['family'], r['level']) == (family, level)])
               for family, level in sorted({(r['family'], r['level']) for r in rows})}
        for name, rows in groups.items()}
    return result


def _read(path):
    return json.loads(Path(path).read_text())


def _inside(name):
    path = (study.ROOT / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(study.ROOT.resolve()):
        raise ValueError('ledger path escapes repository')
    return path


def audit_ledger(directory, files, inputs):
    """Exactly one successful command/log per scored file; failed retries are invalid."""
    records = [json.loads(line) for line in (directory / 'commands.jsonl').read_text().splitlines()]
    matched, logs = Counter(), set()
    expected = {str((directory / item['path']).resolve()) for item in files}
    for record in records:
        command = record.get('command', [])
        if '--phase' not in command or command[command.index('--phase') + 1] != 'holdout':
            continue
        output = str((study.ROOT / command[command.index('--output') + 1]).resolve())
        if output not in expected:
            raise ValueError('unexpected scored ledger output')
        matched[output] += 1
        log = _inside(record['log'])
        if (record.get('returncode') != 0 or log in logs or study.sha(log) != record.get('log_sha256')
                or record.get('evaluation_inputs_sha256') != study.sha(directory / 'evaluation_inputs.json')
                or record.get('terrain_cache_manifest_sha256') != inputs['terrain_cache_manifest_sha256']):
            raise ValueError('invalid scored command/log evidence')
        logs.add(log)
    if matched != Counter({path: 1 for path in expected}):
        raise ValueError('scored command ledger missing/duplicate attempt')
    return dict(scored_commands=len(matched), unique_logs=len(logs), verified=True)


def summarize(directory=ART):
    directory = Path(directory)
    study.verify_frozen()
    if study.sha(directory / 'frozen.json') != study.sha(ART / 'frozen.json'):
        raise ValueError('summary directory freeze differs')
    inputs = study.evaluation_inputs(directory)
    provenance = {key: study.sha(directory / filename) for key, filename in (
        ('experiment_freeze_sha256', 'frozen.json'), ('evaluation_inputs_sha256', 'evaluation_inputs.json'),
        ('terrain_cache_manifest_sha256', 'terrain_cache.json'))}
    cache = _read(directory / 'terrain_cache.json')
    if cache['cache'] != study.cache_snapshot(Path('/tmp/isaaclab/terrains'), [g for g, _ in study.HOLDOUTS]):
        raise ValueError('pinned terrain cache bytes changed')
    for record in cache['preparation']:
        if study.sha(_inside(record['path'])) != record['sha256']:
            raise ValueError('terrain preparation evidence changed')
    # The input helper checks the manifest and pinned pre-scoring ledger prefix.
    result = dict(files=[], provenance=provenance, new_training_transitions=0)
    for scenario, folder in (('mixed', 'evaluations'), ('stones', 'horizon')):
        expected = {f'{c}__geometry{g}_reset{r}.json' for c in study.CONTROLLERS for g, r in study.HOLDOUTS}
        if {p.name for p in (directory / folder).glob('*.json')} != expected:
            raise ValueError('missing or unexpected v19 raw file matrix')
        groups, maps, initial, flats = defaultdict(list), defaultdict(dict), {}, {}
        for controller in study.CONTROLLERS:
            for geometry, reset in study.HOLDOUTS:
                path = directory / folder / f'{controller}__geometry{geometry}_reset{reset}.json'
                data = _read(path)
                if ((data.get('controller'), data.get('geometry_seed'), data.get('reset_seed'),
                     data.get('scenario'), data.get('phase')) != (controller, geometry, reset, scenario, 'holdout')
                        or any(data.get(key) != value for key, value in provenance.items())):
                    raise ValueError('raw condition/input linkage differs')
                if not HISTORY._timestamp(inputs['created_utc']) < HISTORY._timestamp(data['started_utc']) <= HISTORY._timestamp(data['finished_utc']):
                    raise ValueError('score chronology precedes frozen inputs')
                rows = audit(data)
                _paired(initial, (geometry, reset), data)
                if scenario == 'mixed' and controller != 'v16_control':
                    _flat_identity(flats, (geometry, reset), data, rows)
                groups[controller].extend(rows)
                maps[f'geometry{geometry}_reset{reset}'][controller] = rows
                result['files'].append(dict(path=str(path.relative_to(directory)), sha256=study.sha(path), episodes=len(rows)))
        section = _section(groups, scenario == 'mixed')
        section['per_map'] = {name: _section(rows, scenario == 'mixed') for name, rows in maps.items()}
        if scenario == 'mixed':
            section['v5_history_flat_identity'] = dict(verified=True, maps=len(flats), fields=list(FLAT_IDENTITY_FIELDS))
        result[folder] = section
    if len(result['files']) != 24 or sum(f['episodes'] for f in result['files']) != 2220:
        raise ValueError('incomplete v19 24-file/2220-episode audit')
    result['provenance']['ledger'] = audit_ledger(directory, result['files'], inputs)
    result['interpretation'] = ('Descriptive frozen-policy rebaseline on three paired maps, not statistical superiority or a causal reward ablation. '
                                'Single-seed historical checkpoints and simulator-only sensors limit generalization. '
                                'The separate 64s secondary cannot override the 16s primary. No automatic promotion; zero new training transitions.')
    return result


def markdown(result):
    lines = ['# v19 frozen-policy common-map rebaseline', '', result['interpretation'], '']
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
