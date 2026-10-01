# SPDX-License-Identifier: BSD-3-Clause
"""Independent v25 raw evidence audit. No experiment summary/scoring imports.

CPU checks validate recorded evidence and serialized model state, not unobserved
simulator behavior. Synthetic test success is never evidence of completed learning.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('control', 'recovery', 'combined')
SEEDS = (61, 62, 63)
RUNS = tuple(f'{a}{s}' for s in SEEDS for a in ARMS)
LEGACY = ('v5', 'v16_control', 'history_control', 'high53', 'history_high53')
CONTROLLERS = LEGACY + RUNS + tuple('history_' + r for r in RUNS)
MAPS = ((131, 111), (132, 112))
FAMILIES = ('rough', 'slope', 'stairs', 'waves', 'obstacles', 'stepping_stones', 'flat')
PARENT_SHA = '1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc'
TEACHER_SHA = '889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e'
LEGACY_SHA = {'v5': 'fa1ec87ef2b89b698f250d91452685cef0a7f76e340e790650116138aee875c1',
              'v16_control': PARENT_SHA, 'history_control': PARENT_SHA,
              'high53': '5c86a9b6efebf02bca4b37609acf688d69514d1887fb5b85cf28b79f5dc17c65',
              'history_high53': '5c86a9b6efebf02bca4b37609acf688d69514d1887fb5b85cf28b79f5dc17c65'}
GATE_CONFIG = dict(spatial=dict(span_enter=.12, edge_enter=.08, exit_ratio=.6, min_coverage=.9,
    enter_steps=2, min_dwell_seconds=.5, clear_seconds=.3, blend_seconds=.15),
    history_seconds=1., enter_window_seconds=.3, enter_min_seconds=.2, enter_fraction=.8,
    enter_confirm_seconds=.1, exit_window_seconds=.8, exit_min_seconds=.6, exit_fraction=.9,
    exit_confirm_seconds=.3, min_dwell_seconds=.5, blend_seconds=.15, unknown_reset_seconds=.25)

PAIR_FIELDS = ('initial_state_sha256', 'initial_prefix_sha256', 'initial_rng_sha256',
               'terrain_config_sha256', 'family_names', 'difficulties', 'family_indices',
               'level_indices', 'source_sha256')
BOOL_ARRAYS = ('episode_terminated', 'episode_out_of_lane', 'episode_world_exit',
               'episode_strict_one_tile_success', 'episode_strict_all_tiles_success',
               'episode_full_horizon_survival', 'episode_physical_done',
               'episode_physical_timeout', 'episode_window_censored')
NUMBER_ARRAYS = ('forward_distance', 'maximum_distance_m', 'episode_return')
NULLABLE_ARRAYS = ('first_hit_one_seconds', 'first_hit_six_seconds', 'distance_at_snapshot_m')
INT_ARRAYS = ('episode_lengths', 'episode_steps')
ROUTE_NUMBERS = ('episode_alpha_sum', 'episode_v10_duty', 'episode_mean_action_disagreement_rms',
                 'episode_max_action_jump_rms', 'episode_min_coverage', 'episode_max_enter_span',
                 'episode_max_enter_edge', 'episode_max_retain_span', 'episode_max_retain_edge')
ROUTE_INTS = ('episode_active_steps', 'episode_v10_target_steps', 'episode_uncertain_steps', 'episode_switch_count')
ROUTE_NULLABLE = ('episode_first_switch_seconds', 'episode_last_switch_seconds')
ROUTE_NESTED = ('episode_switch_steps', 'episode_switch_to_v10', 'episode_switch_seconds')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite evidence number')
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite(child)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    result = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    finite(result)
    return result


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'invalid SHA256')
    return value


def inside(root, name):
    require(type(name) is str and not Path(name).is_absolute(), 'nonrelative evidence path')
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'evidence path escapes repository')
    return path


def check_hashes(root, entries):
    require(isinstance(entries, dict) and bool(entries), 'empty hash inventory')
    for name, expected in entries.items():
        require(sha(inside(root, name)) == digest(expected), 'changed frozen bytes: ' + name)


def linked(root, item):
    path = inside(root, item['path'])
    require(sha(path) == digest(item['sha256']), 'linked evidence changed: ' + item['path'])
    return read(path)


def timestamp(value):
    require(type(value) is str, 'missing UTC timestamp')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'naive evidence timestamp')
    return result


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def array(obj, key, n, predicate):
    value = obj.get(key)
    require(type(value) is list and len(value) == n and all(predicate(x) for x in value),
            'malformed array: ' + key)
    return value


def window_rows(bundle, seconds):
    """Validate all tracker/routing arrays; recompute success from final state."""
    finite(bundle)
    w, n = bundle['windows'][str(seconds)], bundle['num_envs']
    require(w['window_seconds'] == seconds, 'swapped window label')
    for key in BOOL_ARRAYS:
        array(w, key, n, lambda x: type(x) is bool)
    for key in NUMBER_ARRAYS:
        array(w, key, n, number)
    for key in NULLABLE_ARRAYS:
        array(w, key, n, lambda x: x is None or number(x))
    for key in INT_ARRAYS:
        array(w, key, n, lambda x: type(x) is int and 0 < x <= seconds * 60)
    route = w['routing']
    for key in ROUTE_NUMBERS:
        array(route, key, n, lambda x: number(x) and x >= 0)
    for key in ROUTE_INTS:
        array(route, key, n, lambda x: type(x) is int and x >= 0)
    for key in ROUTE_NULLABLE:
        array(route, key, n, lambda x: x is None or number(x) and x >= 0)
    array(route, 'episode_fall_within_switch_window', n, lambda x: type(x) is bool)
    for key in ROUTE_NESTED:
        array(route, key, n, lambda x: type(x) is list)
    # Unknown arrays cannot evade dimensional/finite checks through schema additions.
    for container in (w, route):
        for key, value in container.items():
            if isinstance(value, list):
                require(len(value) == n, 'extra array shape differs: ' + key)
    require(type(w['captured_step']) is int and w['captured_step'] == max(w['episode_steps']),
            'capture step differs from first-episode completion')
    require(w['capture_reason'] == ('window_cutoff' if w['captured_step'] == seconds * 60
                                   else 'all_first_episodes_finished'), 'capture reason differs')
    rows = []
    for i in range(n):
        distance, steps = w['forward_distance'][i], w['episode_steps'][i]
        fall, lane, world = (w[k][i] for k in BOOL_ARRAYS[:3])
        done, timeout, censored = (w[k][i] for k in BOOL_ARRAYS[-3:])
        require(w['episode_lengths'][i] == steps == route['episode_active_steps'][i], 'episode length disagreement')
        require(timeout == (done and not fall) and (not fall or done), 'invalid physical done/timeout mask')
        require(censored == (seconds == 16 and steps == 960 and not done), 'invalid virtual censor mask')
        require(done or steps == seconds * 60, 'unended short first episode')
        safe = not (fall or lane or world)
        one, six = (safe and f32(distance) >= f32(threshold) for threshold in (13.1, 53.1))
        survival = steps == seconds * 60 and not fall
        require(w['episode_strict_one_tile_success'][i] == one and w['episode_strict_all_tiles_success'][i] == six,
                'saved success mask differs from strict final state')
        require(w['episode_full_horizon_survival'][i] == survival, 'saved survival mask differs')
        require(w['maximum_distance_m'][i] >= distance, 'maximum distance below final distance')
        for field, threshold in (('first_hit_one_seconds', 13.1), ('first_hit_six_seconds', 53.1)):
            hit = w[field][i]
            require(hit is None or 0 < hit <= steps / 60, 'invalid first-hit time')
            require(hit is None or f32(w['maximum_distance_m'][i]) >= f32(threshold), 'hit without maximum crossing')
        duty, count = route['episode_v10_duty'][i], route['episode_switch_count'][i]
        require(duty <= 1 and route['episode_min_coverage'][i] <= 1, 'routing fraction outside unit interval')
        require(math.isclose(duty, route['episode_alpha_sum'][i] / steps, abs_tol=1e-12), 'duty/alpha mismatch')
        require(all(route[k][i] <= steps for k in ('episode_v10_target_steps', 'episode_uncertain_steps')),
                'routing counts exceed episode length')
        ticks, targets, times = (route[k][i] for k in ROUTE_NESTED)
        require(len(ticks) == len(targets) == len(times) == count, 'switch arrays/count differ')
        require(all(type(x) is int and 0 < x <= steps for x in ticks) and ticks == sorted(set(ticks)), 'invalid switch steps')
        require(all(type(x) is bool for x in targets), 'nonboolean switch direction')
        require(all(number(t) and math.isclose(t, (tick - 1) / 60, abs_tol=1e-12) for tick, t in zip(ticks, times)), 'switch seconds differ')
        require(route['episode_first_switch_seconds'][i] == (times[0] if times else None)
                and route['episode_last_switch_seconds'][i] == (times[-1] if times else None), 'switch endpoints differ')
        controller = bundle['controller']
        if not controller.startswith('history_'):
            require(duty == (0 if controller == 'v5' else 1) and count == 0, 'endpoint routing differs')
        rows.append(dict(controller=controller, geometry=bundle['geometry_seed'], reset=bundle['reset_seed'],
                         env=i, seconds=seconds, family=FAMILIES[bundle['family_indices'][i]],
                         level=bundle['level_indices'][i], distance_m=distance, steps=steps,
                         one=one, six=six, fall=fall, lane=lane, world=world, survival=survival,
                         safe_survival=survival and safe, speed_m_s=distance / (steps * bundle['dt']), duty=duty, switches=count))
    return rows


def aggregate(rows):
    require(bool(rows), 'empty aggregate')
    n = len(rows)
    return dict(episodes=n, **{key: sum(r[key] for r in rows) for key in
                ('one', 'six', 'fall', 'lane', 'world', 'survival', 'safe_survival', 'switches')},
                mean_speed_m_s=sum(r['speed_m_s'] for r in rows) / n,
                mean_duty=sum(r['duty'] for r in rows) / n, all_duty_zero=all(r['duty'] == 0 for r in rows))


def groups(rows):
    return {name: aggregate([r for r in rows if (r['family'] == 'flat') == (name == 'flat')])
            for name in ('rough', 'flat')}


def paired(candidate, baseline):
    key = lambda r: (r['geometry'], r['reset'], r['env'])
    left, right = ({key(r): r for r in rows} for rows in (candidate, baseline))
    require(left.keys() == right.keys() and len(left) == len(candidate) == len(baseline), 'unpaired rows')
    output = {}
    for group in ('rough', 'flat'):
        pairs = [(a, right[k]) for k, a in left.items() if (a['family'] == 'flat') == (group == 'flat')]
        cell = {'episodes': len(pairs)}
        for metric in ('one', 'six', 'fall', 'lane', 'world', 'survival'):
            cell[metric] = dict(false_to_true=sum(a[metric] and not b[metric] for a, b in pairs),
                                true_to_false=sum(b[metric] and not a[metric] for a, b in pairs))
        cell.update(six_improved=cell['six']['false_to_true'], six_regressed=cell['six']['true_to_false'],
                    speed_delta_mean_m_s=sum(a['speed_m_s'] - b['speed_m_s'] for a, b in pairs) / len(pairs))
        output[group] = cell
    return output


def retention_gate(candidate, baseline, *, history=False, flat_identity=True):
    """Strict improvement never compensates for a retention failure."""
    a, b = groups(candidate), groups(baseline)
    r, f, br, bf = a['rough'], a['flat'], b['rough'], b['flat']
    checks = {f'rough_{k}_nondecrease': r[k] >= br[k] for k in ('one', 'six')}
    checks.update({f'rough_{k}_nonincrease': r[k] <= br[k] for k in ('fall', 'lane')})
    checks.update({f'flat_{k}_nonincrease': f[k] <= bf[k] for k in ('fall', 'lane')})
    checks['world_zero'] = r['world'] == f['world'] == 0
    checks['strict_rough_improvement'] = (r['one'] > br['one'] or r['six'] > br['six']
                                         )
    if history:
        checks['flat_v5_identity'] = flat_identity and f['all_duty_zero'] and bf['all_duty_zero']
    else:
        checks['flat_speed_nondecrease'] = f['mean_speed_m_s'] >= bf['mean_speed_m_s']
    return {'checks': checks, 'pass': all(checks.values())}


def validate_evaluations(records, freeze):
    development = freeze['phase'] == 'development'
    require(freeze['phase'] in ('development', 'holdout'), 'undeclared audit phase')
    controllers = ('v16_control', 'history_control', 'control61', 'recovery61', 'combined61',
                   'history_control61', 'history_recovery61', 'history_combined61') if development else CONTROLLERS
    maps, count = (((51, 24),), 35) if development else (MAPS, 175)
    require(set(freeze['models']) == set(controllers), 'freeze controller matrix differs')
    expected = {(c, g, r, count) for c in controllers for g, r in maps}
    require({tuple(x) for x in freeze['matrix']} == expected and len(freeze['matrix']) == len(expected), 'freeze matrix differs')
    require(freeze['v5_sha256'] == TEACHER_SHA, 'frozen teacher differs')
    timestamp(freeze['created_utc'])
    indexed = {}
    for b in records:
        finite(b)
        c, g, r, n = b['controller'], b['geometry_seed'], b['reset_seed'], b['num_envs']
        require((c, g, r, n) in expected and (c, g, r) not in indexed, 'undeclared/duplicate evaluation')
        indexed[c, g, r] = b
        require(b['schema'] == 'week03_ant_teammate_v25_eval_v1'
                and b['task'] == 'Week03-Ant-Contact-v16-Eval-v0', 'evaluation task/schema differs')
        require(b['seconds'] == 64 and b['dt'] == 1 / 60 and b['scenario'] == 'mixed'
                and b['scored_episodes'] == count and b['new_training_transitions'] == 0
                and set(b['windows']) == {'16', '64'}, 'evaluation budget/windows differ')
        require(b['family_names'] == list(FAMILIES) and b['difficulties'] == [.2, .4, .6, .8, 1.], 'terrain axes differ')
        family = array(b, 'family_indices', n, lambda x: type(x) is int and 0 <= x < 7)
        level = array(b, 'level_indices', n, lambda x: type(x) is int and 0 <= x < 5)
        require(Counter(zip(family, level)) == Counter({(f, l): count // 35 for f in range(7) for l in range(5)}),
                'family/level matrix differs')
        model = freeze['models'][c]
        require(b['checkpoint_sha256'] == b['checkpoint_sha256_after'] == digest(model['sha256']), 'checkpoint substitution')
        if c in LEGACY_SHA:
            require(model['sha256'] == LEGACY_SHA[c], 'legacy model substitution')
        require(b['v5_sha256'] == TEACHER_SHA and b['teacher_tensor_identity_verified'] is True, 'teacher identity differs')
        require(b['mode'] == model['mode'] == ('hybrid' if c.startswith('history_') else 'v5' if c == 'v5' else 'v10')
                and b['command_mode'] == model['command_mode'] == (None if c == 'v5' else 'conditioned'), 'policy binding differs')
        require(b['source_sha256'] == b['source_sha256_after'], 'source changed during scoring')
        for name, value in b['source_sha256'].items():
            require(freeze['base_source_sha256'].get(name) == digest(value), 'unfrozen evaluation source')
        require(set(b['initial_state_sha256']) == {'root_state', 'joint_pos', 'joint_vel', 'observations'}
                and set(b['initial_rng_sha256']) == {'cpu', 'cuda'}, 'initial state/RNG fields differ')
        for value in (*b['initial_state_sha256'].values(), *b['initial_rng_sha256'].values(),
                      b['initial_prefix_sha256'], b['terrain_config_sha256']):
            digest(value)
        require(b['gate_config'] == GATE_CONFIG, 'fixed history gate configuration differs')
        require(b['success_thresholds_m'] == {'one_tile': 13.1, 'all_tiles': 53.1}, 'scoring thresholds differ')
        require(timestamp(b['started_utc']) < timestamp(b['finished_utc']), 'evaluation chronology differs')
        require(timestamp(freeze['created_utc']) <= timestamp(b['started_utc']), 'holdout preceded model freeze')
        meta = b['v25']
        require(meta['phase'] == meta['model_phase'] == freeze['phase']
                and meta['scoring_training_transitions'] == 0
                and meta['substitutions'] == ['CONTROLLERS', 'SCHEMA', 'resolve_model']
                and meta['adapter_source_sha256'] == meta['adapter_source_sha256_after'] == freeze['source_sha256']
                and meta['base_evaluator_sha256'] == freeze['base_source_sha256']['scripts/evaluate_unseen_terrain.py'],
                'adapter provenance differs')
        for seconds in (16, 64):
            window_rows(b, seconds)
        prefix, full = (b['windows'][str(s)] for s in (16, 64))
        for i, steps in enumerate(full['episode_steps']):
            require(prefix['episode_steps'][i] == min(960, steps), 'windows do not share first episode')
            for key in ('episode_terminated', 'episode_out_of_lane', 'episode_world_exit'):
                require(not prefix[key][i] or full[key][i], 'first-episode flag cleared')
            if steps <= 960:
                for key in (*(k for k in BOOL_ARRAYS if k != 'episode_full_horizon_survival'), *NUMBER_ARRAYS, *INT_ARRAYS, 'first_hit_one_seconds', 'first_hit_six_seconds'):
                    require(prefix[key][i] == full[key][i], 'early-ended windows differ: ' + key)
                for key in prefix['routing']:
                    require(prefix['routing'][key][i] == full['routing'][key][i], 'early-ended routing differs: ' + key)
    require({(*key, b['num_envs']) for key, b in indexed.items()} == expected, 'incomplete evaluation matrix')
    for g, r in maps:
        reference = indexed[controllers[0], g, r]
        for c in controllers:
            b = indexed[c, g, r]
            require(all(b[k] == reference[k] for k in PAIR_FIELDS), 'initial physical/config/RNG pairing differs')
            require(b['gate_config'] == reference['gate_config'], 'history gate config changed')
    for c in (('control61', 'recovery61', 'combined61') if development else RUNS):
        require(freeze['models'][c]['sha256'] == freeze['models']['history_' + c]['sha256'], 'solo/history model mismatch')
        if development:
            require(indexed[c, 51, 24]['windows'] == indexed['v16_control', 51, 24]['windows']
                    and indexed['history_' + c, 51, 24]['windows'] == indexed['history_control', 51, 24]['windows'],
                    'development exact per-environment parity differs')
    return indexed


def raw_flat_identity(candidate, reference, seconds):
    window, ref = candidate['windows'][str(seconds)], reference['windows'][str(seconds)]
    for i, family in enumerate(candidate['family_indices']):
        if family != 6:
            continue
        if window['routing']['episode_v10_duty'][i] != 0 or window['routing']['episode_switch_count'][i] != 0:
            return False
        for key, values in window.items():
            if isinstance(values, list) and values[i] != ref[key][i]:
                return False
        for key, values in window['routing'].items():
            if key != 'episode_mean_action_disagreement_rms' and values[i] != ref['routing'][key][i]:
                return False
    return True


def audit_evaluation_records(records, freeze):
    indexed = validate_evaluations(records, freeze)
    if freeze['phase'] == 'development':
        return dict(phase='development', physical_first_episodes=280, dependent_window_observations=560,
                    exact_per_environment_parity=True)
    output = dict(physical_first_episodes=8050, dependent_window_observations=16100,
                  paired_initial_conditions=350, statistical_independence_claim=False, windows={})
    for seconds in (16, 64):
        rows = [r for c in CONTROLLERS for g, reset in MAPS for r in window_rows(indexed[c, g, reset], seconds)]
        for row in rows:
            if row['family'] == 'flat':
                c, g, reset, i = row['controller'], row['geometry'], row['reset'], row['env']
                window, ref = indexed[c, g, reset]['windows'][str(seconds)], indexed['v5', g, reset]['windows'][str(seconds)]
                row['flat_raw_identity'] = all(values[i] == ref[k][i] for k, values in window.items() if isinstance(values, list)) and all(
                    values[i] == ref['routing'][k][i] for k, values in window['routing'].items() if k != 'episode_mean_action_disagreement_rms')
        controllers = {}
        for c in CONTROLLERS:
            selected = [r for r in rows if r['controller'] == c]
            controllers[c] = dict(total=groups(selected),
                by_map={str(g): groups([r for r in selected if r['geometry'] == g]) for g, _ in MAPS},
                by_family={f: aggregate([r for r in selected if r['family'] == f]) for f in FAMILIES},
                by_level={str(l): groups([r for r in selected if r['level'] == l]) for l in range(5)},
                by_map_family_level={str(g): {f: {str(l): aggregate([r for r in selected if
                    (r['geometry'], r['family'], r['level']) == (g, f, l)]) for l in range(5)} for f in FAMILIES} for g, _ in MAPS})
        contrasts = {}
        for mode, prefix in (('solo', ''), ('history', 'history_')):
            for a, b in (('combined', 'control'), ('recovery', 'control'), ('combined', 'recovery')):
                seeds = {}
                for seed in SEEDS:
                    left = [r for r in rows if r['controller'] == f'{prefix}{a}{seed}']
                    right = [r for r in rows if r['controller'] == f'{prefix}{b}{seed}']
                    gates = {str(g): retention_gate([r for r in left if r['geometry'] == g],
                             [r for r in right if r['geometry'] == g], history=bool(prefix), flat_identity=all(raw_flat_identity(indexed[c, g, reset], indexed['v5', g, reset], seconds)
                                for c in (f'{prefix}{a}{seed}', f'{prefix}{b}{seed}') for gg, reset in MAPS if gg == g)) for g, _ in MAPS}
                    gates['combined'] = retention_gate(left, right, history=bool(prefix), flat_identity=all(
                        raw_flat_identity(indexed[c, g, reset], indexed['v5', g, reset], seconds)
                        for c in (f'{prefix}{a}{seed}', f'{prefix}{b}{seed}') for g, reset in MAPS))
                    seeds[str(seed)] = dict(gates=gates, paired=paired(left, right),
                                           **{'pass': all(x['pass'] for x in gates.values())})
                contrasts[f'{mode}:{a}_vs_{b}'] = dict(seeds=seeds, passed_seeds=sum(v['pass'] for v in seeds.values()),
                                                      **{'pass': all(v['pass'] for v in seeds.values())})
        output['windows'][str(seconds)] = dict(controllers=controllers, contrasts=contrasts, episode_rows=rows)
    output['late_outcomes'] = {}
    for c in CONTROLLERS:
        early = [r for g, reset in MAPS for r in window_rows(indexed[c, g, reset], 16)]
        late = [r for g, reset in MAPS for r in window_rows(indexed[c, g, reset], 64)]
        output['late_outcomes'][c] = paired(late, early)
    return output


def compare_summary(actual, canonical):
    """Compare every computed public summary cell, with no summary imports."""
    for key in ('physical_first_episodes', 'dependent_window_observations', 'paired_initial_conditions',
                'statistical_independence_claim'):
        require(actual[key] == canonical[key], 'summary accounting differs: ' + key)
    for seconds in ('16', '64'):
        published, independent = actual['windows'][seconds], canonical['windows'][seconds]
        require(set(published['controllers']) == set(CONTROLLERS), 'summary controller inventory differs')
        for c in CONTROLLERS:
            a, b = published['controllers'][c], independent['controllers'][c]
            for key in ('total', 'by_map', 'by_family', 'by_level'):
                require(a[key] == b[key], 'summary aggregate differs: ' + c + '/' + key)
            cells = {(x['geometry'], x['family'], x['level']): {k: v for k, v in x.items()
                     if k not in ('geometry', 'family', 'level')} for x in a['by_map_family_level']}
            expected = {(g, f, l): b['by_map_family_level'][str(g)][f][str(l)]
                        for g, _ in MAPS for f in FAMILIES for l in range(5)}
            require(len(a['by_map_family_level']) == 70 and cells == expected, 'summary intersection cells differ')
        for key, contrast in independent['contrasts'].items():
            mode, name = key.split(':')
            pub = published['contrasts'][('history_' if mode == 'history' else '') + name]
            require(pub['seed_passes'] == contrast['passed_seeds'] and pub['verdict'] == ('PASS' if contrast['pass'] else 'FAIL'),
                    'summary arm verdict differs')
            for seed, seed_result in contrast['seeds'].items():
                ps = pub['seeds'][seed]
                require(ps['verdict'] == ('PASS' if seed_result['pass'] else 'FAIL'), 'summary seed verdict differs')
                for scope, gate in seed_result['gates'].items():
                    pg = ps['scopes']['pooled' if scope == 'combined' else scope]
                    require(pg['verdict'] == ('PASS' if gate['pass'] else 'FAIL'), 'summary map verdict differs')
        key_row = lambda r: (r['controller'], r['geometry'], r['reset'], r['env'])
        rows = independent['episode_rows']
        require({key_row(r): r for r in published['episode_rows']} == {key_row(r): r for r in rows}
                and len(published['episode_rows']) == len(rows), 'summary episode rows differ')
        selected = {c: [r for r in rows if r['controller'] == c] for c in CONTROLLERS}
        refs = {f'{c}_vs_{ref}': paired(selected[c], selected[ref]) for c in CONTROLLERS if c not in LEGACY for ref in LEGACY}
        require(published['legacy_descriptive_comparisons'] == refs, 'legacy comparison differs')
        for name, pub in published['contrasts'].items():
            history = name.startswith('history_')
            contrast_name = name.removeprefix('history_')
            arm, baseline_arm = contrast_name.split('_vs_')
            require(arm in ARMS and baseline_arm in ARMS, 'unknown comparison')
            prefix = 'history_' if history else ''
            expected_pooled = {label: groups([r for seed in SEEDS for r in selected[f'{prefix}{a}{seed}']])
                               for label, a in (('candidate', arm), ('baseline', baseline_arm))}
            require(pub['pooled_descriptive'] == expected_pooled, 'pooled descriptive counts differ')
            for seed in SEEDS:
                left, right = selected[f'{prefix}{arm}{seed}'], selected[f'{prefix}{baseline_arm}{seed}']
                ps = pub['seeds'][str(seed)]
                for scope in ('131', '132', 'pooled'):
                    aa = [r for r in left if scope == 'pooled' or r['geometry'] == int(scope)]
                    bb = [r for r in right if scope == 'pooled' or r['geometry'] == int(scope)]
                    gg = retention_gate(aa, bb, history=history,
                        flat_identity=all(r['flat_raw_identity'] for r in aa + bb if r['family'] == 'flat'))
                    renames = {'rough_one_nondecrease': 'rough_one_retained', 'rough_six_nondecrease': 'rough_six_retained',
                        'rough_fall_nonincrease': 'rough_fall_retained', 'rough_lane_nonincrease': 'rough_lane_retained',
                        'flat_fall_nonincrease': 'flat_falls_retained', 'flat_lane_nonincrease': 'flat_lanes_retained',
                        'flat_speed_nondecrease': 'flat_speed_retained'}
                    expected_gate = dict(verdict='PASS' if gg['pass'] else 'FAIL',
                        checks={renames.get(k, k): v for k, v in gg['checks'].items()},
                        candidate=groups(aa), baseline=groups(bb), paired=paired(aa, bb))
                    require(ps['scopes'][scope] == expected_gate, 'summary retention checks/counts differ')
                cells = []
                for g, _ in MAPS:
                    for level in range(5):
                        for family in sorted(FAMILIES):
                            aa = [r for r in left if (r['geometry'], r['level'], r['family']) == (g, level, family)]
                            bb = [r for r in right if (r['geometry'], r['level'], r['family']) == (g, level, family)]
                            aligned = {r['env']: r for r in bb}
                            changes = {k: dict(false_to_true=sum(r[k] and not aligned[r['env']][k] for r in aa),
                                               true_to_false=sum(aligned[r['env']][k] and not r[k] for r in aa))
                                       for k in ('one', 'six', 'fall', 'lane', 'world', 'survival')}
                            cells.append(dict(geometry=g, level=level, family=family,
                                              candidate=aggregate(aa), baseline=aggregate(bb), changes=changes))
                require(ps['paired_map_family_level'] == cells, 'paired intersection counts differ')
    return True


ACTOR = ('actor.0.weight', 'actor.0.bias', 'actor.0.command_weight', 'actor.2.weight', 'actor.2.bias',
         'actor.4.weight', 'actor.4.bias', 'actor.6.weight', 'actor.6.bias')
TRAINABLE = ('std', *ACTOR, *(x.replace('actor.', 'critic.') for x in ACTOR))


def tensor_equal(left, right, label):
    import torch
    require(torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype
            and left.shape == right.shape and torch.equal(left, right), 'tensor mismatch: ' + label)


def model_audit(initial, final, parent, teacher, *, updates=250):
    """Inspect actual tensors and actual Adam moments; no reported flag is proof."""
    import torch
    require(type(updates) is int and updates in (2, 250), 'undeclared model budget')
    init = initial['model_state_dict']
    trained = final['model_state_dict']
    require(initial['iter'] == 0 and final['iter'] == updates - 1, 'checkpoint iteration differs')
    require(initial['optimizer_state_dict']['state'] == {}, 'initializer Adam is not fresh')
    require(set(init) == set(trained) == set(parent['model_state_dict']) and len(init) == 33, 'model tensor inventory differs')
    require(sum(v.numel() for v in init.values()) == 400631, 'model architecture differs')
    tensor_equal(init['std'], torch.full((8,), .2), 'initial std')
    for name, value in init.items():
        require(torch.isfinite(value).all().item() and torch.isfinite(trained[name]).all().item(), 'nonfinite model tensor')
        require(value.shape == trained[name].shape and value.dtype == trained[name].dtype, 'model tensor shape/dtype differs')
        if name != 'std':
            tensor_equal(value, parent['model_state_dict'][name], 'initial parent/' + name)
        if name not in TRAINABLE:
            tensor_equal(value, trained[name], 'frozen teacher/buffer/' + name)
        if name.startswith('teacher.'):
            tensor_equal(value, teacher['model_state_dict'][name.replace('teacher.', 'actor.')], 'original v5/' + name)
    require(bool((trained['std'] > 0).all()), 'nonpositive final std')
    start_opt, end_opt = initial['optimizer_state_dict'], final['optimizer_state_dict']
    require(start_opt['param_groups'] == end_opt['param_groups'] and len(start_opt['param_groups']) == 1, 'Adam groups changed')
    group = start_opt['param_groups'][0]
    require(group['params'] == list(range(27)) and group['lr'] == 1e-4 and tuple(group['betas']) == (.9, .999)
            and group['eps'] == 1e-8 and group['weight_decay'] == 0 and group['amsgrad'] is False,
            'Adam settings/coverage differ')
    require(set(end_opt['state']) == set(range(19)), 'Adam trainable coverage differs')
    for index, name in enumerate(TRAINABLE):
        row = end_opt['state'][index]
        require(set(row) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Adam state fields differ')
        require(row['step'].numel() == 1 and row['step'].item() == updates * 20, 'actual Adam steps differ')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = row[key]
            require(value.shape == trained[name].shape and value.dtype == trained[name].dtype
                    and bool(torch.isfinite(value).all()), 'invalid Adam moments')
        require(bool((row['exp_avg_sq'] >= 0).all()), 'negative Adam second moment')
    return dict(tensors=33, teacher_and_command_buffers_unchanged=True, adam_parameter_states=19,
                adam_steps=updates * 20, final_iteration=updates - 1)


def audit_initial(initial, *, arm, seed, envs, updates):
    finite(initial)
    require(arm in ARMS and seed in SEEDS and (envs, updates) in ((256, 2), (4096, 2), (4096, 250)), 'undeclared training cell')
    require(initial['arm'] == arm and initial['training_seed'] == seed, 'training identity differs')
    require(initial['actual_seed'] == dict.fromkeys(('requested', 'saved_agent', 'saved_environment', 'live_environment'), seed)
            and all(type(v) is int for v in initial['actual_seed'].values()), 'actual seed differs')
    require(initial['num_envs'] == envs and initial['observation_dimensions'] == [envs, 91]
            and initial['dt'] == 1 / 60 and initial['iteration'] == 0 and initial['common_step_counter'] == 0
            and initial['optimizer_state_empty'] is True, 'initial budget/state differs')
    require(initial['initial_policy_parity'] == dict(actor=True, critic=True, teacher=True, device='cuda:1'), 'live parent output parity differs')
    require(set(initial['initial_state_sha256']) == {'root_state', 'joint_pos', 'joint_vel', 'observations'}
            and set(initial['rng_sha256']) == {'cpu', 'cuda'}, 'initial state/RNG inventory differs')
    for h in (*initial['initial_state_sha256'].values(), *initial['rng_sha256'].values(),
              initial['initial_prefix_sha256'], *initial['policy_state_sha256'].values()):
        digest(h)
    parameters = initial['normalized_parameters']
    env, agent = parameters['env'], parameters['agent']
    require(env['seed'] == agent['seed'] == str(seed) and agent['max_iterations'] == str(updates)
            and agent['num_steps_per_env'] == '32' and agent['device'] == 'cuda:1', 'saved seed/budget/device differs')
    require(env['scene']['terrain']['terrain_generator']['seed'] == '130', 'training geometry differs')
    reward = env['rewards']['teammate_recovery']
    require(reward['func'] == 'week03_ant.tasks.teammate_v25_cfg:TeammateRecoveryReward'
            and reward['weight'] == '1.0' and reward['params'] == {'enabled': str(arm != 'control').lower()}, 'recovery config differs')
    algorithm = agent['algorithm']
    require(float(algorithm['entropy_coef']) == (.005 if arm == 'combined' else .002)
            and float(algorithm['teacher_coef']) == .02 and float(algorithm['learning_rate']) == 1e-4
            and algorithm['schedule'] == 'fixed' and algorithm['num_learning_epochs'] == '5'
            and algorithm['num_mini_batches'] == '4', 'PPO coefficients differ')
    return True


def initial_pair(left, right, *, capacity_to_main=False):
    """Project exactly treatment fields or the approved 2->250 budget, never seeds."""
    a, b = deepcopy(left), deepcopy(right)
    if capacity_to_main:
        require(a['arm'] == b['arm'] and a['normalized_parameters']['agent']['max_iterations'] == '2'
                and b['normalized_parameters']['agent']['max_iterations'] == '250', 'capacity projection operands differ')
        a['normalized_parameters']['agent']['max_iterations'] = '250'
    else:
        for x in (a, b):
            x['normalized_parameters']['agent']['algorithm']['entropy_coef'] = '<treatment>'
            x['normalized_parameters']['env']['rewards']['teammate_recovery']['params']['enabled'] = '<treatment>'
            x['arm'] = '<treatment>'
    for key in ('arm', 'training_seed', 'actual_seed', 'normalized_parameters', 'num_envs', 'dt',
                'observation_dimensions', 'initial_state_sha256', 'initial_prefix_sha256', 'policy_state_sha256',
                'rng_sha256', 'initial_policy_parity', 'optimizer_state_empty', 'iteration', 'common_step_counter'):
        require(a[key] == b[key], 'initial pairing differs: ' + key)
    return True


def audit_schedule(schedule, *, arm, seed, envs, updates):
    finite(schedule)
    require(schedule['arm'] == arm and schedule['training_seed'] == seed and schedule['num_envs'] == envs,
            'schedule identity differs')
    require(schedule['policy_steps'] == updates * 32 and schedule['ppo_updates'] == updates
            and schedule['adam_steps'] == updates * 20, 'actual policy/PPO/Adam budget differs')
    require(schedule['entropy_coef'] == (.005 if arm == 'combined' else .002)
            and schedule['teacher_coef'] == .02 and schedule['learning_rate'] == 1e-4, 'runtime coefficients differ')
    runtime = schedule['runtime_initial']
    require(runtime == schedule['runtime_final'] and runtime['entropy_coef'] == schedule['entropy_coef']
            and runtime['teacher_coef'] == .02 and runtime['learning_rate'] == 1e-4
            and runtime['schedule'] == 'fixed' and runtime['desired_kl'] is None, 'runtime endpoint proof differs')
    require(len(runtime['optimizer_groups']) == 1, 'runtime Adam groups differ')
    group = runtime['optimizer_groups'][0]
    require(group['lr'] == 1e-4 and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8
            and group['weight_decay'] == 0 and group['amsgrad'] is False, 'runtime Adam settings differ')
    require(len(schedule['runtime_updates']) == updates, 'missing actual PPO coefficient trace')
    for i, row in enumerate(schedule['runtime_updates'], 1):
        require(row == dict(update=i, adam_steps=i * 20, before=runtime, after=runtime), 'actual coefficient trace differs')
    proof = schedule['episode_randomization']
    require(proof['matches_raw_environment'] is True and digest(proof['assigned_sha256']) == digest(proof['actual_sha256'])
            and type(proof['nonzero']) is int and 0 < proof['nonzero'] <= envs
            and 0 <= proof['minimum'] < proof['maximum'], 'actual random horizons differ')
    require(len(schedule['recovery']) == updates * 32, 'recovery rollout length differs')
    for step, row in enumerate(schedule['recovery'], 1):
        require(row['step'] == step and row['enabled'] is (arm != 'control') and row['dt'] == 1 / 60,
                'reward-time binding differs')
        require(type(row['abstained']) is int and 0 <= row['abstained'] <= envs, 'invalid abstention count')
        components = [row[k] for k in ('clearance_risk', 'tilt_risk', 'action_delta_squared_sum', 'angular_xy_squared_sum')]
        require(all(number(v) and v >= 0 for v in components) and all(v <= envs for v in components[:2]), 'invalid recovery components')
        expected = sum(v * weight for v, weight in zip(components, (-2, -2, -.01, -.025)))
        require(number(row['raw_sum']) and math.isclose(row['raw_sum'], expected, rel_tol=2e-6, abs_tol=1e-5),
                'recovery formula differs')
        require(row['applied_sum'] == (0 if arm == 'control' else row['raw_sum']), 'control-zero/applied recovery differs')
    contact = schedule['contact']
    total = updates * 32
    require(contact['mode'] == 'no_cost' and contact['ramp_steps'] == 4000 and contact['num_envs'] == envs
            and contact['policy_steps'] == total and contact['common_step_counters'] == list(range(1, total + 1)),
            'contact clock/budget differs')
    require(contact['coefficients'] == [0.] * total and contact['coefficient_sum'] == 0
            and contact['scaled_reward_mean'] == [0.] * total and contact['scaled_reward_dt_sum'] == [0.] * total
            and contact['realized_penalty_sum'] == 0, 'additional contact cost not zero')
    for key in ('raw_reward_mean', 'step_dt', 'raw_reward_dt_sum', 'valid_rows'):
        array(contact, key, total, number)
    for i in range(total):
        require(-1 <= contact['raw_reward_mean'][i] <= 0 and contact['step_dt'][i] == 1 / 60
                and type(contact['valid_rows'][i]) is int and 0 <= contact['valid_rows'][i] <= envs
                and math.isclose(contact['raw_reward_dt_sum'][i], contact['raw_reward_mean'][i] * envs / 60,
                                 rel_tol=2e-6, abs_tol=2e-7), 'contact accounting differs')
    return dict(policy_steps=updates * 32, ppo_updates=updates, adam_steps=updates * 20,
                transitions=envs * updates * 32, recovery_rows=updates * 32)


def saved_configs(root, initial):
    import yaml
    logdir = inside(root, initial['log_dir'])
    configs = {}
    for name in ('env', 'agent'):
        path = logdir / 'params' / (name + '.yaml')
        require(sha(path) == initial['saved_parameter_sha256'][name], 'saved YAML changed')
        configs[name] = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    for field in ('log_dir', 'io_descriptors_output_dir'):
        require(Path(configs['env'].pop(field)).resolve() == logdir.resolve(), 'saved output identity differs')
    configs['agent'].pop('run_name')
    require(configs == initial['normalized_parameters'], 'saved vs recorded normalized configs differ')
    return configs


def learning_log(root, record, initial, updates):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    path = inside(root, record['text_log'])
    require(sha(path) == record['text_log_sha256'], 'training console changed')
    text = path.read_text()
    iterations = [(int(a), int(b)) for a, b in re.findall(r'Learning iteration (\d+)/(\d+)', text)]
    require(iterations == [(i, updates) for i in range(updates)], 'actual training iteration sequence differs')
    totals = [int(v) for v in re.findall(r'Total timesteps:\s*(\d+)', text)]
    require(totals == [(i + 1) * initial['num_envs'] * 32 for i in range(updates)], 'actual training transitions differ')
    logdir = inside(root, initial['log_dir'])
    events = {str(p.relative_to(root)): sha(p) for p in logdir.glob('events.out.tfevents.*')}
    require(events and events == record['event_files_sha256'], 'training event inventory changed')
    data = EventAccumulator(str(logdir), size_guidance={'scalars': 0}).Reload()
    tags = data.Tags()['scalars']
    require({'Episode_Reward/adaptive_posture', 'Episode_Reward/teammate_recovery', 'Loss/value_function',
             'Loss/surrogate', 'Loss/prior_loss', 'Loss/learning_rate', 'Train/mean_reward', 'Policy/mean_noise_std'} <= set(tags),
            'missing learning scalars')
    for tag in tags:
        values = data.Scalars(tag)
        require(len(values) == updates and all(math.isfinite(v.value) for v in values), 'nonfinite/incomplete learning scalar')
        if not tag.endswith('/time'):
            require([v.step for v in values] == list(range(updates)), 'scalar update sequence differs')
        if tag == 'Loss/learning_rate':
            require(all(v.value == f32(1e-4) for v in values), 'logged learning rate differs')
    return dict(scalar_tags=len(tags), finite_scalar_values=len(tags) * updates, transitions=totals[-1])


def audit_novelty(root, novelty, original):
    import yaml
    require(novelty['new_training_geometry'] == 130 and novelty['new_holdout_pairs'] == [[131, 111], [132, 112]],
            'novelty planned geometry differs')
    require(len(novelty['training_configs']) == novelty['saved_training_configs'] == 166
            and len(novelty['archived_results']) == novelty['archived_json_files'] == 1158, 'novelty inventory size differs')
    seen = set()
    for item in novelty['training_configs']:
        path = inside(root, item['path'])
        require(item['path'] not in seen and sha(path) == item['sha256'], 'historical training config changed/duplicate')
        seen.add(item['path'])
        data = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        terrain = data.get('scene', {}).get('terrain', {})
        generator = terrain.get('terrain_generator', {}) if isinstance(terrain, dict) else {}
        value = generator.get('seed') if isinstance(generator, dict) else None
        seed = int(value) if value is not None and str(value).isdigit() else None
        require(seed == item['geometry_seed'] and seed not in (130, 131, 132), 'historical training geometry collision')
    def values(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ('geometry', 'geometry_seed', 'terrain_seed', 'training_geometry') and type(child) is int:
                    yield child
                yield from values(child)
        elif isinstance(value, list):
            for child in value:
                yield from values(child)
    paths = set()
    for item in novelty['archived_results']:
        require(item['path'] not in paths and original.get(item['path']) == item['sha256'], 'historical JSON inventory substitution')
        paths.add(item['path'])
        require(not set(values(read(inside(root, item['path'])))) & {130, 131, 132}, 'archived evaluation geometry collision')
    require(paths == {p for p in original if p.startswith('artifacts/') and p.endswith('.json')}, 'novelty archived JSON coverage differs')
    return True


def audit_cache(snapshot, geometries, *, cache_root=Path('/tmp/isaaclab/terrains')):
    require(snapshot['geometries'] == list(geometries) and snapshot['tiles_per_geometry'] == 240, 'cache geometry differs')
    expected = {g * 10007 + col * 131 + row for g in geometries for col in range(30) for row in range(8)}
    require(len(snapshot['tiles']) == len(expected) and {t['tile_seed'] for t in snapshot['tiles']} == expected,
            'cache tile inventory differs')
    for tile in snapshot['tiles']:
        directory = inside(cache_root, tile['cache_key'])
        require(not directory.is_symlink() and set(tile['sha256']) == {'cfg.yaml', 'mesh.obj', 'origin.csv'}, 'cache files differ')
        check_hashes(directory, tile['sha256'])
        match = re.search(r'^seed: (\d+)\s*$', (directory / 'cfg.yaml').read_text(), re.MULTILINE)
        require(match is not None and int(match[1]) == tile['tile_seed'], 'cache seed binding differs')
    return True


def audit_training(root, artifact_dir, *, phase='main'):
    """Verify original bytes, saved configs, live evidence, logs and model tensors."""
    import torch
    require(phase in ('preflight', 'capacity', 'main'), 'undeclared training audit phase')
    torch.set_num_threads(1)
    shared = read(artifact_dir / 'initial_shared.json')
    require(shared['parent_sha256'] == PARENT_SHA and set(shared['models']) == {'61', '62', '63'}, 'parent/initializer inventory differs')
    original = artifact_dir / 'original_publication.sha256'
    require(sha(original) == shared['original_inventory_sha256'], 'original byte inventory changed')
    old = dict((line.split('  ', 1)[1], line.split('  ', 1)[0]) for line in original.read_text().splitlines())
    check_hashes(root, old)
    require(sha(artifact_dir / 'geometry_novelty.json') == shared['geometry_novelty_sha256'], 'geometry novelty evidence changed')
    audit_novelty(root, read(artifact_dir / 'geometry_novelty.json'), old)
    cache = read(artifact_dir / 'training_cache.json')
    check_hashes(root, {cache['preparation']: cache['preparation_sha256']})
    require(read(inside(root, cache['preparation']))['scored_episodes'] == cache['scored_episodes'] == 0, 'training cache preparation scored episodes')
    audit_cache(cache['cache'], (130,))
    import importlib.metadata
    for name, version in shared['runtime'].items():
        require(importlib.metadata.version(name) == version, 'runtime dependency version changed')
    for path, expected in shared['runtime_source_sha256'].items():
        require(sha(path) == digest(expected), 'runtime dependency source changed')
    parent_path = root / 'artifacts/terrain_demo/contact_slip_v16/runs/control/model_249.pt'
    teacher_path = root / 'artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt'
    require(sha(parent_path) == PARENT_SHA and sha(teacher_path) == TEACHER_SHA, 'parent/teacher substitution')
    parent, teacher = (torch.load(p, map_location='cpu', weights_only=False) for p in (parent_path, teacher_path))
    manifest = read(artifact_dir / ('training_validation.json' if phase == 'main' else phase + '.json'))
    runs = RUNS[:3] if phase == 'preflight' else RUNS
    require(set(manifest['records']) == set(runs), 'training run matrix differs')
    envs, updates = (256 if phase == 'preflight' else 4096), (250 if phase == 'main' else 2)
    if phase == 'main':
        freeze_path = artifact_dir / 'training_frozen.json'
        freeze, trained = read(freeze_path), read(artifact_dir / 'trained_models.json')
        require(trained['training_freeze_sha256'] == manifest['training_freeze_sha256'] == sha(freeze_path)
                and trained['training_validation_sha256'] == sha(artifact_dir / 'training_validation.json')
                and set(trained['models']) == set(RUNS), 'final model freeze differs')
        require(freeze['envs'] == 4096 and freeze['iterations'] == 250 and freeze['steps'] == 32
                and freeze['geometry'] == 130 and freeze['seeds'] == list(SEEDS) and freeze['runs'] == list(RUNS)
                and freeze['transitions_per_run'] == 32768000 and freeze['total_training_transitions'] == 294912000
                and freeze['development_training_transitions'] == 2408448, 'frozen training budget differs')
        for key in ('source_sha256', 'all_source_sha256', 'artifacts_sha256', 'expected_main_sha256'):
            check_hashes(root, freeze[key])
        capacity = read(artifact_dir / 'capacity.json')
        commands = [json.loads(line) for line in (artifact_dir / 'commands.jsonl').read_text().splitlines()]
        commands = [row for row in commands if row['label'].startswith('main_')]
        require([row['label'] for row in commands] == ['main_' + run for run in RUNS], 'main execution order differs')
        previous = timestamp(freeze['created_utc'])
        for command in commands:
            require(command['returncode'] == 0 and previous <= timestamp(command['started_utc'])
                    < timestamp(command['finished_utc']) <= timestamp(trained['created_utc']), 'main execution chronology differs')
            previous = timestamp(command['finished_utc'])
            check_hashes(root, {command['log']: command['log_sha256']})
    else:
        check_hashes(root, manifest['source_sha256'])
    results, initial_records = {}, {}
    for run in runs:
        arm = next(a for a in ARMS if run.startswith(a))
        seed = int(run[-2:])
        record = manifest['records'][run]
        initial, schedule, rollout = (linked(root, record[k]) for k in ('initial', 'schedule', 'rollout'))
        audit_initial(initial, arm=arm, seed=seed, envs=envs, updates=updates)
        saved_configs(root, initial)
        actual = audit_schedule(schedule, arm=arm, seed=seed, envs=envs, updates=updates)
        require(rollout['arm'] == arm and rollout['passive_post_action'] is True, 'passive rollout binding differs')
        init_entry = shared['models'][str(seed)]
        init_path = inside(root, init_entry['checkpoint'])
        require(sha(init_path) == init_entry['sha256'], 'initializer changed')
        initializer = torch.load(init_path, map_location='cpu', weights_only=False)
        hashes = {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                  for k, v in initializer['model_state_dict'].items()}
        require(initial['policy_state_sha256'] == hashes == init_entry['state_sha256'], 'live initializer tensor substitution')
        model_path = inside(root, initial['log_dir']) / f'model_{updates - 1}.pt'
        if phase == 'main':
            entry = trained['models'][run]
            require(entry['iteration'] == 249 and entry['training_seed'] == seed and entry['arm'] == arm
                    and entry['transitions'] == 32768000 and Path(entry['checkpoint']).name == 'model_249.pt'
                    and sha(inside(root, entry['checkpoint'])) == entry['sha256'] == sha(model_path), 'final checkpoint binding differs')
            cap_initial = linked(root, capacity['records'][run]['initial'])
            cap_schedule = linked(root, capacity['records'][run]['schedule'])
            initial_pair(cap_initial, initial, capacity_to_main=True)
            require(cap_schedule['episode_randomization'] == schedule['episode_randomization'], 'main/capacity horizons differ')
            require(timestamp(freeze['created_utc']) <= timestamp(initial['created_utc'])
                    <= timestamp(trained['created_utc']), 'main training chronology differs')
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        model_result = model_audit(initializer, state, parent, teacher, updates=updates)
        log_result = learning_log(root, record['learning_log'], initial, updates)
        if seed in initial_records:
            initial_pair(initial_records[seed][0], initial)
            require(initial_records[seed][1] == schedule['episode_randomization'], 'same-seed actual horizons differ')
        else:
            initial_records[seed] = (initial, schedule['episode_randomization'])
        results[run] = dict(actual=actual, model=model_result, logs=log_result, checkpoint_sha256=sha(model_path))
    return dict(phase=phase, runs=results, total_training_transitions=len(runs) * envs * updates * 32,
                original_files_unchanged=len(old), live_gpu_execution_inferred_only_from_recorded_evidence=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--training-artifacts', type=Path)
    parser.add_argument('--phase', choices=('preflight', 'capacity', 'main'), default='main')
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = dict(auditor_sha256=sha(Path(__file__)), synthetic_checks_are_not_live_evidence=True)
    require(args.training_artifacts is not None or args.input_dir is not None, 'no evidence requested')
    if args.training_artifacts:
        result['training'] = audit_training(args.root, args.training_artifacts, phase=args.phase)
    if args.input_dir:
        require(args.freeze is not None, 'evaluation freeze required')
        freeze = read(args.freeze)
        for key in ('source_sha256', 'base_source_sha256'):
            check_hashes(args.root, freeze[key])
        for model in freeze['models'].values():
            check_hashes(args.root, {model['checkpoint']: model['sha256']})
        if freeze['phase'] == 'holdout':
            art = args.training_artifacts or args.root / 'artifacts/terrain_demo/teammate_port_v25'
            trained = read(art / 'trained_models.json')
            require(set(trained['models']) == set(RUNS) and timestamp(trained['created_utc']) <= timestamp(freeze['created_utc']),
                    'all final models did not freeze before holdout')
            for run in RUNS:
                require(trained['models'][run]['sha256'] == freeze['models'][run]['sha256']
                        == freeze['models']['history_' + run]['sha256'], 'holdout final model substitution')
        cache = linked(args.root, freeze['cache_preparation'])
        require(cache['scored_episodes'] == 0 and cache['models'] == freeze['models']
                and cache['source_sha256'] == freeze['source_sha256'], 'evaluation cache preparation binding differs')
        check_hashes(args.root, cache['evidence_sha256'])
        linked(args.root, cache['preparation_freeze'])
        audit_cache(freeze['cache_snapshot'], (51,) if freeze['phase'] == 'development' else (131, 132))
        paths = sorted(args.input_dir.glob('*.json'))
        records = [read(p) for p in paths]
        for path, record in zip(paths, records):
            original = args.input_dir / '_adapter_raw' / path.name
            require(sha(original) == record['v25']['original_raw_sha256']
                    and read(original) == {k: v for k, v in record.items() if k != 'v25'}, 'adapter changed raw evaluation')
        result['evaluation'] = audit_evaluation_records(records, freeze)
        result['raw_sha256'] = {str(p): sha(p) for p in paths}
        if args.summary:
            compare_summary(read(args.summary), result['evaluation'])
            result['summary_verified'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
