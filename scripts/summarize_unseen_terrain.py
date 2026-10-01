# SPDX-License-Identifier: BSD-3-Clause
"""Audit and summarize the predeclared unseen-layout first-episode experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

CONTROLLERS = ('v5', 'v16_control', 'history_control', 'high53', 'history_high53')
PAIRS = ((901, 101), (902, 102))
PAIR_FIELDS = ('initial_state_sha256', 'initial_prefix_sha256', 'initial_rng_sha256',
               'terrain_config_sha256', 'family_names', 'difficulties', 'family_indices',
               'level_indices', 'source_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def episode_rows(bundle, seconds):
    window = bundle['windows'][str(seconds)]
    n = bundle['num_envs']
    fields = ('forward_distance', 'episode_terminated', 'episode_out_of_lane',
              'episode_world_exit', 'episode_steps')
    routing = window['routing']
    require(all(len(window[k]) == n for k in fields), 'incomplete episode arrays')
    require(all(len(routing[k]) == n for k in ('episode_v10_duty', 'episode_switch_count')),
            'incomplete routing arrays')
    rows = []
    for i in range(n):
        distance, steps = window['forward_distance'][i], window['episode_steps'][i]
        fall, lane, world = (window[k][i] for k in fields[1:4])
        duty, switches = routing['episode_v10_duty'][i], routing['episode_switch_count'][i]
        require(all(type(x) is bool for x in (fall, lane, world)), 'nonboolean safety flag')
        require(type(steps) is int and 0 < steps <= seconds * 60, 'invalid episode length')
        require(math.isfinite(distance) and math.isfinite(duty) and 0 <= duty <= 1,
                'nonfinite distance or invalid duty')
        require(type(switches) is int and switches >= 0, 'invalid switch count')
        controller = bundle['controller']
        if not controller.startswith('history_'):
            require(switches == 0 and duty == (0 if controller == 'v5' else 1),
                    'endpoint controller routing mismatch')
        safe = not (fall or lane or world)
        rows.append(dict(controller=controller, geometry=bundle['geometry_seed'],
            reset=bundle['reset_seed'], env=i, seconds=seconds,
            family=bundle['family_names'][bundle['family_indices'][i]],
            level=bundle['level_indices'][i], distance_m=distance, steps=steps,
            one=safe and f32(distance) >= f32(13.1), six=safe and f32(distance) >= f32(53.1),
            fall=fall, lane=lane, world=world, survival=steps == seconds * 60 and not fall,
            safe_survival=steps == seconds * 60 and safe,
            speed_m_s=distance / (steps * bundle['dt']), duty=duty, switches=switches))
    return rows


def validate_bundles(bundles, plan=None):
    if plan is not None:
        require(plan['controllers'] == list(CONTROLLERS) and plan['pairs'] == [list(p) for p in PAIRS]
                and plan['envs_per_bundle'] == 175 and plan['physical_seconds'] == 64
                and plan['dependent_windows'] == [16, 64], 'plan differs from predeclared experiment')
    indexed = {}
    for bundle in bundles:
        key = (bundle['controller'], bundle['geometry_seed'], bundle['reset_seed'])
        require(key not in indexed, 'duplicate bundle')
        indexed[key] = bundle
        require(bundle['schema'] == 'week03_ant_unseen_layout_v1', 'wrong evidence schema')
        require(bundle['num_envs'] == 175 and bundle['seconds'] == 64 and bundle['scenario'] == 'mixed'
                and bundle['scored_episodes'] == 175 and bundle['new_training_transitions'] == 0,
                'wrong physical experiment')
        require(bundle['dt'] == 1 / 60 and set(bundle['windows']) == {'16', '64'}, 'wrong dependent windows')
        require(bundle['checkpoint_sha256_after'] == bundle['checkpoint_sha256']
                and bundle['source_sha256_after'] == bundle['source_sha256'], 'inputs changed during rollout')
        require(set(bundle['initial_state_sha256']) >= {'root_state', 'joint_pos', 'joint_vel', 'observations'}
                and set(bundle['initial_rng_sha256']) == {'cpu', 'cuda'}, 'missing initial pairing evidence')
        require(len(bundle['family_indices']) == 175 and len(bundle['level_indices']) == 175,
                'incomplete assignments')
        for window in bundle['windows'].values():
            require(window['window_seconds'] in (16, 64), 'window label mismatch')
        require(all(bundle['windows'][str(s)]['window_seconds'] == s for s in (16, 64)), 'swapped windows')
        prefix, full = (bundle['windows'][str(s)] for s in (16, 64))
        for i, steps in enumerate(full['episode_steps']):
            require(prefix['episode_steps'][i] == min(960, steps), 'windows do not share first episode lengths')
            if steps <= 960:
                require(all(prefix[k][i] == full[k][i] for k in ('forward_distance',
                    'episode_terminated', 'episode_out_of_lane', 'episode_world_exit')),
                    'post-terminal state differs between dependent windows')
    expected = {(c, g, r) for c in CONTROLLERS for g, r in PAIRS}
    require(set(indexed) == expected, 'requires exactly five controllers and two declared layouts')
    for geometry, reset in PAIRS:
        reference = indexed[('v5', geometry, reset)]
        for controller in CONTROLLERS:
            candidate = indexed[(controller, geometry, reset)]
            require(all(candidate[k] == reference[k] for k in PAIR_FIELDS),
                    f'initial/config pairing mismatch: {controller}/{geometry}')
    for controller, standalone in (('history_control', 'v16_control'), ('history_high53', 'high53')):
        for g, r in PAIRS:
            require(indexed[(controller, g, r)]['checkpoint_sha256'] == indexed[(standalone, g, r)]['checkpoint_sha256'],
                    'history and standalone checkpoints differ')
    for controller in CONTROLLERS:
        require(len({indexed[(controller, g, r)]['checkpoint_sha256'] for g, r in PAIRS}) == 1,
                'checkpoint differs between layouts')
    return indexed


def aggregate(rows):
    n = len(rows)
    return dict(episodes=n, **{k: sum(row[k] for row in rows) for k in
                ('one', 'six', 'fall', 'lane', 'world', 'survival', 'safe_survival', 'switches')},
                mean_speed_m_s=sum(row['speed_m_s'] for row in rows) / n,
                mean_duty=sum(row['duty'] for row in rows) / n,
                all_duty_zero=all(row['duty'] == 0 for row in rows))


def groups(rows):
    return {name: aggregate([r for r in rows if (r['family'] == 'flat') == (name == 'flat')])
            for name in ('rough', 'flat')}


def paired_comparison(candidate, baseline):
    key = lambda r: (r['geometry'], r['reset'], r['env'])
    left, right = ({key(r): r for r in rows} for rows in (candidate, baseline))
    require(left.keys() == right.keys(), 'unpaired comparison')
    result = {}
    for group in ('rough', 'flat'):
        pairs = [(a, right[k]) for k, a in left.items() if (a['family'] == 'flat') == (group == 'flat')]
        result[group] = {'episodes': len(pairs)}
        for metric in ('one', 'six', 'fall', 'lane', 'world', 'survival'):
            # For adverse events an increase is bad; expose neutral false->true naming.
            result[group][metric] = dict(false_to_true=sum(a[metric] and not b[metric] for a, b in pairs),
                true_to_false=sum(b[metric] and not a[metric] for a, b in pairs))
        result[group]['six_improved'] = result[group]['six']['false_to_true']
        result[group]['six_regressed'] = result[group]['six']['true_to_false']
        result[group]['speed_delta_mean_m_s'] = sum(a['speed_m_s'] - b['speed_m_s'] for a, b in pairs) / len(pairs)
    return result


def summarize(bundles, plan=None):
    indexed = validate_bundles(bundles, plan)
    output = dict(schema='week03_ant_unseen_summary_v1', physical_first_episodes=1750,
        dependent_window_observations=3500, paired_initial_conditions=350,
        statistical_independence_claim=False,
        interpretation='Known terrain families with new randomized layouts, not novel OOD families. '
            '16s/64s are dependent windows of the same first physical episode. No tuning or seed reselection.',
        speed_definition='Mean per-first-episode final distance / (steps * dt); visited-episode proxy, not stock flat ID return.',
        success_definition='Float32 final distance >=13.1/53.1m AND no fall/lane/world flag; not maximum distance.',
        survival_definition='Horizon reached without fall; safe_survival additionally excludes lane/world flags.',
        windows={})
    for seconds in (16, 64):
        rows = [row for bundle in indexed.values() for row in episode_rows(bundle, seconds)]
        by_controller = {}
        for controller in CONTROLLERS:
            selected = [r for r in rows if r['controller'] == controller]
            totals = groups(selected)
            require(totals['rough']['episodes'] == 300 and totals['flat']['episodes'] == 50,
                    'rough/flat denominator mismatch')
            for g, _ in PAIRS:
                for level in range(5):
                    cell = groups([r for r in selected if r['geometry'] == g and r['level'] == level])
                    require(cell['rough']['episodes'] == 30 and cell['flat']['episodes'] == 5,
                            'map/level denominator mismatch')
            by_controller[controller] = dict(total=totals,
                by_map={str(g): groups([r for r in selected if r['geometry'] == g]) for g, _ in PAIRS},
                by_family={f: aggregate([r for r in selected if r['family'] == f]) for f in sorted({r['family'] for r in selected})},
                by_level={str(level): groups([r for r in selected if r['level'] == level]) for level in range(5)})
        comparisons = {}
        for candidate, baseline in (('history_control', 'v16_control'), ('history_high53', 'high53'),
                *((c, 'v5') for c in CONTROLLERS if c != 'v5')):
            comparisons[f'{candidate}_vs_{baseline}'] = paired_comparison(
                [r for r in rows if r['controller'] == candidate], [r for r in rows if r['controller'] == baseline])
        output['windows'][str(seconds)] = dict(controllers=by_controller, paired_comparisons=comparisons)
    return output


def markdown(summary):
    lines = ['# 미사용 무작위 지형 배치 평가', '',
        '알려진 지형 종류의 새 무작위 배치이며 새로운 OOD 지형 종류 검증은 아닙니다.',
        '물리적 첫 에피소드 **1,750개**에서 종속적인 16/64초 관측창 **3,500개**를 평가했습니다.',
        '성공은 최종 거리와 낙상·차선 이탈·월드 이탈 없음으로 재계산했습니다. 후속 리셋은 제외합니다.',
        '평지 속도는 각 첫 에피소드의 최종거리/경과시간 평균이며 stock flat ID return이 아닙니다.', '']
    for seconds, data in summary['windows'].items():
        lines += [f'## {seconds}초', '', '| 컨트롤러 | 험지 1타일/300 | 험지 6타일/300 | 낙상 | 차선 | 월드 | 평지 속도 m/s | 평지 낙상/50 | 평지의 험지 expert 비중 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for controller in CONTROLLERS:
            totals = data['controllers'][controller]['total']
            r, f = totals['rough'], totals['flat']
            lines.append(f"| {controller} | {r['one']} | {r['six']} | {r['fall']} | {r['lane']} | {r['world']} | {f['mean_speed_m_s']:.3f} | {f['fall']} | {f['mean_duty']:.6f} |")
        lines += ['', '동일 체크포인트의 history 전환 대비 단독 모델(험지 6타일 개선/퇴행):']
        for name in ('history_control_vs_v16_control', 'history_high53_vs_high53'):
            p = data['paired_comparisons'][name]['rough']
            lines.append(f"- {name}: {p['six_improved']} / {p['six_regressed']}")
        lines.append('')
    lines += ['지형별 이름으로 모델을 고르는 방식이 아니라, 고정된 깊이 이력 게이트가 평지 기본 정책과 험지 전문가를 혼합합니다.',
              '두 배치·기존 학습 시드에 한정된 결과이며 새 결과로 모델·임계값을 재선택하지 않았습니다.']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('output directory must be new; refusing overwrite')
    files = sorted(args.input_dir.glob('*.json'))
    plan_path = args.input_dir.parent / 'plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    result = summarize([json.loads(p.read_text()) for p in files], plan)
    result['input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if plan is not None:
        result['plan_sha256'] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    with (args.output_dir / 'summary.ko.md').open('x') as stream:
        stream.write(markdown(result))


if __name__ == '__main__':
    main()
