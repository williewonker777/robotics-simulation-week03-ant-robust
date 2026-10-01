# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic independent checks for unseen-layout evidence auditing."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/summarize_unseen_terrain.py'
spec = importlib.util.spec_from_file_location('summary', PATH)
assert spec is not None and spec.loader is not None
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def bundles():
    result = []
    for controller in summary.CONTROLLERS:
        for geometry, reset in summary.PAIRS:
            key = 'control' if controller.endswith('control') else 'high53' if controller.endswith('high53') else 'v5'
            b = dict(schema='week03_ant_unseen_layout_v1', controller=controller, geometry_seed=geometry,
                reset_seed=reset, num_envs=175, seconds=64, scenario='mixed', scored_episodes=175,
                new_training_transitions=0, dt=1 / 60, checkpoint_sha256=key, checkpoint_sha256_after=key,
                source_sha256={'a': 'hash'}, source_sha256_after={'a': 'hash'},
                initial_state_sha256={k: 'hash' for k in ('root_state', 'joint_pos', 'joint_vel', 'observations')},
                initial_prefix_sha256='prefix', initial_rng_sha256={'cpu': 'cpu', 'cuda': 'cuda'},
                terrain_config_sha256=str(geometry), family_names=['rough', 'flat'], difficulties=[0, .25, .5, .75, 1],
                family_indices=[0] * 150 + [1] * 25, level_indices=list(range(5)) * 35, windows={})
            for seconds in (16, 64):
                b['windows'][str(seconds)] = dict(window_seconds=seconds, forward_distance=[60.] * 175,
                    episode_terminated=[False] * 175, episode_out_of_lane=[False] * 175,
                    episode_world_exit=[False] * 175, episode_steps=[seconds * 60] * 175,
                    aggregates={'strict_all_tiles_successes': -999},
                    routing=dict(episode_v10_duty=[0. if controller in ('v5', 'history_control', 'history_high53') else 1.] * 175,
                                 episode_switch_count=[0] * 175))
            result.append(b)
    return result


def test_counts_and_aggregates_not_trusted():
    result = summary.summarize(bundles())
    assert result['physical_first_episodes'] == 1750
    assert result['dependent_window_observations'] == 3500
    assert result['paired_initial_conditions'] == 350
    assert result['statistical_independence_claim'] is False
    assert 'independent_episodes' not in result
    total = result['windows']['64']['controllers']['v5']['total']
    assert total['rough']['six'] == 300
    assert total['flat']['episodes'] == 50
    assert total['flat']['mean_speed_m_s'] == 60 / 64
    assert total['flat']['all_duty_zero']
    assert '1,750' in summary.markdown(result)


def test_final_flags_defeat_previously_reached_distance_and_threshold_is_f32():
    b = bundles()[0]
    w = b['windows']['16']
    w['episode_terminated'][0] = True
    w['episode_out_of_lane'][1] = True
    w['episode_world_exit'][2] = True
    w['forward_distance'][3] = summary.f32(53.1)
    w['forward_distance'][4] = 53.099
    rows = summary.episode_rows(b, 16)
    assert not any(row['six'] for row in rows[:3])
    assert rows[3]['six'] and not rows[4]['six']
    assert rows[1]['survival'] and not rows[1]['safe_survival']


@pytest.mark.parametrize('field', summary.PAIR_FIELDS)
def test_pairing_rejected(field):
    data = bundles()
    data[2][field] = deepcopy(data[2][field])
    if isinstance(data[2][field], dict):
        data[2][field]['extra'] = 'changed'
    elif isinstance(data[2][field], list):
        data[2][field][0] = 'changed'
    else:
        data[2][field] = 'changed'
    with pytest.raises(ValueError):
        summary.summarize(data)


@pytest.mark.parametrize('field,value', [('checkpoint_sha256_after', 'bad'), ('num_envs', 1),
    ('new_training_transitions', 1), ('geometry_seed', 903), ('seconds', 16)])
def test_invalid_experiment_rejected(field, value):
    data = bundles()
    data[0][field] = value
    with pytest.raises(ValueError):
        summary.summarize(data)


def test_endpoint_routing_rejected_and_paired_regressions():
    data = bundles()
    data[0]['windows']['16']['routing']['episode_v10_duty'][0] = .1
    with pytest.raises(ValueError, match='routing mismatch'):
        summary.summarize(data)
    data = bundles()
    history = next(b for b in data if b['controller'] == 'history_high53')
    history['windows']['64']['episode_terminated'][0] = True
    result = summary.summarize(data)
    p = result['windows']['64']['paired_comparisons']['history_high53_vs_high53']['rough']
    assert p['six_regressed'] == 1 and p['six_improved'] == 0
    assert p['fall']['false_to_true'] == 1


def test_import_and_nooverwrite(tmp_path):
    code = f"import runpy,sys; runpy.run_path({str(PATH)!r},run_name='test'); assert 'torch' not in sys.modules; assert 'isaaclab' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True)
    inputs, output = tmp_path / 'inputs', tmp_path / 'summary'
    inputs.mkdir()
    for i, b in enumerate(bundles()):
        (inputs / f'{i}.json').write_text(json.dumps(b))
    args = ['--input-dir', str(inputs), '--output-dir', str(output)]
    summary.main(args)
    original = (output / 'summary.json').read_bytes()
    with pytest.raises(SystemExit):
        summary.main(args)
    assert (output / 'summary.json').read_bytes() == original
    with pytest.raises(ValueError):
        summary.summarize(bundles()[:-1])


def test_same_physical_first_episode_required():
    data = bundles()
    data[0]['windows']['64']['episode_steps'][0] = 100
    with pytest.raises(ValueError, match='first episode lengths'):
        summary.summarize(data)

def test_plan_validation():
    with pytest.raises(ValueError, match='plan differs'):
        summary.summarize(bundles(), dict(controllers=[], pairs=[], envs_per_bundle=175,
                                        physical_seconds=64, dependent_windows=[16, 64]))
