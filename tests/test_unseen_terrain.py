# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only contracts for the new opt-in unseen-layout runner."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/evaluate_unseen_terrain.py'
spec = importlib.util.spec_from_file_location('unseen', PATH)
assert spec is not None and spec.loader is not None
unseen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unseen)


def argv(tmp_path):
    return ['--controller', 'history_high53', '--geometry', '901', '--seed', '101',
            '--output', str(tmp_path / 'result.json')]


def test_import_has_no_simulator_or_tensor_import():
    code = f"import runpy,sys; runpy.run_path({str(PATH)!r}, run_name='test'); assert not any(x.startswith(('isaac','torch')) for x in sys.modules)"
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('name,expected', [
    ('v5', ('original', 'v5', None)),
    ('v16_control', ('control', 'v10', 'conditioned')),
    ('history_control', ('control', 'hybrid', 'conditioned')),
    ('high53', ('seed53', 'v10', 'conditioned')),
    ('history_high53', ('seed53', 'hybrid', 'conditioned')),
])
def test_bindings(name, expected):
    assert unseen.controller_binding(name) == expected


@pytest.mark.parametrize('extra', [
    ['--geometry', '-1'], ['--seed', '-1'], ['--num-envs', '0'],
    ['--num-envs', '36'], ['--seconds', '32'], ['--checkpoint', 'other.pt'],
    ['--prepare', '--record', 'demo.mp4'], ['--record', 'demo.avi'],
])
def test_invalid_requests(tmp_path, extra):
    with pytest.raises(SystemExit):
        unseen.parse_args(argv(tmp_path) + extra)


def test_new_paths_and_scenarios(tmp_path):
    for scenario in ('obstacles', 'stones', 'flat'):
        args = unseen.parse_args(argv(tmp_path) + ['--scenario', scenario, '--num-envs', '1'])
        assert args.level == 4
    for path in ('result.json', 'demo.mp4'):
        (tmp_path / path).write_text('preserve')
    with pytest.raises(SystemExit):
        unseen.parse_args(argv(tmp_path))
    with pytest.raises(SystemExit):
        unseen.parse_args(argv(tmp_path) + ['--output', str(tmp_path / 'new.json'),
                                           '--record', str(tmp_path / 'demo.mp4')])
    assert (tmp_path / 'result.json').read_text() == 'preserve'


def test_schema_and_caption(tmp_path):
    args = unseen.parse_args(argv(tmp_path))
    lines = '\n'.join(unseen.caption(args, 'obstacles', 60, .5, 2, 8.))
    assert '901' in lines and '101' in lines and 'resets=2' in lines
    assert 'UNCUT' in lines and 'NOT benchmark successes' in lines
    assert args.kit_args == '--/renderer/multiGpu/enabled=false'
    assert 'unseen' in unseen.SCHEMA and 'v19' not in unseen.SCHEMA


def test_window_renaming():
    class Tracker:
        def to_dict(self):
            return {'distance_at_16s': [1], 'episode_full_64s_survival': [True],
                    'aggregates': {'full_64s_survivals': 1}}
    data = unseen.single_window(Tracker())
    assert data['window_seconds'] == 16
    assert data['distance_at_snapshot_m'] == [1]
    assert data['aggregates'] == {'full_horizon_survivals': 1}


def test_runtime_contracts_are_explicit():
    source = PATH.read_text()
    assert 'PairedHorizonTracker(args.num_envs' in source
    assert 'raw.render(recompute=True)' in source
    assert 'raw.sim.render()' in source
    assert 'torch.equal(state, robot.root_state_w)' in source
    assert "args.output.open('x')" in source
    assert "args.record.open('xb')" in source
    assert 'policy.eval().requires_grad_(False)' in source
    assert source.count('env.get_observations()') == 2
    assert 'model_249.pt' in source
