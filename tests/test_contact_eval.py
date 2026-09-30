"""Fresh-map v16 command matrix and phase boundaries without a simulator."""

import importlib.util
from pathlib import Path
import sys

import pytest

from week03_ant.contact_study import CONTROLLERS, HOLDOUTS, ROOT, model_key

sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('v16_eval_runner_test', ROOT / 'scripts/run_contact_eval.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_exact_fresh_map_controller_matrix_and_command_contract():
    assert HOLDOUTS == ((99, 62), (100, 63))
    assert CONTROLLERS == ('v15_control', 'control', 'slip', 'history_original', 'history_control', 'history_slip')
    assert [model_key(value) for value in CONTROLLERS] == [
        'v15_control', 'control', 'slip', 'original', 'control', 'slip',
    ]
    for controller in CONTROLLERS:
        for geometry, reset in HOLDOUTS:
            for scenario, seconds, envs in (('mixed', '16', '175'), ('stones', '64', '10')):
                cmd = runner.eval_command(controller, geometry, reset, scenario, 'holdout',
                                          Path('result.json'), Path('model.pt'))
                assert cmd[cmd.index('--controller') + 1] == controller
                assert cmd[cmd.index('--geometry') + 1] == str(geometry)
                assert cmd[cmd.index('--seed') + 1] == str(reset)
                assert cmd[cmd.index('--scenario') + 1] == scenario
                assert cmd[cmd.index('--seconds') + 1] == seconds
                assert cmd[cmd.index('--num_envs') + 1] == envs


def test_no_default_model_mutation_or_unregistered_output():
    source = (ROOT / 'scripts/run_contact_eval.py').read_text()
    assert 'git push' not in source and 'README.md' not in source
    assert 'save_json(ART / "frozen.json"' in source
    assert 'evaluation_inputs()' in source
    assert 'preholdout_ledger.json' in source
    with pytest.raises(ValueError):
        model_key('history_v15_control')
