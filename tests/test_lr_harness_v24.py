"""Same-seed pairing, fixed budgets, and fail-closed LR-study boundaries."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from week03_ant import lr_study_v24 as study
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_lr_training_v24 as harness


def initial(seed, condition):
    reference = study.read(study.high.ART / 'capacity.json')['records'][f'seed{seed}']['initial']['path']
    value = study.read(study.ROOT / reference)
    value['lr_condition'] = condition
    value['learning_rate'] = study.learning_rate(condition)
    value['normalized_parameters']['agent']['algorithm']['learning_rate'] = str(study.learning_rate(condition))
    return value


@pytest.mark.parametrize('seed', [51, 52, 53])
def test_same_seed_lr_only_initial_projection(seed):
    high, low = initial(seed, 'high'), initial(seed, 'low')
    before = deepcopy(low)
    harness.paired_initial(high, low, seed)
    assert low == before


@pytest.mark.parametrize('field', ['rng_sha256', 'initial_state_sha256', 'initial_prefix_sha256', 'policy_state_sha256'])
def test_prelearning_mismatch_rejected(field):
    high, low = initial(51, 'high'), initial(51, 'low')
    low[field] = 'changed'
    with pytest.raises(ValueError): harness.paired_initial(high, low, 51)


def test_seed_is_not_projected():
    with pytest.raises(ValueError): harness.paired_initial(initial(51, 'high'), initial(52, 'low'), 51)


def test_only_lr_config_projected():
    high, low = initial(51, 'high'), initial(51, 'low')
    low['normalized_parameters']['agent']['algorithm']['entropy_coef'] = '0.1'
    with pytest.raises(ValueError): harness.paired_initial(high, low, 51)


@pytest.mark.parametrize('name', ['high51', 'low51', 'low52', 'low53'])
def test_main_only_iteration_projection(name):
    condition = 'high' if name == 'high51' else 'low'
    raw = initial(int(name[-2:]), condition)
    projected = harness.project_main_initial(raw, int(name[-2:]))
    assert raw['normalized_parameters']['agent']['max_iterations'] == '2'
    projected['normalized_parameters']['agent']['max_iterations'] = '2'
    assert projected == raw


@pytest.mark.parametrize('condition,seed,envs,iterations', [('high', 52, 4096, 250), ('high', 53, 4096, 250),
    ('low', 52, 256, 2), ('low', 51, 4096, 249), ('low', True, 4096, 250)])
def test_undeclared_training_command_rejected(condition, seed, envs, iterations):
    with pytest.raises(ValueError):
        harness.train_command(seed, Path('i'), Path('s'), condition=condition, envs=envs, iterations=iterations)


def test_declared_command_lr_and_init_bindings():
    for condition in ('high', 'low'):
        command = harness.train_command(51, Path('i'), Path('s'), condition=condition)
        assert command[command.index('--lr-condition') + 1] == condition
        assert command[command.index('--load_run') + 1] == f'v24_{condition}_init'


def test_exact_study_budget_and_inventory():
    assert study.NEW_TRANSITIONS == 3 * 32768000 + 32768000 + 2 * 256 * 32 * 2 + 6 * 4096 * 32 * 2
    assert len(study.CONTROLLERS) == 14 and len(study.COMPARISONS) == 18
    assert len(study.SOURCE_FILES) == 12 and len(study.TRAINING_SOURCES) == 8
    assert study.FIRST_EPISODES == 28 * 175 and study.WINDOW_OBSERVATIONS == 9800
    assert study.EXPECTED_GPU_COMMANDS == 52
    assert study.COMPARISONS[:3] == (('low51', 'seed51'), ('low52', 'seed52'), ('low53', 'seed53'))


def test_outputs_are_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'WORK', tmp_path)
    (tmp_path / 'duplicate.log').write_text('preserve')
    with pytest.raises(FileExistsError): harness.run(['never_execute'], 'duplicate')
    assert (tmp_path / 'duplicate.log').read_text() == 'preserve'


def test_no_low_checkpoint_fallback_to_high():
    assert study.model_key('history_low52') == 'low52'
    assert study.model_learning_rate('history_low52') == 1.e-5
    assert study.model_learning_rate('seed52') == 1.e-4
    assert study.model_learning_rate('parent') is None


def test_actual_horizon_pairing_is_not_postlearning_equality():
    a = {'episode_randomization': {'actual_sha256': 'same'}, 'raw_reward_mean': [1]}
    b = {'episode_randomization': {'actual_sha256': 'same'}, 'raw_reward_mean': [9]}
    harness.paired_randomization(a, b)
    b['episode_randomization']['actual_sha256'] = 'changed'
    with pytest.raises(ValueError): harness.paired_randomization(a, b)


def test_ledger_requires_full_thirteen_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'ART', tmp_path)
    (tmp_path / 'training_commands.jsonl').write_text('')
    with pytest.raises(ValueError, match='count/order'): harness.verify_training_ledger()


def test_historical_same_seed_reference_inventory_exists():
    for phase in ('preflight', 'capacity', 'training_validation'):
        proof = study.read(study.high.ART / f'{phase}.json')
        assert set(proof['records']) == {'seed51', 'seed52', 'seed53'}
        for seed in study.TRAIN_SEEDS:
            initial_proof = study.read(study.ROOT / proof['records'][f'seed{seed}']['initial']['path'])
            assert initial_proof['training_seed'] == seed
            assert initial_proof['normalized_parameters']['agent']['seed'] == str(seed)
