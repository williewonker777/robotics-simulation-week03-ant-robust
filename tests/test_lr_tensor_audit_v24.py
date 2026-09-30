"""Corruption tests for the independent CPU v24 optimizer audit."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_lr_tensor', ROOT / 'scripts/audit_lr_v24_training.py')
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture(scope='module')
def states():
    torch.set_num_threads(1)
    original = torch.load(audit.INIT, map_location='cpu', weights_only=False)
    learned = torch.load(ROOT / 'artifacts/terrain_demo/seed_continuation_v22/runs/seed51/model_249.pt',
                         map_location='cpu', weights_only=False)
    teacher = torch.load(audit.V5, map_location='cpu', weights_only=False)['model_state_dict']
    return original, learned, teacher


def trace(groups):
    groups = audit.json_form(groups)
    obs = dict(saved_learning_rate=1.e-4, algorithm_learning_rate=1.e-4,
               group_learning_rates=[1.e-4], schedule='fixed', desired_kl=None,
               parameter_names=list(audit.PARAMETERS), trainable_parameter_names=list(audit.TRAINABLE))
    return dict(lr_condition='high', learning_rate=1.e-4, expected_updates=2, update_count=2,
                adam_step_count=40, teacher_unchanged=True, passed=True,
                initial=deepcopy(obs), final=deepcopy(obs),
                initial_optimizer_groups=deepcopy(groups), final_optimizer_groups=deepcopy(groups),
                updates=[dict(index=i, before=deepcopy(obs), after=deepcopy(obs), step_start=i*20,
                              step_end=(i+1)*20) for i in range(2)],
                adam_steps=[dict(index=i, update_index=i//20, before=deepcopy(obs), after=deepcopy(obs))
                            for i in range(40)])


def test_no_experiment_imports():
    tree = ast.parse((ROOT / 'scripts/audit_lr_v24_training.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module and not node.module.startswith(('week03_ant', 'run_', 'summarize_'))
        elif isinstance(node, ast.Import):
            assert all(not item.name.startswith(('week03_ant', 'run_', 'summarize_')) for item in node.names)


def test_initializer_only_lr_change(states):
    original, _, _ = states
    low = deepcopy(original)
    low['optimizer_state_dict']['param_groups'][0]['lr'] = 1.e-5
    audit.initializer_pair(original, deepcopy(original), low)
    low['model_state_dict']['std'][0] += .01
    with pytest.raises(ValueError, match='tensor differs'):
        audit.initializer_pair(original, deepcopy(original), low)


@pytest.mark.parametrize('field', ['infos', 'adam_state', 'ordering', 'option', 'lr', 'iteration'])
def test_initializer_ancestry_and_optimizer_corruption(states, field):
    original, _, _ = states
    low = deepcopy(original)
    low['optimizer_state_dict']['param_groups'][0]['lr'] = 1.e-5
    if field == 'infos':
        low['infos']['seed'] = 52
    elif field == 'adam_state':
        low['optimizer_state_dict']['state'][0] = {'step': torch.tensor(1.)}
    elif field == 'ordering':
        low['optimizer_state_dict']['param_groups'][0]['params'].reverse()
    elif field == 'option':
        low['optimizer_state_dict']['param_groups'][0]['eps'] = 1.e-5
    elif field == 'lr':
        low['optimizer_state_dict']['param_groups'][0]['lr'] = 1.e-4
    else:
        low['iter'] = 1
    with pytest.raises(ValueError):
        audit.initializer_pair(original, deepcopy(original), low)


def test_historical_learned_checkpoint_cpu_evidence(states):
    original, learned, teacher = states
    result = audit.checkpoint(learned, original, teacher, 1.e-4, 250)
    assert result['adam_steps'] == 5000
    assert result['trainable_optimizer_states'] == 19
    assert result['teacher_tensors_exact_v5'] == 8


@pytest.mark.parametrize('field', ['step', 'lr', 'teacher', 'std', 'coverage', 'moment_shape', 'nonfinite', 'negative'])
def test_checkpoint_corruption_rejected(states, field):
    original, learned, teacher = states
    state = deepcopy(learned)
    optimizer = state['optimizer_state_dict']
    if field == 'step':
        optimizer['state'][0]['step'] -= 1
    elif field == 'lr':
        optimizer['param_groups'][0]['lr'] = 1.e-5
    elif field == 'teacher':
        state['model_state_dict']['teacher.0.bias'][0] += .1
    elif field == 'std':
        state['model_state_dict']['std'][0] = 0
    elif field == 'coverage':
        optimizer['state'].pop(18)
    elif field == 'moment_shape':
        optimizer['state'][0]['exp_avg'] = torch.zeros(1)
    elif field == 'nonfinite':
        optimizer['state'][0]['exp_avg'][0] = float('nan')
    else:
        optimizer['state'][0]['exp_avg_sq'][0] = -1.
    with pytest.raises(ValueError):
        audit.checkpoint(state, original, teacher, 1.e-4, 250)


def test_complete_two_update_lr_trace(states):
    groups = states[0]['optimizer_state_dict']['param_groups']
    result = audit.lr_trace(trace(groups), 1.e-4, 2, groups, groups)
    assert result['checked_observations'] == 86


@pytest.mark.parametrize('field', ['missing_step', 'hidden_drift', 'off_by_one', 'missing_std',
                                  'changed_options', 'short_update', 'wrong_schedule', 'vacuous_counts'])
def test_trace_corruption_rejected(states, field):
    groups = states[0]['optimizer_state_dict']['param_groups']
    value = trace(groups)
    if field == 'missing_step':
        value['adam_steps'].pop()
    elif field == 'hidden_drift':
        value['adam_steps'][10]['after']['group_learning_rates'][0] = 1.e-5
    elif field == 'off_by_one':
        value['adam_steps'][20]['update_index'] = 2
    elif field == 'missing_std':
        value['updates'][1]['before']['trainable_parameter_names'].remove('std')
    elif field == 'changed_options':
        value['final_optimizer_groups'][0]['eps'] = 1.e-5
    elif field == 'short_update':
        value['updates'][0]['step_end'] = 19
    elif field == 'wrong_schedule':
        value['initial']['schedule'] = 'adaptive'
    else:
        value['adam_step_count'] = 0
    with pytest.raises(ValueError):
        audit.lr_trace(value, 1.e-4, 2, groups, groups)


def test_real_historical_scalar_encoding_and_time_axes():
    records = audit.read(ROOT / 'artifacts/terrain_demo/seed_continuation_v22/preflight.json')['records']
    initial = audit.read(ROOT / records['seed51']['initial']['path'])
    result = audit.scalar_series(ROOT / initial['log_dir'], 2, 1.e-4)
    assert len(result) == 38
    assert len(set(result) - audit.TIMING) == 33


@pytest.mark.parametrize('corruption', [None, 'dt', 'reward', 'policy', 'output_directory'])
def test_full_record_cpu_integration_uses_synthetic_v24_provenance(tmp_path, states, corruption):
    """Copied v22 development bytes test the verifier, not a new training claim."""
    import yaml  # type: ignore[import-untyped]
    original, _, teacher = states
    old = audit.read(ROOT / 'artifacts/terrain_demo/seed_continuation_v22/preflight.json')['records']['seed51']
    data = {name: audit.read(ROOT / old[name]['path']) for name in ('initial', 'schedule', 'rollout')}
    oldlog = ROOT / data['initial']['log_dir']
    logdir = tmp_path / 'logs/synthetic'
    (logdir / 'params').mkdir(parents=True)
    for name in ('env.yaml', 'agent.yaml'):
        shutil.copyfile(oldlog / 'params' / name, logdir / 'params' / name)
    agent = yaml.load((logdir / 'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    agent.update(experiment_name='week03_ant_lr_continuation_v24', load_run='v24_high_init')
    env = yaml.load((logdir / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    env.update(log_dir=str(logdir), io_descriptors_output_dir=str(logdir))
    if corruption == 'dt':
        env['sim']['dt'] = '0.5'
    elif corruption == 'reward':
        env['rewards']['progress']['weight'] = '0.5'
    elif corruption == 'policy':
        agent['policy']['activation'] = 'relu'
    elif corruption == 'output_directory':
        env['io_descriptors_output_dir'] = str(tmp_path / 'wrong')
    (logdir / 'params/agent.yaml').write_text(yaml.safe_dump(agent))
    (logdir / 'params/env.yaml').write_text(yaml.safe_dump(env))
    shutil.copyfile(oldlog / 'model_1.pt', logdir / 'model_1.pt')
    initial = data['initial']
    initial.update(lr_condition='high', learning_rate=1.e-4, log_dir='logs/synthetic')
    initial['saved_parameter_sha256'] = {name: audit.sha(logdir / 'params' / (name + '.yaml')) for name in ('agent', 'env')}
    data['lr'] = trace(original['optimizer_state_dict']['param_groups'])
    record = {}
    for name, value in data.items():
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(value))
        record[name] = dict(path=path.name, sha256=audit.sha(path))
    console = tmp_path / 'training.log'
    shutil.copyfile(ROOT / old['learning_log']['text_log'], console)
    events = {}
    for filename in old['learning_log']['event_files_sha256']:
        target = logdir / Path(filename).name
        shutil.copyfile(ROOT / filename, target)
        events[str(target.relative_to(tmp_path))] = audit.sha(target)
    record['learning_log'] = dict(text_log='training.log', text_log_sha256=audit.sha(console), event_files_sha256=events)
    if corruption is not None:
        with pytest.raises(ValueError, match='configuration|directory'):
            audit.audit_record(tmp_path, record, 'high', 51, 256, 2, original, teacher, original)
        return
    result, _, _, _ = audit.audit_record(tmp_path, record, 'high', 51, 256, 2, original, teacher, original)
    assert result['transitions'] == 16384
    assert result['learning_rate_trace']['adam_steps'] == 40


def test_prelearning_projection_is_only_lr_or_iteration(states):
    record = audit.read(ROOT / 'artifacts/terrain_demo/seed_continuation_v22/capacity.json')['records']['seed52']
    high = audit.read(ROOT / record['initial']['path'])
    low = deepcopy(high)
    low['normalized_parameters']['agent']['algorithm']['learning_rate'] = '1.0e-05'
    audit.initial_pair(low, high, project_lr=True)
    wrong = deepcopy(low)
    wrong['normalized_parameters']['env']['seed'] = '51'
    with pytest.raises(ValueError, match='pre-learning pairing'):
        audit.initial_pair(wrong, high, project_lr=True)
    main = deepcopy(low)
    main['normalized_parameters']['agent']['max_iterations'] = '250'
    audit.initial_pair(low, main, project_iterations=True)
    main['rng_sha256']['cpu'] = 'a' * 64
    with pytest.raises(ValueError, match='pre-learning pairing'):
        audit.initial_pair(low, main, project_iterations=True)
