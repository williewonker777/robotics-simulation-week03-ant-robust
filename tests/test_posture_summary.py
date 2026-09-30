"""Adversarial v13 evidence tests; no simulator or trained outcome assumptions."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from week03_ant.posture_study import CONTROLLERS, HOLDOUTS, SCHEMA, V5_SHA

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load('posture_history_fixtures', ROOT / 'tests/test_history_summary.py')
summary = load('posture_summary_under_test', ROOT / 'scripts/summarize_posture_v13.py')


def fixture(controller='history_adaptive', seconds=16):
    mode = 'history' if controller.startswith('history_') else ('v5' if controller == 'v5' else 'v10')
    data = old.fixture(mode)
    if seconds == 64:
        base = old.base_tests.fixture(seconds=64, mode=data['mode'])
        data.update(base)
    steps = seconds * 60
    data.update(schema=SCHEMA, controller=controller, phase='smoke', scenario='mixed' if seconds == 16 else 'stones',
                v5_sha256=V5_SHA, checkpoint_sha256='a' * 64, checkpoint='model.pt',
                difficulties=[.2, .4, .6, .8, 1.], gate_config=asdict(HistoryGateConfig()),
                posture_config=asdict(PostureConfig()), episode_v10_target_steps=[steps if mode == 'v10' else 0],
                family_names=list(summary.FAMILIES), family_indices=[6], level_indices=[4])
    state = {key: [0.] for key in summary.FIELDS}
    state.update(steps=[steps], low_speed_steps=[0], foot_samples=[steps * 4])
    for key, value in dict(body_clearance=.44, target_height=.44, forward_speed=2.,
                           local_coverage=1., front_coverage=1., flat_speed_bonus=1 / 3, reward=1 / 6).items():
        state[key + '_sum'] = [steps * value]
    empty = {key: [0 if key in summary.COUNTS else 0.] for key in state}
    data['posture_telemetry'] = dict(pre_action_first_episode_only=True, active_steps=[steps],
                                   motion_steps=[steps], low_speed_steps=[0], groups={
        'all_valid': copy.deepcopy(state), 'clear': copy.deepcopy(state),
        'rough': copy.deepcopy(empty), 'intermediate': copy.deepcopy(empty)})
    return data


def test_all_controller_modes_and_horizons():
    for controller in CONTROLLERS:
        for seconds in (16, 64):
            rows = summary.audit(fixture(controller, seconds))
            assert rows[0]['steps'] == seconds * 60
            assert rows[0]['posture']['all_valid']['steps'] == seconds * 60


@pytest.mark.parametrize('key,value', [
    ('schema', 'old'), ('controller', 'unregistered'), ('task', 'changed'),
    ('mode', 'v10'), ('phase', 'prepare'), ('teacher_tensor_identity_verified', False),
    ('v5_sha256', 'a' * 64), ('checkpoint_sha256', 'invalid'),
    ('difficulties', [.2]), ('episode_lengths', []),
])
def test_reject_changed_contract(key, value):
    data = fixture()
    data[key] = value
    with pytest.raises(ValueError):
        summary.audit(data)


def test_reject_config_hash_and_holdout_drift():
    for mutate in (
        lambda d: d['gate_config'].update(enter_fraction=.1),
        lambda d: d['posture_config'].update(clear_height=.3),
        lambda d: d['initial_state_sha256'].pop('observations'),
        lambda d: d.update(phase='holdout', experiment_freeze_sha256='a' * 64, geometry_seed=51, reset_seed=24),
        lambda d: d.update(phase='holdout', experiment_freeze_sha256='a' * 64, geometry_seed=78, reset_seed=52),
    ):
        data = fixture()
        mutate(data)
        with pytest.raises(ValueError):
            summary.audit(data)


@pytest.mark.parametrize('group,key,value', [
    ('all_valid', 'steps', 961), ('all_valid', 'steps', True),
    ('all_valid', 'low_speed_steps', -1), ('all_valid', 'foot_samples', 3841),
    ('all_valid', 'body_clearance_sum', float('nan')),
    ('all_valid', 'reward_sum', 0), ('all_valid', 'foot_cost_sum', 1),
    ('clear', 'body_clearance_sum', 300), ('clear', 'local_coverage_sum', 800),
    ('rough', 'foot_clearance_sum', 1), ('rough', 'steps', 1),
])
def test_malicious_posture_sums_counts_partitions(group, key, value):
    data = fixture()
    data['posture_telemetry']['groups'][group][key] = [value]
    with pytest.raises(ValueError):
        summary.audit(data)


@pytest.mark.parametrize('key,value', [
    ('pre_action_first_episode_only', False), ('active_steps', [959]), ('motion_steps', [961]),
    ('motion_steps', [0]), ('low_speed_steps', [961]), ('low_speed_steps', [1]),
])
def test_reject_global_first_episode_and_motion_occupancy(key, value):
    data = fixture()
    data['posture_telemetry'][key] = value
    with pytest.raises(ValueError):
        summary.audit(data)


def test_unknown_motion_coverage_and_no_imputation():
    data = fixture()
    telemetry = data['posture_telemetry']
    for group in summary.GROUPS:
        for key in summary.FIELDS + summary.COUNTS:
            telemetry['groups'][group][key] = [0 if key in summary.COUNTS else 0.]
    telemetry['motion_steps'] = [900]
    telemetry['low_speed_steps'] = [100]
    result = summary.aggregate(summary.audit(data))
    assert result['unknown_motion_steps'] == 60
    assert result['low_speed_fraction_known_motion'] == 100 / 900
    assert result['motion_coverage'] == 900 / 960
    assert result['posture']['all_valid']['means']['body_clearance'] is None


def test_switch_history_is_reused_not_just_trusted():
    data = fixture()
    history = old.event_data()
    for key in ('episode_switch_steps', 'episode_switch_to_v10', 'episode_switch_count',
                'history_switch_events', 'episode_alpha_sum', 'episode_v10_duty', 'episode_v10_target_steps'):
        data[key] = history[key]
    assert summary.audit(data)
    data['history_switch_events'][0][0]['history']['rough_votes'] = 1
    with pytest.raises(ValueError):
        summary.audit(data)


def aggregate_fixture():
    rows = summary.audit(fixture())
    flat = copy.deepcopy(rows[0])
    flat.update(family='flat', distance=32., falls=0)
    return summary.aggregate([*rows, flat])


def test_actor_and_hybrid_gates_have_different_flat_speed_contracts():
    control = aggregate_fixture()
    actor = dict(control, flat_mean_episode_speed=2.1)
    assert summary.improvement(actor, control)['passed']
    assert not summary.improvement(control, control)['passed']
    control = dict(control, falls=1)
    hybrid = dict(control, falls=0)
    assert summary.improvement(hybrid, control, hybrid=True)['passed']
    assert not summary.improvement(control, control, hybrid=True)['passed']
    assert not summary.improvement(dict(hybrid, flat_mean_v10_duty=.1), control, hybrid=True)['passed']
    assert not summary.improvement(dict(hybrid, flat_mean_episode_speed=1.9), control, hybrid=True)['passed']
    assert not summary.improvement(dict(actor, flat_world=1), control)['passed']


def test_secondary_requires_strict_gain_without_using_flat_speed():
    control = aggregate_fixture()
    control.update(flat_n=0, flat_mean_episode_speed=None, falls=1)
    assert not summary.improvement(control, control, primary=False)['passed']
    assert summary.improvement(dict(control, falls=0), control, primary=False)['passed']


def test_flat_speed_keeps_falls_and_is_mean_per_episode_not_pooled():
    rows = summary.audit(fixture())
    fast, fall = copy.deepcopy(rows[0]), copy.deepcopy(rows[0])
    fast.update(family='flat', distance=32., steps=960)
    fall.update(family='flat', distance=6., steps=60, falls=1)
    result = summary.aggregate([fast, fall])
    assert result['flat_mean_episode_speed'] == 4.
    assert result['flat_falls'] == 1


@pytest.fixture
def complete_study(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    directory = root / 'study'
    directory.mkdir()
    monkeypatch.setattr(summary, 'ROOT', root)

    def file(name, content):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return summary.HISTORY.sha(path)

    sources = {name: file(name, name) for name in summary.SOURCE_FILES}
    legacy = {'sources': {f'old/source{i}': file(f'old/source{i}', str(i)) for i in range(72)},
              'models': {f'old/model{i}': file(f'old/model{i}', str(i)) for i in range(4)}}
    models = {name: {'checkpoint': f'{name}.pt', 'sha256': file(f'{name}.pt', name)}
              for name in ('original', 'control', 'adaptive')}
    for arm in ('control', 'adaptive'):
        models[arm].update(source_checkpoint=f'logs/{arm}/model_249.pt', iteration=249, transitions=32768000,
                           initial_audit_sha256=file(f'study/training/{arm}_initial.json', json.dumps({'arm': arm})))
    training_sha = file('study/training_frozen.json', '{}')
    trained_sha = file('study/trained_models.json', json.dumps({
        'completed_utc': '2026-09-21T23:00:00+00:00',
        'models': {arm: models[arm] for arm in ('control', 'adaptive')},
        'total_training_transitions': 65536000, 'training_freeze_sha256': training_sha,
    }))
    monkeypatch.setattr(summary, 'START_SHA', models['original']['sha256'])
    monkeypatch.setattr(summary, 'legacy_hashes', lambda: copy.deepcopy(legacy))
    frozen = dict(holdouts=[list(v) for v in HOLDOUTS], controllers=list(CONTROLLERS),
                  gate_config=asdict(HistoryGateConfig()), posture_config=asdict(PostureConfig()),
                  source_sha256=sources, legacy=legacy, models=models,
                  training_freeze_sha256=training_sha, trained_models_sha256=trained_sha,
                  frozen_at='2026-09-22T00:00:00+00:00')
    frozen_sha = file('study/frozen.json', json.dumps(frozen))
    cache_sha = file('study/terrain_cache.json', json.dumps({'cache': 'synthetic pinned manifest'}))
    input_sha = file('study/evaluation_inputs.json', json.dumps({
        'experiment_freeze_sha256': frozen_sha,
        'terrain_cache_manifest_sha256': cache_sha,
        'created_utc': '2026-09-22T00:30:00+00:00',
    }))
    for folder, seconds, n in (('evaluations', 16, 175), ('horizon', 64, 10)):
        for controller in CONTROLLERS:
            for geometry, reset in HOLDOUTS:
                data = fixture(controller, seconds)
                for key, value in list(data.items()):
                    if isinstance(value, list) and len(value) == 1:
                        data[key] = [copy.deepcopy(value[0]) for _ in range(n)]
                telemetry = data['posture_telemetry']
                for key in ('active_steps', 'motion_steps', 'low_speed_steps'):
                    telemetry[key] *= n
                for values in telemetry['groups'].values():
                    for key in values:
                        values[key] *= n
                data.update(num_envs=n, phase='holdout', geometry_seed=geometry, reset_seed=reset,
                            experiment_freeze_sha256=frozen_sha, evaluation_inputs_sha256=input_sha,
                            terrain_cache_manifest_sha256=cache_sha, started_utc='2026-09-22T01:00:00+00:00')
                model = models['original' if controller == 'v5' else controller.removeprefix('history_')]
                data.update(checkpoint=model['checkpoint'], checkpoint_sha256=model['sha256'])
                if folder == 'evaluations':
                    cells = [(family, level) for family in range(7) for level in range(5) for _ in range(5)]
                    data['family_indices'] = [family for family, _ in cells]
                    data['level_indices'] = [level for _, level in cells]
                file(f'study/{folder}/{controller}__geometry{geometry}_reset{reset}.json', json.dumps(data))
    return directory


def test_complete_inventory_and_separate_summary(complete_study):
    result = summary.summarize(complete_study)
    assert len(result['files']) == 28
    assert sum(file['episodes'] for file in result['files']) == 2590
    assert result['evaluations']['groups']['adaptive']['n'] == 300
    assert result['horizon']['groups']['adaptive']['n'] == 20
    assert len(result['evaluations']['family_level']['adaptive']) == 35
    assert 'conditional on visited' in result['interpretation']
    assert 'Secondary' in summary.markdown(result)
    assert result['provenance']['evaluation_inputs_sha256'] == summary.HISTORY.sha(complete_study / 'evaluation_inputs.json')
    assert result['provenance']['terrain_cache_manifest_sha256'] == summary.HISTORY.sha(complete_study / 'terrain_cache.json')


@pytest.mark.parametrize('mutation', ['pairing', 'freeze', 'model', 'source', 'missing', 'training', 'phase', 'shape'])
def test_full_audit_rejects_provenance_pairing_and_inventory(complete_study, mutation):
    path = complete_study / 'evaluations/adaptive__geometry78_reset52.json'
    data = json.loads(path.read_text())
    if mutation == 'pairing':
        data['initial_state_sha256']['observations'] = 'f' * 64
    elif mutation == 'freeze':
        data['experiment_freeze_sha256'] = 'f' * 64
    elif mutation == 'model':
        data['checkpoint_sha256'] = 'f' * 64
    elif mutation == 'source':
        (summary.ROOT / summary.SOURCE_FILES[0]).write_text('changed')
    elif mutation == 'missing':
        path.unlink()
    elif mutation == 'training':
        (complete_study / 'training_frozen.json').write_text('changed')
    elif mutation == 'phase':
        data['phase'] = 'smoke'
    elif mutation == 'shape':
        data['num_envs'] = 1
    if mutation != 'missing':
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summary.summarize(complete_study)


@pytest.mark.parametrize('key', ['evaluation_inputs_sha256', 'terrain_cache_manifest_sha256'])
@pytest.mark.parametrize('value', [None, '', 'not-a-hash'])
def test_holdout_requires_input_and_cache_hashes(key, value):
    data = fixture()
    data.update(phase='holdout', geometry_seed=78, reset_seed=52,
                experiment_freeze_sha256='a' * 64, evaluation_inputs_sha256='b' * 64,
                terrain_cache_manifest_sha256='c' * 64)
    data[key] = value
    with pytest.raises(ValueError, match='SHA256'):
        summary.audit(data)


@pytest.mark.parametrize('mutation', ['manifest', 'inputs', 'frozen_link', 'cache_link',
                                      'secondary_result_inputs', 'secondary_result_cache', 'replaced_both'])
def test_evaluation_manifests_pinned_across_phases(complete_study, mutation):
    inputs_path = complete_study / 'evaluation_inputs.json'
    cache_path = complete_study / 'terrain_cache.json'
    inputs = json.loads(inputs_path.read_text())
    if mutation == 'manifest':
        cache_path.write_text('changed cache')
    elif mutation == 'inputs':
        inputs['created_utc'] = '2026-09-23T00:00:00+00:00'
        inputs_path.write_text(json.dumps(inputs))
    elif mutation in ('frozen_link', 'cache_link'):
        key = 'experiment_freeze_sha256' if mutation == 'frozen_link' else 'terrain_cache_manifest_sha256'
        inputs[key] = 'f' * 64
        inputs_path.write_text(json.dumps(inputs))
    elif mutation == 'replaced_both':
        cache_path.write_text('changed cache with updated inputs record')
        inputs['terrain_cache_manifest_sha256'] = summary.HISTORY.sha(cache_path)
        inputs_path.write_text(json.dumps(inputs))
    else:
        path = complete_study / 'horizon/adaptive__geometry78_reset52.json'
        data = json.loads(path.read_text())
        key = 'evaluation_inputs_sha256' if mutation == 'secondary_result_inputs' else 'terrain_cache_manifest_sha256'
        data[key] = 'f' * 64
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summary.summarize(complete_study)


def refresh_freeze_links(directory):
    """Relink synthetic records so a test exercises content, not only stale SHA."""
    frozen_path = directory / 'frozen.json'
    frozen_sha = summary.HISTORY.sha(frozen_path)
    inputs_path = directory / 'evaluation_inputs.json'
    inputs = json.loads(inputs_path.read_text())
    inputs['experiment_freeze_sha256'] = frozen_sha
    inputs_path.write_text(json.dumps(inputs))
    input_sha = summary.HISTORY.sha(inputs_path)
    for folder in ('evaluations', 'horizon'):
        for path in (directory / folder).glob('*.json'):
            data = json.loads(path.read_text())
            data.update(experiment_freeze_sha256=frozen_sha, evaluation_inputs_sha256=input_sha)
            path.write_text(json.dumps(data))


@pytest.mark.parametrize('mutation', ['manifest_bytes', 'manifest_freeze', 'manifest_budget',
                                      'manifest_models', 'final_iteration', 'arm_budget',
                                      'initial_audit_bytes', 'initial_audit_arm', 'initial_audit_digest'])
def test_trained_model_manifest_final_selection_and_training_trace(complete_study, mutation):
    trained_path = complete_study / 'trained_models.json'
    frozen_path = complete_study / 'frozen.json'
    trained = json.loads(trained_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    if mutation == 'manifest_bytes':
        trained['completed_utc'] = 'changed'
        trained_path.write_text(json.dumps(trained))
    elif mutation == 'initial_audit_bytes':
        (complete_study / 'training/control_initial.json').write_text('{}')
    else:
        if mutation == 'manifest_freeze':
            trained['training_freeze_sha256'] = 'f' * 64
        elif mutation == 'manifest_budget':
            trained['total_training_transitions'] -= 1
        elif mutation == 'manifest_models':
            trained['models']['control']['sha256'] = 'f' * 64
        else:
            model = frozen['models']['control']
            if mutation == 'final_iteration':
                model['iteration'] = 248
            elif mutation == 'arm_budget':
                model['transitions'] -= 1
            elif mutation == 'initial_audit_digest':
                model['initial_audit_sha256'] = 'f' * 64
            elif mutation == 'initial_audit_arm':
                audit_path = complete_study / 'training/control_initial.json'
                audit_path.write_text(json.dumps({'arm': 'adaptive'}))
                model['initial_audit_sha256'] = summary.HISTORY.sha(audit_path)
            trained['models']['control'] = copy.deepcopy(model)
        trained_path.write_text(json.dumps(trained))
        frozen['trained_models_sha256'] = summary.HISTORY.sha(trained_path)
        frozen_path.write_text(json.dumps(frozen))
        refresh_freeze_links(complete_study)
    with pytest.raises(ValueError):
        summary.summarize(complete_study)
