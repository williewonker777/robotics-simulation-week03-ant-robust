"""Adversarial v14 metadata, projection, matched-training and provenance tests."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig
from week03_ant.command_study import CONTROLLERS, HOLDOUTS, SCHEMA, TASK

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load('command_posture_fixtures', ROOT / 'tests/test_posture_summary.py')
summary = load('command_summary_under_test', ROOT / 'scripts/summarize_command_v14.py')


def fixture(controller='history_conditioned', seconds=16):
    data = old.fixture(summary.PROJECTED_CONTROLLERS[controller], seconds)
    mode = controller.removeprefix('history_')
    mode = mode if mode in ('masked', 'conditioned') else None
    data.update(schema=SCHEMA, task=TASK, controller=controller, command_mode=mode,
                command_schema_version=1, initial_prefix_sha256='b' * 64,
                initial_rng_sha256={'cpu': 'c' * 64, 'cuda': 'd' * 64})
    data['episode_return'] = [1.]
    data['condition'].update(observations=91, expert_observations=91 if mode else 88)
    return data


def test_all_modes_horizons_and_immutable_projection():
    for controller in CONTROLLERS:
        for seconds in (16, 64):
            data = fixture(controller, seconds)
            untouched = copy.deepcopy(data)
            assert summary.audit(data)[0]['steps'] == seconds * 60
            assert data == untouched
            assert data['condition']['observations'] == 91


@pytest.mark.parametrize('key,value', [
    ('schema', 'old'), ('task', 'changed'), ('controller', 'adaptive'),
    ('command_mode', 'masked'), ('command_mode', None), ('command_schema_version', True),
    ('command_schema_version', 2), ('initial_prefix_sha256', 'invalid'),
    ('initial_rng_sha256', {'cpu': 'a' * 64}), ('phase', 'prepare'),
])
def test_reject_v14_metadata(key, value):
    data = fixture()
    data[key] = value
    with pytest.raises(ValueError):
        summary.audit(data)


@pytest.mark.parametrize('key,value', [
    ('observations', 88), ('observations', 91.), ('expert_observations', 88),
    ('v5_observations', 91), ('actions', 9),
])
def test_reject_dimension_impersonation(key, value):
    data = fixture()
    data['condition'][key] = value
    with pytest.raises(ValueError):
        summary.audit(data)


def test_new_holdout_contract_checked_before_projection():
    data = fixture()
    data.update(phase='holdout', geometry_seed=78, reset_seed=52, num_envs=175,
                experiment_freeze_sha256='a' * 64, evaluation_inputs_sha256='b' * 64,
                terrain_cache_manifest_sha256='c' * 64)
    with pytest.raises(ValueError, match='holdout'):
        summary.audit(data)


def test_old_posture_and_routing_checks_not_bypassed():
    data = fixture()
    data['posture_telemetry']['groups']['all_valid']['reward_sum'][0] = 0
    with pytest.raises(ValueError, match='identity'):
        summary.audit(data)


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
    legacy = {'sources': {f'old/source{i}': file(f'old/source{i}', str(i)) for i in range(89)},
              'models': {f'old/model{i}': file(f'old/model{i}', str(i)) for i in range(6)}}
    models = {name: {'checkpoint': f'{name}.pt', 'sha256': file(f'{name}.pt', name)}
              for name in ('original', 'v13', 'masked', 'conditioned')}
    for arm in ('masked', 'conditioned'):
        models[arm].update(source_checkpoint=f'logs/{arm}/model_249.pt', iteration=249, transitions=32768000,
                           initial_audit_sha256=file(f'study/training/{arm}_initial.json', json.dumps({'arm': arm})))
    initial = {
        'initial_state_sha256': {key: 'a' * 64 for key in summary.HISTORY.INITIAL_KEYS},
        'initial_prefix_sha256': 'b' * 64, 'rng_sha256': {'cpu': 'c' * 64, 'cuda': 'd' * 64},
        'policy_state_sha256': {'actor.0.weight': 'e' * 64}, 'normalized_parameters': {'same': True},
        'dt': 1 / 60, 'num_envs': 4096, 'reward_weight': 1., 'observation_dimensions': [4096, 91],
        'iteration': 0, 'optimizer_state_empty': True, 'default_config_parity': True,
    }
    for arm in ('masked', 'conditioned'):
        models[arm]['initial_audit_sha256'] = file(f'study/training/{arm}_initial.json', json.dumps(dict(initial, arm=arm)))
        file(models[arm]['source_checkpoint'], arm)
    training = dict(
        frozen_at='2026-09-21T22:00:00+00:00',
        source_sha256={key: sources[key] for key in summary.TRAINING_SOURCES}, legacy=legacy,
        initial_checkpoints={arm: dict(checkpoint=f'initial_{arm}.pt', sha256=file(f'initial_{arm}.pt', arm))
                             for arm in ('masked', 'conditioned')},
        starting_checkpoint_sha256=models['original']['sha256'], training_seed=46, geometry=85,
        iterations_per_arm=250, envs=4096, steps_per_env=32, transitions_per_arm=32768000,
        selection='only final249', posture_config=asdict(PostureConfig()), history_config=asdict(HistoryGateConfig()),
        arms=['masked', 'conditioned'], terrain_cache={},
        capacity_sha256=file('study/capacity.json', '{}'), preflight_sha256=file('study/preflight.json', '{}'),
    )
    training_sha = file('study/training_frozen.json', json.dumps(training))
    tags = ['Episode_Reward/adaptive_posture', 'Loss/value_function', 'Loss/surrogate',
            'Loss/prior_loss', 'Train/mean_reward', 'Policy/mean_noise_std']
    validation = {'records': {arm: dict(
        iterations_logged=250, transitions_logged=32768000, all_scalars_finite=True,
        scalar_counts={tag: 250 for tag in tags}, text_log=f'logs/{arm}.log',
        text_log_sha256=file(f'logs/{arm}.log', 'finite logged iterations'),
        event_files_sha256={f'logs/{arm}.events': file(f'logs/{arm}.events', 'finite scalars')},
    ) for arm in ('masked', 'conditioned')}}
    validation_sha = file('study/training_validation.json', json.dumps(validation))
    trained_sha = file('study/trained_models.json', json.dumps({
        'training_validation_sha256': validation_sha,
        'completed_utc': '2026-09-21T23:00:00+00:00',
        'models': {arm: models[arm] for arm in ('masked', 'conditioned')},
        'total_training_transitions': 65536000, 'training_freeze_sha256': training_sha,
    }))
    monkeypatch.setattr(summary, 'START_SHA', models['original']['sha256'])
    monkeypatch.setattr(summary, 'V13_SHA', models['v13']['sha256'])
    monkeypatch.setattr(summary, 'legacy_hashes', lambda: copy.deepcopy(legacy))
    frozen = dict(holdouts=[list(v) for v in HOLDOUTS], controllers=list(CONTROLLERS),
                  gate_config=asdict(HistoryGateConfig()), posture_config=asdict(PostureConfig()),
                  source_sha256=sources, legacy=legacy, models=models,
                  training_freeze_sha256=training_sha, trained_models_sha256=trained_sha,
                  primary=dict(files=12, episodes=2100, seconds=16, envs=175),
                  secondary=dict(files=12, episodes=120, seconds=64, envs=10),
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
                model = models[controller.removeprefix('history_')]
                data.update(checkpoint=model['checkpoint'], checkpoint_sha256=model['sha256'])
                if folder == 'evaluations':
                    cells = [(family, level) for family in range(7) for level in range(5) for _ in range(5)]
                    data['family_indices'] = [family for family, _ in cells]
                    data['level_indices'] = [level for _, level in cells]
                file(f'study/{folder}/{controller}__geometry{geometry}_reset{reset}.json', json.dumps(data))
    return directory


def test_complete_inventory_and_separate_summary(complete_study):
    result = summary.summarize(complete_study)
    assert len(result['files']) == 24
    assert sum(file['episodes'] for file in result['files']) == 2220
    assert result['evaluations']['groups']['conditioned']['n'] == 300
    assert result['horizon']['groups']['conditioned']['n'] == 20
    assert len(result['evaluations']['family_level']['conditioned']) == 35
    assert 'conditional on visited' in result['interpretation']
    assert 'Secondary' in summary.markdown(result)
    assert result['provenance']['evaluation_inputs_sha256'] == summary.HISTORY.sha(complete_study / 'evaluation_inputs.json')
    assert result['provenance']['terrain_cache_manifest_sha256'] == summary.HISTORY.sha(complete_study / 'terrain_cache.json')


@pytest.mark.parametrize('mutation', ['pairing', 'freeze', 'model', 'source', 'missing', 'training', 'phase', 'shape'])
def test_full_audit_rejects_provenance_pairing_and_inventory(complete_study, mutation):
    path = complete_study / 'evaluations/conditioned__geometry86_reset58.json'
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
        path = complete_study / 'horizon/conditioned__geometry86_reset58.json'
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
        (complete_study / 'training/masked_initial.json').write_text('{}')
    else:
        if mutation == 'manifest_freeze':
            trained['training_freeze_sha256'] = 'f' * 64
        elif mutation == 'manifest_budget':
            trained['total_training_transitions'] -= 1
        elif mutation == 'manifest_models':
            trained['models']['masked']['sha256'] = 'f' * 64
        else:
            model = frozen['models']['masked']
            if mutation == 'final_iteration':
                model['iteration'] = 248
            elif mutation == 'arm_budget':
                model['transitions'] -= 1
            elif mutation == 'initial_audit_digest':
                model['initial_audit_sha256'] = 'f' * 64
            elif mutation == 'initial_audit_arm':
                audit_path = complete_study / 'training/masked_initial.json'
                audit_path.write_text(json.dumps({'arm': 'conditioned'}))
                model['initial_audit_sha256'] = summary.HISTORY.sha(audit_path)
            trained['models']['masked'] = copy.deepcopy(model)
        trained_path.write_text(json.dumps(trained))
        frozen['trained_models_sha256'] = summary.HISTORY.sha(trained_path)
        frozen_path.write_text(json.dumps(frozen))
        refresh_freeze_links(complete_study)
    with pytest.raises(ValueError):
        summary.summarize(complete_study)


@pytest.mark.parametrize('key', ['initial_prefix_sha256', 'initial_rng_sha256'])
def test_evaluation_pairing_includes_prefix_and_rng(complete_study, key):
    path = complete_study / 'evaluations/conditioned__geometry86_reset58.json'
    data = json.loads(path.read_text())
    if key == 'initial_rng_sha256':
        data[key]['cuda'] = 'f' * 64
    else:
        data[key] = 'f' * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='paired controllers'):
        summary.summarize(complete_study)


def relink_training(directory, training=None, trained=None, frozen=None):
    frozen_path = directory / 'frozen.json'
    frozen = frozen or json.loads(frozen_path.read_text())
    trained_path = directory / 'trained_models.json'
    trained = trained or json.loads(trained_path.read_text())
    if training is not None:
        (directory / 'training_frozen.json').write_text(json.dumps(training))
        digest = summary.HISTORY.sha(directory / 'training_frozen.json')
        trained['training_freeze_sha256'] = frozen['training_freeze_sha256'] = digest
    trained_path.write_text(json.dumps(trained))
    frozen['trained_models_sha256'] = summary.HISTORY.sha(trained_path)
    frozen_path.write_text(json.dumps(frozen))
    refresh_freeze_links(directory)


@pytest.mark.parametrize('key,value', [
    ('training_seed', 45), ('geometry', 75), ('envs', 64), ('iterations_per_arm', 249),
    ('steps_per_env', 31), ('transitions_per_arm', 1), ('arms', ['conditioned', 'masked']),
    ('starting_checkpoint_sha256', 'f' * 64), ('source_sha256', {}),
    ('initial_checkpoints', {}), ('capacity_sha256', 'f' * 64), ('preflight_sha256', 'f' * 64),
    ('frozen_at', '2026-09-23T00:00:00+00:00'),
])
def test_training_freeze_content_rejected_after_relink(complete_study, key, value):
    training = json.loads((complete_study / 'training_frozen.json').read_text())
    training[key] = value
    relink_training(complete_study, training=training)
    with pytest.raises(ValueError):
        summary.summarize(complete_study)


@pytest.mark.parametrize('key', ['initial_state_sha256', 'initial_prefix_sha256', 'rng_sha256',
                                 'policy_state_sha256', 'normalized_parameters', 'dt'])
def test_training_actual_pairing_rejected_after_relink(complete_study, key):
    path = complete_study / 'training/conditioned_initial.json'
    data = json.loads(path.read_text())
    if isinstance(data[key], dict):
        field = next(iter(data[key]))
        data[key][field] = 'f' * 64
    else:
        data[key] = .1 if key == 'dt' else 'f' * 64
    path.write_text(json.dumps(data))
    frozen = json.loads((complete_study / 'frozen.json').read_text())
    frozen['models']['conditioned']['initial_audit_sha256'] = summary.HISTORY.sha(path)
    trained = json.loads((complete_study / 'trained_models.json').read_text())
    trained['models'] = {arm: frozen['models'][arm] for arm in ('masked', 'conditioned')}
    relink_training(complete_study, frozen=frozen, trained=trained)
    with pytest.raises(ValueError, match='training arms differ'):
        summary.summarize(complete_study)


@pytest.mark.parametrize('mutation', ['primary', 'secondary', 'source_model', 'initial_model',
                                      'completion_time', 'input_time', 'start_time'])
def test_fixed_inventory_source_and_chronology(complete_study, mutation):
    frozen_path = complete_study / 'frozen.json'
    frozen = json.loads(frozen_path.read_text())
    if mutation in ('primary', 'secondary'):
        frozen[mutation]['episodes'] -= 1
        frozen_path.write_text(json.dumps(frozen))
        refresh_freeze_links(complete_study)
    elif mutation == 'source_model':
        (summary.ROOT / frozen['models']['masked']['source_checkpoint']).write_text('changed')
    elif mutation == 'initial_model':
        (summary.ROOT / 'initial_masked.pt').write_text('changed')
    elif mutation == 'completion_time':
        trained = json.loads((complete_study / 'trained_models.json').read_text())
        trained['completed_utc'] = '2026-09-23T00:00:00+00:00'
        relink_training(complete_study, trained=trained)
    elif mutation == 'input_time':
        path = complete_study / 'evaluation_inputs.json'
        inputs = json.loads(path.read_text())
        inputs['created_utc'] = '2026-09-21T00:00:00+00:00'
        path.write_text(json.dumps(inputs))
        refresh_freeze_links(complete_study)
    else:
        path = complete_study / 'evaluations/conditioned__geometry86_reset58.json'
        data = json.loads(path.read_text())
        data['started_utc'] = '2026-09-22T00:15:00+00:00'
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summary.summarize(complete_study)


@pytest.mark.parametrize('key', ['forward_distance', 'episode_return'])
def test_hybrid_raw_flat_identity_rejects_mean_preserving_swap(complete_study, key):
    path = complete_study / 'evaluations/history_conditioned__geometry86_reset58.json'
    data = json.loads(path.read_text())
    before = sum(data[key][:25])
    data[key][0] += .25
    data[key][1] -= .25
    assert sum(data[key][:25]) == before
    if key == 'forward_distance':
        data['maximum_distance_m'][0] += .25
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='per-environment flat'):
        summary.summarize(complete_study)


def test_exact_flat_identity_observable_in_primary_judgment(complete_study):
    result = summary.summarize(complete_study)
    primary = result['evaluations']
    identity = primary['hybrid_flat_raw_identity']
    assert identity['verified'] is True
    assert identity['maps'] == 2
    assert identity['flat_episodes_per_controller'] == 50
    assert identity['controllers'] == ['history_original', 'history_masked', 'history_conditioned']
    assert primary['hybrid_conditioned_vs_masked']['checks']['fixed_v5_per_environment_flat_raw_identity'] is True
    assert 'hybrid_flat_raw_identity' not in result['horizon']


def test_unused_expert_disagreement_not_part_of_flat_identity(complete_study):
    path = complete_study / 'evaluations/history_conditioned__geometry86_reset58.json'
    data = json.loads(path.read_text())
    data['episode_mean_action_disagreement_rms'][0] += .1
    path.write_text(json.dumps(data))
    assert summary.summarize(complete_study)['evaluations']['hybrid_flat_raw_identity']['verified']


@pytest.mark.parametrize('mutation', ['digest', 'arms', 'finite', 'iterations', 'transitions',
                                      'tag_missing', 'scalar_count', 'empty_events', 'log_drift', 'event_drift'])
def test_training_log_validation_provenance(complete_study, mutation):
    path = complete_study / 'training_validation.json'
    validation = json.loads(path.read_text())
    record = validation['records']['conditioned']
    if mutation == 'log_drift':
        (summary.ROOT / record['text_log']).write_text('changed')
    elif mutation == 'event_drift':
        (summary.ROOT / next(iter(record['event_files_sha256']))).write_text('changed')
    else:
        if mutation == 'arms':
            validation['records'].pop('masked')
        elif mutation == 'finite':
            record['all_scalars_finite'] = False
        elif mutation == 'iterations':
            record['iterations_logged'] = 249
        elif mutation == 'transitions':
            record['transitions_logged'] -= 1
        elif mutation == 'tag_missing':
            record['scalar_counts'].pop('Loss/prior_loss')
        elif mutation == 'scalar_count':
            record['scalar_counts']['Loss/prior_loss'] = 249
        elif mutation == 'empty_events':
            record['event_files_sha256'] = {}
        else:
            validation['extra'] = 'changed bytes'
        path.write_text(json.dumps(validation))
        if mutation != 'digest':
            trained = json.loads((complete_study / 'trained_models.json').read_text())
            trained['training_validation_sha256'] = summary.HISTORY.sha(path)
            relink_training(complete_study, trained=trained)
    with pytest.raises(ValueError):
        summary.summarize(complete_study)
