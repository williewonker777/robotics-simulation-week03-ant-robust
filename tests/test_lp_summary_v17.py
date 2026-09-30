"""v17 schema must be audited before frozen v13 physical scoring projection."""

import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from week03_ant.lp_study_v17 import CONTROLLERS, HOLDOUTS, SCHEMA, EVAL_TASK as TASK
from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load('v17_direction_fixture', ROOT / 'tests/test_direction_summary.py')
summary = load('v17_summary_under_test', ROOT / 'scripts/summarize_lp_v17.py')
V16_CONTROL_EQUIVALENTS = {
    'v16_control': 'v14', 'fixed': 'control', 'lp': 'stable',
    'history_original': 'history_original', 'history_fixed': 'history_control',
    'history_lp': 'history_stable',
}


def fixture(controller='history_lp', seconds=16):
    data = old.fixture(V16_CONTROL_EQUIVALENTS[controller], seconds)
    expert = controller.removeprefix('history_')
    data.update(schema=SCHEMA, task=TASK, controller=controller,
                command_mode=None if expert == 'original' else 'conditioned',
                contact_config=copy.deepcopy(summary.CONTACT_CONFIG))
    data['contact_telemetry'] = dict(
        pre_action_first_episode_only=True, reward_unweighted_not_dt_integrated=True,
        active_steps=[seconds * 60], valid_steps=[seconds * 60 - 1],
        no_contact_steps=[0], contact_foot_samples=[seconds * 60 - 1],
        speed_capped_foot_samples=[0], contact_fraction_sum=[(seconds * 60 - 1) / 4],
        contacted_tip_speed_sum=[0.], bounded_cost_sum=[0.],
    )
    data['gate_config'] = asdict(HistoryGateConfig())
    data['posture_config'] = asdict(PostureConfig())
    return data


def test_all_modes_and_horizons_preserve_raw_evidence():
    for controller in CONTROLLERS:
        for seconds in (16, 64):
            data = fixture(controller, seconds)
            original = copy.deepcopy(data)
            rows = summary.audit(data)
            assert len(rows) == 1 and rows[0]['contact']['valid_steps'] == seconds * 60 - 1
            assert data == original


@pytest.mark.parametrize('key,value', [
    ('schema', 'old'), ('task', 'old'), ('controller', 'adaptive'),
    ('command_schema_version', True), ('initial_prefix_sha256', 'short'),
    ('contact_config', {'force_threshold_n': 0}),
])
def test_rejects_metadata_impersonation(key, value):
    data = fixture()
    data[key] = value
    with pytest.raises(ValueError):
        summary.audit(data)


def test_rejects_holdout_substitution_and_contact_fraud():
    data = fixture()
    data.update(phase='holdout', geometry_seed=97, reset_seed=61, num_envs=175,
                experiment_freeze_sha256='a' * 64, evaluation_inputs_sha256='b' * 64,
                terrain_cache_manifest_sha256='c' * 64)
    with pytest.raises(ValueError, match='holdout'):
        summary.audit(data)
    assert HOLDOUTS == ((102, 64), (103, 65))
    data = fixture()
    data['contact_telemetry']['contact_fraction_sum'] = [1.]
    with pytest.raises(ValueError, match='fraction'):
        summary.audit(data)


def test_flat_identity_detects_per_environment_drift():
    data = fixture()
    data['family_indices'] = [0]
    data['level_indices'] = [0]
    rows = summary.audit(data)
    initial = {}
    summary._flat_identity(initial, (102, 64), data, rows)
    data['forward_distance'] = [data['forward_distance'][0] + 1]
    with pytest.raises(ValueError, match='flat'):
        summary._flat_identity(initial, (102, 64), data, rows)


def test_development_proofs_reject_changed_smoke_and_parity_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, 'ROOT', tmp_path)
    monkeypatch.setattr(summary, 'audit', lambda data: [{'steps': 960}])
    directory = tmp_path / 'study'
    directory.mkdir()

    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n')
        return summary.HISTORY.sha(path)

    state = dict(initial_state_sha256={'root_state': 'a' * 64, 'joint_pos': 'b' * 64,
                                       'joint_vel': 'c' * 64, 'observations': 'd' * 64},
                 initial_prefix_sha256='e' * 64, initial_rng_sha256={'cpu': 'f' * 64, 'cuda': '0' * 64})
    fields = list(summary.PARITY_FIELDS)
    raw = dict(state, checkpoint_sha256=summary.V16_CONTROL_SHA, controller='v16_control',
               **{field: i for i, field in enumerate(fields) if field not in state})
    old_sha = write('old.json', dict(raw, controller='control'))
    new_sha = write('new.json', raw)
    probe = dict(configured_targets=['/World/ground/terrain/mesh', '/World/flatPlane/GroundPlane/CollisionPlane'],
                 collision_prims=['/World/ground/terrain/mesh', '/World/flatPlane/GroundPlane/CollisionPlane'],
                 measures=[{'per_family': {family: 1. for family in summary.FAMILIES}}])
    probe_sha = write('probe.json', probe)
    parity = dict(exact_reference_parity=True, development_only=True,
                  two_live_collision_prims=probe['collision_prims'],
                  reference_path='old.json', reference_sha256=old_sha,
                  new_path='new.json', new_sha256=new_sha,
                  sensor_probe_path='probe.json', sensor_probe_sha256=probe_sha,
                  equal_fields=fields)
    parity_sha = write('study/evaluator_parity.json', parity)
    models, records = {}, []
    for controller in CONTROLLERS:
        key = controller.removeprefix('history_')
        models.setdefault(key, dict(checkpoint=f'{key}.pt', sha256=(key[0] * 64)))
        data = dict(state, controller=controller, checkpoint=models[key]['checkpoint'],
                    checkpoint_sha256=models[key]['sha256'], evaluation_plan_sha256='1' * 64,
                    phase='smoke', scenario='mixed', geometry_seed=51, reset_seed=24,
                    num_envs=35, condition={'seconds': 16})
        name = f'smoke_{controller}.json'
        digest = write(name, data)
        records.append(dict(path=name, sha256=digest, controller=controller,
                            checkpoint_sha256=models[key]['sha256']))
    smokes_sha = write('study/final_model_smokes.json', dict(
        passed=True, development_only=True, exact_initial_pairing=True,
        v16_reference_parity_sha256=parity_sha, records=records,
        created_utc='2026-09-23T01:00:00+00:00'))
    frozen = dict(final_model_smokes_sha256=smokes_sha, evaluator_parity_sha256=parity_sha,
                  source_sha256={'docs/experiment_plans/learning_progress_v17.md': '1' * 64},
                  frozen_at='2026-09-23T02:00:00+00:00')
    assert summary.audit_development_proofs(directory, frozen, models)['six_smokes_verified']
    mislabeled = json.loads((tmp_path / 'smoke_history_fixed.json').read_text())
    mislabeled['controller'] = 'fixed'
    changed_digest = write('smoke_history_fixed.json', mislabeled)
    smokes = json.loads((directory / 'final_model_smokes.json').read_text())
    next(item for item in smokes['records'] if item['controller'] == 'history_fixed')['sha256'] = changed_digest
    frozen['final_model_smokes_sha256'] = write('study/final_model_smokes.json', smokes)
    with pytest.raises(ValueError, match='smoke model'):
        summary.audit_development_proofs(directory, frozen, models)
    original = dict(state, controller='history_fixed', checkpoint=models['fixed']['checkpoint'],
                    checkpoint_sha256=models['fixed']['sha256'], evaluation_plan_sha256='1' * 64,
                    phase='smoke', scenario='mixed', geometry_seed=51, reset_seed=24,
                    num_envs=35, condition={'seconds': 16})
    restored_digest = write('smoke_history_fixed.json', original)
    next(item for item in smokes['records'] if item['controller'] == 'history_fixed')['sha256'] = restored_digest
    frozen['final_model_smokes_sha256'] = write('study/final_model_smokes.json', smokes)
    (tmp_path / 'smoke_lp.json').write_text('{}\n')
    with pytest.raises(ValueError, match='smoke raw'):
        summary.audit_development_proofs(directory, frozen, models)
    write('smoke_lp.json', dict(state, controller='lp', checkpoint=models['lp']['checkpoint'],
        checkpoint_sha256=models['lp']['sha256'], evaluation_plan_sha256='1' * 64,
        phase='smoke', scenario='mixed', geometry_seed=51, reset_seed=24, num_envs=35,
        condition={'seconds': 16}))
    (tmp_path / 'probe.json').write_text('{}\n')
    with pytest.raises(ValueError, match='parity raw'):
        summary.audit_development_proofs(directory, frozen, models)


@pytest.mark.parametrize('key', ['episode_strict_one_tile_success', 'episode_strict_all_tiles_success'])
def test_rejects_fabricated_success_arrays(key):
    data = fixture()
    data[key] = [not data[key][0]]
    with pytest.raises(ValueError):
        summary.audit(data)


def test_gate_requires_flat_speed_gain_and_rejects_lane_regression():
    baseline = dict(n=300, one=280, six=175, falls=20, lane=1, world=0,
                    flat_n=50, flat_falls=3, flat_lane=0, flat_world=0,
                    flat_mean_episode_speed=10., flat_mean_v10_duty=0.)
    assert not summary.improvement(baseline, baseline)['passed']
    faster = dict(baseline, flat_mean_episode_speed=10.1)
    assert summary.improvement(faster, baseline)['passed']
    assert not summary.improvement(dict(faster, lane=2), baseline)['passed']
    assert not summary.improvement(baseline, baseline, hybrid=True)['passed']
    assert summary.improvement(dict(baseline, six=176), baseline, hybrid=True)['passed']
    assert not summary.improvement(baseline, baseline, primary=False)['passed']
    assert not summary.improvement(dict(baseline, six=176, world=1), baseline, primary=False)['passed']


def test_pairing_rejects_observation_or_rng_changes():
    data = fixture()
    initial = {}
    summary._paired(initial, (102, 64), data)
    changed = copy.deepcopy(data)
    changed['initial_rng_sha256']['cpu'] = '1' * 64
    with pytest.raises(ValueError, match='initial state'):
        summary._paired(initial, (102, 64), changed)


@pytest.mark.parametrize('path', ['/tmp/external.json', '../escape.json'])
def test_provenance_rejects_external_paths(path):
    with pytest.raises(ValueError, match='provenance'):
        summary._inside(path)


def test_sampling_provenance_requires_two_default_size_lp_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, 'ROOT', tmp_path)
    folder = tmp_path / 'study' / 'training'
    folder.mkdir(parents=True)
    records, models = {}, {}
    for arm in ('fixed', 'lp'):
        data = dict(mode=arm, stage_size=1024, stage_updates=2, stages=[{}, {}],
                    first_episode_exclusions=4096, invalid_nonfinite_count=0,
                    transition_steps=32768000)
        path = folder / f'{arm}_sampling.json'
        path.write_text(json.dumps(data))
        digest = summary.HISTORY.sha(path)
        records[arm] = dict(data, path=str(path.relative_to(tmp_path)), sha256=digest)
        models[arm] = {'sampling_sha256': digest}
    validation = {'sampling_summaries': records}
    summary.audit_sampling_proofs(tmp_path / 'study', validation, models)
    path = folder / 'lp_sampling.json'
    data = json.loads(path.read_text())
    data.update(stage_updates=1, stages=[{}])
    path.write_text(json.dumps(data))
    digest = summary.HISTORY.sha(path)
    records['lp'].update(stage_updates=1, sha256=digest)
    models['lp']['sampling_sha256'] = digest
    with pytest.raises(ValueError, match='two default-size'):
        summary.audit_sampling_proofs(tmp_path / 'study', validation, models)
    records['lp']['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='digest'):
        summary.audit_sampling_proofs(tmp_path / 'study', validation, models)
