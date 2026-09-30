"""v16 schema must be audited before frozen v13 physical scoring projection."""

import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from week03_ant.contact_study import CONTROLLERS, HOLDOUTS, SCHEMA, TASK
from week03_ant.history_gate import HistoryGateConfig
from week03_ant.posture_math import PostureConfig

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load('v16_direction_fixture', ROOT / 'tests/test_direction_summary.py')
summary = load('v16_summary_under_test', ROOT / 'scripts/summarize_contact_v16.py')
V15_EQUIVALENTS = {
    'v15_control': 'v14', 'control': 'control', 'slip': 'stable',
    'history_original': 'history_original', 'history_control': 'history_control',
    'history_slip': 'history_stable',
}


def fixture(controller='history_slip', seconds=16):
    data = old.fixture(V15_EQUIVALENTS[controller], seconds)
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
    assert HOLDOUTS == ((99, 62), (100, 63))
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
    summary._flat_identity(initial, (99, 62), data, rows)
    data['forward_distance'] = [data['forward_distance'][0] + 1]
    with pytest.raises(ValueError, match='flat'):
        summary._flat_identity(initial, (99, 62), data, rows)


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
    fields = [f'field{i}' for i in range(10)]
    raw = dict(state, checkpoint_sha256=summary.V15_SHA, controller='v15_control',
               **{field: i for i, field in enumerate(fields)})
    old_sha = write('old.json', raw)
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
        v15_reference_parity_sha256=parity_sha, records=records,
        created_utc='2026-09-23T01:00:00+00:00'))
    frozen = dict(final_model_smokes_sha256=smokes_sha, evaluator_parity_sha256=parity_sha,
                  source_sha256={'docs/experiment_plans/contact_slip_v16.md': '1' * 64},
                  frozen_at='2026-09-23T02:00:00+00:00')
    assert summary.audit_development_proofs(directory, frozen, models)['six_smokes_verified']
    mislabeled = json.loads((tmp_path / 'smoke_history_control.json').read_text())
    mislabeled['controller'] = 'control'
    changed_digest = write('smoke_history_control.json', mislabeled)
    smokes = json.loads((directory / 'final_model_smokes.json').read_text())
    next(item for item in smokes['records'] if item['controller'] == 'history_control')['sha256'] = changed_digest
    frozen['final_model_smokes_sha256'] = write('study/final_model_smokes.json', smokes)
    with pytest.raises(ValueError, match='smoke model'):
        summary.audit_development_proofs(directory, frozen, models)
    original = dict(state, controller='history_control', checkpoint=models['control']['checkpoint'],
                    checkpoint_sha256=models['control']['sha256'], evaluation_plan_sha256='1' * 64,
                    phase='smoke', scenario='mixed', geometry_seed=51, reset_seed=24,
                    num_envs=35, condition={'seconds': 16})
    restored_digest = write('smoke_history_control.json', original)
    next(item for item in smokes['records'] if item['controller'] == 'history_control')['sha256'] = restored_digest
    frozen['final_model_smokes_sha256'] = write('study/final_model_smokes.json', smokes)
    (tmp_path / 'smoke_slip.json').write_text('{}\n')
    with pytest.raises(ValueError, match='smoke raw'):
        summary.audit_development_proofs(directory, frozen, models)
    write('smoke_slip.json', dict(state, controller='slip', checkpoint=models['slip']['checkpoint'],
        checkpoint_sha256=models['slip']['sha256'], evaluation_plan_sha256='1' * 64,
        phase='smoke', scenario='mixed', geometry_seed=51, reset_seed=24, num_envs=35,
        condition={'seconds': 16}))
    (tmp_path / 'probe.json').write_text('{}\n')
    with pytest.raises(ValueError, match='parity raw'):
        summary.audit_development_proofs(directory, frozen, models)
