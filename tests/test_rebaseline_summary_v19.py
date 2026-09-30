"""CPU-only audits of copies; historical development evidence stays untouched."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from week03_ant import rebaseline_study_v19 as study

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('rebaseline_summary', ROOT / 'scripts/summarize_rebaseline_v19.py')
assert spec is not None and spec.loader is not None
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture(controller='history_original'):
    file = ('final_smoke_history_original.json' if controller in ('v5', 'history_original')
            else 'legacy_parity_v16_control.json')
    data = json.loads((ROOT / 'outputs/terrain_style_v18_20260923' / file).read_text())
    data.update(schema=study.SCHEMA, task=study.TASK, controller=controller,
                evaluation_plan_sha256=study.sha(study.PLAN), new_training_transitions=0)
    return data


@pytest.mark.parametrize('controller', ['history_original', 'v16_control'])
def test_frozen_development_copy_audits_without_mutation(controller):
    data = fixture(controller)
    before = copy.deepcopy(data)
    rows = summary.audit(data)
    assert len(rows) == 35
    assert data == before


@pytest.mark.parametrize('key,value', [
    ('schema', 'old'), ('task', 'old'), ('controller', 'control'),
    ('command_schema_version', True), ('new_training_transitions', 1),
    ('new_training_transitions', False), ('initial_prefix_sha256', 'short'),
    ('checkpoint_sha256', '0' * 64), ('evaluation_plan_sha256', '0' * 64),
    ('mode', 'v5'), ('reset_seed', 68), ('default_config_parity', False),
])
def test_rejects_metadata_corruption(key, value):
    data = fixture()
    data[key] = value
    with pytest.raises(ValueError):
        summary.audit(data)


@pytest.mark.parametrize('key', ['episode_strict_all_tiles_success', 'episode_strict_one_tile_success'])
def test_recomputes_first_episode_success(key):
    data = fixture()
    data[key][0] = not data[key][0]
    with pytest.raises(ValueError):
        summary.audit(data)


def test_rejects_nonfinite_and_contact_fraud():
    data = fixture()
    data['wall_seconds'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        summary.audit(data)
    data = fixture()
    data['contact_telemetry']['valid_steps'][0] += 10000
    with pytest.raises(ValueError):
        summary.audit(data)


def group():
    return dict(n=150, flat_n=25, one=80, six=20, falls=5, lane=4, world=0,
                flat_world=0, flat_falls=0, flat_lane=0, flat_mean_episode_speed=3.)


def test_both_comparisons_allow_equal_flat_speed_but_require_strict_rough_gain():
    ref = group()
    assert not summary.improvement(ref, ref)['passed']
    candidate = dict(ref, six=21)
    assert summary.improvement(candidate, ref)['passed']
    assert not summary.improvement(dict(candidate, flat_mean_episode_speed=2.99), ref)['passed']
    assert not summary.improvement(dict(candidate, flat_lane=1), ref)['passed']
    assert not summary.improvement(dict(candidate, falls=6), ref)['passed']
    assert not summary.improvement(dict(candidate, flat_world=1), ref)['passed']
    assert summary.improvement(dict(candidate, flat_n=0, flat_mean_episode_speed=None),
                               dict(ref, flat_n=0, flat_mean_episode_speed=None), primary=False)['passed']


def test_paired_six_keeps_rough_only_and_gain_loss_direction():
    ref = [dict(family='stairs', level=1, six=0), dict(family='stairs', level=2, six=1),
           dict(family='flat', level=0, six=0)]
    candidate = [dict(r, six=1-r['six']) for r in ref]
    assert summary.paired_six(candidate, ref) == dict(n=2, gains=1, losses=1, both_success=0, both_failure=0)
    with pytest.raises(ValueError):
        summary.paired_six(list(reversed(candidate)), ref)


def test_v5_projection_and_model_mapping():
    assert summary.PROJECT['v5'] == 'v5'
    assert study.model_key('v5') == study.model_key('history_original') == 'original'
    assert study.model_key('v16_control') == study.model_key('history_control') == 'control'
    assert study.policy_mode('v5') == 'v5'
    data = fixture('v5')
    # Relabeling a hybrid without changing its routing must NOT pass as v5.
    with pytest.raises(ValueError):
        summary.audit(data)


def test_flat_identity_ignores_unused_expert_disagreement_but_detects_physical_drift():
    data = fixture()
    rows = summary.audit(data)
    initial = {}
    summary._flat_identity(initial, (51, 24), data, rows)
    flat = next(i for i, row in enumerate(rows) if row['family'] == 'flat')
    data['episode_mean_action_disagreement_rms'][flat] += 1
    summary._flat_identity(initial, (51, 24), data, rows)
    data['forward_distance'][flat] += 1
    with pytest.raises(ValueError, match='flat outcomes'):
        summary._flat_identity(initial, (51, 24), data, rows)


def test_ledger_rejects_missing_duplicate_failed_and_modified_log(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    (tmp_path / 'evaluation_inputs.json').write_text('{}')
    (tmp_path / 'run.log').write_text('success')
    files = [{'path': 'evaluations/result.json'}]
    inputs = {'terrain_cache_manifest_sha256': 'a' * 64}
    record = dict(command=['python', 'evaluate.py', '--phase', 'holdout', '--output', 'evaluations/result.json'],
                  returncode=0, log='run.log', log_sha256=study.sha(tmp_path / 'run.log'),
                  evaluation_inputs_sha256=study.sha(tmp_path / 'evaluation_inputs.json'),
                  terrain_cache_manifest_sha256='a' * 64)
    ledger = tmp_path / 'commands.jsonl'
    ledger.write_text(json.dumps(record) + '\n')
    assert summary.audit_ledger(tmp_path, files, inputs)['verified']
    for records in ([], [record, record], [dict(record, returncode=1)], [dict(record, log_sha256='b' * 64)]):
        ledger.write_text(''.join(json.dumps(r) + '\n' for r in records))
        with pytest.raises(ValueError):
            summary.audit_ledger(tmp_path, files, inputs)


def test_initial_pairing_rejects_rng_drift():
    data = fixture()
    initial = {}
    summary._paired(initial, (51, 24), data)
    data['initial_rng_sha256']['cpu'] = '0' * 64
    with pytest.raises(ValueError, match='initial state'):
        summary._paired(initial, (51, 24), data)


def test_missing_or_extra_raw_matrix_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, 'ART', tmp_path)
    monkeypatch.setattr(study, 'verify_frozen', lambda: {})
    monkeypatch.setattr(study, 'evaluation_inputs', lambda directory: {})
    monkeypatch.setattr(study, 'cache_snapshot', lambda *args: {})
    for name in ('frozen.json', 'evaluation_inputs.json'):
        (tmp_path / name).write_text('{}')
    (tmp_path / 'terrain_cache.json').write_text('{"cache": {}, "preparation": []}')
    with pytest.raises(ValueError, match='file matrix'):
        summary.summarize(tmp_path)
    (tmp_path / 'evaluations').mkdir()
    (tmp_path / 'evaluations' / 'unexpected.json').write_text('{}')
    with pytest.raises(ValueError, match='file matrix'):
        summary.summarize(tmp_path)
