"""Adversarial CPU tests for versioned history-selector result auditing."""
import copy
from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest

from week03_ant.history_gate import HistoryGateConfig
from week03_ant.hybrid_gate import PRESETS

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_tests = load("history_base_fixtures", ROOT / "tests/test_hybrid_summary.py")
summary = load("history_summary_under_test", ROOT / "scripts/summarize_history_hybrid.py")


def fixture(controller="history"):
    mode = controller if controller in ("v5", "v10") else "hybrid"
    d = base_tests.fixture(mode=mode)
    d.update(task="Week03-Ant-Prior-Lanes-Eval-v10", difficulties=[.2, .4, .6, .8, 1.],
             episode_v10_target_steps=[960 if controller == "v10" else 0],
             backend_result_sha256="e" * 64, backend_evaluator_sha256="b" * 64,
             backend_result_local="outputs/missing-backend.json", schema="week03_ant_history_switch_v12_v1", base_contract_schema="week03_ant_depth_switch_v11_v1",
             controller=controller, selector_variant="v12_history" if controller == "history" else
             ("v11_instant" if controller == "instant" else "fixed_policy"),
             gate_config=asdict(HistoryGateConfig()) if controller == "history" else asdict(PRESETS["cautious"]),
             teacher_tensor_identity_verified=True, new_training_transitions=0,
             history_switch_events=[[]], source_sha256={p: "a" * 64 for p in summary.NEW_SOURCES},
             original_source_sha256={f"old{i}": "b" * 64 for i in range(63)},
             initial_state_sha256={k: "c" * 64 for k in summary.INITIAL_KEYS})
    return d


def event_data():
    d = fixture()
    d["episode_switch_steps"] = [[12]]
    d["episode_switch_to_v10"] = [[True]]
    d["episode_switch_count"] = [1]
    d["history_switch_events"] = [[dict(step=12, target_v10=True,
        features=dict(enter_span=.3, enter_edge=.01, retain_span=.3, retain_edge=.01,
                      enter_coverage=1., retain_coverage=1., enter_edge_coverage=1., retain_edge_coverage=1.),
        history=dict(rough_votes=12, clear_votes=0, rough_fraction=1., clear_fraction=0.,
                     enter_history_samples=12, exit_history_samples=12, history_samples=12,
                     rough_evidence_seconds=.2, clear_evidence_seconds=0.,
                     consecutive_rough_seconds=.2, consecutive_clear_seconds=0.,
                     unknown_seconds=0., history_expired=False))]]
    consistent_routing(d)
    return d


def consistent_routing(d):
    import struct
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
    for i, length in enumerate(d["episode_lengths"]):
        events = dict(zip(d["episode_switch_steps"][i], d["episode_switch_to_v10"][i]))
        target = False
        alpha = total = 0.0
        count = 0
        for step in range(1, length + 1):
            if d["mode"] in ("v5", "v10"):
                target = d["mode"] == "v10"
                alpha = float(target)
            else:
                target = events.get(step, target)
                alpha = min(1., max(0., f32(alpha + f32((1 if target else -1) / 9))))
            total += alpha
            count += int(target)
        d["episode_alpha_sum"][i] = total
        d["episode_v10_duty"][i] = total / length
        d["episode_v10_target_steps"][i] = count


def test_valid_all_controllers_and_switch_evidence():
    for controller in ("v5", "v10", "instant", "history"):
        assert len(summary.audit(fixture(controller))) == 1
    assert len(summary.audit(event_data())) == 1


@pytest.mark.parametrize("key,value", [
    ("schema", "v11"), ("mode", "v10"), ("selector_variant", "v11_instant"),
    ("teacher_tensor_identity_verified", False), ("new_training_transitions", 1),
    ("base_contract_schema", "unknown"), ("episode_lengths", []),
])
def test_reject_changed_contract_and_corrupt_arrays(key, value):
    d = fixture()
    d[key] = value
    with pytest.raises(ValueError):
        summary.audit(d)


def test_config_identity_and_initial_hash_contract():
    d = fixture()
    d["gate_config"]["enter_min_seconds"] = .1
    with pytest.raises(ValueError):
        summary.audit(d)
    d = fixture()
    d["initial_state_sha256"].pop("observations")
    with pytest.raises(ValueError):
        summary.audit(d)


@pytest.mark.parametrize("field,value", [
    ("rough_votes", 11), ("rough_fraction", .7), ("rough_evidence_seconds", .1),
    ("enter_history_samples", 18), ("history_samples", 13),
    ("consecutive_rough_seconds", .05), ("history_expired", True),
    ("unknown_seconds", .1),
])
def test_reject_short_support_or_inconsistent_history(field, value):
    d = event_data()
    d["history_switch_events"][0][0]["history"][field] = value
    with pytest.raises(ValueError):
        summary.audit(d)


def test_reject_event_telemetry_mismatch_and_invalid_spatial_feature():
    d = event_data()
    d["history_switch_events"][0][0]["step"] = 13
    with pytest.raises(ValueError):
        summary.audit(d)
    d = event_data()
    d["history_switch_events"][0][0]["features"]["enter_span"] = .01
    with pytest.raises(ValueError):
        summary.audit(d)


def test_exposure_denominator_uses_terrain_first_episode_steps_only():
    rows = summary.audit(fixture())
    rows[0].update(switches=2, steps=60)
    other = copy.deepcopy(rows[0]); other.update(steps=180, switches=1)
    flat = copy.deepcopy(rows[0]); flat.update(family="flat", steps=960, switches=100)
    result = summary.aggregate([*rows, other, flat])
    assert result["active_seconds"] == 4
    assert result["switches"] == 3
    assert result["switches_per_100_active_seconds"] == 75
    assert result["flat_active_seconds"] == 16


def test_improvement_requires_safety_and_progress_not_just_fewer_switches():
    rows = summary.audit(fixture())
    rows[0].update(switches=2, steps=960)
    instant = summary.aggregate(rows)
    history = dict(instant, switches=1, switches_per_100_active_seconds=6.25)
    assert summary.improvement(history, instant, primary=False)["passed"]
    history["six"] = 0
    assert not summary.improvement(history, instant, primary=False)["passed"]
    history = dict(instant, switches=1, switches_per_100_active_seconds=6.25, flat_world=1)
    assert not summary.improvement(history, instant, primary=False)["passed"]


def test_initial_state_pairing_hard_fails_even_one_observation_hash():
    initial = {}
    d = fixture()
    summary.check_initial(initial, (72, 46), d)
    changed = copy.deepcopy(d)
    changed["initial_state_sha256"]["observations"] = "d" * 64
    with pytest.raises(ValueError):
        summary.check_initial(initial, (72, 46), changed)


def test_actual_gate_switch_trace_satisfies_audit():
    import torch
    from week03_ant.foothold_math import foothold_grid
    from week03_ant.history_gate import HistoryDepthPolicyGate

    d = fixture()
    gate = HistoryDepthPolicyGate(1, "cpu", 1 / 60, HistoryGateConfig())
    rough = (foothold_grid(device="cpu").reshape(-1, 2)[:, 0] > .5).float()[None] * .3
    valid = torch.ones_like(rough, dtype=torch.bool)
    events = []
    for step in range(1, 151):
        out = gate.step(rough if step <= 60 else torch.zeros_like(rough), valid,
                        torch.zeros(1, 8), torch.ones(1, 8))
        if out["switched"].item():
            events.append(dict(step=step, target_v10=out["target_v10"].item(),
                               features={k: out[k].item() for k in summary.FEATURES},
                               history={k: out[k].item() for k in summary.HISTORY_FIELDS}))
    assert [event["target_v10"] for event in events] == [True, False]
    d["episode_switch_steps"] = [[e["step"] for e in events]]
    d["episode_switch_to_v10"] = [[e["target_v10"] for e in events]]
    d["episode_switch_count"] = [len(events)]
    d["history_switch_events"] = [events]
    consistent_routing(d)
    assert len(summary.audit(d)) == 1


def directory_fixture(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(summary, "ROOT", tmp_path)
    files = [*summary.NEW_SOURCES, *[f"old{i}" for i in range(62)], "scripts/evaluate_hybrid.py"]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    sources = {name: summary.sha(tmp_path / name) for name in summary.NEW_SOURCES}
    old = {name: summary.sha(tmp_path / name) for name in files if name not in summary.NEW_SOURCES}
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen-v10")
    parent = tmp_path / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"frozen-v5")
    checkpoints = {"42": {"checkpoint": "model.pt", "sha256": summary.sha(checkpoint)}}
    v11_path = tmp_path / "artifacts/terrain_demo/hybrid_v11/frozen.json"
    v11_path.parent.mkdir(parents=True)
    v11_path.write_text(json.dumps(dict(source_sha256=old, checkpoints=checkpoints, v5_sha256=summary.sha(parent))))
    frozen = dict(holdouts=[[72, 46], [73, 47]], controllers=list(summary.CONTROLLERS),
                  history_config=asdict(HistoryGateConfig()), instant_config=asdict(PRESETS["cautious"]),
                  source_sha256=sources, v11_source_sha256=old, v11_frozen_sha256=summary.sha(v11_path),
                  checkpoints=checkpoints, v5_sha256=summary.sha(parent), frozen_at="2026-09-22T01:00:00+00:00")
    directory = tmp_path / "results"
    directory.mkdir()
    (directory / "frozen.json").write_text(json.dumps(frozen))
    families = ["flat", "rough", "slope", "stairs", "waves", "obstacles", "stepping_stones"]
    for folder, scenario, seconds, n in (("evaluations", "mixed", 16, 175), ("horizon", "stones", 64, 10)):
        (directory / folder).mkdir()
        for controller in summary.CONTROLLERS:
            for geom, reset in frozen["holdouts"]:
                d = fixture(controller)
                for key, value in list(d.items()):
                    if isinstance(value, list) and len(value) == 1:
                        d[key] = [copy.deepcopy(value[0]) for _ in range(n)]
                d.update(num_envs=n, family_names=families, difficulties=[.2, .4, .6, .8, 1.],
                         family_indices=[i // 25 for i in range(n)] if scenario == "mixed" else [6] * n,
                         level_indices=[i // 5 % 5 for i in range(n)] if scenario == "mixed" else [4] * n,
                         policy_seed=None if controller == "v5" else 42, geometry_seed=geom, reset_seed=reset,
                         scenario=scenario, phase="holdout", checkpoint_sha256=checkpoints["42"]["sha256"],
                         v5_sha256=frozen["v5_sha256"], source_sha256=sources, original_source_sha256=old,
                         v11_frozen_sha256=frozen["v11_frozen_sha256"], backend_evaluator_sha256=old["scripts/evaluate_hybrid.py"], started_utc="2026-09-22T02:00:00+00:00")
                d["condition"].update(seconds=seconds, max_steps=seconds * 60, snapshot_seconds=8 if seconds == 16 else 16)
                d["episode_lengths"] = d["episode_active_steps"] = [seconds * 60] * n
                d["episode_alpha_sum"] = [float(controller == "v10") * seconds * 60] * n
                d["episode_v10_target_steps"] = [seconds * 60 if controller == "v10" else 0] * n
                path = directory / folder / f"{controller}__geometry{geom}_reset{reset}.json"
                path.write_text(json.dumps(d))
    return directory


def test_complete_directory_and_exact_pairing_failure(tmp_path, monkeypatch):
    import json

    directory = directory_fixture(tmp_path, monkeypatch)
    result = summary.summarize(directory)
    assert len(result["files"]) == 16
    assert sum(row["episodes"] for row in result["files"]) == 1480
    assert result["evaluations"]["groups"]["history"]["n"] == 300
    assert result["horizon"]["groups"]["history"]["n"] == 20
    assert not result["evaluations"]["history_vs_instant"]["passed"]
    assert "Per-family" in summary.markdown(result)
    path = directory / "evaluations/history__geometry72_reset46.json"
    d = json.loads(path.read_text())
    d["initial_state_sha256"]["observations"] = "f" * 64
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="different initial"):
        summary.summarize(directory)


def test_switch_to_v10_cannot_have_zero_duty_or_occupancy():
    d = event_data()
    d["episode_v10_duty"] = [0.]
    d["episode_alpha_sum"] = [0.]
    d["episode_v10_target_steps"] = [0]
    with pytest.raises(ValueError):
        summary.audit(d)


@pytest.mark.parametrize("key,value", [("task", "different"), ("difficulties", [.2, 1.])])
def test_reject_task_or_difficulty_contract(key, value):
    d = fixture()
    d[key] = value
    with pytest.raises(ValueError):
        summary.audit(d)


def test_reject_wrong_snapshot_contract():
    d = fixture()
    d["condition"]["snapshot_seconds"] = 7
    with pytest.raises(ValueError):
        summary.audit(d)


def test_target_occupancy_is_checked_independently():
    d = event_data()
    d["episode_v10_target_steps"] = [0]
    with pytest.raises(ValueError, match="occupancy"):
        summary.audit(d)


def test_local_backend_digest_is_optional_but_verified_if_present(tmp_path, monkeypatch):
    directory = directory_fixture(tmp_path, monkeypatch)
    backend = tmp_path / "outputs/missing-backend.json"
    backend.parent.mkdir()
    backend.write_text("corrupted ignored backend")
    with pytest.raises(ValueError, match="backend result digest"):
        summary.summarize(directory)


def test_backend_hash_metadata_must_be_sha256():
    d = fixture()
    d["backend_result_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA256"):
        summary.audit(d)
