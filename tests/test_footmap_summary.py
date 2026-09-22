import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("footmap_summary", Path(__file__).parents[1] / "scripts/summarize_footmap.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def fixture():
    return {
        "num_envs": 5, "completed_episodes": 5, "boundary_evidence_available": True,
        "forward_distance": [60., 60., 60., 60., 100.],
        "episode_terminated": [False, True, False, False, False],
        "episode_out_of_lane": [False, False, True, False, False],
        "episode_world_exit": [False, False, False, True, False],
        "lane_family_indices": [0, 0, 0, 0, 1], "lane_level_indices": [4] * 5,
        "lane_family_names": ["stones", "flat"],
        "episode_returns": [1.] * 5, "episode_lengths": [960] * 5, "step_dt_seconds": 1 / 60,
        "traversal_geometry": {"one_terrain_tile_clearance_m": 13.1,
                               "all_terrain_tiles_clearance_m": 53.1, "lane_containment_foot_margin_m": 1.1},
        "episode_cleared_one_terrain_tile": [True, False, False, False, None],
        "episode_cleared_all_terrain_tiles": [True, False, False, False, None],
        "one_terrain_tile_crossings": 1, "all_terrain_tiles_crossings": 1,
    }


def test_audit_disqualifies_falls_and_both_exits_despite_distance():
    one, six = summary.recompute(fixture())
    assert one == six == [True, False, False, False, None]
    r = summary.aggregate([fixture()])
    assert (r["n"], r["one"], r["six"], r["falls"], r["lane"], r["world"], r["flat_n"]) == (4, 1, 1, 1, 1, 1, 1)


@pytest.mark.parametrize("key,value", [("completed_episodes", 4), ("boundary_evidence_available", False),
                                       ("one_terrain_tile_crossings", 4)])
def test_audit_rejects_missing_or_inconsistent_evidence(key, value):
    data = fixture()
    data[key] = value
    with pytest.raises(ValueError):
        summary.recompute(data)


def test_gate_compares_reference_rates_without_duplicate_episodes():
    r = dict(n=300, one=250, six=80, falls=20, world=0, flat_n=50, flat_falls=2)
    groups = {mode: copy.deepcopy(r) for mode in ("frozen_v5", "blind", "height", "footmap")}
    for mode in ("blind", "height", "footmap"):
        groups[mode] = {key: value * 3 for key, value in groups[mode].items()}
    groups["footmap"]["one"] += 3
    seeds = {str(seed): {"height": {"one": 250}, "footmap": {"one": 251}} for seed in range(3)}
    assert summary.gate(groups, seeds)["passed"]
    groups["footmap"]["six"] -= 1
    assert not summary.gate(groups, seeds)["passed"]
