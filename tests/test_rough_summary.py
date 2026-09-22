import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import summarize_rough_v5 as rough_summary  # noqa: E402


def _family(episodes, crossings, *, flat=False, exits=(0, 0)):
    entry = {
        "episodes": episodes,
        "falls": 1,
        "forward_speed_mean": 2.0,
        "episode_return_mean": 3.0,
        "levels": {"0:0.2": {"episodes": episodes, "falls": 1, "forward_speed_mean": 2.0}},
    }
    if flat:
        entry.update(
            terrain_crossing_episodes=0,
            one_terrain_tile_crossings=None,
            all_terrain_tiles_crossings=None,
            complete_lane_loops=None,
            out_of_lane_exits=None,
            world_exits=None,
        )
    else:
        one, all_tiles = crossings
        entry.update(
            terrain_crossing_episodes=episodes,
            one_terrain_tile_crossings=one,
            all_terrain_tiles_crossings=all_tiles,
            complete_lane_loops=all_tiles,
            out_of_lane_exits=exits[0],
            world_exits=exits[1],
        )
    return entry


def _run(terrain_episodes, crossings, *, boundary=True, old=False):
    terrain = _family(terrain_episodes, crossings, exits=(1, 2))
    flat = _family(terrain_episodes, (0, 0), flat=True)
    run = {
        "completed_episodes": terrain_episodes * 2,
        "terminated_episodes": 2,
        "episode_return_mean": 3.0,
        "boundary_evidence_available": boundary,
        "family_breakdown": {"flat": flat, "rough": terrain},
    }
    if old:
        run.pop("boundary_evidence_available")
        for entry in run["family_breakdown"].values():
            for key in list(entry):
                if "terrain" in key or "crossing" in key or key in {
                    "complete_lane_loops", "out_of_lane_exits", "world_exits"
                }:
                    entry.pop(key)
    return run


def _write(directory, name, run):
    (directory / f"policy__{name}.json").write_text(json.dumps(run))


def test_summary_weights_crossings_by_episode_count_and_excludes_flat(tmp_path):
    _write(tmp_path, "lanes_seed24", _run(2, (1, 1)))
    _write(tmp_path, "lanes_seed25", _run(8, (4, 2)))

    summary = rough_summary.summarize(tmp_path, "policy")
    rough = summary["families"]["rough"]
    assert rough["terrain_crossing_episodes"] == 10
    assert rough["one_terrain_tile_crossings"] == 5
    assert rough["one_terrain_tile_crossing_rate"] == 0.5
    assert rough["all_terrain_tiles_crossings"] == 3
    assert rough["all_terrain_tiles_crossing_rate"] == 0.3
    assert rough["out_of_lane_exits"] == 2
    assert rough["world_exits"] == 4
    assert summary["families"]["flat"]["terrain_crossing_episodes"] is None
    assert summary["families"]["flat"]["one_terrain_tile_crossings"] is None
    assert summary["overall"]["terrain_crossing_episodes"] == 10


def test_old_run_makes_traversal_and_boundary_evidence_explicitly_unknown(tmp_path):
    _write(tmp_path, "lanes_seed24", _run(2, (1, 1)))
    _write(tmp_path, "lanes_seed25", _run(8, (4, 2), old=True))

    summary = rough_summary.summarize(tmp_path, "policy")
    rough = summary["families"]["rough"]
    assert rough["terrain_crossing_episodes"] is None
    assert rough["one_terrain_tile_crossings"] is None
    assert rough["boundary_evidence_available"] is False
    assert rough["out_of_lane_exits"] is None
    assert rough["world_exits"] is None
    rendered = rough_summary.markdown([summary])
    assert "unknown" in rendered
    assert "successful episodes / applicable episodes; distinct from survival" in rendered


def test_missing_boundary_evidence_does_not_turn_exits_into_zero(tmp_path):
    _write(tmp_path, "lanes_seed24", _run(2, (1, 1)))
    _write(tmp_path, "lanes_seed25", _run(8, (4, 2), boundary=False))

    summary = rough_summary.summarize(tmp_path, "policy")
    rough = summary["families"]["rough"]
    assert rough["one_terrain_tile_crossings"] == 5
    assert rough["boundary_evidence_available"] is False
    assert rough["out_of_lane_exits"] is None
    assert rough["world_exits"] is None
