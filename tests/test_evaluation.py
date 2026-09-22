import pytest
import torch

from week03_ant.evaluation import FirstEpisodeAccumulator, LaneTraversalGeometry, classify_lane_traversal


GEOMETRY = LaneTraversalGeometry(tile_length=8.0, terrain_tiles=6, foot_reach=1.1)


def test_accumulates_only_first_episode_per_environment():
    accumulator = FirstEpisodeAccumulator(num_envs=2, device="cpu")
    accumulator.update(torch.tensor([1.0, 2.0]), torch.tensor([True, False]))
    accumulator.update(torch.tensor([100.0, 3.0]), torch.tensor([False, True]))

    summary = accumulator.summary().to_dict()

    assert summary["completed_episodes"] == 2
    assert summary["episode_returns"] == [1.0, 5.0]
    assert summary["episode_lengths"] == [1.0, 2.0]
    assert summary["episode_return_mean"] == pytest.approx(3.0)
    assert summary["episode_return_std"] == pytest.approx(2.0)


def test_rejects_incomplete_batch():
    accumulator = FirstEpisodeAccumulator(num_envs=2, device="cpu")
    accumulator.update(torch.ones(2), torch.tensor([True, False]))

    with pytest.raises(RuntimeError, match="1 environments"):
        accumulator.summary()


@pytest.mark.parametrize("distance", [0.0, 4.0, 13.099])
def test_traversal_does_not_count_standstill_portal_or_insufficient_clearance(distance):
    classified = classify_lane_traversal(
        torch.tensor([distance]), torch.tensor([False]), torch.tensor([False]), GEOMETRY
    )

    assert not bool(classified.cleared_one_tile[0])
    assert not bool(classified.distance_cleared_one_tile[0])


def test_traversal_thresholds_come_from_lane_geometry():
    assert GEOMETRY.one_tile_clearance == pytest.approx(13.1)
    assert GEOMETRY.all_tiles_clearance == pytest.approx(53.1)
    assert GEOMETRY.lane_period == pytest.approx(56.0)

    classified = classify_lane_traversal(
        torch.tensor([13.1, 53.1]),
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        GEOMETRY,
    )
    assert classified.cleared_one_tile.tolist() == [True, True]
    assert classified.cleared_all_tiles.tolist() == [False, True]


def test_fall_after_crossing_is_distance_evidence_not_success():
    classified = classify_lane_traversal(
        torch.tensor([54.0]), torch.tensor([True]), torch.tensor([False]), GEOMETRY
    )

    assert bool(classified.distance_cleared_all_tiles[0])
    assert not bool(classified.cleared_one_tile[0])
    assert not bool(classified.cleared_all_tiles[0])


def test_loop_wrap_counts_complete_periods_and_requires_no_termination():
    classified = classify_lane_traversal(
        torch.tensor([55.999, 56.0, 112.5, 70.0]),
        torch.tensor([False, False, False, True]),
        torch.tensor([False, False, False, False]),
        GEOMETRY,
    )

    assert classified.distance_complete_lane_loops.tolist() == [0, 1, 2, 1]
    assert classified.complete_lane_loops.tolist() == [0, 1, 2, 0]


def test_plane_distance_is_explicitly_inapplicable_to_terrain_crossing():
    classified = classify_lane_traversal(
        torch.tensor([1000.0]), torch.tensor([False]), torch.tensor([True]), GEOMETRY
    )

    assert not bool(classified.applicable[0])
    assert not bool(classified.distance_cleared_one_tile[0])
    assert classified.distance_complete_lane_loops.tolist() == [0]


def test_nonfinite_distances_are_never_traversal_evidence_or_success():
    classified = classify_lane_traversal(
        torch.tensor([float("nan"), float("inf"), float("-inf")]),
        torch.tensor([False, False, False]),
        torch.tensor([False, False, False]),
        GEOMETRY,
    )

    assert classified.distance_cleared_one_tile.tolist() == [False, False, False]
    assert classified.distance_cleared_all_tiles.tolist() == [False, False, False]
    assert classified.distance_complete_lane_loops.tolist() == [0, 0, 0]
    assert classified.cleared_one_tile.tolist() == [False, False, False]
    assert classified.cleared_all_tiles.tolist() == [False, False, False]
    assert classified.complete_lane_loops.tolist() == [0, 0, 0]


@pytest.mark.parametrize("out_of_lane,world_exit", [(True, False), (False, True)])
def test_boundary_exit_keeps_distance_evidence_but_excludes_success(out_of_lane, world_exit):
    classified = classify_lane_traversal(
        torch.tensor([60.0]),
        torch.tensor([False]),
        torch.tensor([False]),
        GEOMETRY,
        out_of_lane=torch.tensor([out_of_lane]),
        world_exit=torch.tensor([world_exit]),
    )

    assert bool(classified.distance_cleared_all_tiles[0])
    assert classified.distance_complete_lane_loops.tolist() == [1]
    assert not bool(classified.cleared_one_tile[0])
    assert classified.complete_lane_loops.tolist() == [0]
