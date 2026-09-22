import pytest
import torch

from week03_ant.horizon import HorizonEpisodeTracker


def sample(tracker, step, *, distances, dones=(False, False), finals=(0.0, 0.0),
           terminated=(False, False), lane=(False, False), last_lane=(False, False),
           world=(False, False), last_world=(False, False)):
    tracker.update(
        step=step,
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor(dones),
        reset_terminated=torch.tensor(terminated),
        current_distance=torch.tensor(distances),
        last_distance=torch.tensor(finals),
        current_out_of_lane=torch.tensor(lane),
        last_out_of_lane=torch.tensor(last_lane),
        current_world_exit=torch.tensor(world),
        last_world_exit=torch.tensor(last_world),
    )


def make_tracker(max_steps=4, snapshot_seconds=2.0):
    return HorizonEpisodeTracker(
        2, "cpu", dt=1.0, max_steps=max_steps, snapshot_seconds=snapshot_seconds,
        one_tile_threshold=13.1, all_tiles_threshold=53.1,
    )


def test_autoreset_uses_saved_final_evidence_and_ignores_later_episodes():
    tracker = make_tracker()
    sample(tracker, 1, distances=(10.0, 4.0))
    # Env 0 has already auto-reset to 999 m; its real final evidence is 14 m.
    sample(tracker, 2, distances=(999.0, 8.0), dones=(True, False), finals=(14.0, 0.0),
           terminated=(True, False), last_lane=(True, False))
    sample(tracker, 3, distances=(5000.0, 54.0), lane=(False, False))
    sample(tracker, 4, distances=(9000.0, 0.0), dones=(True, True), finals=(9000.0, 55.0),
           last_world=(True, False))

    result = tracker.to_dict()
    assert result["forward_distance"] == [14.0, 55.0]
    assert result["maximum_distance_m"] == [14.0, 55.0]
    assert result["episode_steps"] == [2, 4]
    assert result["episode_return"] == [2.0, 8.0]
    assert result["episode_terminated"] == [True, False]
    assert result["episode_out_of_lane"] == [True, False]
    assert result["episode_world_exit"] == [False, False]


def test_crossing_then_fall_retains_hit_but_is_not_strict_success():
    tracker = make_tracker()
    sample(tracker, 1, distances=(54.0, 12.0))
    sample(tracker, 2, distances=(0.0, 20.0), dones=(True, False), finals=(60.0, 0.0),
           terminated=(True, False))
    sample(tracker, 3, distances=(0.0, 54.0))
    sample(tracker, 4, distances=(0.0, 0.0), dones=(False, True), finals=(0.0, 54.0))

    result = tracker.to_dict()
    assert result["first_hit_six_seconds"] == [1.0, 3.0]
    assert result["episode_strict_all_tiles_success"] == [False, True]
    assert result["aggregates"]["distance_only_all_tiles_hits"] == 2
    assert result["aggregates"]["strict_all_tiles_successes"] == 1


def test_early_termination_censors_16_second_snapshot():
    tracker = make_tracker()
    sample(tracker, 1, distances=(4.0, 5.0), dones=(True, False), finals=(4.0, 0.0),
           terminated=(True, False))
    sample(tracker, 2, distances=(999.0, 9.0))
    sample(tracker, 3, distances=(999.0, 12.0))
    sample(tracker, 4, distances=(999.0, 0.0), dones=(False, True), finals=(0.0, 14.0))

    result = tracker.to_dict()
    assert result["distance_at_16s"] == [None, 9.0]


def test_boundary_evidence_accumulates_and_blocks_strict_success():
    tracker = make_tracker()
    sample(tracker, 1, distances=(20.0, 20.0), lane=(True, False), world=(False, True))
    sample(tracker, 2, distances=(30.0, 30.0), lane=(False, False), world=(False, False))
    sample(tracker, 3, distances=(54.0, 54.0))
    sample(tracker, 4, distances=(0.0, 0.0), dones=(True, True), finals=(60.0, 60.0),
           last_lane=(True, False), last_world=(False, True))

    result = tracker.to_dict()
    assert result["episode_out_of_lane"] == [True, False]
    assert result["episode_world_exit"] == [False, True]
    assert result["episode_strict_all_tiles_success"] == [False, False]
    assert result["episode_full_64s_survival"] == [True, True]


def test_requires_all_first_episodes_to_finish():
    tracker = make_tracker()
    sample(tracker, 1, distances=(1.0, 1.0))
    with pytest.raises(RuntimeError, match="2 first episodes unfinished"):
        tracker.to_dict()


def test_rejects_snapshot_off_control_step():
    with pytest.raises(ValueError, match="control step"):
        HorizonEpisodeTracker(1, "cpu", dt=0.3, max_steps=100, snapshot_seconds=16.0)
