import pytest
import torch

from week03_ant.evaluation import FirstEpisodeAccumulator


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
