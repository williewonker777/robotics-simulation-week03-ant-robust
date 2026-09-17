"""Pure accumulation utilities shared by the evaluation script and tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EvaluationSummary:
    returns: torch.Tensor
    lengths: torch.Tensor

    def to_dict(self) -> dict[str, object]:
        returns = self.returns.detach().cpu().to(torch.float64)
        lengths = self.lengths.detach().cpu().to(torch.float64)
        return {
            "completed_episodes": int(returns.numel()),
            "episode_return_mean": float(returns.mean()),
            "episode_return_std": float(returns.std(unbiased=False)),
            "episode_return_min": float(returns.min()),
            "episode_return_max": float(returns.max()),
            "episode_length_mean": float(lengths.mean()),
            "episode_length_std": float(lengths.std(unbiased=False)),
            "episode_returns": returns.tolist(),
            "episode_lengths": lengths.tolist(),
        }


class FirstEpisodeAccumulator:
    """Accumulate exactly one episode per vectorized environment."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.returns = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.finished = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        rewards = rewards.reshape(-1)
        dones = dones.reshape(-1).to(torch.bool)
        active = ~self.finished
        self.returns[active] += rewards[active]
        self.lengths[active] += 1
        self.finished |= dones & active

    @property
    def complete(self) -> bool:
        return bool(torch.all(self.finished).item())

    def summary(self) -> EvaluationSummary:
        if not self.complete:
            missing = int((~self.finished).sum().item())
            raise RuntimeError(f"evaluation incomplete: {missing} environments have no finished episode")
        return EvaluationSummary(self.returns.clone(), self.lengths.clone())
