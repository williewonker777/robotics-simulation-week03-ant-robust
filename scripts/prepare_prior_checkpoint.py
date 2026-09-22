"""Initialize a v10 frozen-teacher checkpoint from the unchanged selected v5."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from week03_ant.foothold_math import OBSERVATIONS
from week03_ant.prior_policy import PRIOR_MODES, PriorActorCritic, warmstart_prior_from_60d
from week03_ant.prior_ppo import PINNED_PPO_SHA256, PINNED_PPO_VERSION


SOURCE_SHA256 = "889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e"


def prepare_checkpoint(source_path, destination, mode, seed=42):
    if mode not in PRIOR_MODES:
        raise ValueError(f"unknown prior mode: {mode}")
    source_path = Path(source_path)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != SOURCE_SHA256:
        raise ValueError("source is not the pinned original v5 checkpoint")

    torch.manual_seed(seed)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    obs = {"policy": torch.zeros(1, OBSERVATIONS)}
    policy = PriorActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["policy"]},
        8,
        prior_mode=mode,
        input_mode="targets",
        init_noise_std=0.2,
    )
    warmstart_prior_from_60d(policy, source["model_state_dict"], std=0.2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": 0,
            "infos": {
                "prior_mode": mode,
                "teacher_coef": 0.0 if mode == "free" else 0.02,
                "source": str(source_path.resolve()),
                "source_sha256": source_sha256,
                "source_iteration": source.get("iter"),
                "seed": seed,
                "observation_dimensions": OBSERVATIONS,
                "teacher_observation_dimensions": 60,
                "observation_layout": {"base": 60, "feet_xyz": 12, "foothold_targets": 16},
                "action_dimensions": 8,
                "action_std": 0.2,
                "reference_std": 0.2,
                "optimizer": {"name": "Adam", "learning_rate": 1e-4},
                "rsl_rl_version": PINNED_PPO_VERSION,
                "ppo_source_sha256": PINNED_PPO_SHA256,
            },
        },
        destination,
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--mode", choices=PRIOR_MODES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare_checkpoint(args.source, args.destination, args.mode, args.seed))


if __name__ == "__main__":
    main()
