"""Convert an unchanged 60D Ant checkpoint to the explicit v9 contract."""

import argparse
import hashlib
from pathlib import Path

import torch

from week03_ant.foothold_math import OBSERVATIONS
from week03_ant.foothold_policy import FootholdActorCritic, INPUT_MODES, warmstart_from_60d


def prepare_checkpoint(source_path, destination, mode, seed=42):
    if mode not in INPUT_MODES:
        raise ValueError(f"unknown input mode: {mode}")
    source_path = Path(source_path)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    torch.manual_seed(seed)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    obs = {"policy": torch.zeros(1, OBSERVATIONS)}
    policy = FootholdActorCritic(
        obs, {"policy": ["policy"], "critic": ["policy"]}, 8,
        input_mode=mode, init_noise_std=0.2,
    )
    warmstart_from_60d(policy, source["model_state_dict"], std=0.2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": 0,
        "infos": {
            "input_mode": mode,
            "source": str(source_path.resolve()),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_iteration": source.get("iter"),
            "seed": seed,
            "observation_dimensions": OBSERVATIONS,
            "observation_layout": {"base": 60, "feet_xyz": 12, "foothold_targets": 16},
            "action_dimensions": 8,
            "action_std": 0.2,
            "optimizer": {"name": "Adam", "learning_rate": 1e-4},
        },
    }, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--mode", choices=INPUT_MODES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare_checkpoint(args.source, args.destination, args.mode, args.seed))


if __name__ == "__main__":
    main()
