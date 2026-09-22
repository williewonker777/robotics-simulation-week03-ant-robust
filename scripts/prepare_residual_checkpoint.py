"""Initialize a v8 checkpoint without modifying the original v5 artifact."""

import argparse
import hashlib
from pathlib import Path

import torch

from week03_ant.footmap_math import FOOTMAP_OBSERVATIONS
from week03_ant.residual_policy import ResidualActorCritic, warmstart_residual_from_60d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--mode", choices=("blind", "footmap"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    torch.manual_seed(args.seed)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    obs = {"policy": torch.zeros(1, FOOTMAP_OBSERVATIONS)}
    policy = ResidualActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 8, input_mode=args.mode)
    warmstart_residual_from_60d(policy, source["model_state_dict"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iter": 0,
        "infos": {"input_mode": args.mode, "residual_limit": 0.5, "encoder_seed": args.seed,
                  "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  "source_iteration": source.get("iter"), "base_frozen": True},
    }, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
