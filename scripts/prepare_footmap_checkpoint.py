"""Convert an unchanged 60D Ant checkpoint to the explicit v7 policy contract."""

import argparse
import hashlib
from pathlib import Path

import torch
from week03_ant.footmap_math import FOOTMAP_OBSERVATIONS
from week03_ant.footmap_policy import FootMapActorCritic, INPUT_MODES, warmstart_from_60d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--mode", choices=INPUT_MODES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--std", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    if not torch.isfinite(torch.tensor(args.learning_rate)) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    torch.manual_seed(args.seed)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    obs = {"policy": torch.zeros(1, FOOTMAP_OBSERVATIONS)}
    policy = FootMapActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 8, input_mode=args.mode)
    warmstart_from_60d(policy, source["model_state_dict"], args.std)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iter": 0,
        "infos": {"input_mode": args.mode, "source": str(args.source.resolve()),
                  "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  "encoder_seed": args.seed, "source_iteration": source.get("iter"),
                  "observation_dimensions": FOOTMAP_OBSERVATIONS, "action_std": args.std},
    }, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
