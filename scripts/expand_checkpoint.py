#!/usr/bin/env python3
"""Expand a 60D Ant checkpoint for the local height-scan observation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="checkpoint with the original observation width")
    parser.add_argument("destination", type=Path, help="expanded checkpoint path")
    parser.add_argument("--extra-dims", type=int, default=54, help="number of appended observation features")
    parser.add_argument("--std", type=float, default=0.20, help="initial policy action-noise standard deviation")
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--reset-iteration", action="store_true",
                        help="Start a new fine-tuning run at iteration zero (legacy default preserves it).")
    args = parser.parse_args()
    if args.extra_dims <= 0:
        parser.error("--extra-dims must be positive")

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    for key in ("actor.0.weight", "critic.0.weight"):
        weight = state[key]
        if weight.ndim != 2 or weight.shape[1] != 60:
            raise ValueError(f"{key} must be a 60D input layer, got {tuple(weight.shape)}")
        padding = weight.new_zeros((weight.shape[0], args.extra_dims))
        state[key] = torch.cat((weight, padding), dim=1)

    state["std"].fill_(args.std)
    optimizer = copy.deepcopy(checkpoint["optimizer_state_dict"])
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = args.learning_rate
    checkpoint["optimizer_state_dict"] = optimizer
    if checkpoint.get("infos") is None:
        checkpoint["infos"] = {}
    checkpoint["infos"].update(
        {
            "expanded_observation_dims": args.extra_dims,
            "expanded_from": str(args.source.resolve()),
        }
    )
    if args.reset_iteration:
        checkpoint["infos"]["source_iteration"] = checkpoint.get("iter")
        checkpoint["iter"] = 0

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.destination)
    print(f"wrote {args.destination} ({args.destination.stat().st_size} bytes)")
    print(f"actor input width: {state['actor.0.weight'].shape[1]}")
    print(f"critic input width: {state['critic.0.weight'].shape[1]}")


if __name__ == "__main__":
    main()
