#!/usr/bin/env python3
"""Prepare a 114D terrain checkpoint for the v4 recovery curriculum.

The v4 scanner is shifted forward but keeps the v3 tensor shape.  This utility
preserves the learned policy weights, resets PPO's stale adaptive optimizer state,
and restores enough action noise for the policy to discover gap/pit recovery
motions.  ``--zero-scan`` is available for experiments where the old centered
scanner semantics should be ignored completely.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="v3 checkpoint with a 114D policy input")
    parser.add_argument("destination", type=Path, help="output checkpoint path")
    parser.add_argument("--std", type=float, default=0.20, help="initial action-noise standard deviation")
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument(
        "--zero-scan",
        action="store_true",
        help="zero the final 54 scanner columns before the first v4 update",
    )
    args = parser.parse_args()
    if args.std <= 0.0:
        parser.error("--std must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    for key in ("actor.0.weight", "critic.0.weight"):
        weight = state[key]
        if weight.ndim != 2 or weight.shape[1] != 114:
            raise ValueError(f"{key} must be a 114D input layer, got {tuple(weight.shape)}")
        if args.zero_scan:
            weight[:, 60:] = 0.0

    state["std"].fill_(args.std)
    optimizer = copy.deepcopy(checkpoint.get("optimizer_state_dict", {}))
    optimizer["state"] = {}
    for group in optimizer.get("param_groups", []):
        group["lr"] = args.learning_rate
    checkpoint["optimizer_state_dict"] = optimizer
    if checkpoint.get("infos") is None:
        checkpoint["infos"] = {}
    checkpoint["infos"].update(
        {
            "prepared_for": "Week03-Ant-Terrain-Extreme-Recovery-v4",
            "prepared_from": str(args.source.resolve()),
            "scanner_layout": "forward_offset_x_0.60",
            "scanner_columns_zeroed": bool(args.zero_scan),
            "optimizer_state_reset": True,
            "learning_rate": args.learning_rate,
        }
    )

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.destination)
    print(f"wrote {args.destination} ({args.destination.stat().st_size} bytes)")
    print(f"actor input width: {state['actor.0.weight'].shape[1]}")
    print(f"critic input width: {state['critic.0.weight'].shape[1]}")
    print(f"action std: {float(state['std'].mean()):.4f}")
    print(f"scanner columns zeroed: {args.zero_scan}")


if __name__ == "__main__":
    main()
