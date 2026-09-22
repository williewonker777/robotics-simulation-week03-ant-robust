#!/usr/bin/env python3
"""Prepare a converged 60D Ant checkpoint for fine-tuning on the v5 terrain lanes.

After 1,000+ PPO iterations the course policies keep an action-noise standard
deviation of 0.003--0.2 and the adaptive learning rate sits at its 1e-5 floor, so
PPO barely explores new terrain. This utility keeps the actor/critic weights,
restores a moderate action noise, clears the stale optimizer state and restarts
the iteration counter so the v5 run numbers its checkpoints from zero.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="60D course-interface checkpoint")
    parser.add_argument("destination", type=Path, help="output checkpoint path")
    parser.add_argument("--std", type=float, default=0.3, help="action-noise standard deviation to restore")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    args = parser.parse_args()
    if args.std <= 0.0:
        parser.error("--std must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    for key in ("actor.0.weight", "critic.0.weight"):
        if state[key].shape[1] != 60:
            raise ValueError(f"{key} must take the 60D course observation, got {tuple(state[key].shape)}")
    previous_std = state["std"].clone()
    state["std"].fill_(args.std)
    optimizer = copy.deepcopy(checkpoint.get("optimizer_state_dict", {}))
    optimizer["state"] = {}
    for group in optimizer.get("param_groups", []):
        group["lr"] = args.learning_rate
    checkpoint["optimizer_state_dict"] = optimizer
    source_iteration = checkpoint.get("iter")
    checkpoint["iter"] = 0
    checkpoint["infos"] = dict(checkpoint.get("infos") or {})
    checkpoint["infos"].update(
        {
            "prepared_for": "Week03-Ant-Rough-Lanes-v5",
            "prepared_from": str(args.source.resolve()),
            "source_iteration": source_iteration,
            "optimizer_state_reset": True,
            "learning_rate": args.learning_rate,
            "action_std": args.std,
        }
    )

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.destination)
    print(f"wrote {args.destination} ({args.destination.stat().st_size} bytes)")
    print(f"source iteration: {source_iteration}")
    print(f"action std: {[round(v, 4) for v in previous_std.tolist()]} -> {args.std}")


if __name__ == "__main__":
    main()
