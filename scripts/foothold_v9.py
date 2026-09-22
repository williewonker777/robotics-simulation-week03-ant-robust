"""Opt-in v9 launcher preserving original train/evaluate/demo source hashes."""

import argparse
from pathlib import Path
import runpy
import sys

from week03_ant.foothold_policy import INPUT_MODES, register_foothold_policy
import week03_ant.tasks.foothold_v9  # noqa: F401


def training_arguments(forwarded):
    """Keep the persisted input mode and shaping arm inseparable in training."""
    forwarded = list(forwarded)

    def values(key):
        return [arg.split("=", 1)[1] for arg in forwarded if arg.lstrip("+").startswith(key + "=")]

    if values("agent.policy.require_config_match"):
        raise ValueError("training mode guard cannot be disabled")
    modes = values("agent.policy.input_mode")
    if len(modes) > 1 or (modes and modes[0] not in INPUT_MODES):
        raise ValueError("expected one valid v9 input mode")
    mode = modes[0] if modes else "guided"
    expected = -1.0 if mode == "guided" else 0.0
    weights = values("env.rewards.foothold_support.weight")
    if len(weights) > 1 or (weights and float(weights[0]) != expected):
        raise ValueError("support shaping must be -1 for guided and 0 for other modes")
    if not weights:
        forwarded.append(f"env.rewards.foothold_support.weight={expected}")
    return [*forwarded, "agent.policy.require_config_match=true"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "evaluate", "demo"))
    args, forwarded = parser.parse_known_args()
    if args.operation == "train":
        forwarded = training_arguments(forwarded)
    script = Path(__file__).with_name({"train": "train.py", "evaluate": "play_one_episode.py", "demo": "demo_terrains.py"}[args.operation])
    register_foothold_policy()
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
