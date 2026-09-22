"""Opt-in v8 launcher around unchanged train/evaluate/demo scripts.

Usage: ../run-python scripts/residual_v8.py train [normal train.py arguments]
The separate entry point preserves every archived v7 implementation hash.
"""

import argparse
from pathlib import Path
import runpy
import sys

from week03_ant.residual_policy import register_residual_policy
import week03_ant.tasks.residual_v8  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "evaluate", "demo"))
    args, forwarded = parser.parse_known_args()
    if args.operation == "train":
        if any(arg.startswith("agent.policy.require_config_match=") for arg in forwarded):
            raise ValueError("training checkpoint/config matching cannot be disabled")
        forwarded.append("agent.policy.require_config_match=true")
    filename = {"train": "train.py", "evaluate": "play_one_episode.py", "demo": "demo_terrains.py"}[args.operation]
    script = Path(__file__).with_name(filename)
    register_residual_policy()
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
