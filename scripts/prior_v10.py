"""Opt-in v10 launcher; archived train/eval/demo code stays immutable."""

import argparse
from pathlib import Path
import runpy
import sys

from week03_ant.prior_policy import PRIOR_COEFFICIENTS, PRIOR_MODES, register_prior_components
import week03_ant.tasks.prior_v10  # noqa: F401


def training_arguments(forwarded):
    """Fail closed on treatment mismatches and nonzero support shaping."""
    forwarded = list(forwarded)

    def values(key):
        return [arg.split("=", 1)[1] for arg in forwarded if arg.lstrip("+").startswith(key + "=")]

    if values("agent.policy.require_config_match"):
        raise ValueError("training mode guard cannot be disabled")
    modes = values("agent.policy.prior_mode")
    if len(modes) > 1 or (modes and modes[0] not in PRIOR_MODES):
        raise ValueError("expected one valid v10 prior mode")
    mode = modes[0] if modes else "anchored"
    expected = PRIOR_COEFFICIENTS[mode]
    for key, value, numeric in (
        ("agent.policy.input_mode", "targets", False),
        ("agent.algorithm.teacher_coef", expected, True),
        ("env.rewards.foothold_support.weight", 0., True),
        ("agent.policy.class_name", "PriorActorCritic", False),
        ("agent.algorithm.class_name", "PriorPPO", False),
    ):
        supplied = values(key)
        if len(supplied) > 1 or (supplied and (float(supplied[0]) if numeric else supplied[0]) != value):
            raise ValueError(f"v10 training contract mismatch: {key}")
        if not supplied:
            forwarded.append(f"{key}={value}")
    return [*forwarded, "agent.policy.require_config_match=true"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "evaluate", "demo"))
    args, forwarded = parser.parse_known_args()
    if args.operation == "train":
        forwarded = training_arguments(forwarded)
    script = Path(__file__).with_name({
        "train": "train.py", "evaluate": "play_one_episode.py", "demo": "demo_terrains.py",
    }[args.operation])
    register_prior_components()
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
