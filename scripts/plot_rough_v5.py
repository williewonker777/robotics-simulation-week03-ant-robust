#!/usr/bin/env python3
"""Plot the v5 terrain-lane benchmark (fall rate and forward speed per family)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FAMILY_ORDER = ("flat", "rough", "slope", "stairs", "waves", "obstacles", "stepping_stones")
COLORS = ("#9aa5b1", "#d68a3c", "#2f7ab8", "#3a9e6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="JSON written by summarize_rough_v5.py --json")
    parser.add_argument("output", type=Path)
    parser.add_argument("--names", nargs="*", help="display names, one per summary entry")
    args = parser.parse_args()
    summaries = json.loads(args.summary.read_text())
    names = args.names or [summary["label"] for summary in summaries]
    families = [family for family in FAMILY_ORDER if family in summaries[0]["families"]]
    x = np.arange(len(families))
    width = 0.8 / len(summaries)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for index, (summary, name) in enumerate(zip(summaries, names)):
        offset = (index - (len(summaries) - 1) / 2) * width
        falls = [100.0 * summary["families"][family]["fall_rate"] for family in families]
        speed = [summary["families"][family]["speed"] for family in families]
        color = COLORS[index % len(COLORS)]
        axes[0].bar(x + offset, falls, width, label=name, color=color)
        axes[1].bar(x + offset, speed, width, label=name, color=color)
    axes[0].set_ylabel("fall rate (%)")
    axes[0].set_title("Falls per episode (lower is better)")
    axes[1].set_ylabel("mean forward speed (m/s)")
    axes[1].set_title("Forward speed (higher is better)")
    for axis in axes:
        axis.set_xticks(x, [family.replace("_", "\n") for family in families])
        axis.grid(axis="y", alpha=0.3)
        axis.set_axisbelow(True)
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle("v5 terrain-lane benchmark: 5 difficulty levels x 5 envs x 3 seeds per family", fontsize=11)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
