#!/usr/bin/env python3
"""Publication-only plots of audited v25 results; never selects a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit_teammate_v25_raw import compare_summary

ARMS = ("control", "recovery", "combined")
SEEDS = (61, 62, 63)
COLORS = ("#64748b", "#059669", "#7c3aed")


def plot_window(summary, seconds, output):
    controllers = summary["windows"][str(seconds)]["controllers"]
    metrics = (("rough", "six", "Strict six-tile success (%)"),
               ("rough", "fall", "Falls (%)"),
               ("rough", "lane", "Lane exits (%)"),
               ("flat", "mean_speed_m_s", "Flat mean speed (m/s)"))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
    for row, prefix in enumerate(("", "history_")):
        refs = ("v16_control", "high53") if not prefix else ("history_control", "history_high53")
        for column, (group, metric, label) in enumerate(metrics):
            axis = axes[row, column]
            def value(controller):
                cell = controllers[controller]["total"][group]
                return cell[metric] if metric == "mean_speed_m_s" else cell[metric] * 100 / cell["episodes"]
            for index, arm in enumerate(ARMS):
                values = [value(f"{prefix}{arm}{seed}") for seed in SEEDS]
                mean = sum(values) / len(values)
                axis.bar(index, mean, color=COLORS[index], alpha=.72, width=.62)
                axis.errorbar(index, mean, yerr=[[mean - min(values)], [max(values) - mean]],
                              fmt="none", color="#334155", capsize=5)
                for shift, datum in zip((-.12, 0, .12), values):
                    axis.scatter(index + shift, datum, color="#0f172a", s=20, zorder=3)
            for ref, color, style, name in zip(refs, ("#dc2626", "#d97706"), ("--", ":"),
                                              ("v16 parent", "v22 high53")):
                axis.axhline(value(ref), color=color, linestyle=style, linewidth=1.4, label=name)
            axis.set_xticks(range(3), ARMS, rotation=12)
            axis.set_title(label)
            axis.set_ylim(bottom=0)
            if metric == "six":
                axis.set_ylim(0, 100)
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
            if column == 0:
                axis.set_ylabel("Solo" if not prefix else "Unchanged history gate")
            if column == 3:
                axis.legend(fontsize=8)
    fig.suptitle(f"v25 | {seconds}s of the same first physical episode | heldout maps 131 / 132", fontsize=14)
    fig.text(.5, .015, "Bars: three-training-seed means; dots: all three seeds; whiskers: min/max, NOT confidence intervals.\n"
             "Per seed: 300 rough + 50 flat episodes. Common initial conditions; 16/64s windows are dependent. No model promotion.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    if output.exists():
        raise FileExistsError(output)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_learning(curves, output):
    metrics = ("Train/mean_reward", "Policy/mean_noise_std", "Loss/prior_loss")
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), squeeze=False)
    for column, arm in enumerate(ARMS):
        for row, metric in enumerate(metrics):
            axis = axes[row, column]
            for seed, style in zip(SEEDS, ("-", "--", ":")):
                points = curves["runs"][f"{arm}{seed}"][metric]
                if len(points) != 250 or [p["step"] for p in points] != list(range(250)):
                    raise ValueError("incomplete training scalar sequence")
                axis.plot([p["step"] for p in points], [p["value"] for p in points],
                          color=COLORS[column], linestyle=style, label=f"seed{seed}", linewidth=1)
            axis.set_title(f"{arm}: {metric}")
            axis.grid(alpha=.2)
            axis.set_xlabel("PPO iteration")
            if row == 0:
                axis.legend(fontsize=8)
    fig.suptitle("v25 learning diagnostics | all nine fixed final249 runs", fontsize=14)
    fig.text(.5, .012, "Reward definitions differ between arms: training return is NOT a performance ranking.\n"
             "Entropy coefficient: .002 / .002 / .005. Frozen teacher prior coefficient: .02. Evaluate using heldout strict traversal.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    if output.exists():
        raise FileExistsError(output)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--training-curves", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    compare_summary(summary, summary["independent_raw_audit"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seconds in (16, 64):
        plot_window(summary, seconds, args.output_dir / f"comparison_{seconds}s.png")
    if args.training_curves:
        plot_learning(json.loads(args.training_curves.read_text()), args.output_dir / "learning_diagnostics.png")


if __name__ == "__main__":
    main()
