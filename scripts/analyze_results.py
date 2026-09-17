#!/usr/bin/env python3
"""Generate tables and figures from staged TensorBoard and evaluation artifacts."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
EVALUATIONS = ROOT / "artifacts" / "evaluations"
PLOTS = ROOT / "artifacts" / "plots"


def smooth(values: np.ndarray, width: int = 25) -> np.ndarray:
    if len(values) < width:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    padded = np.pad(values, (width - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_training_curves() -> list[dict[str, object]]:
    curves = []
    for manifest_path in sorted(RUNS.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        accumulator = EventAccumulator(str(manifest_path.parent)).Reload()
        scalars = accumulator.Scalars("Train/mean_reward")
        curves.append(
            {
                **manifest,
                "steps": np.asarray([point.step for point in scalars], dtype=np.int64),
                "rewards": np.asarray([point.value for point in scalars], dtype=np.float64),
            }
        )
    return curves


def plot_training(curves: list[dict[str, object]]) -> None:
    colors = {"baseline": "#2563eb", "friction": "#f59e0b", "robust": "#059669"}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for curve in curves:
        grouped[str(curve["variant"])].append(curve)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for variant in ("baseline", "friction", "robust"):
        items = grouped.get(variant, [])
        if not items:
            continue
        minimum = min(len(item["rewards"]) for item in items)
        stacked = np.stack([smooth(item["rewards"][:minimum]) for item in items])
        steps = items[0]["steps"][:minimum]
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        ax.plot(steps, mean, label=f"{variant} (n={len(items)} seeds)", color=colors[variant], linewidth=2.2)
        if len(items) > 1:
            ax.fill_between(steps, mean - std, mean + std, color=colors[variant], alpha=0.18)

    ax.set_title("Isaac Ant PPO learning curves (25-iteration moving average)")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Mean episode reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(PLOTS / "training_curves.png", dpi=180)
    plt.close(fig)


def evaluation_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_run = []
    pooled: dict[tuple[str, str], dict[str, object]] = {}
    manifests = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in RUNS.glob("*/manifest.json")
    }
    for path in sorted(EVALUATIONS.glob("*__*.json")):
        label, scenario = path.stem.split("__", 1)
        if label not in manifests:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        manifest = manifests[label]
        row = {
            "label": label,
            "variant": manifest["variant"],
            "seed": manifest["seed"],
            "scenario": scenario,
            "mean": result["episode_return_mean"],
            "std": result["episode_return_std"],
            "length_mean": result["episode_length_mean"],
            "completed": result["completed_episodes"],
        }
        per_run.append(row)
        key = (str(manifest["variant"]), scenario)
        group = pooled.setdefault(key, {"returns": [], "lengths": [], "seed_means": []})
        group["returns"].extend(result["episode_returns"])
        group["lengths"].extend(result["episode_lengths"])
        group["seed_means"].append(result["episode_return_mean"])

    summary = []
    for (variant, scenario), values in sorted(pooled.items()):
        returns = np.asarray(values["returns"], dtype=np.float64)
        lengths = np.asarray(values["lengths"], dtype=np.float64)
        seed_means = np.asarray(values["seed_means"], dtype=np.float64)
        summary.append(
            {
                "variant": variant,
                "scenario": scenario,
                "training_seeds": len(seed_means),
                "episodes": len(returns),
                "return_mean": returns.mean(),
                "return_std": returns.std(),
                "seed_mean_std": seed_means.std(),
                "length_mean": lengths.mean(),
            }
        )
    return per_run, summary


def write_tables(per_run: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    csv_path = EVALUATIONS / "evaluation_per_run.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_run[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_run)

    summary_csv = EVALUATIONS / "evaluation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "| Variant | Scenario | Seeds | Episodes | Return mean ± std | Mean length |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['variant']} | {row['scenario']} | {row['training_seeds']} | "
            f"{row['episodes']} | {row['return_mean']:.2f} ± {row['return_std']:.2f} | "
            f"{row['length_mean']:.1f} |"
        )
    (EVALUATIONS / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_evaluation(summary: list[dict[str, object]]) -> None:
    scenarios = ["id", "low_friction", "heavy", "push"]
    variants = ["baseline", "friction", "robust"]
    colors = {"baseline": "#2563eb", "friction": "#f59e0b", "robust": "#059669"}
    index = {(row["variant"], row["scenario"]): row for row in summary}
    x = np.arange(len(scenarios))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for offset, variant in enumerate(variants):
        means = [index.get((variant, scenario), {}).get("return_mean", np.nan) for scenario in scenarios]
        errors = [index.get((variant, scenario), {}).get("return_std", 0.0) for scenario in scenarios]
        ax.bar(x + (offset - 1) * width, means, width, yerr=errors, capsize=3, label=variant, color=colors[variant])
    ax.set_title("100-environment first-episode evaluation")
    ax.set_ylabel("Episode return (population mean ± std)")
    ax.set_xticks(x, ["ID", "Low friction", "+30% torso mass", "Strong pushes"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(PLOTS / "evaluation_returns.png", dpi=180)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    curves = load_training_curves()
    if curves:
        plot_training(curves)
    per_run, summary = evaluation_rows()
    if per_run and summary:
        write_tables(per_run, summary)
        plot_evaluation(summary)
    print(f"training_runs={len(curves)} evaluation_runs={len(per_run)}")


if __name__ == "__main__":
    main()
