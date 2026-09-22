#!/usr/bin/env python3
"""Aggregate v5 benchmark JSON files into per-family tables (Markdown and CSV)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

LANE_SEEDS = ("lanes_seed24", "lanes_seed25", "lanes_seed26")
COURSE = ("id", "low_friction", "heavy", "push")
TRAVERSAL_COUNTS = (
    "one_terrain_tile_crossings",
    "all_terrain_tiles_crossings",
    "complete_lane_loops",
)


def load(directory: Path, label: str, name: str) -> dict | None:
    path = directory / f"{label}__{name}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def aggregate_traversal(entries: list[dict], *, applicable: bool, boundary_evidence: bool) -> dict:
    """Aggregate count metrics without averaging already-rounded per-run rates."""
    empty = {
        "terrain_crossing_episodes": None,
        "one_terrain_tile_crossings": None,
        "one_terrain_tile_crossing_rate": None,
        "all_terrain_tiles_crossings": None,
        "all_terrain_tiles_crossing_rate": None,
        "complete_lane_loops": None,
        "out_of_lane_exits": None,
        "world_exits": None,
        "boundary_evidence_available": False,
    }
    if not applicable:
        return empty

    required = ("terrain_crossing_episodes",) + TRAVERSAL_COUNTS
    if not all(all(entry.get(key) is not None for key in required) for entry in entries):
        return empty

    episodes = sum(entry["terrain_crossing_episodes"] for entry in entries)
    result = dict(empty)
    result["terrain_crossing_episodes"] = episodes
    for count_key, rate_key in (
        ("one_terrain_tile_crossings", "one_terrain_tile_crossing_rate"),
        ("all_terrain_tiles_crossings", "all_terrain_tiles_crossing_rate"),
    ):
        count = sum(entry[count_key] for entry in entries)
        result[count_key] = count
        result[rate_key] = count / episodes if episodes else None
    result["complete_lane_loops"] = sum(entry["complete_lane_loops"] for entry in entries)

    has_boundary_counts = all(
        entry.get("out_of_lane_exits") is not None and entry.get("world_exits") is not None for entry in entries
    )
    if boundary_evidence and has_boundary_counts:
        result["boundary_evidence_available"] = True
        result["out_of_lane_exits"] = sum(entry["out_of_lane_exits"] for entry in entries)
        result["world_exits"] = sum(entry["world_exits"] for entry in entries)
    return result


def run_traversal(run: dict) -> dict:
    """Copy the machine-readable traversal fields from a single course run."""
    applicable = run.get("terrain_crossing_episodes") is not None
    return aggregate_traversal(
        [run], applicable=applicable, boundary_evidence=run.get("boundary_evidence_available") is True
    )


def summarize(directory: Path, label: str) -> dict:
    runs = [run for run in (load(directory, label, name) for name in LANE_SEEDS) if run is not None]
    if not runs:
        raise FileNotFoundError(f"no lane benchmark JSON for {label} in {directory}")
    families = list(runs[0]["family_breakdown"])
    table = {}
    for family in families:
        entries = [run["family_breakdown"][family] for run in runs]
        episodes = sum(entry["episodes"] for entry in entries)
        family_summary = {
            "episodes": episodes,
            "fall_rate": sum(entry["falls"] for entry in entries) / episodes,
            "speed": mean(entry["forward_speed_mean"] for entry in entries),
            "return": mean(entry["episode_return_mean"] for entry in entries),
            "levels": {
                level: {
                    "fall_rate": sum(run["family_breakdown"][family]["levels"][level]["falls"] for run in runs)
                    / sum(run["family_breakdown"][family]["levels"][level]["episodes"] for run in runs),
                    "speed": mean(run["family_breakdown"][family]["levels"][level]["forward_speed_mean"] for run in runs),
                }
                for level in entries[0]["levels"]
            },
        }
        family_summary.update(
            aggregate_traversal(
                entries,
                applicable=family != "flat",
                boundary_evidence=all(run.get("boundary_evidence_available") is True for run in runs),
            )
        )
        table[family] = family_summary
    terrain = [family for family in families if family != "flat"]
    overall = {
        "seeds": len(runs),
        "episodes": sum(run["completed_episodes"] for run in runs),
        "fall_rate": sum(run["terminated_episodes"] for run in runs) / sum(run["completed_episodes"] for run in runs),
        "terrain_fall_rate": mean(table[family]["fall_rate"] for family in terrain),
        "terrain_speed": mean(table[family]["speed"] for family in terrain),
        "return_seed_means": [run["episode_return_mean"] for run in runs],
    }
    overall.update(
        aggregate_traversal(
            [table[family] for family in terrain],
            applicable=bool(terrain),
            boundary_evidence=all(table[family]["boundary_evidence_available"] for family in terrain),
        )
    )
    course = {}
    for name in ("lanes100_seed24",) + COURSE:
        run = load(directory, label, name)
        if run is not None:
            course[name] = {
                "return_mean": run["episode_return_mean"],
                "return_std": run["episode_return_std"],
                "length_mean": run["episode_length_mean"],
                "terminated": run.get("terminated_episodes"),
            }
            course[name].update(run_traversal(run))
    return {"label": label, "families": table, "overall": overall, "course": course}


def markdown(summaries: list[dict]) -> str:
    families = list(summaries[0]["families"])
    lines = ["| Policy | " + " | ".join(families) + " | terrain mean |", "|---" * (len(families) + 2) + "|"]
    for summary in summaries:
        cells = [
            f"{100 * summary['families'][f]['fall_rate']:.0f}% / {summary['families'][f]['speed']:.2f}" for f in families
        ]
        overall = summary["overall"]
        cells.append(f"{100 * overall['terrain_fall_rate']:.0f}% / {overall['terrain_speed']:.2f}")
        lines.append(f"| {summary['label']} | " + " | ".join(cells) + " |")
    terrain_families = [family for family in families if family != "flat"]
    lines += [
        "",
        "Terrain traversal success (successful episodes / applicable episodes; distinct from survival):",
        "",
        "| Policy | Family | one-tile clearance | full six-tile clearance | lane loops | lane exits | world exits |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for family in terrain_families:
            entry = summary["families"][family]
            denominator = entry["terrain_crossing_episodes"]

            def success_cell(count_key: str, rate_key: str) -> str:
                count = entry[count_key]
                if count is None or denominator is None:
                    return "unknown"
                return f"{count}/{denominator} ({100 * entry[rate_key]:.1f}%)"

            boundary = (
                (str(entry["out_of_lane_exits"]), str(entry["world_exits"]))
                if entry["boundary_evidence_available"]
                else ("unknown", "unknown")
            )
            loops = "unknown" if entry["complete_lane_loops"] is None else str(entry["complete_lane_loops"])
            lines.append(
                f"| {summary['label']} | {family} | "
                f"{success_cell('one_terrain_tile_crossings', 'one_terrain_tile_crossing_rate')} | "
                f"{success_cell('all_terrain_tiles_crossings', 'all_terrain_tiles_crossing_rate')} | "
                f"{loops} | {boundary[0]} | {boundary[1]} |"
            )
    lines += ["", "| Policy | lanes 100-env | ID | low friction | heavy | push |", "|---|---:|---:|---:|---:|---:|"]
    for summary in summaries:
        cells = []
        for name in ("lanes100_seed24",) + COURSE:
            entry = summary["course"].get(name)
            cells.append("-" if entry is None else f"{entry['return_mean']:.2f} ± {entry['return_std']:.2f}")
        lines.append(f"| {summary['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+")
    parser.add_argument(
        "--directory", type=Path, default=Path("artifacts/terrain_demo/evaluations/rough_v5"), help="JSON directory"
    )
    parser.add_argument("--csv", type=Path, help="optional per-family CSV output")
    parser.add_argument("--json", type=Path, help="optional machine-readable summary output")
    args = parser.parse_args()
    summaries = [summarize(args.directory, label) for label in args.labels]
    seed_counts = sorted({summary["overall"]["seeds"] for summary in summaries})
    seed_text = str(seed_counts[0]) if len(seed_counts) == 1 else "/".join(map(str, seed_counts))
    print(f"Cells: fall rate / mean forward speed (m/s); {seed_text} available lane seed(s)\n")
    print(markdown(summaries))
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "policy", "family", "episodes", "fall_rate", "forward_speed_mps", "return_mean",
                    "terrain_crossing_episodes", "one_terrain_tile_crossings", "one_terrain_tile_crossing_rate",
                    "all_terrain_tiles_crossings", "all_terrain_tiles_crossing_rate", "complete_lane_loops",
                    "boundary_evidence_available", "out_of_lane_exits", "world_exits",
                ]
            )
            for summary in summaries:
                for family, entry in summary["families"].items():
                    writer.writerow(
                        [
                            summary["label"], family, entry["episodes"], f"{entry['fall_rate']:.4f}",
                            f"{entry['speed']:.4f}", f"{entry['return']:.4f}",
                            entry["terrain_crossing_episodes"], entry["one_terrain_tile_crossings"],
                            entry["one_terrain_tile_crossing_rate"], entry["all_terrain_tiles_crossings"],
                            entry["all_terrain_tiles_crossing_rate"], entry["complete_lane_loops"],
                            entry["boundary_evidence_available"], entry["out_of_lane_exits"], entry["world_exits"],
                        ]
                    )
    if args.json:
        args.json.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    for summary in summaries:
        seed_means = summary["overall"]["return_seed_means"]
        print(
            f"\n{summary['label']}: lane return per seed {[round(v, 2) for v in seed_means]}"
            f" (seed-mean std {pstdev(seed_means):.2f}), overall fall rate {100 * summary['overall']['fall_rate']:.1f}%"
        )


if __name__ == "__main__":
    main()
