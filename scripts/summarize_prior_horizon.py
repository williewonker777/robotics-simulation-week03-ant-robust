"""Audit and summarize the separate 64s diagnostic, never a promotion metric."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

from run_prior_experiment import ROOT, EVAL_PAIRS, sha, write_json, check_sources


ARRAYS = (
    "forward_distance", "episode_terminated", "episode_lengths", "episode_out_of_lane", "episode_world_exit",
    "first_hit_one_seconds", "first_hit_six_seconds", "distance_at_16s", "maximum_distance_m",
    "episode_full_64s_survival", "episode_strict_one_tile_success", "episode_strict_all_tiles_success",
)


def audit(data):
    if (data["schema"], data["task"]) != (
            "week03_ant_prior_v10_horizon_v1", "Week03-Ant-Prior-Lanes-Eval-v10"):
        raise ValueError("unexpected diagnostic schema/task")
    n = data["num_envs"]
    condition = data["condition"]
    if n != 10 or any(len(data[key]) != n for key in ARRAYS):
        raise ValueError("unmatched diagnostic episode arrays")
    if (condition["terrain_family"], condition["terrain_level_index"], condition["terrain_difficulty"],
            condition["episode_length_seconds"], condition["max_steps"], condition["distance_snapshot_seconds"]) != (
            "stepping_stones", 4, 1.0, 64., 3840, 16.):
        raise ValueError("changed long-horizon condition")
    if (condition["one_tile_clearance_m"], condition["all_tiles_clearance_m"]) != (13.1, 53.1):
        raise ValueError("changed clearance thresholds")
    if not math.isclose(condition["step_dt_seconds"], 1 / 60, abs_tol=1e-10):
        raise ValueError("changed diagnostic control step")
    if data["scope"]["promotion_metric"] or data["scope"]["primary_175_env_assignment_reused"]:
        raise ValueError("diagnostic scope was misrepresented")
    if data["policy_observation_dimensions"] != 88:
        raise ValueError("wrong policy dimensions")
    rows = []
    for i in range(n):
        values = {key: data[key][i] for key in ARRAYS}
        distance, maximum, steps = (values[key] for key in ("forward_distance", "maximum_distance_m", "episode_lengths"))
        if not math.isfinite(distance) or not math.isfinite(maximum) or maximum < distance or not 0 < steps <= 3840:
            raise ValueError("invalid distance/length evidence")
        safe = not any(values[key] for key in ("episode_terminated", "episode_out_of_lane", "episode_world_exit"))
        one, six = (safe and distance >= float(np.float32(threshold)) for threshold in (13.1, 53.1))
        full = steps == 3840 and not values["episode_terminated"]
        if (one, six, full) != (values["episode_strict_one_tile_success"],
                                values["episode_strict_all_tiles_success"], values["episode_full_64s_survival"]):
            raise ValueError("strict success or survival disagrees with raw first-episode arrays")
        snapshot = values["distance_at_16s"]
        if (snapshot is None) != (steps <= 960) or (snapshot is not None and not math.isfinite(snapshot)):
            raise ValueError("16s snapshot censoring mismatch")
        for key, threshold in (("first_hit_one_seconds", 13.1), ("first_hit_six_seconds", 53.1)):
            hit = values[key]
            if (hit is None) != (maximum < float(np.float32(threshold))):
                raise ValueError("first-hit evidence disagrees with maximum distance")
            if hit is not None and (not math.isfinite(hit) or not 0 < hit <= steps / 60 + 1e-9):
                raise ValueError("invalid first-hit time")
        rows.append(dict(n=1, one=int(one), six=int(six), falls=int(values["episode_terminated"]),
                         lane=int(values["episode_out_of_lane"]), world=int(values["episode_world_exit"]),
                         survival=int(full), first_hit_one=values["first_hit_one_seconds"],
                         first_hit_six=values["first_hit_six_seconds"], distance16=snapshot,
                         final_distance=distance, maximum_distance=maximum))
    for key, raw in (("one", "strict_one_tile_successes"), ("six", "strict_all_tiles_successes"),
                     ("falls", "posture_terminations"), ("lane", "out_of_lane_exits"),
                     ("world", "world_exits"), ("survival", "full_64s_survivals")):
        if sum(row[key] for row in rows) != data["aggregates"][raw]:
            raise ValueError(f"stored aggregate mismatch: {raw}")
    return rows


def aggregate(rows):
    result = {key: sum(row[key] for row in rows) for key in ("n", "one", "six", "falls", "lane", "world", "survival")}
    for key in ("first_hit_one", "first_hit_six", "distance16", "final_distance", "maximum_distance"):
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = {"observed_n": len(values), "mean": mean(values) if values else None,
                       "median": median(values) if values else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    frozen = json.loads((args.directory / "frozen.json").read_text())
    check_sources(frozen["source_sha256"])
    groups, seeds, files = defaultdict(list), defaultdict(lambda: defaultdict(list)), []
    for record in [frozen["reference"], *frozen["runs"]]:
        checkpoint = ROOT / record["checkpoint"]
        if sha(checkpoint) != record["sha256"]:
            raise ValueError("frozen checkpoint changed")
        for geometry, reset in EVAL_PAIRS:
            path = args.directory / "horizon" / f"{checkpoint.parent.name}__geometry{geometry}_reset{reset}.json"
            data = json.loads(path.read_text())
            if not data["scope"]["predeclared_holdout_pair"]:
                raise ValueError("diagnostic is not a predeclared held-out pair")
            if (data["geometry_seed"], data["reset_seed"], data["checkpoint_sha256"], data["policy_prior_mode"]) != (
                    geometry, reset, record["sha256"], record["mode"]):
                raise ValueError("unpaired condition or checkpoint")
            if datetime.fromisoformat(data["started_utc"]) <= datetime.fromisoformat(frozen["frozen_at"]):
                raise ValueError("diagnostic ran before final model freeze")
            rows = audit(data)
            groups[record["group"]].extend(rows)
            if record["training_seed"] is not None:
                seeds[str(record["training_seed"])][record["group"]].extend(rows)
            files.append({"file": str(path), "sha256": sha(path), "episodes": len(rows)})
    totals = {group: aggregate(rows) for group, rows in groups.items()}
    if {group: row["n"] for group, row in totals.items()} != {"frozen_v5": 20, "free": 60, "anchored": 60}:
        raise ValueError("diagnostic denominators differ from plan")
    result = {"totals": totals, "seeds": {seed: {g: aggregate(r) for g, r in by_group.items()} for seed, by_group in seeds.items()},
              "scope": "Non-gating64s stone1.0-only diagnostic; two maps, not140 independent maps; never combine with primary.",
              "time_interpretation": "Hit-time statistics are conditional on reaching distance, even if later falling/exiting; not unconditional traversal time.",
              "verification": {"status": "pass", "files": files, "first_episodes": 140}}
    write_json(args.directory / "horizon_summary.json", result)
    lines = ["# Separate64s stone1.0 diagnostic", "", result["scope"], "",
             "| Policy | Strict1tile | Strict6tiles | Falls | Lane | World | Full64survival |", "|---|---:|---:|---:|---:|---:|---:|"]
    for group, row in totals.items():
        lines.append(f"| {group} | " + " | ".join(f"{row[key]}/{row['n']}" for key in ("one", "six", "falls", "lane", "world", "survival")) + " |")
    lines += ["", "## Conditional times and censored distances", "", result["time_interpretation"], "",
              "Distanceat16s is null if the first episode ended at/before16s. Final distance is at64s OR earlier termination."]
    for group, row in totals.items():
        lines += ["", f"### {group}"]
        for key in ("first_hit_one", "first_hit_six", "distance16", "final_distance", "maximum_distance"):
            value = row[key]
            lines.append(f"- {key}: observed{value['observed_n']}/{row['n']}; mean={value['mean']}; median={value['median']}")
    (args.directory / "horizon_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
