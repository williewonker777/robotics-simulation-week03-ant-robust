"""Audit frozen v7 raw episode arrays and report every seed, not the best seed."""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def recompute(data):
    n = data["num_envs"]
    keys = ("forward_distance", "episode_terminated", "episode_out_of_lane", "episode_world_exit",
            "lane_family_indices", "lane_level_indices", "episode_returns", "episode_lengths")
    if data["completed_episodes"] != n or not data["boundary_evidence_available"]:
        raise ValueError("missing completed first episodes or containment evidence")
    if any(len(data[key]) != n for key in keys):
        raise ValueError("invalid raw array lengths")
    if any(not math.isfinite(v) for key in ("forward_distance", "episode_returns", "episode_lengths") for v in data[key]):
        raise ValueError("nonfinite episode evidence")
    g = data["traversal_geometry"]
    if (g["one_terrain_tile_clearance_m"], g["all_terrain_tiles_clearance_m"],
            g["lane_containment_foot_margin_m"]) != (13.1, 53.1, 1.1):
        raise ValueError("changed traversal contract")
    one, six = [], []
    for i, distance in enumerate(data["forward_distance"]):
        flat = data["lane_family_names"][data["lane_family_indices"][i]] == "flat"
        safe = not any(data[key][i] for key in ("episode_terminated", "episode_out_of_lane", "episode_world_exit"))
        one.append(None if flat else bool(distance >= float(np.float32(13.1)) and safe))
        six.append(None if flat else bool(distance >= float(np.float32(53.1)) and safe))
    if one != data["episode_cleared_one_terrain_tile"] or six != data["episode_cleared_all_terrain_tiles"]:
        raise ValueError("stored strict crossings disagree with independent raw-array recomputation")
    if sum(v is True for v in one) != data["one_terrain_tile_crossings"]:
        raise ValueError("stored aggregate disagrees")
    if sum(v is True for v in six) != data["all_terrain_tiles_crossings"]:
        raise ValueError("stored six-tile aggregate disagrees")
    return one, six


def aggregate(runs, family=None, level=None):
    result = dict(n=0, one=0, six=0, falls=0, lane=0, world=0, flat_n=0, flat_falls=0)
    speed_sum = 0.0
    for data in runs:
        one, six = recompute(data)
        for i in range(data["num_envs"]):
            name = data["lane_family_names"][data["lane_family_indices"][i]]
            if name == "flat":
                result["flat_n"] += 1
                result["flat_falls"] += int(data["episode_terminated"][i])
                continue
            if family is not None and name != family:
                continue
            if level is not None and data["lane_level_indices"][i] != level:
                continue
            result["n"] += 1
            result["one"] += int(one[i])
            result["six"] += int(six[i])
            result["falls"] += int(data["episode_terminated"][i])
            result["lane"] += int(data["episode_out_of_lane"][i])
            result["world"] += int(data["episode_world_exit"][i])
            speed_sum += data["forward_distance"][i] / (data["episode_lengths"][i] * data["step_dt_seconds"])
    result["speed_m_s"] = speed_sum / result["n"] if result["n"] else None
    return result


def gate(groups, seeds):
    def rate(group, metric):
        return Fraction(groups[group][metric], groups[group]["n"])
    checks = {}
    for baseline in ("blind", "height", "frozen_v5"):
        for metric in ("one", "six", "falls"):
            a, b = rate("footmap", metric), rate(baseline, metric)
            checks[f"{metric}_vs_{baseline}"] = a <= b if metric == "falls" else a >= b
    checks["zero_world_exit"] = groups["footmap"]["world"] == 0
    checks["flat_falls_vs_frozen_v5"] = (
        Fraction(groups["footmap"]["flat_falls"], groups["footmap"]["flat_n"])
        <= Fraction(groups["frozen_v5"]["flat_falls"], groups["frozen_v5"]["flat_n"]))
    checks["majority_seed_pairs_better_than_height"] = sum(
        row["footmap"]["one"] > row["height"]["one"] for row in seeds.values()) >= 2
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((args.directory / "frozen.json").read_text())
    source_hashes = dict(frozen["source_sha256"])
    source_hashes.update(json.loads((args.directory / "source_at_train_start.json").read_text()))
    for name, expected in source_hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source changed since training: {name}")
    by_mode, by_seed = defaultdict(list), defaultdict(lambda: defaultdict(list))
    audit = []
    for record in [frozen["reference"], *frozen["runs"]]:
        checkpoint = root / record["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("checkpoint hash changed")
        name = checkpoint.parent.name
        group = "frozen_v5" if record["training_seed"] is None else record["mode"]
        for geometry, reset in frozen["heldout_pairs"]:
            path = args.directory / "evaluations" / f"{name}__geometry{geometry}_reset{reset}.json"
            data = json.loads(path.read_text())
            if (data["terrain_generator_seed"], data["seed"], data["num_envs"], data["max_steps"],
                    data["policy_observation_dimensions"], data["footmap_policy"]["input_mode"]) != (
                    geometry, reset, 175, 960, 514, record["mode"]):
                raise ValueError(f"unmatched evaluation conditions: {path}")
            if data["checkpoint_sha256"] != record["sha256"]:
                raise ValueError("evaluation checkpoint identity mismatch")
            if data["lane_family_names"] != ["rough", "slope", "stairs", "waves", "obstacles", "stepping_stones", "flat"]:
                raise ValueError("unexpected benchmark families")
            for family in range(7):
                for level in range(5):
                    count = sum(f == family and l == level for f, l in
                                zip(data["lane_family_indices"], data["lane_level_indices"]))
                    if count != 5:
                        raise ValueError("unbalanced family/difficulty benchmark")
            recompute(data)
            # Independently check every stored family/level count as well.
            for family, saved in data["family_breakdown"].items():
                if family == "flat":
                    continue
                for lvl, stored in [(None, saved), *[(int(k.split(":")[0]), v) for k, v in saved["levels"].items()]]:
                    counts = aggregate([data], family=family, level=lvl)
                    for field, rawfield in (("one", "one_terrain_tile_crossings"), ("six", "all_terrain_tiles_crossings"),
                                          ("falls", "falls"), ("lane", "out_of_lane_exits"), ("world", "world_exits")):
                        if counts[field] != stored[rawfield]:
                            raise ValueError(f"family/level mismatch {path} {family}/{lvl}/{field}")
            by_mode[group].append(data)
            if record["training_seed"] is not None:
                by_seed[str(record["training_seed"])][group].append(data)
            audit.append({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "episodes": 175})
    totals = {mode: aggregate(runs) for mode, runs in by_mode.items()}
    seeds = {seed: {mode: aggregate(runs) for mode, runs in row.items()} for seed, row in by_seed.items()}
    hard = {mode: {str(level): aggregate(runs, level=level) for level in (3, 4)} for mode, runs in by_mode.items()}
    families = by_mode["footmap"][0]["lane_family_names"][:-1]
    per_family = {mode: {family: {str(level): aggregate(runs, family, level) for level in (3, 4)}
                         for family in families} for mode, runs in by_mode.items()}
    result = {"totals": totals, "seeds": seeds, "hard": hard, "hard_per_family": per_family,
              "promotion_gate": gate(totals, seeds),
              "interpretation": "matched three-seed architecture/input study; ideal scan, not real depth. "
                                "Reference evaluated once per condition, not replicated as independent data. "
                                "Training seed is paired with training geometry; no broad-generalization guarantee."}
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# FootMap v7 — frozen, equal-budget comparison", "",
             "750 iterations x4096 environments x32 steps per run; 3 training seeds per arm.",
             "Held-out geometry/reset60/34 and61/35; first episode only, at most16s.", "",
             "| Policy | One tile | Six tiles | Falls | Flat falls | Speed m/s |", "|---|---:|---:|---:|---:|---:|"]
    for mode in ("frozen_v5", "blind", "height", "footmap"):
        r = totals[mode]
        lines.append(f"| {mode} | {r['one']}/{r['n']} | {r['six']}/{r['n']} | {r['falls']}/{r['n']} | "
                     f"{r['flat_falls']}/{r['flat_n']} | {r['speed_m_s']:.3f} |")
    lines += ["", "## Highest difficulty1.0", "", "| Policy | One tile | Six tiles | Falls |", "|---|---:|---:|---:|"]
    for mode in ("frozen_v5", "blind", "height", "footmap"):
        r = hard[mode]["4"]
        lines.append(f"| {mode} | {r['one']}/{r['n']} | {r['six']}/{r['n']} | {r['falls']}/{r['n']} |")
    lines += ["", "## Every training seed", "", "| Seed | Mode | One tile | Six tiles | Falls |", "|---|---|---:|---:|---:|"]
    for seed, row in seeds.items():
        for mode, r in row.items():
            lines.append(f"| {seed} | {mode} | {r['one']}/{r['n']} | {r['six']}/{r['n']} | {r['falls']}/{r['n']} |")
    lines += ["", f"Promotion gate: **{'PASS' if result['promotion_gate']['passed'] else 'FAIL'}**", "",
              result["interpretation"], "", "See summary.json for every family/level and individual gate check."]
    (args.directory / "summary.md").write_text("\n".join(lines) + "\n")
    (args.directory / "verification.json").write_text(json.dumps(
        {"status": "pass", "mismatches": 0, "files": audit, "source_hashes_verified": source_hashes,
         "total_first_episodes": sum(r["episodes"] for r in audit)}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
