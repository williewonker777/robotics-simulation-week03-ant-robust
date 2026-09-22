"""Audit the frozen v10 first-episode evidence and retain failed gates."""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from summarize_footmap import aggregate, recompute


GROUPS = ("frozen_v5", "free", "anchored")
FAMILIES = ["rough", "slope", "stairs", "waves", "obstacles", "stepping_stones", "flat"]
TASK = "Week03-Ant-Prior-Lanes-Eval-v10"


def directional_gate(groups, seeds):
    candidate, free, baseline = (groups[name] for name in ("anchored", "free", "frozen_v5"))
    checks = {
        "fewer_falls_than_free": Fraction(candidate["falls"], candidate["n"]) < Fraction(free["falls"], free["n"]),
        "fewer_lane_exits_than_free": Fraction(candidate["lane"], candidate["n"]) < Fraction(free["lane"], free["n"]),
        "joint_safety_improvement_majority": sum(
            row["anchored"]["falls"] < row["free"]["falls"]
            and row["anchored"]["lane"] < row["free"]["lane"] for row in seeds.values()
        ) >= 2,
        "six_at_least_v5": Fraction(candidate["six"], candidate["n"]) >= Fraction(baseline["six"], baseline["n"]),
        "zero_world_exit": candidate["world"] == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def promotion_gate(groups, seeds):
    checks = {}
    for baseline in GROUPS[:-1]:
        for metric in ("one", "six", "falls", "lane"):
            candidate = Fraction(groups["anchored"][metric], groups["anchored"]["n"])
            reference = Fraction(groups[baseline][metric], groups[baseline]["n"])
            checks[f"{metric}_vs_{baseline}"] = candidate <= reference if metric in ("falls", "lane") else candidate >= reference
    checks["zero_world_exit"] = groups["anchored"]["world"] == 0
    checks["flat_falls_vs_frozen_v5"] = (
        Fraction(groups["anchored"]["flat_falls"], groups["anchored"]["flat_n"])
        <= Fraction(groups["frozen_v5"]["flat_falls"], groups["frozen_v5"]["flat_n"])
    )
    checks["fewer_falls_than_free_majority"] = sum(
        row["anchored"]["falls"] < row["free"]["falls"] for row in seeds.values()
    ) >= 2
    return {"passed": all(checks.values()), "checks": checks}


def _verify_family_breakdown(data, path):
    for family, saved in data["family_breakdown"].items():
        if family == "flat":
            continue
        levels = [(None, saved), *[(int(key.split(":")[0]), value) for key, value in saved["levels"].items()]]
        for level, stored in levels:
            counts = aggregate([data], family=family, level=level)
            for field, raw in (
                ("one", "one_terrain_tile_crossings"), ("six", "all_terrain_tiles_crossings"),
                ("falls", "falls"), ("lane", "out_of_lane_exits"), ("world", "world_exits"),
            ):
                if counts[field] != stored[raw]:
                    raise ValueError(f"family/level mismatch: {path}/{family}/{level}/{field}")


def summarize(directory):
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((directory / "frozen.json").read_text())
    if frozen["heldout_pairs"] != [[66, 40], [67, 41]]:
        raise ValueError("changed held-out geometry/reset pairs")
    for name, expected in frozen["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source changed since training: {name}")
    groups = defaultdict(list)
    by_seed = defaultdict(lambda: defaultdict(list))
    audit = []
    records = [frozen["reference"], *frozen["runs"]]
    if len(records) != 7 or len({row["checkpoint"] for row in records}) != 7:
        raise ValueError("expected 7 distinct frozen policies")
    for record in records:
        checkpoint = root / record["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("frozen checkpoint changed")
        if record["policy_class"] != "PriorActorCritic" or record["group"] not in GROUPS:
            raise ValueError("unexpected frozen policy metadata")
        for geometry, reset in frozen["heldout_pairs"]:
            path = directory / "evaluations" / f"{checkpoint.parent.name}__geometry{geometry}_reset{reset}.json"
            data = json.loads(path.read_text())
            observed = (
                data["terrain_generator_seed"], data["seed"], data["num_envs"], data["max_steps"],
                data["policy_observation_dimensions"], data["task"],
            )
            if observed != (geometry, reset, 175, 960, 88, TASK):
                raise ValueError(f"unmatched evaluation conditions: {path}")
            if "depth_observation" in data:
                raise ValueError("v10 raw evidence unexpectedly contains a depth-observation contract")
            if data["checkpoint_sha256"] != record["sha256"]:
                raise ValueError("evaluation checkpoint identity mismatch")
            if data["lane_family_names"] != FAMILIES:
                raise ValueError("unexpected benchmark families")
            for family in range(7):
                for level in range(5):
                    count = sum(
                        f == family and l == level
                        for f, l in zip(data["lane_family_indices"], data["lane_level_indices"])
                    )
                    if count != 5:
                        raise ValueError("unbalanced family/difficulty cells")
            recompute(data)
            _verify_family_breakdown(data, path)
            groups[record["group"]].append(data)
            if record["training_seed"] is not None:
                by_seed[str(record["training_seed"])][record["group"]].append(data)
            audit.append({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "episodes": 175})
    totals = {group: aggregate(groups[group]) for group in GROUPS}
    if totals["frozen_v5"]["n"] != 300 or any(totals[group]["n"] != 900 for group in GROUPS[1:]):
        raise ValueError("unexpected primary denominators")
    seeds = {seed: {group: aggregate(runs) for group, runs in row.items()} for seed, row in by_seed.items()}
    if set(seeds) != {"42", "43", "44"}:
        raise ValueError("missing paired training seed")
    if any(set(row) != set(GROUPS[1:]) or any(values["n"] != 300 for values in row.values())
           for row in seeds.values()):
        raise ValueError("unbalanced per-seed policy groups")
    hard = {group: {str(level): aggregate(runs, level=level) for level in (3, 4)}
            for group, runs in groups.items()}
    per_family = {
        group: {
            family: {str(level): aggregate(runs, family, level) for level in range(5)}
            for family in FAMILIES[:-1]
        }
        for group, runs in groups.items()
    }
    result = {
        "totals": totals, "seeds": seeds, "hard": hard, "per_family": per_family,
        "promotion_gate": promotion_gate(totals, seeds),
        "directional_gate": directional_gate(totals, seeds),
        "interpretation": (
            "Matched three-seed fixed-budget training-only v5 mean-prior study. Frozen v5 is evaluated once per condition, "
            "not replicated. Two held-out terrain maps are not thousands of independent maps. The scan and "
            "distal-center heuristic are ideal simulation signals, not real RGB-D or certified foothold support."
        ),
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Frozen-v5 training-only mean prior v10", "",
        "6 runs: 750 iterations x4096 environments x32 steps; 3 paired seeds/mode.",
        "Fresh geometry/reset66/40 and67/41. 14 evaluations /2450 first episodes, at most16s.", "",
        "| Policy | One tile | Six tiles | Falls | Lane exits | Flat falls | Speed m/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        row = totals[group]
        lines.append(
            f"| {group} | {row['one']}/{row['n']} | {row['six']}/{row['n']} | {row['falls']}/{row['n']} | "
            f"{row['lane']}/{row['n']} | {row['flat_falls']}/{row['flat_n']} | {row['speed_m_s']:.3f} |"
        )
    lines += ["", "## Highest difficulty 1.0", "", "| Policy | One tile | Six tiles | Falls |",
              "|---|---:|---:|---:|"]
    for group in GROUPS:
        row = hard[group]["4"]
        lines.append(f"| {group} | {row['one']}/{row['n']} | {row['six']}/{row['n']} | {row['falls']}/{row['n']} |")
    lines += ["", "## Every training seed", "", "| Seed | Policy | One tile | Six tiles | Falls |",
              "|---|---|---:|---:|---:|"]
    for seed, row in seeds.items():
        for group, values in row.items():
            lines.append(
                f"| {seed} | {group} | {values['one']}/{values['n']} | {values['six']}/{values['n']} | "
                f"{values['falls']}/{values['n']} |"
            )
    lines += ["", f"Promotion gate: **{'PASS' if result['promotion_gate']['passed'] else 'FAIL'}**", "",
              f"Directional gate: **{'PASS' if result['directional_gate']['passed'] else 'FAIL'}**", "",
              result["interpretation"], "", "See summary.json for every family/level and individual gate check."]
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
    (directory / "verification.json").write_text(json.dumps({
        "status": "pass", "mismatches": 0, "files": audit,
        "source_hashes_verified": frozen["source_sha256"],
        "total_primary_first_episodes": sum(row["episodes"] for row in audit),
    }, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    print((args.directory / "summary.md").read_text(), end="")
    if not result["promotion_gate"]["passed"]:
        print("v10 remains experimental; retain frozen v5.")


if __name__ == "__main__":
    main()
