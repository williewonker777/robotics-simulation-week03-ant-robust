"""Audit v8 frozen first-episode evidence; retain all seeds and failed gates."""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from summarize_footmap import aggregate, recompute

GROUPS = ("frozen_v5", "v7_footmap", "residual_blind", "residual_footmap")


def promotion_gate(groups, seeds):
    checks = {}
    for baseline in GROUPS[:-1]:
        for metric in ("one", "six", "falls"):
            candidate = Fraction(groups["residual_footmap"][metric], groups["residual_footmap"]["n"])
            reference = Fraction(groups[baseline][metric], groups[baseline]["n"])
            checks[f"{metric}_vs_{baseline}"] = candidate <= reference if metric == "falls" else candidate >= reference
    candidate, reference = groups["residual_footmap"], groups["frozen_v5"]
    checks["zero_world_exit"] = candidate["world"] == 0
    checks["flat_falls_vs_frozen_v5"] = (Fraction(candidate["flat_falls"], candidate["flat_n"])
                                          <= Fraction(reference["flat_falls"], reference["flat_n"]))
    checks["majority_seed_pairs_better_than_blind"] = sum(
        row["residual_footmap"]["one"] > row["residual_blind"]["one"] for row in seeds.values()) >= 2
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((args.directory / "frozen.json").read_text())
    for name, expected in frozen["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source changed since training: {name}")
    groups, by_seed, audit = defaultdict(list), defaultdict(lambda: defaultdict(list)), []
    records = [frozen["reference"], *frozen["comparisons"], *frozen["runs"]]
    if len(records) != 10 or len({r["checkpoint"] for r in records}) != 10:
        raise ValueError("expected 10 distinct frozen policies")
    for record in records:
        checkpoint = root / record["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("frozen checkpoint changed")
        for geometry, reset in frozen["heldout_pairs"]:
            path = args.directory / "evaluations" / f"{checkpoint.parent.name}__geometry{geometry}_reset{reset}.json"
            data = json.loads(path.read_text())
            if (data["terrain_generator_seed"], data["seed"], data["num_envs"], data["max_steps"],
                    data["policy_observation_dimensions"], data["depth_observation"]["mode"],
                    data["depth_observation"]["noise"], data["depth_observation"]["rays"]) != (
                    geometry, reset, 175, 960, 514, "actual", 0, 221):
                raise ValueError(f"unmatched evaluation conditions: {path}")
            if data["checkpoint_sha256"] != record["sha256"]:
                raise ValueError("evaluation checkpoint identity mismatch")
            families = ["rough", "slope", "stairs", "waves", "obstacles", "stepping_stones", "flat"]
            if data["lane_family_names"] != families:
                raise ValueError("unexpected benchmark families")
            for family in range(7):
                for level in range(5):
                    if sum(f == family and l == level for f, l in
                           zip(data["lane_family_indices"], data["lane_level_indices"])) != 5:
                        raise ValueError("unbalanced family/difficulty cells")
            recompute(data)
            for family, saved in data["family_breakdown"].items():
                if family == "flat":
                    continue
                for level, stored in [(None, saved), *[(int(k.split(":")[0]), v) for k, v in saved["levels"].items()]]:
                    counts = aggregate([data], family=family, level=level)
                    for field, raw in (("one", "one_terrain_tile_crossings"), ("six", "all_terrain_tiles_crossings"),
                                       ("falls", "falls"), ("lane", "out_of_lane_exits"), ("world", "world_exits")):
                        if counts[field] != stored[raw]:
                            raise ValueError(f"family/level mismatch: {path}/{family}/{level}/{field}")
            groups[record["group"]].append(data)
            if record["training_seed"] is not None:
                by_seed[str(record["training_seed"])][record["group"]].append(data)
            audit.append({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "episodes": 175})
    totals = {group: aggregate(groups[group]) for group in GROUPS}
    if any(row["n"] != (300 if group == "frozen_v5" else 900) for group, row in totals.items()):
        raise ValueError("unexpected primary denominators")
    seeds = {seed: {group: aggregate(runs) for group, runs in row.items()} for seed, row in by_seed.items()}
    if set(seeds) != {"42", "43", "44"}:
        raise ValueError("missing seed")
    if any(set(row) != set(GROUPS[1:]) or any(values["n"] != 300 for values in row.values())
           for row in seeds.values()):
        raise ValueError("unbalanced per-seed policy groups")
    hard = {group: {str(level): aggregate(runs, level=level) for level in (3, 4)} for group, runs in groups.items()}
    per_family = {group: {family: {str(level): aggregate(runs, family, level) for level in range(5)}
                         for family in families[:-1]} for group, runs in groups.items()}
    diagnostics = {}
    for path in sorted((args.directory / "diagnostics").glob("*.json")):
        data = json.loads(path.read_text())
        diagnostics[path.stem] = aggregate([data])
    result = {"totals": totals, "seeds": seeds, "hard": hard, "per_family": per_family,
              "promotion_gate": promotion_gate(totals, seeds), "diagnostics_not_in_gate": diagnostics,
              "interpretation": "Matched fixed-budget residual comparison, all three seeds. Frozen v5 is evaluated "
              "once per condition, not replicated. Two held-out terrain maps, not thousands of independent maps. "
              "Ideal raycasts, not real RGB-D. A frozen base/bounded mean is not a safety guarantee."}
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Frozen-base residual v8", "", "6 new runs: 750 iterations x4096 environments x32 steps; 3 seeds/mode.",
             "Fresh geometry/reset62/36 and63/37. 20 primary evaluations /3500 first episodes, at most16s.", "",
             "| Policy | One tile | Six tiles | Falls | Flat falls | Speed m/s |", "|---|---:|---:|---:|---:|---:|"]
    for group, row in totals.items():
        lines.append(f"| {group} | {row['one']}/{row['n']} | {row['six']}/{row['n']} | {row['falls']}/{row['n']} | "
                     f"{row['flat_falls']}/{row['flat_n']} | {row['speed_m_s']:.3f} |")
    lines += ["", "## Highest difficulty1.0", "", "| Policy | One tile | Six tiles | Falls |", "|---|---:|---:|---:|"]
    for group in GROUPS:
        row = hard[group]["4"]
        lines.append(f"| {group} | {row['one']}/{row['n']} | {row['six']}/{row['n']} | {row['falls']}/{row['n']} |")
    lines += ["", "## Every seed", "", "| Seed | Policy | One tile | Six tiles | Falls |", "|---|---|---:|---:|---:|"]
    for seed, row in seeds.items():
        for group, values in row.items():
            lines.append(f"| {seed} | {group} | {values['one']}/{values['n']} | {values['six']}/{values['n']} | "
                         f"{values['falls']}/{values['n']} |")
    lines += ["", f"Promotion gate: **{'PASS' if result['promotion_gate']['passed'] else 'FAIL'}**", "", result["interpretation"],
              "", "Depth zero/shuffle diagnostic episodes are not included in primary totals or promotion."]
    (args.directory / "summary.md").write_text("\n".join(lines) + "\n")
    (args.directory / "verification.json").write_text(json.dumps(
        {"status": "pass", "mismatches": 0, "files": audit, "source_hashes_verified": frozen["source_sha256"],
         "total_primary_first_episodes": sum(row["episodes"] for row in audit)}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
