#!/usr/bin/env python3
"""Compare fresh-process scores (one checkpoint per Isaac Sim process, as the official script runs) with the
main v28 evaluation (several checkpoints per process) and write ``artifacts/combo_v28/results/fresh_process_*``.

Inputs: ``outputs/combo_v28/submission_fresh`` and ``outputs/combo_v28/fresh_top`` (terrain seed 2028).
For a checkpoint scored fresh on all 28 conditions its fresh demo score is the plain 28-condition mean.
For a checkpoint scored fresh only on some conditions, the "partly fresh" score replaces only those conditions.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from week03_ant import combo_v28_analysis as A  # noqa: E402

OUT = ROOT / "outputs" / "combo_v28"
RESULTS = ROOT / "artifacts" / "combo_v28" / "results"


def load(folder: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for path in folder.glob("*/*.json"):
        if path.parent.name in ("jobs", "logs"):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[(row["checkpoint_id"], row["terrain"])] = row
    return rows


def main() -> None:
    fresh = {**load(OUT / "submission_fresh"), **load(OUT / "fresh_top")}
    main_rows = load(OUT / "eval")
    per_condition, checkpoints = [], {}
    for (cid, terrain), row in sorted(fresh.items()):
        ref = main_rows[(cid, terrain)]
        if row["checkpoint_sha256"] != ref["checkpoint_sha256"]:
            raise SystemExit(f"SHA differs for {cid} on {terrain}")
        same = row["returns"] == ref["returns"]
        per_condition.append([cid, terrain, f"{row['return_mean']:.6f}", f"{ref['return_mean']:.6f}",
                              f"{row['return_mean'] - ref['return_mean']:+.6f}", same])
        checkpoints.setdefault(cid, {})[terrain] = (row["return_mean"], ref["return_mean"], same)

    summary = {}
    for cid, conditions in checkpoints.items():
        main_all = {name: main_rows[(cid, name)]["return_mean"] for name in A.CONDITIONS}
        mixed = dict(main_all)
        for name, (value, _, _) in conditions.items():
            mixed[name] = value
        summary[cid] = {
            "fresh_conditions": len(conditions),
            "identical_conditions": sum(same for *_, same in conditions.values()),
            "differing_conditions": sorted(name for name, (*_, same) in conditions.items() if not same),
            "max_abs_condition_diff": max(abs(v - m) for v, m, _ in conditions.values()),
            "main_demo": st.mean(main_all.values()),
            "fresh_demo" if len(conditions) == len(A.CONDITIONS) else "partly_fresh_demo": st.mean(mixed.values()),
        }

    recipes = {}
    for cid, item in summary.items():
        recipe, _ = A.split_checkpoint_id(cid)
        key = "fresh_demo" if "fresh_demo" in item else "partly_fresh_demo"
        recipes.setdefault((recipe, key), []).append((item["main_demo"], item[key]))
    recipe_rows = []
    for (recipe, key), values in sorted(recipes.items(), key=lambda kv: -st.mean(v for _, v in kv[1])):
        main_mean = st.mean(m for m, _ in values)
        new_mean = st.mean(v for _, v in values)
        recipe_rows.append({"recipe": recipe, "checkpoints": len(values), "kind": key,
                            "main_demo": main_mean, key: new_mean, "difference": new_mean - main_mean})

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "fresh_process_conditions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["checkpoint", "condition", "fresh_return", "main_return", "difference", "per_env_identical"])
        writer.writerows(per_condition)
    report = {
        "rule": "fresh = one checkpoint per new Isaac Sim process (the official play_one_episode situation), terrain seed 2028; "
                "main = the v28 evaluation with several checkpoints per process",
        "checkpoints": summary,
        "recipes": recipe_rows,
    }
    (RESULTS / "fresh_process_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in recipe_rows:
        key = row["kind"]
        print(f"{row['recipe']:<24s} n={row['checkpoints']} main {row['main_demo']:7.3f}  {key} {row[key]:7.3f}  ({row['difference']:+.3f})")
    for cid, item in sorted(summary.items()):
        print(f"  {cid:<26s} identical {item['identical_conditions']}/{item['fresh_conditions']}  differs {item['differing_conditions']}")


if __name__ == "__main__":
    main()
