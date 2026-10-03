#!/usr/bin/env python3
"""Aggregate the v28 demo evaluation into tables, plots and a summary JSON.

Reads ``outputs/combo_v28/eval*/<terrain>/<checkpoint>.json`` and the run records, and writes
``artifacts/combo_v28/`` (results CSV/JSON, consolidated raw evaluations, plots).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from week03_ant import combo_v28_analysis as A  # noqa: E402

OUT = ROOT / "outputs" / "combo_v28"
ART = ROOT / "artifacts" / "combo_v28"

TERRAIN_KO = {
    "flat": "평지", "stick": "Stick 험지", "lim": "Lim 박스", "sticklim": "Stick+Lim 지형",
    "mine": "내 v5 지형", "all": "전체 혼합 지형",
}
REFERENCE_KO = {
    "ref_stick_flat": "Stick flat (원본)",
    "ref_stick_rough": "Stick rough (원본)",
    "ref_stick_control": "Stick control (원본)",
    "ref_stick_recovery": "Stick recovery (원본)",
    "ref_lim_e0": "Lim E0 (원본)",
    "ref_lim_e4": "Lim E4 (원본)",
    "ref_lim_e15": "Lim E15 (원본)",
    "ref_lim_f3a": "Lim F3a (원본)",
}
CONDITION_KO = {
    "flat": "평지", "flat_mu05": "평지 μ0.5", "flat_mu01": "평지 μ0.1", "flat_mu02_mult": "평지 μ0.2×",
    "boxes_5": "박스 5", "boxes_10": "박스 10", "boxes_15": "박스 15", "boxes_fine_10": "잔박스 10",
    "boxes_10_mu02": "박스10 μ0.2", "boxes_10_mu02_mult": "박스10 μ0.2×",
    "rough_5": "요철 5", "rough_10": "요철 10", "stick_rough": "Stick 험지", "wave_15": "물결 15",
    "slope_20": "경사↑", "slope_inv_20": "경사↓", "stairs_10": "계단↑", "stairs_inv_10": "계단↓",
    "isaaclab_rough": "IsaacLab rough",
    "obstacles_10": "장애물", "rails_8": "레일", "cylinders_10": "원기둥", "cones_12": "원뿔",
    "tilted_blocks_8": "기운 블록", "platform_10": "단상",
    "gaps_20": "틈", "pits_15": "구덩이", "stones": "징검다리",
}


def recipe_label(recipe: str) -> str:
    """Human label: ``lim_e5_d1+stock`` -> ``Lim 박스 + entropy + 내 랜덤화 · +600it 원래 보상``."""
    if recipe in REFERENCE_KO:
        return REFERENCE_KO[recipe]
    base, _, arm = recipe.partition("+")
    terrain, entropy, dr = base.split("_")
    if base == "flat_e0_d0":
        label = "제공 baseline"
    elif base == "flat_e0_d1":
        label = "내 Robust42"
    else:
        parts = [TERRAIN_KO[terrain]]
        if entropy == "e5":
            parts.append("entropy")
        if dr == "d1":
            parts.append("내 랜덤화")
        label = " + ".join(parts)
    if arm == "stock":
        label += " · +600it"
    elif arm == "recovery":
        label += " · +600it 회복보상"
    return label


def load_results(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("*/*.json")):
        if path.parent.name in ("jobs", "logs"):
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def recipe_rows(entries: dict[str, A.RecipeScore]) -> list[list]:
    rows = []
    for entry in A.rank(entries):
        demo_mean, demo_std = entry.demo
        away_mean, _ = entry.away
        balanced_mean, _ = entry.balanced
        worst_name, worst_value = entry.worst_condition
        categories = entry.category_means()
        conditions = entry.condition_means()
        rows.append([
            entry.recipe, recipe_label(entry.recipe), len(entry.checkpoints),
            " ".join(str(seed) for seed in entry.seeds),
            fmt(demo_mean), fmt(demo_std), fmt(balanced_mean), fmt(away_mean),
            fmt(conditions["boxes_10"]), fmt(conditions["flat"]),
            fmt(entry.fall_rate, 3), fmt(entry.distance), worst_name, fmt(worst_value),
            *[fmt(categories[name]) for name in A.CATEGORIES],
        ])
    return rows


RECIPE_HEADER = [
    "recipe", "label", "checkpoints", "seeds", "demo_mean", "demo_seed_std", "category_balanced",
    "away_from_home_mean", "boxes_10", "flat", "fall_rate", "distance_m", "worst_condition", "worst_return",
    *[f"cat_{name}" for name in A.CATEGORIES],
]


def condition_rows(entries: dict[str, A.RecipeScore]) -> list[list]:
    rows = []
    for entry in A.rank(entries):
        means = entry.condition_means()
        rows.append([entry.recipe, recipe_label(entry.recipe), *[fmt(means[name]) for name in A.CONDITIONS]])
    return rows


def checkpoint_rows(scores: dict[str, A.CheckpointScore]) -> list[list]:
    rows = []
    for key, score in sorted(scores.items(), key=lambda item: -item[1].demo_score if item[1].complete else 0):
        if not score.complete:
            continue
        rows.append([key, fmt(score.demo_score), fmt(score.category_balanced), fmt(score.away_score),
                     fmt(score.fall_rate, 3), *[fmt(score.returns[name]) for name in A.CONDITIONS]])
    return rows


def training_rows() -> list[list]:
    rows = []
    for path in sorted((OUT / "run_records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        final = record["final_checkpoint"]
        init = record.get("init_checkpoint") or {}
        rows.append([
            record["run_name"], record["task"], record["seed"], record["entropy_coef"], record["max_iterations"],
            record["transitions"], fmt(record["train_seconds"], 1), final["sha256"],
            record.get("direct_sha256_match"), init.get("sha256", ""), final["iter"],
            " ".join(f"{value:.3f}" for value in (final.get("action_std") or [])),
        ])
    return rows


def consolidate_raw(rows: list[dict], path: Path) -> None:
    """All raw results in one file; checkpoint paths become repository-relative (the only change)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(ROOT) + "/"
    # mtime=0 keeps the gzip bytes reproducible for the same results.
    with gzip.GzipFile(path, "wb", mtime=0) as raw:
        for row in sorted(rows, key=lambda item: (item["terrain"], item["checkpoint_id"])):
            row = dict(row, checkpoint=row["checkpoint"].replace(prefix, ""))
            raw.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))


def summarize(folder: Path, label: str) -> dict:
    rows = load_results(folder)
    scores = A.collect(rows)
    entries = A.recipes(scores)
    results = ART / "results"
    write_csv(results / f"{label}_recipes.csv", RECIPE_HEADER, recipe_rows(entries))
    write_csv(results / f"{label}_conditions.csv", ["recipe", "label", *A.CONDITIONS], condition_rows(entries))
    write_csv(results / f"{label}_checkpoints.csv",
              ["checkpoint", "demo", "category_balanced", "away_from_home", "fall_rate", *A.CONDITIONS],
              checkpoint_rows(scores))
    consolidate_raw(rows, results / f"{label}_evaluations.jsonl.gz")
    ranked = A.rank(entries)
    return {
        "results": len(rows),
        "checkpoints_complete": sum(score.complete for score in scores.values()),
        "ranking": [
            {"recipe": entry.recipe, "label": recipe_label(entry.recipe), "seeds": entry.seeds,
             "demo_mean": entry.demo[0], "demo_seed_std": entry.demo[1], "fall_rate": entry.fall_rate,
             "category_means": entry.category_means()}
            for entry in ranked
        ],
        "main_effects": A.main_effects(entries),
        "entries": entries,
        "scores": scores,
    }


def markdown_tables(selection: dict, confirmation: dict | None) -> str:
    """README-ready tables: ranking of every recipe and the per-category breakdown."""
    entries = selection["entries"]
    ranked = A.rank(entries)
    lines = ["| 순위 | 조합 | seed | 데모 점수 (28조건 평균) | 범주 균형 | 넘어짐률 | 박스 ±10 cm | 평지 | 최악 조건 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for index, entry in enumerate(ranked, start=1):
        mean, std = entry.demo
        conditions = entry.condition_means()
        worst, worst_value = entry.worst_condition
        seeds = len(entry.checkpoints) if not entry.recipe.startswith("ref_") else "원본"
        spread = f" ± {std:.1f}" if len(entry.checkpoints) > 1 else ""
        lines.append(
            f"| {index} | {recipe_label(entry.recipe)} | {seeds} | {mean:.1f}{spread} | {entry.balanced[0]:.1f} | "
            f"{entry.fall_rate * 100:.0f}% | {conditions['boxes_10']:.1f} | {conditions['flat']:.1f} | "
            f"{CONDITION_KO[worst]} {worst_value:.1f} |"
        )
    lines += ["", "| 조합 | " + " | ".join(A.CATEGORY_LABELS_KO[c] for c in A.CATEGORIES) + " |",
              "|---|" + "---:|" * len(A.CATEGORIES)]
    for entry in ranked:
        categories = entry.category_means()
        lines.append(f"| {recipe_label(entry.recipe)} | " + " | ".join(f"{categories[c]:.1f}" for c in A.CATEGORIES) + " |")
    if confirmation and confirmation["entries"]:
        lines += ["", "| 확인 평가 순위 | 조합 | 데모 점수 (새 지형 seed 2029) | 넘어짐률 |", "|---:|---|---:|---:|"]
        for index, entry in enumerate(A.rank(confirmation["entries"]), start=1):
            mean, std = entry.demo
            spread = f" ± {std:.1f}" if len(entry.checkpoints) > 1 else ""
            lines.append(f"| {index} | {recipe_label(entry.recipe)} | {mean:.1f}{spread} | {entry.fall_rate * 100:.0f}% |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", action="store_true", help="also render the PNG figures")
    args = parser.parse_args()

    summary = {"selection_terrain_seed": 2028, "confirmation_terrain_seed": 2029}
    selection = summarize(OUT / "eval", "selection")
    summary["selection"] = {k: v for k, v in selection.items() if k not in ("entries", "scores")}
    confirmation = None
    if (OUT / "eval_confirm").exists():
        confirmation = summarize(OUT / "eval_confirm", "confirmation")
        summary["confirmation"] = {k: v for k, v in confirmation.items() if k not in ("entries", "scores")}
    write_csv(ART / "results" / "training_runs.csv",
              ["run", "task", "seed", "entropy_coef", "iterations", "transitions", "train_seconds",
               "final_sha256", "direct_read_sha_match", "init_sha256", "final_iter", "action_std"],
              training_rows())
    (ART / "results" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    (ART / "results" / "tables.md").write_text(markdown_tables(selection, confirmation), encoding="utf-8")
    if args.plots:
        from combo_v28_plots import render_all  # noqa: E402  (scripts/ on sys.path)

        render_all(selection, confirmation, ART / "plots", recipe_label, CONDITION_KO)
    top = summary["selection"]["ranking"][:8]
    for item in top:
        print(f"{item['demo_mean']:7.2f} ± {item['demo_seed_std']:5.2f}  {item['recipe']:<24s} {item['label']}")


if __name__ == "__main__":
    main()
