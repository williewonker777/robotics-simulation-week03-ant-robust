"""Static figures for the v28 README (matplotlib, light surface).

Palette (validated with the dataviz checker, light mode): accent blue for the selected recipe,
orange for the teammates' original checkpoints, aqua for my original models (provided baseline and
Robust42), recessive gray for every other recipe.  Values are direct-labelled where a static image has
no tooltip; the full numbers live in the CSV tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from week03_ant import combo_v28_analysis as A  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"
TEAMMATE = "#eb6834"
MINE = "#1baf7a"
OTHER = "#c9c8c1"
BLUE_RAMP = ["#f0f6fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _role(recipe: str, winner: str) -> str:
    if recipe == winner:
        return "winner"
    if recipe.startswith("ref_"):
        return "teammate"
    if recipe in ("flat_e0_d0", "flat_e0_d1"):
        return "mine"
    return "other"


COLORS = {"winner": ACCENT, "teammate": TEAMMATE, "mine": MINE, "other": OTHER}


def ranking(entries: dict, label, path: Path, title: str, winner: str | None = None) -> None:
    ranked = A.rank(entries)
    if not ranked:
        return
    v28 = [entry for entry in ranked if not entry.recipe.startswith("ref_")]
    winner = winner or (v28[0].recipe if v28 else ranked[0].recipe)
    labelled = {entry.recipe for entry in v28[:3]} | {winner, "flat_e0_d0", "flat_e0_d1"}
    height = max(4.0, 0.26 * len(ranked) + 1.4)
    fig, ax = plt.subplots(figsize=(11, height))
    y = np.arange(len(ranked))[::-1]
    for row, entry in zip(y, ranked):
        role = _role(entry.recipe, winner)
        mean, std = entry.demo
        ax.barh(row, mean, height=0.62, color=COLORS[role], linewidth=0, zorder=2)
        if len(entry.checkpoints) > 1:
            ax.plot([mean - std, mean + std], [row, row], color=INK2, linewidth=1, zorder=3)
        if role != "other" or entry.recipe in labelled:
            text = f"{mean:.1f}" + (f" ± {std:.1f}" if len(entry.checkpoints) > 1 else "")
            ax.text(mean + std + 0.8 if len(entry.checkpoints) > 1 else mean + 0.8, row, text,
                    va="center", ha="left", fontsize=8.5, color=INK, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([label(entry.recipe) for entry in ranked], fontsize=8.5)
    ax.set_xlabel("28개 조건 평균 점수 (환경 100개, 첫 에피소드 보상 합)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(entry.demo[0] + entry.demo[1] for entry in ranked) * 1.12)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    handles = [Patch(color=ACCENT, label="제출 조합"), Patch(color=TEAMMATE, label="팀원 모델"),
               Patch(color=MINE, label="제공 baseline, 이전 제출 모델"), Patch(color=OTHER, label="다른 조합")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap(entries: dict, recipes: list[str], label, condition_label: dict, path: Path, title: str) -> None:
    rows = [entries[recipe] for recipe in recipes if recipe in entries]
    if not rows:
        return
    data = np.array([[entry.condition_means()[name] for name in A.CONDITIONS] for entry in rows])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    vmax = max(140.0, float(data.max()))
    fig, ax = plt.subplots(figsize=(15, 0.42 * len(rows) + 2.4))
    ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if value > 0.55 * vmax else INK)
    ax.set_xticks(range(len(A.CONDITIONS)))
    ax.set_xticklabels([condition_label[name] for name in A.CONDITIONS], rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label(entry.recipe) for entry in rows], fontsize=8.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    start = 0
    for category, names in A.CATEGORIES.items():
        end = start + len(names)
        if start:
            ax.axvline(start - 0.5, color=SURFACE, linewidth=3)
        ax.text((start + end - 1) / 2, 1.015, A.CATEGORY_LABELS_KO[category], ha="center", va="bottom",
                fontsize=9, color=INK2, transform=ax.get_xaxis_transform())
        start = end
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def factors(effects: dict, path: Path, terrain_label: dict) -> None:
    if not effects:
        return
    panels = [("terrain", "학습 지형"), ("entropy", "PPO entropy"), ("dr", "랜덤화")]
    level_names = {"e0": "0 (원래)", "e5": "0.005", "d0": "끔", "d1": "켬"}
    widths = [len(effects.get(key, {})) or 1 for key, _ in panels]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": widths})
    for ax, (key, name) in zip(axes, panels):
        levels = effects.get(key, {})
        order = sorted(levels, key=levels.get, reverse=True)
        values = [levels[level] for level in order]
        x = np.arange(len(order))
        ax.bar(x, values, width=0.55, color=ACCENT, linewidth=0, zorder=2)
        for xi, value in zip(x, values):
            ax.text(xi, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([terrain_label.get(level, level_names.get(level, level)) for level in order],
                           fontsize=8.5, rotation=20 if key == "terrain" else 0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(name, fontsize=10.5, color=INK, loc="left")
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, max(values) * 1.18 if values else 1)
    axes[0].set_ylabel("평균 데모 점수")
    fig.suptitle("요소별 평균 점수 (첫 비교 24조합, 다른 요소는 평균)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def continuation(entries: dict, label, path: Path) -> None:
    parents = sorted({recipe.split("+")[0] for recipe in entries if "+" in recipe})
    if not parents:
        return
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(parents) + 1.6))
    marks = [("", "1,000번 학습", OTHER), ("+stock", "600번 추가 학습", ACCENT),
             ("+recovery", "회복 보상으로 600번 추가 학습", TEAMMATE)]
    for row, parent in enumerate(parents[::-1]):
        points = []
        for suffix, _, color in marks:
            entry = entries.get(parent + suffix)
            if entry is None:
                continue
            points.append(entry.demo[0])
            ax.scatter(entry.demo[0], row, s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(entry.demo[0], row + 0.22, f"{entry.demo[0]:.1f}", ha="center", fontsize=8, color=INK)
        if points:
            ax.plot([min(points), max(points)], [row, row], color=AXIS, linewidth=2, zorder=2)
    ax.set_yticks(range(len(parents)))
    ax.set_yticklabels([label(parent) for parent in parents[::-1]], fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("28개 조건 평균 점수 (학습 seed 3개 평균)")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(parents) - 0.2)
    ax.legend(handles=[Patch(color=color, label=name) for _, name, color in marks], frameon=False, fontsize=8.5,
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    ax.set_title("추가 학습 비교", loc="left", fontsize=12, color=INK, pad=26)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


LADDER = [
    ("flat_e0_d0", "제공 baseline\n(평지에서 학습)"),
    ("sticklim_e0_d0", "1단계\n험지에서 학습"),
    ("sticklim_e5_d0", "2단계\nentropy 0.005"),
    ("sticklim_e5_d0+stock", "3단계\n600번 추가 학습"),
]
"""Stages from the provided baseline to the submission recipe; every stage is a trained, scored recipe."""

DETOURS = [
    ("sticklim_e5_d1", "sticklim_e5_d0", "2단계에 랜덤화 추가"),
    ("sticklim_e5_d0+recovery", "sticklim_e5_d0+stock", "3단계를 회복 보상으로"),
    ("mine_e5_d0", "sticklim_e5_d0", "1단계를 이전 험지 세트로"),
]
"""Alternatives measured at the same point of the ladder (seed-paired)."""


def ladder(entries: dict, path: Path) -> None:
    """Left: demo score of each ladder stage (3-seed mean, seed spread). Right: seed-paired change of the detours."""
    if any(recipe not in entries for recipe, _ in LADDER):
        return
    fig, (ax, side) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})
    means = [entries[recipe].demo[0] for recipe, _ in LADDER]
    stds = [entries[recipe].demo[1] for recipe, _ in LADDER]
    colors = [MINE] + [OTHER] * (len(LADDER) - 2) + [ACCENT]
    x = np.arange(len(LADDER))
    ax.bar(x, means, width=0.58, color=colors, linewidth=0, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=INK2, elinewidth=1, capsize=0, zorder=3)
    for xi, (mean, std) in enumerate(zip(means, stds)):
        ax.text(xi, mean + std + 1.5, f"{mean:.1f}", ha="center", va="bottom", fontsize=12, color=INK, weight="bold")
    for xi in range(1, len(LADDER)):
        step = A.paired_difference(entries[LADDER[xi][0]], entries[LADDER[xi - 1][0]])
        sign = "+" if step["mean"] >= 0 else ""
        ax.text(xi - 0.5, max(means[xi - 1], means[xi]) + 8, f"{sign}{step['mean']:.1f}",
                ha="center", va="bottom", fontsize=11, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in LADDER], fontsize=9.5)
    ax.set_ylabel("28개 조건 평균 점수")
    ax.set_ylim(0, max(means) * 1.32)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_title("단계별 점수 (학습 seed 3개 평균)", loc="left", fontsize=11.5, color=INK)

    rows = []
    for recipe, base, label in DETOURS:
        if recipe in entries and base in entries:
            diff = A.paired_difference(entries[recipe], entries[base])
            rows.append((label, diff["mean"], diff["wins"], len(diff["seeds"])))
    y = np.arange(len(rows))[::-1]
    for yi, (label, mean, wins, count) in zip(y, rows):
        side.barh(yi, mean, height=0.55, color=OTHER, linewidth=0, zorder=2)
        side.text(mean - 0.6, yi, f"{mean:.1f}", ha="right", va="center", fontsize=10, color=INK)
    side.axvline(0, color=AXIS, linewidth=1, zorder=1)
    side.set_yticks(y)
    side.set_yticklabels([label for label, *_ in rows], fontsize=9.5)
    side.set_xlim(min(row[1] for row in rows) * 1.35, 2)
    side.set_xlabel("같은 단계와 비교한 점수 차이")
    side.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    side.set_axisbelow(True)
    side.tick_params(axis="y", length=0)
    side.set_title("효과가 없었던 방법", loc="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_all(selection: dict, confirmation: dict | None, out_dir: Path, label, condition_label: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = selection["entries"]
    v28 = [entry for entry in A.rank(entries) if not entry.recipe.startswith("ref_")]
    winner = v28[0].recipe if v28 else None
    ranking(entries, label, out_dir / "ranking_selection.png",
            "조합별 28개 조건 평균 점수", winner)
    top = [entry.recipe for entry in v28[:6]]
    refs = ["flat_e0_d0", "flat_e0_d1", "ref_stick_rough", "ref_stick_recovery", "ref_lim_e15", "ref_lim_f3a"]
    heatmap(entries, top + [recipe for recipe in refs if recipe not in top], label, condition_label,
            out_dir / "conditions_selection.png", "조건별 점수 (상위 조합과 비교 모델, 학습 seed 3개 평균)")
    terrain_label = {"flat": "평지", "stick": "Stick 험지", "lim": "Lim 박스", "sticklim": "Stick+Lim",
                     "mine": "이전 험지 세트", "all": "전체 혼합"}
    factors(A.main_effects(entries), out_dir / "factor_effects.png", terrain_label)
    continuation(entries, label, out_dir / "continuation.png")
    ladder(entries, out_dir / "improvement_ladder.png")
    if confirmation and confirmation["entries"]:
        ranking(confirmation["entries"], label, out_dir / "ranking_confirmation.png",
                "새로 만든 지형에서 다시 평가한 점수", winner)
