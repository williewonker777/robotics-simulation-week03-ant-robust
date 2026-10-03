"""Pure-Python aggregation for the v28 demo evaluation (no Isaac Sim import).

A *result* is one checkpoint on one terrain condition: the course ``play_one_episode.py`` statistics
(mean/std of the first-episode return over 100 environments) plus fall rate and forward distance.
The demo score of a checkpoint is the plain mean of its returns over all conditions; a recipe score
is the mean and sample standard deviation of that score over its training seeds.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev

CATEGORIES = {
    "flat_friction": ("flat", "flat_mu05", "flat_mu01", "flat_mu02_mult"),
    "boxes": ("boxes_5", "boxes_10", "boxes_15", "boxes_fine_10", "boxes_10_mu02", "boxes_10_mu02_mult"),
    "rough_wave": ("rough_5", "rough_10", "stick_rough", "wave_15"),
    "slope_stairs": ("slope_20", "slope_inv_20", "stairs_10", "stairs_inv_10", "isaaclab_rough"),
    "obstacles": ("obstacles_10", "rails_8", "cylinders_10", "cones_12", "tilted_blocks_8", "platform_10"),
    "gaps_holes": ("gaps_20", "pits_15", "stones"),
}
CONDITIONS = tuple(name for names in CATEGORIES.values() for name in names)
CATEGORY_OF = {name: category for category, names in CATEGORIES.items() for name in names}
HOME_CONDITIONS = ("flat", "boxes_10", "stick_rough")

CATEGORY_LABELS_KO = {
    "flat_friction": "평지·저마찰",
    "boxes": "박스",
    "rough_wave": "요철·물결",
    "slope_stairs": "경사·계단",
    "obstacles": "장애물",
    "gaps_holes": "틈·구덩이·징검다리",
}


@dataclass
class CheckpointScore:
    checkpoint_id: str
    returns: dict[str, float]
    falls: dict[str, float]
    distances: dict[str, float]

    @property
    def complete(self) -> bool:
        return all(name in self.returns for name in CONDITIONS)

    @property
    def demo_score(self) -> float:
        return mean(self.returns[name] for name in CONDITIONS)

    @property
    def fall_rate(self) -> float:
        return mean(self.falls[name] for name in CONDITIONS)

    @property
    def distance(self) -> float:
        return mean(self.distances[name] for name in CONDITIONS)

    @property
    def worst(self) -> tuple[str, float]:
        name = min(CONDITIONS, key=lambda key: self.returns[key])
        return name, self.returns[name]

    @property
    def away_score(self) -> float:
        """Mean over conditions that are nobody's exact training terrain."""
        return mean(self.returns[name] for name in CONDITIONS if name not in HOME_CONDITIONS)

    def category_scores(self) -> dict[str, float]:
        return {category: mean(self.returns[name] for name in names) for category, names in CATEGORIES.items()}

    @property
    def category_balanced(self) -> float:
        return mean(self.category_scores().values())


def collect(results: list[dict]) -> dict[str, CheckpointScore]:
    """Group raw evaluation records by checkpoint id."""
    scores: dict[str, CheckpointScore] = {}
    for result in results:
        key = result["checkpoint_id"]
        if key.endswith("__repeat"):
            continue
        score = scores.setdefault(key, CheckpointScore(key, {}, {}, {}))
        terrain = result["terrain"]
        if terrain in score.returns:
            raise ValueError(f"duplicate result for {key} on {terrain}")
        score.returns[terrain] = float(result["return_mean"])
        score.falls[terrain] = float(result["fall_rate"])
        score.distances[terrain] = float(result["distance_mean"])
    return scores


def split_checkpoint_id(checkpoint_id: str) -> tuple[str, int | None]:
    """``lim_e5_d0_s42`` -> (``lim_e5_d0``, 42); references keep ``None`` as seed."""
    if checkpoint_id.startswith("ref_"):
        return checkpoint_id, None
    recipe, seed = checkpoint_id.rsplit("_s", 1)
    return recipe, int(seed)


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


@dataclass
class RecipeScore:
    recipe: str
    seeds: list[int] = field(default_factory=list)
    checkpoints: list[CheckpointScore] = field(default_factory=list)

    def _over_seeds(self, getter) -> tuple[float, float]:
        values = [getter(score) for score in self.checkpoints]
        return mean(values), sample_std(values)

    @property
    def demo(self) -> tuple[float, float]:
        return self._over_seeds(lambda score: score.demo_score)

    @property
    def away(self) -> tuple[float, float]:
        return self._over_seeds(lambda score: score.away_score)

    @property
    def balanced(self) -> tuple[float, float]:
        return self._over_seeds(lambda score: score.category_balanced)

    @property
    def fall_rate(self) -> float:
        return mean(score.fall_rate for score in self.checkpoints)

    @property
    def distance(self) -> float:
        return mean(score.distance for score in self.checkpoints)

    def condition_means(self) -> dict[str, float]:
        return {name: mean(score.returns[name] for score in self.checkpoints) for name in CONDITIONS}

    def category_means(self) -> dict[str, float]:
        per_condition = self.condition_means()
        return {category: mean(per_condition[name] for name in names) for category, names in CATEGORIES.items()}

    @property
    def worst_condition(self) -> tuple[str, float]:
        per_condition = self.condition_means()
        name = min(per_condition, key=per_condition.get)
        return name, per_condition[name]


def recipes(scores: dict[str, CheckpointScore]) -> dict[str, RecipeScore]:
    """Group complete checkpoint scores into recipes (references are single-checkpoint recipes).

    ``seeds[i]`` is the training seed of ``checkpoints[i]``.
    """
    grouped: dict[str, RecipeScore] = {}
    for key, score in sorted(scores.items()):
        if not score.complete:
            continue
        recipe, seed = split_checkpoint_id(key)
        entry = grouped.setdefault(recipe, RecipeScore(recipe))
        entry.checkpoints.append(score)
        if seed is not None:
            entry.seeds.append(seed)
    return grouped


def rank(entries: dict[str, RecipeScore], min_seeds: int = 1) -> list[RecipeScore]:
    """Highest mean demo score first; ties broken by lower fall rate, then by name."""
    eligible = [entry for entry in entries.values() if len(entry.checkpoints) >= min_seeds]
    return sorted(eligible, key=lambda entry: (-entry.demo[0], entry.fall_rate, entry.recipe))


def factor_levels(recipe: str) -> dict[str, str] | None:
    """Stage-1 recipe ``terrain_entropy_dr`` -> factor levels (``None`` for other recipes)."""
    if recipe.startswith("ref_") or "+" in recipe:
        return None
    terrain, entropy, dr = recipe.split("_")
    return {"terrain": terrain, "entropy": entropy, "dr": dr}


def main_effects(entries: dict[str, RecipeScore]) -> dict[str, dict[str, float]]:
    """Mean checkpoint demo score per factor level, pooled over the other factors and seeds."""
    pooled: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries.values():
        levels = factor_levels(entry.recipe)
        if levels is None:
            continue
        for factor, level in levels.items():
            pooled[factor][level].extend(score.demo_score for score in entry.checkpoints)
    return {factor: {level: mean(values) for level, values in levels.items()} for factor, levels in pooled.items()}


def paired_difference(a: RecipeScore, b: RecipeScore) -> dict | None:
    """Seed-paired difference of demo scores ``a - b`` (same training seeds)."""
    by_seed_a = dict(zip(a.seeds, a.checkpoints))
    by_seed_b = dict(zip(b.seeds, b.checkpoints))
    common = sorted(set(by_seed_a) & set(by_seed_b))
    if not common:
        return None
    diffs = [by_seed_a[seed].demo_score - by_seed_b[seed].demo_score for seed in common]
    return {
        "seeds": common,
        "differences": diffs,
        "mean": mean(diffs),
        "std": sample_std(diffs),
        "wins": sum(diff > 0 for diff in diffs),
    }


def is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)
