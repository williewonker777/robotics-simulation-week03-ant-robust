import importlib.util
from pathlib import Path
import sys


scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("residual_summary", scripts / "summarize_residual.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evidence():
    groups = {name: dict(n=100, one=80, six=40, falls=10, world=0, flat_falls=1, flat_n=10)
              for name in module.GROUPS}
    groups["residual_footmap"]["one"] = 81
    seeds = {str(seed): {"residual_footmap": {"one": 81}, "residual_blind": {"one": 80}} for seed in (42, 43, 44)}
    return groups, seeds


def test_balanced_improvement_passes():
    assert module.promotion_gate(*evidence())["passed"]


def test_long_traversal_regression_fails_despite_one_tile_improvement():
    groups, seeds = evidence()
    groups["residual_footmap"]["six"] = 39
    result = module.promotion_gate(groups, seeds)
    assert not result["passed"] and not result["checks"]["six_vs_frozen_v5"]


def test_seed_majority_is_required_not_only_pooled_improvement():
    groups, seeds = evidence()
    for seed in ("43", "44"):
        seeds[seed]["residual_footmap"]["one"] = 79
    assert not module.promotion_gate(groups, seeds)["passed"]


def test_distinct_denominators_use_rates():
    groups, seeds = evidence()
    for key in ("n", "one", "six", "falls", "flat_n", "flat_falls"):
        groups["frozen_v5"][key] *= 3
    assert module.promotion_gate(groups, seeds)["passed"]
