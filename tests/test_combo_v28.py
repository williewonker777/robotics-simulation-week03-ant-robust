"""CPU checks for the v28 combination study (no Isaac Sim needed)."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from week03_ant import combo_v28_analysis as analysis

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "src" / "week03_ant" / "tasks" / "combo_v28_cfg.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_combo_v28", ROOT / "scripts" / "run_combo_v28.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_combo_v28"] = module
    spec.loader.exec_module(module)
    return module


def _cfg_conditions() -> list[tuple[str, str]]:
    """(name, category) of every EvalCondition(...) literal in the config module, in order."""
    tree = ast.parse(CFG.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "EvalCondition":
            found.append((node.args[0].value, node.args[1].value))
    return found


def test_condition_lists_agree_across_config_runner_and_analysis():
    runner = _load_runner()
    cfg = _cfg_conditions()
    assert [name for name, _ in cfg] == list(analysis.CONDITIONS) == runner.EVAL_NAMES
    assert len(cfg) == 28
    assert {name: category for name, category in cfg} == analysis.CATEGORY_OF
    assert set(analysis.HOME_CONDITIONS) <= set(analysis.CONDITIONS)


def test_recipe_grid_and_reused_cells():
    runner = _load_runner()
    assert len(runner.ALL_RECIPES) == 6 * 2 * 2
    assert set(runner.REUSED) == {"flat_e0_d0", "flat_e0_d1"}
    assert len(runner.STAGE1_RECIPES) == 22
    assert runner.parse_recipe("sticklim_e5_d1") == ("sticklim", "e5", "d1")
    assert runner.task_id("sticklim", "d1", "recovery") == "Week03-Ant-Combo-v28-Sticklim-D1-Recovery"
    with pytest.raises(ValueError):
        runner.parse_recipe("rocks_e5_d1")
    for recipe in runner.REUSED:
        for seed in runner.SEEDS:
            assert runner.checkpoint_for(recipe, seed).exists()


def test_direct_and_cached_sha_agree(tmp_path):
    runner = _load_runner()
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(range(256)) * 9000 + b"tail")
    try:
        direct = runner.direct_sha256(path)
    except OSError as error:  # filesystems without O_DIRECT
        pytest.skip(f"O_DIRECT unsupported: {error}")
    assert direct == runner.sha256(path)


def _result(checkpoint, terrain, ret, fall=0.0, dist=0.0):
    return {"checkpoint_id": checkpoint, "terrain": terrain, "return_mean": ret, "fall_rate": fall,
            "distance_mean": dist}


def _full(checkpoint, base, bump=None):
    bump = bump or {}
    return [_result(checkpoint, name, base + bump.get(name, 0.0), fall=0.1) for name in analysis.CONDITIONS]


def test_demo_score_is_plain_mean_over_all_conditions():
    scores = analysis.collect(_full("lim_e5_d0_s42", 10.0, {"flat": 28.0}))
    score = scores["lim_e5_d0_s42"]
    assert score.complete
    assert score.demo_score == pytest.approx(11.0)
    assert score.away_score == pytest.approx(10.0)
    assert score.worst[1] == pytest.approx(10.0)
    assert score.category_scores()["flat_friction"] == pytest.approx(17.0)


def test_incomplete_and_repeat_results_are_excluded():
    rows = _full("a_e0_d0_s42", 5.0)[:-1] + [_result("a_e0_d0_s42__repeat", "flat", 99.0)]
    scores = analysis.collect(rows)
    assert "a_e0_d0_s42__repeat" not in scores
    assert not scores["a_e0_d0_s42"].complete
    assert analysis.recipes(scores) == {}


def test_duplicate_result_is_an_error():
    with pytest.raises(ValueError, match="duplicate"):
        analysis.collect([_result("x_s1", "flat", 1.0), _result("x_s1", "flat", 2.0)])


def test_recipe_ranking_seed_statistics_and_effects():
    rows = []
    for seed, value in ((42, 10.0), (43, 12.0), (44, 14.0)):
        rows += _full(f"lim_e5_d0_s{seed}", value)
        rows += _full(f"lim_e0_d0_s{seed}", value - 4.0)
    rows += _full("ref_lim_f3a", 30.0)
    entries = analysis.recipes(analysis.collect(rows))
    assert entries["lim_e5_d0"].demo[0] == pytest.approx(12.0)
    assert entries["lim_e5_d0"].demo[1] == pytest.approx(2.0)
    assert entries["lim_e5_d0"].seeds == [42, 43, 44]
    ranked = analysis.rank(entries, min_seeds=3)
    assert [entry.recipe for entry in ranked] == ["lim_e5_d0", "lim_e0_d0"]
    assert analysis.rank(entries)[0].recipe == "ref_lim_f3a"
    effects = analysis.main_effects(entries)
    assert effects["entropy"]["e5"] - effects["entropy"]["e0"] == pytest.approx(4.0)
    paired = analysis.paired_difference(entries["lim_e5_d0"], entries["lim_e0_d0"])
    assert paired["mean"] == pytest.approx(4.0)
    assert paired["wins"] == 3


def test_checkpoint_ids_split_into_recipe_and_seed():
    assert analysis.split_checkpoint_id("all_e5_d1_s44") == ("all_e5_d1", 44)
    assert analysis.split_checkpoint_id("lim_e5_d0+stock_s43") == ("lim_e5_d0+stock", 43)
    assert analysis.split_checkpoint_id("ref_stick_rough") == ("ref_stick_rough", None)
    assert analysis.factor_levels("mine_e0_d1") == {"terrain": "mine", "entropy": "e0", "dr": "d1"}
    assert analysis.factor_levels("lim_e5_d0+stock") is None
