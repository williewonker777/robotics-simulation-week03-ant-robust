"""CPU regression checks for v18 raw-audit projection and fresh-map guards."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import sys

import pytest

from week03_ant.style_study_v18 import ROOT, SCHEMA, sha

sys.path.insert(0, str(ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("v18_summary_test", ROOT / "scripts/summarize_style_v18.py")
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def _raw():
    source = ROOT / "artifacts/terrain_demo/learning_progress_v17/evaluations/fixed__geometry102_reset64.json"
    data = json.loads(source.read_text())
    data.update(schema=SCHEMA, controller="always", phase="smoke",
                evaluation_plan_sha256=sha(ROOT / "docs/experiment_plans/terrain_style_v18.md"))
    return data


def test_v18_projection_reuses_strict_physical_audit():
    rows = summary.audit(_raw())
    assert len(rows) == 175
    assert sum(row["family"] == "flat" for row in rows) == 25


@pytest.mark.parametrize("field,value", [
    ("schema", "wrong"), ("controller", "lp"), ("phase", "prepare"),
    ("evaluation_plan_sha256", "0" * 64),
])
def test_v18_metadata_mutations_rejected(field, value):
    data = _raw()
    data[field] = value
    with pytest.raises(ValueError):
        summary.audit(data)


def test_v18_fresh_holdout_map_guard():
    data = _raw()
    data["phase"] = "holdout"
    with pytest.raises(ValueError, match="map"):
        summary.audit(data)


def test_v18_physical_mutation_rejected():
    data = deepcopy(_raw())
    data["episode_lengths"][0] = -1
    with pytest.raises(ValueError):
        summary.audit(data)
