"""CPU-only contracts for the independent qualitative terrain renderer."""
import importlib.util
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/render_hybrid_terrains.py"
spec = importlib.util.spec_from_file_location("hybrid_render_helpers", PATH)
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)


def test_import_does_not_launch_simulator():
    assert "isaaclab.app" not in sys.modules


def test_fixed_families_exclude_previously_shown_stones():
    assert render.FAMILIES == ("rough", "slope", "stairs", "waves", "obstacles")


def test_caption_keeps_resets_and_qualitative_warning_visible():
    lines = render.caption("hybrid", "stairs", 120, 1 / 60, .75, 2, -1.5)
    assert "t=2.00s" in lines[0]
    assert "stairs difficulty=1.0" in lines[0]
    assert "applied v10 alpha=0.75" in lines[0]
    assert "resets=2" in lines[0]
    assert "current episode -1.5m" in lines[1]
    assert "NOT benchmark successes" in lines[1]


def test_frozen_source_verification_rejects_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "ROOT", tmp_path)
    source = tmp_path / "source.py"
    source.write_text("frozen\n")
    frozen = {"source_sha256": {"source.py": render.sha(source)}}
    assert render.verify_sources(frozen) == frozen["source_sha256"]
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        render.verify_sources(frozen)
