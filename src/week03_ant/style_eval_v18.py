"""Fresh-map v18 evaluation provenance, separate from frozen training source."""

from __future__ import annotations

from .lp_study_v17 import evaluation_inputs as _evaluation_inputs
from .posture_study import sha
from .style_study_v18 import ART, ROOT, TRAINING_SOURCES

EVALUATION_SOURCES = (
    "src/week03_ant/style_eval_v18.py",
    "scripts/evaluate_style_v18.py",
    "scripts/run_style_eval.py",
    "scripts/summarize_style_v18.py",
    "tests/test_style_eval_v18.py",
)
SOURCE_FILES = (*TRAINING_SOURCES, *EVALUATION_SOURCES)


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def evaluation_inputs(directory=ART):
    return _evaluation_inputs(directory)
