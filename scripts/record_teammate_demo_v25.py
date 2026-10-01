#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Predeclared qualitative replay; one env is not part of the scored matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_teammate_v25 import load_base, controller_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("v16_control", "combined61"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    base = load_base()
    original_resolve = base.resolve_model

    class Legacy:
        resolve_model = staticmethod(original_resolve)

    base.CONTROLLERS = ("v16_control", "combined61")
    base.SCHEMA = "week03_ant_teammate_v25_qualitative_v1"
    base.resolve_model = lambda c: controller_binding(c, "holdout", Legacy)
    # AppLauncher closes this numerical process. The untouched base writes the
    # complete original JSON/video before closing; no post-close code is needed.
    base.launch(["--controller", args.controller, "--geometry", "131", "--seed", "111",
                 "--scenario", "obstacles", "--level", "4", "--num-envs", "1",
                 "--seconds", "16", "--device", "cuda:1", "--headless",
                 "--output", str(args.output), "--record", str(args.record)])


if __name__ == "__main__":
    main()
