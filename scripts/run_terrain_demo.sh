#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COURSE_ROOT="${ROBOTICS_SIM_CLASS_ROOT:-/mnt/ssd970/robotics_simulation_class}"

cd "$ROOT"
export PYTHONUNBUFFERED=1
exec "$COURSE_ROOT/run-python" scripts/demo_terrains.py "$@"
