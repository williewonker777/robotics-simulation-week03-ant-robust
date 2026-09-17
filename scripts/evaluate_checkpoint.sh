#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <run-label> <checkpoint> <cuda-index>" >&2
  exit 2
fi

label=$1
checkpoint=$2
device=$3
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

declare -A tasks=(
  [id]=Week03-Ant-Baseline-v0
  [low_friction]=Week03-Ant-Test-LowFriction-v0
  [heavy]=Week03-Ant-Test-Heavy-v0
  [push]=Week03-Ant-Test-Push-v0
)

mkdir -p "$root/artifacts/evaluations" "$root/artifacts/console"
for scenario in id low_friction heavy push; do
  output="$root/artifacts/evaluations/${label}__${scenario}.json"
  console="$root/artifacts/console/eval_${label}__${scenario}.log"
  echo "Evaluating $label on $scenario (${tasks[$scenario]})"
  "$root/scripts/run_evaluate.sh" \
    --task "${tasks[$scenario]}" \
    --headless \
    --device "cuda:$device" \
    --num_envs 100 \
    --seed 24 \
    --max_steps 960 \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --kit_args=--/renderer/multiGpu/enabled=false 2>&1 | tee "$console"
done
