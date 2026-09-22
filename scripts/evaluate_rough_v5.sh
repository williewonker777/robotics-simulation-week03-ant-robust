#!/usr/bin/env bash
# Evaluate one 60D checkpoint on the v5 terrain-lane benchmark and the course tasks.
#
# Lane benchmark: Week03-Ant-Rough-Lanes-Eval-v5, 175 environments (5 per family x
# level lane) for seeds 24/25/26, plus the assignment-format 100-environment run.
# Course tasks: ID flat ground and the three public OOD holdouts (100 envs, seed 24).
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <label> <checkpoint> <cuda-index>" >&2
  exit 2
fi

label=$1
checkpoint=$2
device=$3
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/artifacts/terrain_demo/evaluations/rough_v5"
console="$root/artifacts/console"
mkdir -p "$out" "$console"

run() {
  local task=$1 envs=$2 seed=$3 name=$4
  echo "Evaluating $label: $name ($task, $envs envs, seed $seed)"
  "$root/scripts/run_evaluate.sh" \
    --task "$task" \
    --headless \
    --device "cuda:$device" \
    --num_envs "$envs" \
    --seed "$seed" \
    --max_steps 960 \
    --checkpoint "$checkpoint" \
    --output "$out/${label}__${name}.json" \
    --kit_args=--/renderer/multiGpu/enabled=false >"$console/eval_rough_v5_${label}__${name}.log" 2>&1
}

for seed in 24 25 26; do
  run Week03-Ant-Rough-Lanes-Eval-v5 175 "$seed" "lanes_seed${seed}"
done
run Week03-Ant-Rough-Lanes-Eval-v5 100 24 "lanes100_seed24"

declare -A course=(
  [id]=Week03-Ant-Baseline-v0
  [low_friction]=Week03-Ant-Test-LowFriction-v0
  [heavy]=Week03-Ant-Test-Heavy-v0
  [push]=Week03-Ant-Test-Push-v0
)
for scenario in id low_friction heavy push; do
  run "${course[$scenario]}" 100 24 "$scenario"
done
echo "Wrote $out/${label}__*.json"
