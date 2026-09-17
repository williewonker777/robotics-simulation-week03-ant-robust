#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <baseline|friction|robust> <seed> <cuda-index>" >&2
  exit 2
fi

variant=$1
seed=$2
device=$3
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

case "$variant" in
  baseline) task=Week03-Ant-Baseline-v0 ;;
  friction) task=Week03-Ant-Friction-v0 ;;
  robust) task=Week03-Ant-Robust-v0 ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac

run_name="${variant}_seed${seed}"
console_dir="$root/artifacts/console"
mkdir -p "$console_dir"
console_log="$console_dir/train_${run_name}.log"

cd "$root"
echo "variant=$variant"
echo "task=$task"
echo "seed=$seed"
echo "device=cuda:$device"
echo "console_log=$console_log"

set -o pipefail
./scripts/run_train.sh \
  --task "$task" \
  --headless \
  --device "cuda:$device" \
  --num_envs 4096 \
  --max_iterations 1000 \
  --seed "$seed" \
  --run_name "$run_name" \
  --kit_args=--/renderer/multiGpu/enabled=false 2>&1 | tee "$console_log"
