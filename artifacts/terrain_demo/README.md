# Multi-terrain Ant demo

The original Week 03 submission evaluates dynamics randomization on the course
flat-ground task. This directory contains an additional terrain-generalization
extension trained in the same Isaac Lab 2.3.0 environment.

## Terrain set

`Week03-Ant-Terrain-v0` keeps the original 60-dimensional observation and
8-dimensional action interfaces, but replaces the plane with four equally
represented procedural families:

- flat plane
- random height-field roughness
- pyramid slope
- pyramid stairs

Torso height observations and fall termination are measured relative to each
terrain patch's origin. This prevents elevated patches from changing the meaning
of the original base-height feature.

## Training and selection

The selected policy was trained from scratch for 1,000 PPO iterations on the
full terrain curriculum, then fine-tuned for 600 iterations on a moderate
curriculum (`difficulty_range=(0.0, 0.45)`). It is stored at:

```text
artifacts/terrain_demo/runs/terrain_mild_seed42/model_1598.pt
```

SHA-256:

```text
1bc0f6d4e255f179aa4ad9b4392e19425c0503a63319730d2b68c338ad59fa2a
```

The full 100-environment evaluation uses seed 24, 25 environments per terrain
family, and one episode of at most 960 control steps per environment.

| Policy | Mean return | Return std | Mean length | Full-length episodes |
|---|---:|---:|---:|---:|
| Original robust checkpoint | 15.71 | 15.50 | 270.95 | 6/100 |
| Full-terrain fine-tune | 21.78 | 15.28 | 504.24 | 15/100 |
| Full-terrain scratch | 31.96 | 15.98 | 574.83 | 11/100 |
| **Selected moderate fine-tune** | **33.00** | **16.36** | **595.15** | **19/100** |

Selected-policy breakdown:

| Terrain | Episodes | Mean return | Return std | Mean length | Full length |
|---|---:|---:|---:|---:|---:|
| Flat | 25 | 33.61 | 15.43 | 580.44 | 2 |
| Random rough | 25 | 35.37 | 14.92 | 610.04 | 6 |
| Slope | 25 | 31.61 | 16.72 | 589.56 | 5 |
| Stairs | 25 | 31.42 | 17.86 | 600.56 | 6 |

Machine-readable results are in [`evaluations/`](evaluations). The original
robust result predates terrain-breakdown logging, so its 6/100 full-length count
was recovered directly from the recorded per-environment episode lengths.

## 후속 개선 정책 (v1)

The first policy used a flat-world style torso-height termination. On slopes and
stairs this can terminate a still-upright robot simply because the ground surface
descends. `Week03-Ant-Terrain-Posture-v1` keeps the same 60D/8D interface but
uses an elevation-independent overturning-angle termination (`1.2 rad`). The
selected checkpoint was continued for 2,000 PPO iterations on the full terrain
curriculum from the moderate policy above, and selected at iteration 2,800
instead of taking the final checkpoint.

```text
artifacts/terrain_demo/runs/terrain_posture_full_seed42/model_2800.pt
```

SHA-256:

```text
dc146b0b8d8b7a8ba60b7831c599b49536bee6bad674bf02a083abb523d39737
```

On the same seed-24, 100-environment evaluation:

| Policy / termination | Mean return | Return std | Mean length | Full-length episodes |
|---|---:|---:|---:|---:|
| v0 moderate policy / terrain-origin height | 33.00 | 16.36 | 595.15 | 19/100 |
| **v1 selected / overturning angle** | **33.96** | **14.81** | **637.17** | **23/100** |

The v1 per-terrain means are flat `34.91 / 633.1` steps (3 full), rough
`34.55 / 624.9` (6 full), slope `32.44 / 609.6` (5 full), and stairs
`33.93 / 681.2` (9 full). The five stored evaluation seeds (7, 24, 42, 43,
44) average `655.4` steps and `26.8/100` full episodes; seed 44 has unusually
large negative energy/action penalties, so return—not just survival—is reported
per seed in the JSON files.

This is a measurable improvement in survival and completion, not a claim of
universal terrain robustness. The terrain families and difficulty range remain
the same as v0.

## Live demo

This command opens four environments (flat, rough, slope, stairs) and switches
the following camera every seven seconds:

```bash
./scripts/run_terrain_demo.sh \
  --task Week03-Ant-Terrain-Posture-v1 \
  --device cuda:0 \
  --seed 7 \
  --cycle-seconds 7 \
  --checkpoint artifacts/terrain_demo/runs/terrain_posture_full_seed42/model_2800.pt \
  --kit_args=--/renderer/multiGpu/enabled=false
```

The fixed visual sample for seed 7 reached episode lengths
`453 / 960 / 960 / 959` on flat / rough / slope / stairs, respectively. It is a
qualitative camera sample, not the statistical result; use the 100-environment
tables above for quantitative comparison.

Close the Isaac Sim window to stop the demo, or terminate its hosting process.

## Reproduction

Full-terrain training:

```bash
./scripts/run_train.sh \
  --task Week03-Ant-Terrain-v0 \
  --headless --device cuda:0 --num_envs 4096 \
  --seed 42 --max_iterations 1000 \
  --run_name terrain_scratch_seed42 \
  --kit_args=--/renderer/multiGpu/enabled=false
```

Moderate-terrain fine-tuning from the full-terrain checkpoint:

```bash
FULL_RUN="$({
  find logs/rsl_rl/week03_ant -maxdepth 1 -type d \
    -name '*_terrain_scratch_seed42' -printf '%f\n'
} | sort | tail -1)"

./scripts/run_train.sh \
  --task Week03-Ant-Terrain-Mild-Train-v0 \
  --headless --device cuda:0 --num_envs 4096 \
  --seed 42 --max_iterations 600 \
  --run_name terrain_mild_seed42 \
  --resume \
  --load_run "$FULL_RUN" \
  --checkpoint model_999.pt \
  --kit_args=--/renderer/multiGpu/enabled=false
```

Evaluation:

```bash
./scripts/run_evaluate.sh \
  --task Week03-Ant-Terrain-Posture-v1 \
  --headless --device cuda:0 --num_envs 100 \
  --seed 24 --max_steps 960 \
  --checkpoint artifacts/terrain_demo/runs/terrain_posture_full_seed42/model_2800.pt \
  --output artifacts/terrain_demo/evaluations/terrain_posture_full_2800_seed24.json \
  --kit_args=--/renderer/multiGpu/enabled=false
```

The v1 continuation used the same `--resume --load_run ... --checkpoint
model_1598.pt` pattern as the moderate fine-tune above, with task
`Week03-Ant-Terrain-Posture-v1`, `--max_iterations 2000`, and run name
`terrain_posture_full_seed42`. Checkpoints are saved every 50 iterations; the
stored iteration-2,800 checkpoint was selected using evaluation seed 24; the
other four seed files provide an additional post-selection robustness check.
