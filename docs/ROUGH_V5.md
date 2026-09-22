# v5: measured rough-terrain traversal

> Historical stable baseline. The later [recovery continuation](ROUGH_RECOVERY.md)
> validates a new recommended policy; the checkpoint and measurements below are
> retained unchanged for comparison.

This extension keeps the course policy interface (60 observations, 8 actions,
action scale 7.5, MLP `[400, 200, 100]`). It does not replace the original
flat-ground submission or the older v0–v4 experiments.

## Result (2026-09-21)

Selected checkpoint: [`rough_v5_traverse1499_seed42/model_1499.pt`](../artifacts/terrain_demo/runs/rough_v5_traverse1499_seed42/model_1499.pt).
SHA-256: `d02b5146a3c334d63fb4e362594f3488bfe4ab9bbabbd46c018bf35ef62dcd03`.

Three reset seeds × 175 environments give **450 terrain episodes plus 75 flat
episodes per policy**, with identical benchmark geometry and strict containment:

| Policy | Terrain falls | Terrain speed | One-tile success | Six-tile success | Footprint lane exits |
|---|---:|---:|---:|---:|---:|
| Original flat robust | 31.3% | 0.88 m/s | 7.6% | 0.0% | 32 |
| Earlier terrain v2 | 8.9% | 1.88 m/s | 74.2% | 0.0% | 12 |
| Resumed v5 parent C | 16.0% | 3.00 m/s | 65.1% | 39.3% | 40 |
| **Selected v5** | **7.1%** | **2.94 m/s** | **82.0%** | **43.8%** | **8** |

All four policies had zero world exits. This does **not** mean zero lane exits,
zero falls, or universal terrain success. A successful one-tile episode clears
an 8 m terrain tile and remains upright/in its lane for the entire 16 seconds.

| Selected-policy terrain | One-tile success | Six-tile success |
|---|---:|---:|
| Rough | 69/75 (92.0%) | 51/75 (68.0%) |
| Slope | 72/75 (96.0%) | 37/75 (49.3%) |
| Stairs | 70/75 (93.3%) | 21/75 (28.0%) |
| Obstacles | 70/75 (93.3%) | 43/75 (57.3%) |
| Waves | 64/75 (85.3%) | 45/75 (60.0%) |
| **Stepping stones — unresolved at high difficulty** | **24/75 (32.0%)** | **0/75** |

Stepping-stone success by difficulty is **12/15, 11/15, 1/15, 0/15, 0/15** for
levels 0.2, 0.4, 0.6, 0.8, 1.0. The high levels remain a genuine limitation,
not a completed recovery. The focused-stone candidate was retained for future
work but rejected as the default because its other-terrain fall rate increased.

Additional checks:

- Course ID return: **134.80 ± 29.67**, retaining **87.3%** of the original robust
  reference's 154.32. Low friction: **157.72 ± 28.36**; heavy torso:
  **142.79 ± 37.24**; pushes: **132.28 ± 34.53** (100 first episodes each).
- Assignment-format 100-lane return: **57.68 ± 33.70**.
- Held-out terrain-generator seeds **52/53**, reset seed 27, after selection:
  **116/150 (77.3%) / 117/150 (78.0%)** terrain successes, **14/150 / 12/150**
  falls, **2.89 / 2.86 m/s**, zero world exits. High stepping stones remain weak.
- **1,375** selected-policy first episodes were evaluated in total; references
  add another **1,575**. These are simulator observations, not hardware validation.

Full evidence: [comparison tables](../artifacts/terrain_demo/evaluations/rough_v5/summary_20260921.md),
[machine summary](../artifacts/terrain_demo/evaluations/rough_v5/summary_20260921.json),
[selection record](../artifacts/terrain_demo/evaluations/rough_v5/selection_20260921.json),
and [seven-family video](../artifacts/terrain_demo/rough_v5_selected_seed7.mp4).

### Run the selected policy

```bash
CHECKPOINT=artifacts/terrain_demo/runs/rough_v5_traverse1499_seed42/model_1499.pt

# Live viewer: seven families at difficulty 0.8, including the difficult stones.
./scripts/run_terrain_demo.sh --task Week03-Ant-Rough-Lanes-Demo-v5 \
  --device cuda:0 --seed 7 --cycle-seconds 8 --checkpoint "$CHECKPOINT" \
  --kit_args=--/renderer/multiGpu/enabled=false

# Reproduce the full fixed-seed benchmark.
./scripts/evaluate_rough_v5.sh reproduced_selected "$CHECKPOINT" 0
```

Validation: 29 CPU regression tests, Python `compileall`, shell syntax checks,
`git diff --check`, three GPU smoke tests, full training/evaluation runs and
checkpoint SHA-256/interface/finite-weight checks passed. An independent code
review approved the boundary-evidence fixes. Ruff/mypy/Pyright/LSP diagnostics
were unavailable in the existing environment; no dependencies were added.

Main changed files: `src/week03_ant/lane_math.py`, `src/week03_ant/tasks/lanes.py`,
`src/week03_ant/tasks/rough_v5_cfg.py`, `src/week03_ant/tasks/__init__.py`, `src/week03_ant/evaluation.py`,
`scripts/play_one_episode.py`, `scripts/summarize_rough_v5.py`,
`scripts/demo_terrains.py`, and the package-local tests. Existing lane/math and
upstream reward utilities were reused; the benchmark itself was not made easier.
[Automated evidence validation](../artifacts/terrain_demo/evaluations/rough_v5/verification_20260921.json)
and [video validation](../artifacts/terrain_demo/evaluations/rough_v5/video_20260921.json)
are retained separately from the training outputs.

## Fixed benchmark

`Week03-Ant-Rough-Lanes-Eval-v5` contains rough ground, slopes, stairs, waves,
discrete obstacles and stepping stones at five difficulty levels (0.2–1.0).
Each lane has an 8 m flat entrance, six 8 m terrain tiles and an 8 m flat exit.
The Ant starts at the entrance centre. Crossing the exit centre wraps it back
by 56 m without changing velocity, orientation or accumulated forward distance.
This removes the finite-world fall that invalidated some older completion counts.

The flat family uses the same PhysX ground-plane surface as the course task,
below the mesh. A mesh plane and a PhysX ground plane are not interchangeable
for this fast, contact-sensitive policy.

Evaluation uses exactly one **first episode** per environment: 175 environments
(5 per family/level) at reset seeds 24, 25 and 26; an additional 100-environment
lane run; and the unchanged course ID/three public OOD tasks. These reset seeds
share terrain-generator seed 51; they are not independent unseen geometries.

### Survival is not traversal

- Survival: no fall in the 960-step / 16-second episode.
- One terrain tile cleared: at least `4 + 8 + 1.1 = 13.1 m` forward, including
  the entrance half-tile and a conservative rear-foot clearance margin.
- Entire six-tile course cleared: at least `4 + 6×8 + 1.1 = 53.1 m` forward.
- Lane lap: 56 m unwrapped forward distance; reported separately.
- Successful clearance additionally requires no fall, no departure from the
  assigned lane and no world exit throughout that episode. Flat-plane episodes
  do not count as terrain traversal.

Containment uses a conservative 1.1 m horizontal foot-reach margin on each side:
the torso must stay within 2.9 m of the centre of its 8 m lane. This prevents a
nominally in-lane torso from borrowing a neighboring terrain with its feet.
Missing boundary evidence makes evaluation fail instead of assuming zero exits.

These are geometric progress checks, not a guarantee of every foot making a
specific contact. Termination reasons, distances and per-level results are kept
in the JSON, including failures and slow survivors.

## Continuation from the interrupted work

The completed September 18 run C (`model_2999`) was better than run D on the
September 21 seed-24 check: 31/150 versus 46/150 terrain falls, and course ID
return 129.21 versus 113.91. C retained 83.7% of the original robust policy's
154.32 course ID return, but stepping-stone progress remained weak.

The separate `Week03-Ant-Rough-Lanes-Safe-Train-v5` task caps only the **terrain
progress reward** at 3 m/s and adds a fall penalty of -10 per event (-600 weight
integrated at 60 Hz). Flat progress stays uncapped. This changes the training
objective, not evaluation rewards, terrain, observations or termination limits.
The parent weights are preserved; a copied checkpoint resets Adam and restores
action standard deviation 0.08, then PPO runs at a fixed learning rate of 5e-5.
All existing dynamics/sensor randomization remains active.

The safety stage reduced falls but did not solve traversal: its iteration-500
policy crossed one tile in 104/150 seed-24 terrain episodes, with 9 lane exits
and only 8/25 stepping-stone successes. The iteration-999 policy had more lane
exits (23), so iteration 500 was selected as the next-stage parent.
Those screening counts used torso-only containment; final evaluation adds the
stricter foot-reach margin above and must not be compared directly to them.

`Week03-Ant-Rough-Lanes-Traverse-Train-v5` adds training-only lane keeping
(squared lateral error outside a 0.5 m deadband), a low-forward-speed penalty
after the first second, half the mesh energy penalty, a -30 terminal penalty,
and an early reset when the torso is within 0.5 m of its lane edge. The flat
family retains its original progress/energy terms and has no lane/stall penalty.
This stage uses gamma 0.995 and entropy coefficient 0.001. These changes are
**not** applied to `Week03-Ant-Rough-Lanes-Eval-v5`.

## Reproduction

Run from the repository with the existing course environment. No dependency
or Isaac Sim/Isaac Lab version change is required.

```bash
COURSE_ROOT=${ROBOTICS_SIM_CLASS_ROOT:-/mnt/ssd970/robotics_simulation_class}

"$COURSE_ROOT/run-python" scripts/prepare_finetune_checkpoint.py \
  artifacts/terrain_demo/runs/rough_v5_parent_c2999/model_2999.pt \
  logs/rsl_rl/week03_ant_rough_v5/init_safe_c2999/model_0.pt \
  --std 0.08 --learning-rate 0.00005

./scripts/run_train.sh --task Week03-Ant-Rough-Lanes-Safe-Train-v5 \
  --headless --device cuda:0 --num_envs 4096 --seed 42 --max_iterations 1000 \
  --resume --load_run init_safe_c2999 --checkpoint model_0.pt \
  --run_name safe_c2999 \
  agent.algorithm.schedule=fixed agent.algorithm.learning_rate=0.00005 \
  agent.algorithm.desired_kl=null agent.save_interval=100 \
  --kit_args=--/renderer/multiGpu/enabled=false
```

Continuation from the selected safety-stage checkpoint:

```bash
"$COURSE_ROOT/run-python" scripts/prepare_finetune_checkpoint.py \
  artifacts/terrain_demo/runs/rough_v5_safe500_seed42/model_500.pt \
  logs/rsl_rl/week03_ant_rough_v5/init_traverse_safe500/model_0.pt \
  --std 0.08 --learning-rate 0.00005

./scripts/run_train.sh --task Week03-Ant-Rough-Lanes-Traverse-Train-v5 \
  --headless --device cuda:0 --num_envs 4096 --seed 42 --max_iterations 1500 \
  --resume --load_run init_traverse_safe500 --checkpoint model_0.pt \
  --run_name traverse_safe500 \
  agent.algorithm.schedule=fixed agent.algorithm.learning_rate=0.00005 \
  agent.algorithm.desired_kl=null agent.algorithm.gamma=0.995 \
  agent.algorithm.entropy_coef=0.001 agent.save_interval=250 \
  --kit_args=--/renderer/multiGpu/enabled=false
```

Use `scripts/evaluate_rough_v5.sh <label> <checkpoint> 0` for the complete
evaluation suite. Do not evaluate with the safe-training task to compare returns.
Checkpoints, configuration snapshots, TensorBoard logs and SHA-256 manifests are
stored under `artifacts/terrain_demo/runs/`; evaluation JSONs are under
`artifacts/terrain_demo/evaluations/rough_v5/`.

### Focused stepping-stone exploration

The traversal stage's iteration 1499 improved the other five terrain families,
but high-difficulty stepping stones still trapped the feet (the diagnostic video
advanced from 5.6 m at 2 s to only 7.3 m at 11 s). The next experiment restores
more exploration and oversamples stones without changing their geometry:

```bash
"$COURSE_ROOT/run-python" scripts/prepare_finetune_checkpoint.py \
  artifacts/terrain_demo/runs/rough_v5_traverse1499_seed42/model_1499.pt \
  logs/rsl_rl/week03_ant_rough_v5/init_stones_traverse1499/model_0.pt \
  --std 0.18 --learning-rate 0.000075

./scripts/run_train.sh --task Week03-Ant-Rough-Lanes-Traverse-Train-v5 \
  --headless --device cuda:0 --num_envs 4096 --seed 42 --max_iterations 800 \
  --resume --load_run init_stones_traverse1499 --checkpoint model_0.pt \
  --run_name stones_focus \
  agent.algorithm.schedule=fixed agent.algorithm.learning_rate=0.000075 \
  agent.algorithm.desired_kl=null agent.algorithm.gamma=0.995 \
  agent.algorithm.entropy_coef=0.003 agent.save_interval=200 \
  env.events.lane_layout.params.family_weights.stepping_stones=6.0 \
  env.events.lane_layout.params.family_weights.waves=2.0 \
  env.rewards.fall.weight=-600.0 env.rewards.stall.weight=-2.0 \
  env.rewards.progress.params.terrain_speed_limit=4.0 \
  env.terminations.lane_departure.params.margin=1.1 \
  --kit_args=--/renderer/multiGpu/enabled=false
```

`--family stepping_stones --level 3` on the v5 demo selects a single challenging
lane instead of cycling through all seven families. Recorded videos show family,
difficulty, simulated time, cumulative resets and current-episode forward
distance; automatic resets are not hidden.

## Limits

The policy is reactive (no terrain scanner). Success on these bounded slopes,
15 cm stairs and stepping-stone gaps is not evidence of arbitrary extreme-terrain
robustness or real-robot readiness. Training return is not a completion metric.
