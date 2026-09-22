# Frozen-base residual v8

**Result: promotion FAIL; retain the existing v5 recommendation.** The residual
recovers much of v7's lost six-tile completion, but does not solve hard stones or
surpass v5. All numbers below use fresh matched geometry62/63, so do not compare
them as if they used v7's earlier geometry60/61 benchmark.

## Why this experiment

The previous v7 footmap experiment reduced falls relative to a height-only CNN,
but degraded six-tile completion versus the original v5. This experiment freezes
that competent v5 actor and learns only an additive correction, rather than
updating the entire actor.

The additive-controller hypothesis is informed by [Residual Reinforcement
Learning for Robot Control](https://arxiv.org/pdf/1812.03201) (Johannink et al., RSS
2019) and [Continuous Versatile Jumping Using Learned Action
Residuals](https://proceedings.mlr.press/v211/yang23b/yang23b.pdf) (Yang et al., L4DC
2023). Their controllers/algorithms differ from this Ant PPO implementation;
neither is evidence that this particular residual guarantees stability.

## Policy contract

```text
mean = frozen_v5(observation[:60])
       + 0.5 * tanh(residual_MLP([observation[:60], CNN(spatial_map)]))
training_action ~ Normal(mean, learned_std)
evaluation_action = mean
```

- Four legs, eight direct-effort actions. All old checkpoints are preserved.
- Frozen base is checkpoint-self-contained and explicitly nontrainable after
  construction and strict loading. No runtime teacher file or routing signal.
- Residual output layer starts at zero: exact parent actor behavior initially.
- The bound is per action-mean component at the **same observation**, not a
  closed-loop trajectory bound. At the unchanged action scale7.5, it corresponds
  to up to3.75 additional effort; Gaussian exploration is not bounded by0.5.
  Existing simulator actuator limits/wrapper behavior are unchanged.
- Reuse v7's514D observations,221-ray ideal height scan, validity and four foot
  centers. Spatial encoder:6→8→16 convolution channels,32D embedding. Actor
  correction and critic MLPs:400/200/100 ELU. The critic is zero-column
  warm-started from v5; its CNN is separate and uses the same mode as the actor.
- Same architecture in `blind` (all spatial channels zero) and `footmap` modes;
  initial paired tensors differ only by the persistent mode buffer.
- Real RGB-D images, recurrent memory, full capsule support maps, new rewards,
  action penalties and curriculum changes are **not** introduced here.

## Fixed experimental design

Two modes × training seeds42/43/44, paired training geometry51/58/59. Each run
uses750iterations ×4096environments ×32steps =98,304,000transitions; six runs
total589,824,000. Same reward/action/termination/physics/PPO contracts as v7.

All final iteration749 checkpoints are frozen before evaluation. Fresh geometry/
reset pairs62/36 and63/37,175 balanced first episodes each,16s maximum. One-tile
success requires13.1m and six-tile53.1m, with no terminal condition, whole-footprint
lane exit or world exit; footprint margin1.1m. No threshold relaxation.

The frozen v5 reference and all three previous v7 footmap policies are rerun under
these same conditions:20 primary files/3500episodes. Baseline samples are not
duplicated to match the three trained seeds. Only two independent held-out maps
are sampled, not thousands. Representative seed42 is fixed in advance for video
and zero/shuffle-depth diagnostics. Those diagnostics retain actual foot maps,
are out-of-distribution sensitivity tests, and never enter promotion totals.

Promotion requires no aggregate one/six/fall regression versus v5, blind residual
and v7 footmap, zero world exits, no flat-fall regression versus v5, and greater
one-tile success than blind residual in at least two of three paired seeds.
Small aggregate differences do not establish statistical superiority.

## Reproduction

```bash
../run-python -m pytest -q
../run-python scripts/run_residual_experiment.py --phase train --device cuda:1
../run-python scripts/run_residual_experiment.py --phase evaluate --device cuda:1
../run-python scripts/run_residual_experiment.py --phase diagnostics --device cuda:1
../run-python scripts/summarize_residual.py artifacts/terrain_demo/residual_v8
```

The batch runner fails closed rather than overwriting experiment evidence.
For individual invocations use `scripts/residual_v8.py train|evaluate|demo` with
the ordinary script arguments and a `Week03-Ant-Residual-Lanes-*-v8` task.
This opt-in launcher leaves the complete archived v7 source set unchanged.

## Verification / results

Before long training:98CPU regression tests,64env×2iteration PPO smoke and35env
saved-checkpoint evaluation passed. Actual saved environment/PPO configs match
v7 (except the intended64vs4096 smoke batch); frozen base is bit-identical after
PPO. Source/checkpoint preservation and initial paired/action/value equivalence
are checked. Final metrics are recorded in the experiment's frozen artifacts;
implementation correctness is not a claim of improved terrain performance.

All six runs finished; final checkpoints were frozen at
`2026-09-21T17:36:23.944014+00:00` before any fresh evaluation. No checkpoint or
hyperparameter reselection/retraining followed the results.

| Policy | Strict one tile | Six tiles | Terrain falls | Flat falls |
|---|---:|---:|---:|---:|
| Original v5 |259/300 (86.3%)|136/300 (45.3%)|32/300 (10.7%)|3/50|
| Previous v7 footmap |753/900 (83.7%)|219/900 (24.3%)|94/900 (10.4%)|6/150|
| Blind residual v8 |712/900 (79.1%)|374/900 (41.6%)|99/900 (11.0%)|10/150|
| Terrain/footmap residual v8 |747/900 (83.0%)|370/900 (41.1%)|96/900 (10.7%)|10/150|

The terrain residual increases one-tile success over the blind residual in all
three paired training seeds (245vs237,248vs239,254vs236, each/300), but its six-tile
score is slightly lower (118vs121,123vs123,129vs130). This is not sufficient to
replace the original policy. All world exits are zero.

| Difficulty1.0 stepping stones | One tile | Six tiles | Falls |
|---|---:|---:|---:|
| Original v5 |8/10|0/10|0/10|
| v7 footmap |16/30|0/30|3/30|
| Blind residual v8 |8/30|0/30|2/30|
| Terrain residual v8 |12/30|0/30|6/30|

Difficulty0.8 stone one-tile counts are respectively6/10,10/30,9/30,12/30;
all six-tile counts are zero. Hard-stone generalization remains unresolved.
The small baseline subgroup is not inflated into30 independent samples.

The predeclared gate fails on one/six vs v5; one/falls vs v7; six vs blind
residual; and flat falls vs v5. Passing implementation tests or improving a
single metric is not promotion.

### Depth sensitivity (predeclared seed42 only)

| Inference scan | One tile | Six tiles | Terrain falls |
|---|---:|---:|---:|
| Actual |245/300|118/300|32/300|
| All scan height/validity zero |226/300|50/300|43/300|
| Scan shuffled between environments |236/300|111/300|48/300|

Actual foot maps remain in all three cases. The700 extra diagnostic episodes
(including flat) are excluded from the3500 primary samples/gate. These changes
show sensitivity to the terrain input, not a guarantee of useful perception on
unseen distributions or real cameras. Shuffling breaks scan/foot consistency;
zero is also out-of-distribution and not the same as training a blind policy.

## Files and reproducibility

New policy/entry points: `src/week03_ant/residual_policy.py`,
`src/week03_ant/tasks/residual_v8.py`, `tasks/agents/residual_v8_cfg.py`,
`scripts/{residual_v8,prepare_residual_checkpoint,run_residual_experiment,
summarize_residual,render_residual_comparison,build_residual_viewer}.py`.
Regression tests: `tests/test_residual.py`, `tests/test_residual_summary.py`.

Reuse rather than replace v7's observations, CNN, critic, environments, original
training/evaluation/demo scripts and strict raw-array aggregation. This keeps all
17 training-source hashes (including the unchanged v7 sources) auditable. No
dependency installation, remote Git, commit, hardware actuation, or alteration of
old checkpoint artifacts was performed.

Artifacts: `artifacts/terrain_demo/residual_v8/` contains the frozen manifest,
six trained policies and reference, commands, actual configs, raw evaluations,
summary, diagnostics, verification, independent review and comparison viewer.
Raw console evidence stays in ignored `outputs/residual_v8_20260922/`.

Remaining limits: only3 training seeds and2 new maps; ideal scans rather than
real RGB-D; no real robot validation; no robust hard-stone solution. A possible
next experiment is explicit foothold planning or revisiting reward alignment,
not simply making the same CNN larger. These are hypotheses, not demonstrated
causes of failure. Ruff/mypy/Pyright/LSP are unavailable; no tooling installed.
