# v7: controlled foot-aware spatial encoding

Status: all nine training runs and20 held-out evaluations completed on2026-09-22(KST).
**Promotion gate FAIL: keep the existing v5 recommendation.**
The footmap arm reduces falls compared with height-only CNN, but does not solve
hard stones or preserve the old policy's sustained traversal performance.

## Research basis and bounded scope

The user's literature-followup approval is implemented as a first controlled
experiment, not a wholesale reproduction of a paper:

- [Hwang et al., Foot Position Maps and Stability Rewards (2026 preprint)](https://arxiv.org/html/2604.02744v1):
  explicit per-foot spatial maps motivate the four-channel representation.
  The paper's CoP reward is **not** ported to the Ant's long-capsule feet.
- [Motion Priors Reimagined (2025)](https://arxiv.org/html/2505.16084v2) and
  [DreamWaQ++ (TRO 2026)](https://arxiv.org/html/2409.19709v2) motivate possible
  later residual/memory experiments, **not features of this implementation**.

v7 changes the observation encoding only. Terrain geometry, 8 direct-effort
commands with scale7.5, training rewards/family weights, episode limits and
strict crossing definitions are preserved. No GRU, residual action, adaptive
curriculum, CoP reward, new dependencies, real robot command, remote Git
operation or automatic commit is introduced. Old v0-v6 tasks/checkpoints remain.

## Observation and policy contract

514D flat observation, in this order:

1. Original60 features, including the inherited simulation-local base-height
   measurement; the blind arm is therefore not a claim of pure deployable proprioception.
2. 221 relative heights and221 validity bits.
3. Four distal capsule-center positions xyz, relative to the torso in its
   yaw-only frame. Foot order: front-left, front-right, back-left, back-right.
   Link origins are not used as foot tips; local offsets match the existing
   Ant capsule geometry (+/-.4,+/-.4,0). There is no terrain-family/level oracle.

The idealized raycaster uses a17x13 grid at.2m resolution, x[-1.2,2.0],
y[-1.2,1.2]. Compared with v6 it extends rear/side coverage to include the rear
feet; all v7 arms share this exact scene. This is **not a rendered depth camera**:
no lens, occlusion, latency or real mapping pipeline. Nearest valid intersection
of the existing terrain mesh and physical flat plane is encoded; missed rays
remain unknown, not safe flat. Training height noise is+/-.02m, eval noise0.

Isaac Lab `ordering='xy'` flattens y-rows/x-columns: the policy image is
13rows x17columns, not17x13. The simulator probe compares every generated ray
start to the policy coordinates. Reset and treadmill-wrap scans are refreshed.

Four Gaussian foot maps (sigma.15m) use the kinematic x/y coordinates. Outside
or nonfinite centers create zero maps, not edge-clamped fictitious contacts.
The z coordinates are retained in the observation/diagnostics but this first
policy uses only x/y maps; contact footprint extent is not modeled.

Actor and critic have separate6->8->16 stride2 CNNs, flattened spatial features
projected to32D. Concatenate with original60 ->400/200/100 ELU heads ->8 actions
or one value. They are feed-forward networks with no temporal memory.

All three arms use the same architecture and three-seed training budget:

| Checkpoint mode | Height/valid channels | Four foot maps |
|---|---|---|
| blind | zero | zero |
| height | actual | zero |
| footmap | actual | actual |

The mode is a strict checkpoint int64 buffer and restored at load, so an eval
CLI default cannot silently turn a blind policy into a perceptive one. Training
also checks its configured mode against the resumed checkpoint. RSL-RL3.0.1
resolves custom policy classes in its runner namespace; registration is limited
to v7 entry paths. Generic upstream MLP-only ONNX/JIT export is not supported
for this CNN; use this project's runner-based inference.

Warmstart copies the existing v5 actor/critic entirely and zeroes the32 added
first-layer feature columns. The initial deterministic actions are preserved
within floating-point precision; PPO then trains both backbone and encoders.
This is **not** a frozen/residual policy and does not guarantee retained behavior
once training starts. Action std resets to.20, optimizer/iteration restart.

## Frozen experiment design

- Modes blind/height/footmap; trainingseed/geometry42/51,43/58,44/59.
- Each4096env x32steps x750iterations =98,304,000 transitions;9runs total.
- Fixed PPO LR1e-4,gamma.995,lambda.95,entropy.002,5epochs/4minibatches.
- Same recovery training: stones weight4, flat2, other families1; stall-2,
  stone-bodyheight weight-3/target.55, swing-.5/trap0, fall-1800, lane margin1.1.
- Final iteration749 only, all seeds reported. No best-checkpoint or best-seed
  selection. Representative video seed42 is chosen before performance results.
- Every final checkpoint is hashed and frozen **before** held-out geometry/reset
  60/34 and61/35. 175 balanced first episodes each, at most16s.
- Reference is an action-preserving v7 conversion of the frozen v5 model in the
  **same v7 scene**, evaluated once per condition, not duplicated into fake
  independent samples. Historical v6 scores are not a matched encoder ablation.
- Strict1-tile13.1m /6-tile53.1m requires no terminal, whole-footprint lane exit,
  or world exit over the first episode. Report both, falls, flat preservation,
  and each family at difficulty0.8/1.0. Survival/distance-only is not success.

Promotion requires pooled footmap one/six at least both trained controls, falls
no greater, zero worldexit, and one-tile improvement vs height in at least2/3
seed pairs. It must also avoid one/six/fall/flat-fall rate regression against
frozen v5. A failed gate keeps v5 recommended. Finite shared evaluation maps and
3 training seeds are not independent terrain population samples or a hardware
safety guarantee. Training seed and training geometry are coupled; this first
experiment does not isolate those two sources of variability.

## Verification and reproducibility

Verification:80CPU tests passed (75 before full training,5 additional audit tests),64env2iteration warmstart PPO and35env
checkpoint evaluation passed; simulator221-ray514D probe verified all four feet
visible at reset, nonzero yaw/roll/pitch frame alignment, reset freshness,
wrap scan/foot-coordinate errors0. New models cannot load into the original60D
course task. No dependencies were installed.

From the project root, after activating the existing course environment:

```bash
../run-python -m pytest -q
../run-python scripts/probe_footmap.py --headless --device cuda:1 --output outputs/footmap_probe.json
../run-python scripts/run_footmap_experiment.py --phase train --device cuda:1
../run-python scripts/run_footmap_experiment.py --phase evaluate --device cuda:1
../run-python scripts/summarize_footmap.py artifacts/terrain_demo/footmap_v7
```

The batch refuses to overwrite existing evidence. Original run logs/commands
are in ignored `outputs/footmap_v7_20260921/`; final checkpoints, parameters,
freeze manifest, raw evaluations and report are under
`artifacts/terrain_demo/footmap_v7/`. `--device` and `agent.device` are explicitly
matched by the batch. One experiment GPU job runs at a time; GPU1 was selected
to avoid another user-owned GPU0 simulator, which was not stopped.

## Frozen results

All9runs completed750iterations (884,736,000 transitions total). Freeze timestamp
2026-09-21T15:34:48.680171+00:00 precedes every held-out evaluation. No retraining
or checkpoint selection followed these results. Every first episode is counted,
including failures; final checkpoint749 was used for every arm/seed.

| Policy | Strict one tile | Six tiles | Terrain falls | Flat falls |
|---|---:|---:|---:|---:|
| Frozen v5 reference |257/300 (85.7%)|135/300 (45.0%)|29/300 (9.7%)|7/50|
| Equal-budget blind |746/900 (82.9%)|326/900 (36.2%)|87/900 (9.7%)|17/150|
| Height CNN |755/900 (83.9%)|144/900 (16.0%)|108/900 (12.0%)|13/150|
| **Height CNN + foot maps** |**760/900 (84.4%)**|**216/900 (24.0%)**|**85/900 (9.4%)**|**16/150**|

Each trained arm pools3 independently trained policies on the same two held-out
geometry/reset conditions. The frozen reference is evaluated once per condition
(300terrain+50flat), not duplicated3times as independent evidence. Total recorded
first episodes:3,500. All world exits0.

The added footmap channels improve observed falls and sustained crossing vs the
matched height-only arm, but one-tile improvement is only5/900 episodes. Effects
vary across seeds: seed42 loses17one-tile successes vsheight, seed43 gains1 and
seed44 gains21. This is not a claim of statistically established superiority.

At maximum difficulty1.0, strict successes are44/60 frozenv5,136/180 blind,
133/180 height,137/180 footmap; respective falls9/60,25/180,39/180,28/180.
On maximum-difficulty **stepping stones**, success is3/10 frozenv5,14/30 blind,
18/30 height,16/30 footmap. At difficulty0.8stones it is5/10,13/30,12/30,12/30.
Thus foot maps **did not improve the high-stone success objective over height-only**
in this experiment. High-stone6-tile success is0 for every arm.

The predeclared promotion gate fails on one/six vsfrozenv5 and six vsblind.
Retain all9experimental models, but do not replace
`rough_v5_portal_rehearsal4_seed43/model_round_4.pt`.

Artifacts: [full report](../artifacts/terrain_demo/footmap_v7/summary.md),
[all family/level/seed counts](../artifacts/terrain_demo/footmap_v7/summary.json),
[raw-array/source/hash audit](../artifacts/terrain_demo/footmap_v7/verification.json),
[freeze/checkpoints](../artifacts/terrain_demo/footmap_v7/frozen.json).
Independent reviewer approved implementation and equal-budget controls;80tests,
compileall, shellsyntax and diff checks pass. Ruff/mypy/Pyright/LSP were unavailable;
no package was installed to mask that gap.

### Interpretation and next experiment, not an implemented feature

The present model adapts all old MLP weights during PPO. A frozen-v5 plus limited
terrain residual remains a reasonable **next hypothesis** for retaining sustained
walking; it has not been tested here. Do not treat this small result as proof that
foot-aware perception cannot work. Full capsule-foot contact geometry, memory,
sensor occlusion and an adaptive difficulty curriculum remain unimplemented.

There is also an existing training-objective tension: terrain progress reward
saturates at3.0m/s, while53.1m within16s requires mean forward speed>=3.319m/s.
This is not a physical speed limit and has not been proven to cause the regression,
but it warrants a separate reward-alignment experiment rather than changing
the evaluation threshold or quietly changing rewards inside the current comparison.

## Visual comparison and changed files

[Open the local comparison viewer](../artifacts/terrain_demo/footmap_v7/index.html).
Two uncut16s/480frame/30fps/1280x720 clips use the predeclared maximum stone
condition geometry60/reset34, reference vsseed42footmap. Full decode passed.
At15.9s the displayed episode distances are16.7m vs15.9m, bothreset0. This is
a qualitative case, not another independent benchmark or a success-rate claim.
Video hashes, model hashes and exact commands are in
[video_manifest.json](../artifacts/terrain_demo/footmap_v7/video_manifest.json).

Implementation files: `src/week03_ant/footmap_math.py`, `footmap_policy.py`,
`tasks/footmap_v7_cfg.py`; task registration and the new runner config; v7-only
registration hooks in `scripts/train.py`, `play_one_episode.py`, `demo_terrains.py`.
Tools: `prepare_footmap_checkpoint.py`, `probe_footmap.py`,
`run_footmap_experiment.py`, `summarize_footmap.py`. Tests:
`tests/test_footmap.py`, `tests/test_footmap_summary.py`. README and this document
explain the experiment. Existing lane physics/rewards and old policies are reused,
not replaced. No broad refactor or dependency addition was performed.
