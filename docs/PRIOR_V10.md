# v10 — training-only frozen-v5 mean prior

## Status
All six training runs, primary evaluation and the separate64-second diagnostic are
complete and independently audited. Primary promotion **FAIL** (lane exits versus
v5); directional hypothesis **PASS**. Frozen v5 remains recommended. Long-horizon
fall rates remain high; this is not complete hard-terrain recovery.

## Why this method
The v9 targets-only arm improved six-tile traversal over the feet-only arm in every
paired training seed, but increased aggregate lane departures and did not beat the
frozen v5 safety baseline. More support shaping in v9 did not fix falls. V10 tests a
training-time behavioral prior, not another reward or sensor change. These observations
motivate a hypothesis; they do not prove catastrophic forgetting as the failure cause.

The approach is inspired by [Kickstarting Deep Reinforcement Learning](https://arxiv.org/html/1803.03835):
a teacher supplies an auxiliary policy objective on student-generated states while
reinforcement learning continues. [Policy Distillation](https://arxiv.org/html/1511.06295)
provides related teacher/student distribution-matching context. V10 is **not an exact
reproduction**: it uses a constant coefficient and an equal-fixed-variance Gaussian
mean loss, averaged over actions. Kickstarting's original method can decay teacher
influence; that is not tested in this fixed study.

## Treatment and invariants
- Student: unchanged v9 targets88D (old60 + distal FK12 + four target-relativeXYZ/valid16),
  MLP400/200/100 ELU, eight effort-action means, no observation normalization.
- Sensor: unchanged ideal825-ray, .1m grid, yaw-aligned height scan; not an RGB-D camera.
- Teacher: original frozen v5's60D actor, exact unchanged raw observation prefix.
  All teacher parameters remain bit-identical to the hash-pinned original model.
- Training on student rollout minibatches only:
  `L_prior = mean_batch,action((mu_student - mu_v5)^2) / (2 * 0.2^2)`.
  `L_total = L_PPO + lambda * L_prior`; lambda0 (free) versus0.02 (anchored).
  The latter is0.25 times the mean-squared action-mean error.
- Fixed0.2 is the prior's reference sigma, **not** the student's learnable exploration
  standard deviation. Prior gradients touch actor means only, not teacher/critic/std.
- The teacher is not called by inference. Unlike v8's frozen base plus bounded residual,
  every student actor layer may change and there is no inference-time safety wrapper.
- Lambda0.02 is a local predeclared engineering hypothesis, not a paper-prescribed value.
  It may overconstrain the teacher's weak stone gait and prevent useful recovery actions.
- Loss compares pre-action-transform means, not the actual clipped-action distribution.
  Saturation and teacher/student mean RMS are logged; the anchor is not a safety bound.
- Both arms store the same frozen teacher and consume identical initialization RNG.
  Both warm-start v5's60 columns and zero the additional28; std0.2, reset Adam1e-4.
- Physical terrain, rewards (support shaping exactly0), terminations, observations/noise,
  action mapping and existing PPO settings remain unchanged from v9 targets.
- Mode, coefficient and fixed sigma are typed checkpoint buffers with strict resume
  guards. Training validates the loaded arm. Every frozen final teacher is checked
  against original v5 actor tensors, not merely trusted from a mode label.

## Implementation isolation
New opt-in `scripts/prior_v10.py` registers `PriorActorCritic` and `PriorPPO` without
editing installed libraries or archived train/evaluate/demo code. The specialized
PPO update supports feedforward, fixed-rate, single-GPU, no RND/symmetry. It rejects
unsupported combinations and pins RSL-RL3.0.1 PPO source SHA256
`deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c`.
Coefficient-zero loss/state numerical parity with the installed PPO is a pretraining
requirement. The base implementation still owns rollout storage/returns/optimizer.

## Predeclared study
-2arms x3seeds42/43/44, training geometry51/58/59 respectively.
-750iterations x4096env x32steps =98,304,000 transitions/run;589,824,000 total.
-All six final749 checkpoints freeze before holdout; no best-seed/checkpoint selection.
-Fresh geometry66/reset40 and67/reset41,175env/condition, first episode only, at most16s.
-Seven policies including one frozen v5 reference:14JSON/2450episodes including flat;
  terrain denominators300 v5 and900 per learned arm. Reference counts are not replicated.
-Strict one/six crossing: final first-episode odometer distance>=13.1/53.1m with no
  posture termination, accumulated lane departure or world exit; margin1.1 unchanged.
-Only two held-out maps, not2450 independent maps; no broad-generalization guarantee.

### Strict promotion gate
Anchored one/six rates no worse than **both** v5 and free; terrain fall/lane rates no
worse than both; flat fall rate no worse than v5; world0; fewer falls than free in at
least2/3 paired training seeds. Every condition must pass, otherwise retain v5.

### Separate directional hypothesis gate
Aggregate falls **and** lane departures strictly lower than free; both improve jointly
in at least2/3 paired seeds; six-tile rate at least v5; world0. This practical hypothesis
verdict cannot by itself authorize promotion and is not a statistical significance test.

## Separate64-second diagnostic
Same seven frozen policies and two maps,10env per map, all stones difficulty1.0,
64s/3840steps.14JSON/140firstepisodes:20v5,60free,60anchored. The10env reset assignment
is paired across policies but differs from the175env primary assignment. Never merge
these episodes with primary or use them for tuning/promotion.

Record first time-to13.1/53.1m, distanceat16s (null if already ended at/before16s), final
first-episode distance at64s or termination, maxdistance, falls, lane/world, survival.
A first hit followed by a fall/exit is **not strict final success**. Hit-time averages
are conditional on reaching distance; they do not describe all episodes.53.1m/16s
requires3.32m/s, so zero16s six-tile successes alone cannot distinguish inability from
slow traversal. Auto-reset samples use saved `LaneState.last_*` evidence.

## Reproduction
From this repository (course environment already installed):

```bash
../run-python -m pytest
../run-python scripts/run_prior_experiment.py --phase train --device cuda:1
../run-python scripts/run_prior_experiment.py --phase evaluate --device cuda:1
../run-python scripts/summarize_prior.py artifacts/terrain_demo/prior_v10
../run-python scripts/run_prior_experiment.py --phase horizon --device cuda:1
../run-python scripts/summarize_prior_horizon.py artifacts/terrain_demo/prior_v10
source ../activate.sh
../run-python scripts/render_prior_comparison.py --device cuda:1
../run-python scripts/build_prior_viewer.py
```

Study runner refuses to overwrite frozen evidence. These are original-study commands,
not instructions to overwrite an existing completed run. Reports/checkpoints live in
`artifacts/terrain_demo/prior_v10/`; raw logs/preflight in ignored
`outputs/prior_v10_20260922/`. No dependency installation, remote operations, commits,
or physical robot control are part of this experiment.

## Presentation contract
Predeclared anchoredseed42 versus frozenv5, geometry66/reset40/stones1.0, uncut16s
and64s. Failures/resets retained. Separate1env qualitative cases are never added to
benchmark counts. Both videos require exact frame counts and complete decode checks.

## Independently audited primary results (two fresh maps,16s)

All six final749 models were frozen at `2026-09-22T07:03:57.926864+00:00` before any
held-out evaluation.589,824,000 training transitions;14rawJSON/2450firstepisodes.

| Policy | One tile | Six tiles | Falls | Lane exits | Flat falls |
|---|---:|---:|---:|---:|---:|
| frozen_v5 | 260/300 (86.7%) | 132/300 (44.0%) | 30/300 (10.0%) | 0/300 (0.0%) | 4/50 |
| free | 744/900 (82.7%) | 422/900 (46.9%) | 96/900 (10.7%) | 61/900 (6.8%) | 13/150 |
| anchored | 802/900 (89.1%) | 454/900 (50.4%) | 73/900 (8.1%) | 15/900 (1.7%) | 3/150 |

Anchored versus the matched free learner:58 more one-tile successes,32 more six-tile
successes,23 fewer falls,46 fewer lane departures. Falls and six-tile counts improve
in all three paired training seeds. Joint fall/lane improvement holds in seeds42/44;
seed43 lane departures regress3→4, so this is not uniform improvement.

**Promotion fails only `lane_vs_frozen_v5`:15/900 versus0/300.** All other10checks
pass. The separately predeclared directional gate passes; this does not overrule
promotion or prove safety/generalization. No tuning or model reselection followed.

### Hardest terrain remains a limitation

Across all terrain families at difficulty1.0, free has24/180 six-tile successes versus
anchored13/180, despite anchored having fewer falls26→21. Aggregate improvements
do not imply improvement on every difficult terrain.

On stones1.0 specifically, one-tile success is4/10 v5,24/30 free,20/30 anchored;
falls1/10,4/30,3/30 and lane departures0/10,2/30,3/30. All have zero six-tile
successes within16s. Stones0.8: one8/10,23/30,24/30; falls0/10,6/30,1/30.

### Was the training prior active?

Last100iteration teacher/student action-difference RMS perseed42/43/44: free
0.294/0.283/0.299, anchored0.150/0.149/0.156. Weighted loss was0 for free and
0.00565/0.00552/0.00612 for anchored. This telemetry uses each policy's own training
state distribution, not paired held-out states, and does not alone establish the
reason for a performance change. See `training_telemetry.json`.

### Verification

207CPUtests; coefficient-zero bit-exact PPO loss/state parity;64env2iteration training
and35env saved-model evaluation;4096env capacity;64s10env non-heldout smoke;39actual
saved-configuration checks pertrainingrun. Independent no-summary-import audit:
53source/PLAN/testhashes,35archivedsources,7frozenmodels,6configs,teacher identity,
paired initializations, freeze chronology, all14rawfiles and every cell/gate pass.

Two issues in the independent auditor (extra-command chronology and malformed
raw-array rejection) were hardened and re-reviewed before final evaluation data
were available; no frozen experiment source/model was changed. Primary snapshot:
`independent_audit_primary.json` / `independent_review_primary.md`.

[Primary results](../artifacts/terrain_demo/prior_v10/summary.md) ·
[Independent review](../artifacts/terrain_demo/prior_v10/independent_review.md) ·
[Local comparison page](../artifacts/terrain_demo/prior_v10/index.html)

## Independently audited64s hard-stone diagnostic

All first episodes on stones1.0;10env/map, two maps,140episodes kept separate from
primary. No retraining/selection or relaxed primary gate.

| Policy | Strict1tile | Strict6tiles | Falls | Lane exits | Survived64s |
|---|---:|---:|---:|---:|---:|
| frozen_v5 | 13/20 (65.0%) | 0/20 (0.0%) | 5/20 (25.0%) | 0/20 (0.0%) | 15/20 (75.0%) |
| free | 19/60 (31.7%) | 14/60 (23.3%) | 29/60 (48.3%) | 12/60 (20.0%) | 31/60 (51.7%) |
| anchored | 23/60 (38.3%) | 15/60 (25.0%) | 31/60 (51.7%) | 5/60 (8.3%) | 29/60 (48.3%) |

Both learned policies sometimes traverse all six tiles when given64s, unlike the
frozen reference in this small diagnostic. However, anchored gains only one strict
six-tile success over free14→15/60, and falls **increase29→31/60**. Versus v5,
anchored falls51.7% are also worse than25%. The primary fall advantage therefore
does **not** establish long-duration stability. Do not call the hardest stones solved.

| Training seed | Free six / falls / lane | Anchored six / falls / lane |
|---|---|---|
|42|4 /13 /3 (n20)|6 /10 /1 (n20)|
|43|3 /10 /6 (n20)|5 /10 /1 (n20)|
|44|7 /6 /3 (n20)|4 /11 /3 (n20)|

Anchoring is worse on seed44: fewer six-tile successes and five additional falls.
This dissenting seed is retained; no model was selected after seeing the results.

Distance-only53.1m hits: v50/20, free23/60, anchored21/60. The strictly successful
counts are0/20,14/60,15/60 because later failures/exits invalidate earlier hits.
Conditional mean time-to53.1m is51.67s free and56.72s anchored; these are times
among those reaching the distance, including later failures, not universal times.

Final first-episode distance means (64s or earlier termination) are20.84m v5,41.73m
free,39.82m anchored. Mean16s distances are14.19/18.33/16.59m only among17/20,
47/60,49/60 uncensored survivors respectively, so survivor bias must not be ignored.

All primary and horizon raw-array audits passed with zero mismatches. The two
maps and repeated initializations are limited evidence, not a real-camera,
real-robot or broad-distribution safety guarantee.

[64s full results](../artifacts/terrain_demo/prior_v10/horizon_summary.md) ·
[64s machine-readable results](../artifacts/terrain_demo/prior_v10/horizon_summary.json)

## Presentation and final checks

Four predeclared videos completed:16s/480frames and64s/1920frames perpolicy,
30fps,1280x720, complete decoder pass and content hashes retained. Each includes
all failures/resets. At15.9s the visible captions show v5 reset2/current-episode8.3m
versus anchored reset1/17.6m; at63.9s v5 reset2/43.6m versus anchored reset1/68.1m.
These are **post-reset qualitative trajectories**, not first-episode benchmark
successes or evidence that either clip was fall-free. No video was reselected.

Final207CPUtests, compileall, shell syntax, whitespace/diff checks, viewer JavaScript
syntax and all12local viewer links pass. Standalone Ruff/mypy/Pyright are not
installed; no such lint/typecheck result is claimed and no package was added.
All49logged experiment/render/decode commands exited0; all project GPU jobs ended.
The independent audit was rerun after the complete49-command ledger. No old model,
archived35-source implementation, threshold, dependency or physical robot was changed.

Compatibility: v10 checkpoints require the opt-in88D v10 task/policy loader. They
are not drop-in replacements for the original60D course interface or a real robot.
