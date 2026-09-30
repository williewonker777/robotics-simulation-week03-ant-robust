# v17 preregistered plan — learning-progress terrain curriculum

## Question, chosen reference, and stop rule
After v16 contact-conditioned slip reward failed all four matched promotion comparisons,
test ONE genuinely different method: training-task sampling rather than another
locomotion reward or actor sensor. Li, Li & Hutter (2026), *Scaling Rough Terrain
Locomotion with Automatic Curriculum Reinforcement Learning*, Eq. 5–7, estimates
signed per-task improvement in episodic return and softmax-samples informative
tasks (https://arxiv.org/html/2601.17428v1). This Ant reset sampler is an
adaptation, not a reproduction: the paper uses ANYmal-D, other terrain/tasks,
and its own complete training setup. Hard stepping stones are not demonstrated
there. Do not infer efficacy from the paper; final Ant passage/fall gates decide.
Stop after this one preregistered paired experiment and raw audit, PASS or FAIL.
Do not tune coefficient, seed, checkpoint, or holdout after seeing fresh scores.

## Existing evidence and source
Source is frozen v16 **control** final249
`artifacts/terrain_demo/contact_slip_v16/runs/control/model_249.pt`, SHA-256
`1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc`.
It has 91D conditioned actor/critic and learned command U, frozen 60D v5 teacher,
adaptive posture reward, and no v15 direction or v16 slip cost. Preserve every
learned tensor except scalar exploration std reset to .2; fresh Adam1e-4,
empty state, iteration0. Both v17 arms use the same v16 sensor-enabled training
scene, existing rewards and physics, with contact_slip.weight=0. Inference remains
91D, depth-history gate and physical score unchanged. No v16 frozen file edits.

## One treatment: signed learning-progress reset sampling
Both arms have the same NEW per-episode lane-resampling wrapper. Existing weighted
startup assignment remains identical (five other mesh families weight1,
stepping_stones weight4, flat weight2, each with five levels). Reset wrapper
records old-lane completed return and duration BEFORE changing lane, invokes the
old root reset once, and translates the new reset pose by entrance/center/ground
origin deltas. Flat/mesh crossing changes spawn z correctly; no teleport progress
reward. Initial construction reset never samples. Because RSL randomizes initial
episode counters without past rewards, the first post-learn completion of EACH
environment is resampled but excluded from learning-progress statistics. Only
subsequent true completed episodes with finite reward are eligible. Process each
simultaneous reset batch as one aggregate before updating distribution.

- `fixed`: on every completed reset, sample the original task weights with a
  private seed49017 generator. This is the causal control for per-reset
  resampling; old v16 static-startup control is descriptive only.
- `lp`: same startup, private generator, reset and physics; the sole difference
  is signed learning-progress updates of mesh family×level probabilities.
  Flat mass is fixed at 2/11 to retain the course gait; its five physically
  equivalent lane IDs are sampled equally and logged but not LP-updated.
  Mesh mass remains 9/11. Across 30 mesh tasks, base probability is the
  inherited family×level weight normalized (stones4, other families1).
  After at least 1,024 eligible mesh completions AND at least four samples in
  every mesh task, calculate current task-average full episodic return. The
  first window establishes the previous mean and leaves LP=0. Later windows:

      LP_i = mean_return_current_i - mean_return_previous_i
      q_i = normalize(base_i * exp(clip(LP_i / 5, -2, 2)))
      p_mesh_i = 0.4 * base_i + 0.6 * q_i

  This is a base-weighted signed softmax with floor and fixed flat quota;
  beta5, clip2 and floor.4 are OUR preregistered adaptations, not paper
  constants. No EMA, absolute LP, fake zero for missing task, or holdout
  feedback. If a window lacks four samples/task, delay the update and keep
  accumulating. Log window counts, means, LP, p, excluded initial episodes,
  sampled episodes and actual transition occupancy. Stage size16 may be used
  ONLY in development preflight to exercise updates; full training uses1,024.

## Equal-budget training and controls
Seed49, training geometry101, 4,096 env×32 steps×250 PPO iterations/arm =
32,768,000 transitions each, 65,536,000 total. Train `fixed` and `lp` sequentially
on cuda:1 under existing GPU lock; one heavy GPU job at a time. Save only
final model249 for holdouts, not a cherry-picked intermediate. Exact full91D
initial observations/root/joints/CPU+CUDA RNG/policy/teacher and normalized
configs must match except arm mode/output paths; both modes have identical
initial lane layout and private sampler state. Train logs all250 iterations,
final model+Adam finite, and learned command U/teacher contract checked.
Before training freeze: pure CPU tests, independent code review, 256-env, 300-iteration
late-stage development smoke (two valid LP windows using declared development
stage16), and 4,096-env two-iteration paired capacity test. The latter checks
physical batch feasibility, not LP convergence. Full training must show >=2
valid default-size LP windows before holdout; if no update, report an invalid
mechanism rather than tuning after seeing holdouts.

## Fresh evaluation, prior references, and decision criteria
Holdouts have never been used in earlier versions: geometry102/reset64 and
geometry103/reset65. Prewarm both OBJ caches without scoring, then freeze
new v17 source, prior source/model hashes, final249 models, cache and command
prefix BEFORE scoring. Use unchanged v16 sensor-enabled eval physics/task,
no training LP wrapper. Six controllers: `v16_control` (static-startup source
reference), `fixed`, `lp`, `history_original` (v5+v10), `history_fixed`,
`history_lp`. Original history expert88D; other new experts91D, all with
fixed v5 teacher60D and unchanged v12 depth-history gate. Same initial
state/prefix/RNG per map/scenario must match exactly. Evaluate every controller
on both maps: primary 16s mixed175env = 12 files/2,100 first episodes
(rough300+flat50 per controller); separate secondary 64s level4 stepping
stones10env = 12 files/120 first episodes. Total24 files/2,220. Strict one/six
=13.1/53.1m with no fall/lane/world exit; first episode only. V16 references
are descriptive, not post-result substitutes for paired contrasts.

Primary causal contrasts: `lp` vs same-budget `fixed`; `history_lp` vs
`history_fixed`. Actor promotion requires rough one/six not lower, falls/lane
not higher, world0 including flat, flat falls not higher, AND flat mean
per-episode speed strictly higher. Hybrid requires rough nonregression,
flat nonregression, exact per-environment v5-branch flat raw identity and
at least one strict rough terrain improvement. Secondary 64s requires
one/six/falls/lane/world nonregression plus >=1 strict gain. Secondary cannot
override primary failure. Report all family/level outcomes and raw 24 JSON,
not merely weighted average. A policy is not called universally better from
one fine-tune seed/two maps. No default switch absent promotion.

## Verification and risks
CPU tests: normalized signed LP/floor/flat quota; missing-task postponement;
permutation-invariant batch update; private RNG parity; initial and first
randomized-horizon exclusions; old-lane return attribution incl terminal
reward once; mesh↔flat spawn translation and unchanged velocity/orientation;
zero teleport-progress spike and post-reset observation destination. Guard
nonfinite returns, unexpected task layout, version changes, stale/overwritten
outputs. Training and eval source SHA freeze; 24-file independent raw audit;
full CPU suite, compileall, diff/static checks; record optional-tool gaps.
Any v16 source/model/cache/score change fails. No dependency, default-controller
change, commit or remote Git operation. If no improvement, report FAIL honestly.

## Pre-training, pre-holdout development amendment — 2026-09-23
The originally planned 256-env/120-iteration stage16 fixed-arm smoke completed
without a simulator error but produced only ONE valid LP window: 903 eligible
episodes overall, with the first window covering 412 mesh episodes and one rare
mesh task still at zero in the remaining window. Its retained raw log/sampler
artifact is `outputs/learning_progress_v17_20260923/preflight_fixed*`; no LP-arm
development run, full training, fresh-map evaluation, or holdout scoring had
occurred. The missing second update is an exposure/coverage issue from the
predeclared four-per-task gate, not a score comparison. Extend only the paired,
development-only 256-env smoke to 300 iterations in BOTH arms, starting from
the same checkpoint/seed and preserving all treatment/window constants. Retain
the failed 120-iteration evidence; never use either development smoke for model
selection. Full training budget and holdout protocol remain unchanged.
