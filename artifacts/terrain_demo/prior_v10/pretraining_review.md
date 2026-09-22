# v10 independent pretraining review

**Verdict: APPROVE**

The frozen-v5 Gaussian-mean-prior implementation and the predeclared study match
`outputs/prior_v10_20260922/PLAN.md`. No CRITICAL, HIGH, MEDIUM, or LOW unresolved
findings remain. This is approval of implementation/protocol readiness only, not a
claim that the method will improve terrain performance.

## Scope

18 v10 files reviewed: the plan; policy/PPO/horizon core; task registration and
agent configuration; checkpoint preparation, launcher, experiment runner, primary
and horizon summarizers, horizon probe, renderer/viewer; and four focused test
modules.

## Stage 1 — specification and protocol

- `src/week03_ant/prior_policy.py:19-127`: full 88D student, exact 60D prefix
  teacher, frozen/eval teacher, student-only inherited inference, persisted mode,
  coefficient and fixed-sigma buffers, and strict load/config guards match the
  plan.
- `src/week03_ant/prior_ppo.py:41-193`: the auxiliary loss is the per-batch and
  per-action fixed-equal-sigma Gaussian mean term; it is added only at coefficient
  0.02. The student's learned PPO standard deviation remains the upstream PPO
  variable. Teacher means are detached. Unsupported recurrent/adaptive/RND/
  symmetry/multi-GPU variants fail closed.
- `scripts/prepare_prior_checkpoint.py:19-69`: the immutable selected-v5 SHA is
  required; student actor/value warm-start and teacher identity are recorded;
  optimizer state starts empty; overwrite is refused.
- `scripts/prior_v10.py:12-38` and
  `src/week03_ant/tasks/agents/prior_v10_cfg.py:9-34`: treatment mode,
  coefficient, targets input, zero foothold shaping, PPO class and strict training
  load guard are locked.
- `scripts/run_prior_experiment.py:96-149`: saved configuration is compared against
  the archived v9 targets arm, including the complete PPO configuration and the
  unchanged observation/reward/action/termination/event/curriculum/robot/scanner/
  simulation/terrain contract apart from the declared treatment and seeds.
- `scripts/run_prior_experiment.py:180-247`: all six final-749 checkpoints are
  frozen before the two held-out pairs; the single v5 reference is not replicated
  as training; checkpoint architecture, scalar treatment contract, teacher-v5
  bit identity, hashes and transition counts are checked.
- `scripts/summarize_prior.py:18-48,66-179`: the strict all-comparator promotion
  gate and the separate directional joint-safety gate implement the authoritative
  plan. Raw arrays, family/level balance, 300/900/900 terrain denominators,
  checkpoint identities, source hashes and all family/level/seed tables are
  independently recomputed.
- `src/week03_ant/horizon.py:68-195`, `scripts/probe_prior_horizon.py:71-221`, and
  `scripts/summarize_prior_horizon.py:23-127`: the 64-second diagnostic is isolated
  from promotion, preserves first-episode evidence across auto-reset, censors the
  16-second snapshot as declared, distinguishes first hit from strict final
  success, and re-audits 20/60/60 denominators.
- No broad catch, silent default, alternate success route, or fallback masks a
  broken primary contract.

## Resolved during review

- `scripts/run_prior_experiment.py:250-257` now refuses the horizon phase until a
  passing 2,450-episode primary verification exists.
- `scripts/summarize_prior_horizon.py:23-26,97-106` now verifies schema/task,
  predeclared held-out-pair scope, model pairing and post-freeze timing.

## Stage 2 — code quality and safety

- No credentials, network writes, shell interpolation, unsafe deserialization of
  untrusted input, or mutation of the installed RSL-RL package was introduced.
- Installed RSL-RL 3.0.1 PPO source/version are pinned and checked before training.
- Source review found no inference-time teacher call and no prior gradient path to
  teacher, critic, or exploration standard deviation.
- `basedpyright` was run over all reviewed Python files. This repository has no
  project Pyright configuration/stubs for Isaac Lab, TensorDict, or RSL-RL's
  dynamic runner registration, so it reports framework/dynamic-JSON false
  positives; those diagnostics were inspected. No actionable v10 type defect was
  found, and all reviewed files pass `compileall` and executable tests.

## Verification evidence

- Focused v10 tests: **56 passed** after the final protocol hardening.
- Full CPU suite: **207 passed** after the final protocol hardening.
- Zero-coefficient `PriorPPO` update is bit-exact with installed upstream PPO on
  seeded identical rollout storage.
- 64-env/two-iteration training plus save/load 35-env evaluation passed.
- 4,096-env/two-iteration capacity smoke passed.
- Non-holdout 10-env/64-second horizon smoke and independent raw-array
  recomputation passed.
- All 35 archived v9 source hashes match; the pinned v5 parent hash matches.
- Formal held-out performance remains unknown until the fixed training and frozen
  evaluation complete.

## Recommendation

**APPROVE** the fixed study. Do not alter the source/plan/test hash set after the
formal train-start freeze; retain v5 unless the predeclared promotion gate passes.
