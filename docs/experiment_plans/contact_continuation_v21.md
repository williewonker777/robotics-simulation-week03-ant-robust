# v21 — Matched no-contact-cost continuation control

Preregistered 2026-09-28 before new training or fresh-map scoring. Direct bounded
experiment in the established architecture; preserve all v0–v20 evidence/defaults.

## Question, evidence and limits

v20's immediate and ramped contact-cost continuations both failed the overall gates.
There was no equal-budget no-cost continuation arm, so a decline from the untrained
parent cannot isolate cost from continued learning. Add that missing control before
inventing another reward. The motivation remains v20's partial adaptation of
[Aractingi et al. (2023), reward curriculum](https://arxiv.org/html/2309.16683v1).
This control is an experimental-design repair, NOT another claimed paper algorithm.

Reuse v20's fixed immediate/ramped final249 models, without retraining/selecting them.
A newly trained no_cost model has the same start, seed, budget, sensors and all other
settings. Historical cohort execution remains a limitation: an exact 2-iteration
replay supports compatibility, NOT proof of identical full-length historical execution
or statistical causality. One training seed and two maps cannot establish general
algorithm superiority or real-robot safety. No automatic default promotion.

## Fixed training and compatibility gates

- Parent: unchanged v16 control249, SHA
  `1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc`.
- Copy the v20 initial checkpoint BYTE-FOR-BYTE into new experiment/v21_init/model_0.pt,
  SHA `10d8485d6d18f1dd6c01724ff69fc6a2ced3bbcd3d7b0db1285b7097cb5c21bd`.
  Do not rewrite checkpoint infos. Verify all32 non-std tensors equal parent, std0.2,
  empty Adam1e-4, iter0, frozen-v5 teacher and exact source/copy hashes.
- New no_cost only:4096env×32steps×250iterations=32,768,000 NEW main transitions,
  seed51/geometry110, final249 only. Existing immediate/ramped each32,768,000 are
  REUSED v20 continuations, not new v21 training. Parent has0 continuation transitions.
- Keep91D policy/value,8D actions, CommandPriorActorCritic conditioned/anchored/targets,
  teacher coefficient0.02, fixed physics/rewards/terminations and geometry110 cache.
- Compute the exact v20 raw contact geometry every call even for zero factor;
  RewardManager weight remains1. no_cost factor0 always. No sensor/reward skipping.
  Record global counters1..N, factor/raw/scaled/valid/dt evidence and passive contact
  telemetry. Preserve writable real-environment episode-length randomization forwarding.
- Development only permits immediate factor1 as a POSITIVE CONTROL; forbid its
  full250-iteration launch. New main no_cost coefficient mass and realized penalty0.
- Before GPU replay, newly pin old v20 immediate preflight/capacity model_1.pt bytes
  plus old raw/log/reference hashes. These checkpoint1 hashes were not in the original
  v20 development freeze; disclose that they are newly pinned historical evidence.
- Initialize training cache without stepping; require exact existing v20 geometry110
  cache hashes, never regenerate a different cache under the same comparison label.
- Preflight: new immediate then no_cost,256env×2iterations/ramp_steps32 metadata.
  Capacity: same pair4096env×2iterations/ramp_steps4000 metadata. New development
  training total557,056 transitions; do not select these models.
- Initial full91D obs/root/joints,88Dprefix, policy tensors, CPU/CUDA RNG, normalized
  config and REAL randomized horizons must equal corresponding old v20 immediate.
  Canonicalize ONLY declared coefficient mode, new reward function identifier,
  experiment_name/load_run and existing output identifiers; validate exact expected
  values before normalization. Every other field is unchanged.
- For both immediate positive-control runs require exact reward-time trajectory,
  passive rollout, final model1/Adam/iteration/infos and all learning/reward scalar
  values against newly pinned historical files. Exclude only the three Perf timing/FPS
  tags and duplicate Train/*/time series with wall-time x axes from exact scalar replay;
  their finiteness/completeness remains audited. No tolerance relaxation or aggregate-only proof.
- Main no_cost must match old v20 MAIN initial proof and real randomized horizons.
  Require8000 actual reward calls, zero scaled reward/dt sum, finite raw costs/weights/
  Adam and250 complete finite scalar logs. Freeze all experimental definitions,
  reference pins, initialization, cache and development evidence before main.

## Fresh evaluation, fixed before scores

- Eight controllers: parent,no_cost,immediate,ramped,history_parent,history_no_cost,
  history_immediate,history_ramped. History is unchanged v12 gate with frozen-v5
  teacher and the named91D expert; no inference changes.
- New geometry/reset **113/73,114/74**. Prior111/71,112/72 are NOT reused as holdouts.
- Primary16s175env/map/controller: rough300+flat50 per controller,16files2800 episodes.
- Separate64s10env/map/controller highest-level stepping stones:20 per controller,
  16files160 episodes. Total32files2960 scored FIRST episodes. No horizon mixing.
- Development51/24/35env:8runs280 episodes plus init-only cache preparation. All six
  reused controllers must exactly reproduce their v20 development physics/routing
  fields; all eight initial-state/RNG proofs pair; four history flat branches match.
- Initialize all480 new-map tiles without scoring, then freeze cache/source/models/
  command-ledger prefix before one-pass scored execution. No hidden retry or selection.
- Strict1/6tile remains13.1/53.1m AND no first-episode fall/footprint lane/world exit.
  Keep every low-speed/no-contact episode in denominators; report contact/posture
  diagnostics, per-map/family/level and paired6tile gains/losses.
- Primary contrasts: no_cost-parent; immediate-no_cost; ramped-no_cost;
  ramped-immediate; each corresponding history contrast. Also report both cost arms
  versus matching parent, for12 descriptive gates total per horizon.
- Gate: roughone/six nonlower, falls/lane nonhigher, worldzero includingflat,
  flatfalls/lane nonhigher, flatmean episode speed nonlower, at least one strict
  rough improvement. Secondary omits flat checks. Disclose ALL per-map failures.

## Completion and preservation

Stop after the one new fixed-budget control and full32-file matrix, even if it fails.
No reward retuning, earlier checkpoint selection, held-out map replacement or
outcome-driven rerun. Any replay/config/source/cache mismatch stops main/scoring;
preserve raw failure evidence. Only explicit recorded infrastructure repairs can
receive new development attempts before freeze, never relaxed success criteria.

Validate CPU regressions first, then positive replays, full training and fixed eval.
Independent raw arithmetic must recompute counts/gates and proof linkages; separately
verify CPU tensors/teacher/Adam. Record main/development/reused budgets distinctly.
Run available lint/type/AST/compile/whitespace/doc-link tests and disclose tool limits.
Preserve all old uncommitted work. One GPU-heavy job at a time on assigned GPU1;
no changes to unrelated robot processes, default policy, dependencies, commits or
remote Git/GitHub state.
