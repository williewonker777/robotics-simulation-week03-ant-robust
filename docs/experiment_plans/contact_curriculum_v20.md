# v20 — Immediate versus ramped contact-slip penalty

Preregistered 2026-09-28, before training or new-map scoring. Direct bounded
experiment using the existing training/evaluation architecture, not a refactor.

## Evidence and question

v16's immediate slip cost previously reduced traversal compared with its no-cost
control. v19 established that the accumulated no-cost v16 controller improves
pooled16s traversal over v5, but64s stones still have lane exits and per-map
regressions. These observations do NOT establish that slip or penalty onset caused
the failures. Test whether introducing the SAME contact cost gradually is better
than imposing it immediately during a matched continuation.

[Aractingi et al., *Controlling the Solo12 Quadruped Robot with Deep Reinforcement
Learning* (2023), Reward curriculum](https://arxiv.org/html/2309.16683v1) uses a
linear0→1 factor on penalties to avoid learning to stand still instead of tracking
velocity. Here ONLY the existing contact-slip proxy receives that factor; all
other rewards/noise/terrain and Ant torque actions remain unchanged. This is a
partial warm-start adaptation, not the paper's full method, hardware or schedule.

## Frozen treatment and budget

- Parent: v16 no-slip-cost control final249, SHA
  `1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc`.
- Shared copy preserves ALL non-std tensors, including command conditioning and
  frozen-v5 teacher. Reset std to0.2, fresh Adam1e-4, iteration0 for BOTH arms.
- `immediate`: contact factor1 for every step.
- `ramped`: for one-based global policy step t, factor
  `clamp((t-1)/(4000-1),0,1)`. First factor0, step4000exact1, remaining steps1.
  The common/global clock must NOT reset when individual episodes reset or wrap.
- The existing bounded contact reward and sensors run in BOTH arms at term weight1;
  multiply only its scalar output. No sensor skipping at factor0, no RNG consumption.
- Each arm:4096env×32steps×250iterations=32,768,000transitions;
  total65,536,000. Train geometry110/seed51. Final249 only, no checkpoint search.
- Coefficient-step mass is8000 vs6000. This is NOT realized penalty dose, which
  additionally depends on visited states, contact cost and dt. Both trajectory and
  coefficient mass differ: this is NOT a pure onset effect with matched dose.
  First-half4000step ramp and final coefficient1 are preregistered engineering
  choices, not numbers taken from the paper. Last125iterations have factor1 in both.
- Keep91D actor/critic production,8Dactions, conditioned/anchored targets policy,
  teacher coefficient0.02, fixed v5 teacher, existing physics/rewards/terminations.

## Alternatives and decision

1. Another static reward weight: smaller code surface, but does not answer whether
   a gradual introduction helps; already negative immediate-cost evidence exists.
2. Ramped existing contact cost (selected): one coefficient trajectory changes,
   physical sensor implementation and inference stay fixed; dose differs explicitly.
3. New hard lane constraint/controller: more directly constrains exits but changes
   termination/dynamics or inference and needs a different question/validation scope.

Principles: preserve old evidence; paired inputs/budget; distinguish motion from
survival; report adverse outcomes; no default promotion. Main drivers are minimal
causal ambiguity, preserved baseline performance, and measured long-horizon exits.

## Gates before main training

- Regression tests for schedule boundaries/monotonicity/sum, invalid inputs,
  per-episode reset immunity, identical raw contact cost at factor1, CLI overrides,
  tensor/optimizer normalization and evidence guards. No dependency installation.
- Initialize training geometry110 cache WITHOUT stepping before any paired run;
  pin all240tiles/720files. Same requirement for development51 and all new eval maps.
- Preflight:256env×2iterations; both use declared development ramp_steps32 to
  exercise0→1→plateau in64steps. This override is permitted ONLY for this dev shape.
- Capacity:4096env×2iterations, main ramp_steps4000. Main also4000 exactly.
- Full91D observation,88Dprefix/root/joints, actor/critic/teacher outputs, policy
  weights, CPU/CUDA RNG and normalized config must pair at start. Adam empty.
- Record every actual reward coefficient, source common_step_counter and raw/scaled
  reward summaries; verify no negative rewards escape[-1,0], no reset clock restart,
  exact step count/monotone schedule and full-budget dose. Keep post-physics contact
  sampling and exclusion of reset/wrap rows; record contact/no-contact statistics.
- Preserve all250iteration scalar logs, finite policy/Adam tensors and teacher
  identity. Freeze source, parent/init/model/config/cache/log links. One GPU job.

## Evaluation frozen before new scores

- Six controllers: `parent`, `immediate`, `ramped`, `history_parent`,
  `history_immediate`, `history_ramped`. Parent is v16 control; history always means
  the unchanged v12 gate between v5 teacher and the named expert.
- New paired maps **111/71,112/72**; no previous v19 maps as holdouts.
- Primary16s175env/map: controller has rough300+flat50 first episodes.
- Separate secondary64s10env/map, highest-level stones: controller has20 first
  episodes. Total24files/2220episodes. No mixing horizons or flat/rough denominators.
- Development51/24/35env: evaluator parent must exactly match v19 v16_control,
  history_parent must match v19 history_control. All6newcontrollers must have paired
  fullinitialstate/prefix/RNG and byte-identical flat physical/routing history branch.
- One-time scoring after all **480 new holdout tiles** are initialized without
  stepping, frozen sources/models/pre-score ledger and cache hashes are fixed.
- Strict1/6tile remains13.1/53.1m plus NO first-episode fall, footprint lane exit or
  world exit. Report per-map/family/level, flat speed, low-speed and no-contact
  visited-state telemetry, and paired6tile gains/losses. No stall/no-contact
  conditioning may exclude episodes from traversal/fall/exit denominators.
- Primary comparison: ramped vs immediate; separate history_ramped vs
  history_immediate. Descriptive gate: roughone/six not lower, falls/lane not higher,
  worldzero includingflat, flatfalls/lane not higher, flatmean episode speed not
  lower, and at least one strict rough improvement. Secondary excludes flat rules.
- ALSO report the same gates against each matching frozen parent. Beating a newly
  degraded immediate arm does not establish improvement over the current parent.
  No statistical/algorithm superiority or automatic model promotion from one training
  seed and two maps, even if pooled gates pass; disclose every per-map failure.

## Stop, verification and preservation

Abort scored execution on source/model/cache/init mismatch; preserve all failed
evidence. No silent output overwrite, holdout retry, map substitution or retuning.
Development-only infrastructure fixes require recorded reason and fresh proof before
freeze, not a relaxed criterion. Do not change v0–v19 source/model bytes, unrelated
robot processes, defaults, dependencies, Git commits or remote state.

Stop after this one paired training and its fixed evaluation matrix, regardless of
success. Recompute raw strict counts/gates independently of the reporting arithmetic;
check all checkpoint/hash/ledger links. Run fullCPUtests, available bounded type/lint
checks, compileall/AST, new-file whitespace and diff checks. Report unavailable tools
and limits. Save commands, results, methods/paper differences, failure history and
next hypotheses, without claiming physical-robot safety.

## Review repairs before implementation freeze

Use a writable forwarding `episode_length_buf` property in the new telemetry proxy:
the underlying RSL runner assigns randomized initial horizons through this wrapper.
Verify the assigned tensor reaches the real environment unchanged and pair its SHA.
Preserve the old v16 source even though its read-only proxy lacks this forwarding.
Thus this is a valid new paired comparison, not a claim to reproduce v16's training
trajectory. Capture actual reward-time unscaled/scaled values inside the reward
term, not after-reset passive telemetry. Assert initial global counter0 and actual
reward counters1..N; geometry-only diagnostics must not advance schedule history.
