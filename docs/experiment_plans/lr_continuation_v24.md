# v24 — Fixed global learning-rate sensitivity of no-cost continuation

Preregistered before any v24 GPU execution, 2026-09-28. This is a bounded,
additive extension of the existing experiment architecture. Preserve every
frozen v0–v23 source/model/artifact byte and the current default policy.

## Question and evidence boundary

V22/v23 found short-window regressions after continuation under three seeds.
V23 also found that some history combinations improve only at the longer window.
Neither result diagnoses a learning-rate cause. Test whether a tenfold reduction
of FIXED GLOBAL PPO LR, 1e-4 to 1e-5, changes these outcomes at equal data budget.

[PPO v2](https://arxiv.org/pdf/1707.06347v2), §§2–3 and §5/Algorithm1, motivates
controlling repeated optimization and distinguishes clipped objectives from a
hard constrained update. Its AppendixA uses a different benchmark configuration;
it does not validate this continuation LR choice. [What Matters in On-Policy RL?
v1](https://arxiv.org/pdf/2006.05990v1), §2, §3.2, §3.7 and AppendixJ.1/Fig.69,
shows conditional LR/initial-std sensitivity. Its Adam LR sweep starts at3e-5;
nearby1e-5 is an epsilon candidate, NOT an LR recommendation. Our1e-5 and3seeds
are preregistered engineering choices, not literature guarantees.

Global LR affects actor, critic and trainable std through their optimizer, not
only actor drift. Smaller LR does not guarantee smaller eventual parameter drift,
better robustness or a particular mechanism. Retaining learned initial std,
changing PPO clipping/KL/prior/reward or shortening training are separate,
deferred interventions. No selection or tuning on old/new holdout outcomes.

## Fixed arms, initialization and environment

- Training seeds51/52/53, training geometry110, samev22parent v16control249.
- High controls are the three frozen v22 final249 checkpoints (LR1e-4).
  Parent appears once. A repeated high51 training is a validation replay, NOT a
  fourth independent seed and NOT a replacement for an unsuccessful replay.
- Low51/52/53 are NEW fixed-LR1e-5 continuations. Same4096env×32steps×250iterations
  =32,768,000transitions each,98,304,000new low-arm transitions. Final249 ONLY.
- Same91D conditioned/anchored/targets network,8Dactions, fixed-v5 teacher,
  prior0.02, unchanged no-cost contact term, physics, curriculum/random horizons,
  minibatches/epochs, entropy, clipping and every other PPO parameter.
- Starting high checkpoint is a byte-identical copy of the frozen v22/v21 init.
  Low init is the same deserialized object with ONLY each empty Adam param-group
  `lr` changed1e-4→1e-5. All33policy tensors, std0.2, iteration0, empty optimizer
  states, param ordering/options and historical infos are preserved. Infos are
  ancestry, not the current LR authority; the new manifest records the LR change.
  No cross-serialization file-SHA equality requirement; independently pin each
  file and prove deserialized LR-only difference.
- Normal checkpoint loading must yield the declared saved-config LR,
  `alg.learning_rate` and EVERY Adam group LR. Never silently coerce/repair a
  wrong loaded LR. Verify exact optimizer parameter coverage including actor,
  critic and std, and unchanged frozen teacher tensors.
- Check LR/schedule/fixed/desired_kl=None before and after EACH PPO update AND
  every actual Adam step; distinguish250PPO updates from5000optimizer steps
  (2update development has40Adam steps). Persist all per-update and per-step
  LR observations/counts, initial/final optimizer metadata and final group LR.
  Wrong checkpoint/config/runtime LR must fail before the first update.

## References, paired development and training freeze

Before GPU, pin v22 init, all three same-seed preflight/capacity/main records,
checkpoints, logs/TensorBoard, normalized configs, caches and v23 development.
Rehash237legacy source files and22checkpoint paths. No prior-byte modification.
Implement12new definitions (7training Python+plan+4evaluation Python), run CPU
tests/static checks and independent implementation review before development.

GPU order, one job at a time on cuda:1:
1. Training cache initialization-only preparation, geometry110,0scored episodes.
2. High51 then low51 small preflight:256env×32×2iterations, ramp32.
3. For each seed51,52,53: high capacity then low capacity,4096env×32×2, ramp4000.
4. Freeze ALL12training/evaluation definitions, plan, references, initialization,
   development, cache and expected-main records BEFORE any250-iteration run.
5. High51 full250-iteration positive replay. Must exactly reproduce frozen v22
   model/Adam/metadata, reward/contact sequences and all33non-time scalar series
   before any low main run. All38scalar series must be finite. Only5time series
   are excluded from exact equality; record them explicitly. Serialized SHA
   equality is an observation, not a substitute for/deserialized replay gate.
6. Low51→low52→low53 full250-iteration runs, each from its low initializer.

Every high capacity seed51/52/53 must exactly replay ITS SAME-SEED v22 capacity,
not merely seed51. High51 small also exactly replays its v22 small record.
All references already exist; no added historical claim or rewritten reference.

Low/high pairing is SAME-seed PRE-learning physical state/observations/policy/RNG,
actual randomized initial episode horizons and common configuration. Normalize
only enumerated opt-in task/experiment/start-directory identities; preserve
actual seed and LR in raw proof. For a separate comparison copy, project exactly
the LR field to compare common non-LR settings. No broad text replacement.
Do NOT require post-learning low/high trajectories, rewards, contacts or final
weights to match; those are consequences of the intervention.

Each full run pairs to its own same-arm/same-seed4096capacity initializer; ONLY
`max_iterations`2→250 is projected to a pinned expected-main record. Check live
requested/saved agent/environment seed and actual horizon assignment, not config
intent alone. Full high51 also pairs to historical high51 main recorded evidence.

New training accounting: low mains98,304,000 + diagnostic high replay32,768,000
+ development1,605,632 = **132,677,632** transitions. Development is2×256×32×2
+6×4096×32×2. Historical high controls98,304,000transitions are reused ancestry,
not new training and not extra independent seeds.

## Evaluation: keep v23 paired horizons and all arms/seeds

Controllers, exact order:
parent,seed51,seed52,seed53,low51,low52,low53,history_parent,history_seed51,
history_seed52,history_seed53,history_low51,history_low52,history_low53.
`seedNN` means unchanged historical high LR; `lowNN` means new low LR final249.

Fresh geometry/reset119/79 and120/80 (no previous scored use found locally).
Each controller/map:175mixed environments,7families×5levels×5starts, physical64s,
dt1/60,max3840steps. Passive16s/64s windows from SAME first episode, immutable
v23tracker, no virtual reset. Strict final-distance13.1/53.1+no first-episode
termination/lane/world violations. Float32 thresholds preserved. Timeout is not
a fall; first-hit/max-distance is not strict success. Auxiliary8s/16s snapshots
remain separately named. Preserve physical/censor flags, early-completion,
JSON-deep snapshots and state/policy/history/RNG passivity proofs.

Evaluation development uses geometry51/reset24/35env:
- Initialization-only cache prep,0scores.
- Eight paired64s runs, order parent,history_parent,low51,low52,low53,
  history_low51,history_low52,history_low53:280first episodes/560dependentwindows.
- BOTH windows of parent/history_parent exactly reproduce all35PARITY_FIELDS
  from corresponding v23 development. All8 initial state/prefix/RNG proofs and
  the4history flat branches pair. Low final outputs have no historical equality
  target; validate all raw semantics/provenance rather than fabricate one.
- Substituting for6historical-high development GPU runs requires exact whole
  evaluator AST equivalence to v23 after this ENUMERATED allowlist ONLY: module
  docstring, study-module import identity, version-labelled CLI/error/print text,
  and explicit new raw `lr_condition`/`model_learning_rate` provenance fields.
  No physics/inference/sensor/step/reset/tracker/snapshot/passivity change is
  allowed. Reuse frozen v23tracker unchanged. Exhaustive14controller→checkpoint
  path/SHA→expert/history mode tests, including swapped-high/low negative tests,
  are mandatory. If this equivalence fails, reject reduced development; do not
  silently add runs or weaken the comparison after training-source freeze.
- Freeze final models and validated development, initialize both fresh terrains
  without scoring, pin480tiles and preholdout ledger prefix, then score.

Holdouts:28physical bundle files,56nested window records,
**4,900first episodes /9,800DEPENDENT window observations**, not9800episodes.
Each controller/window300rough+50flat; highest-stones subset10. Do not duplicate
parent or treat histories/maps/windows as independent training replicates.

## Preregistered reporting and comparisons

Exact18directed comparisons, same order at both windows:
1. low51_vs_seed51;2.low52_vs_seed52;3.low53_vs_seed53;
4.history_low51_vs_history_seed51;5.history_low52_vs_history_seed52;
6.history_low53_vs_history_seed53;
7.low51_vs_parent;8.low52_vs_parent;9.low53_vs_parent;
10.history_low51_vs_history_parent;11.history_low52_vs_history_parent;
12.history_low53_vs_history_parent;
13.seed51_vs_parent;14.seed52_vs_parent;15.seed53_vs_parent;
16.history_seed51_vs_history_parent;17.history_seed52_vs_history_parent;
18.history_seed53_vs_history_parent.

Same existing full-MIXED gates at BOTH windows: rough1/6 nonlower, falls/lane
nonhigher,world0includingflat; flatfalls/lane nonhigher and flat mean episode
speed nonlower; at least one strict rough improvement. Family/family-level rough
subgroups omit flat checks; full level subgroups retain available flat checks.
Report pooled/per-map/family/level/family-level and same-row controller-six pairs.
Seed ranges for high/low actor/history separately; parent counted once, matched
low-minus-high deltas per seed. Report all3seeds, not a selected favorable one.

For every controller, preserve49same-row16→64subgroups per pooled/map view:
four-cell strict-six neither/late-gain/late-loss/both; overlapping new terminated,
lane/world or final-distance-loss flags; first-six-hit(16,64] separately. Report
16s PRIMARY and64s SECONDARY; longer-window gains cannot override short regression.

## Validation, failure semantics and stop

CPU tests cover LR-only initialization, loaded-LR rejection before learning,
all-update/Adam-step LR guards, same-seed pairing/projection, positive replay,
all14bindings/18comparisons, full evaluator AST allowlist, strict/censoring/time
pairing, raw tampering, ordered budget/provenance and exclusive output writes.
Independent stdlib raw arithmetic/provenance audit and independent reviewer
check final evidence; independently deserialize trained checkpoints/optimizer
and verify learning/LR/teacher/scalar/step-budget evidence on CPU.

Expected GPU commands **52**:1training prep+2small+6capacity+1full high replay
+3low mains+1eval prep+8eval development+2holdout prep+28holdouts.
No extra training, checkpoint selection, map replacement or scored retries.
Any mismatch stops downstream work. A separately documented infrastructure
repair is possible ONLY before training-source freeze with preserved failed
attempt evidence and a new explicitly accounted attempt. After that freeze,
no code repair/restart under the same frozen identity; report the incomplete
study and the failure rather than replacing evidence or continuing ungated.

Run full CPU and available lint/type/AST/compile/diff verification. Preserve
local console/cache/runtime evidence and portable plan/results. No dependencies,
default-policy changes, commits, remote Git/GitHub operations or robot changes.
Stop after this single fixed study and complete reporting regardless of outcome.
One parent/training terrain and3seeds/2eval maps cannot establish significance,
general robustness, actor-only mechanism or physical/indefinite safety. TASK298's
older separate demonstration/publication scope remains separate.
