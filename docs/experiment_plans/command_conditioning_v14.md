# v14 — explicit terrain-command conditioning (pretraining plan)

User request: try another method informed by papers, implement and test. Continue
body-height adaptation intent: higher for obstacles, lower/faster on clear ground.
One bounded new experiment, not an automatic hyperparameter search. Direct scoped
execution with independent design/implementation review; no formal OMX mode.

## Literature and hypothesis
Walk These Ways (Margolis & Agrawal, CoRL2022/PMLR2023) makes body-height and
footswing/gait behavior commands policy inputs as well as reward parameters:
https://proceedings.mlr.press/v205/margolis23a/margolis23a.pdf (sections3.1–3.2).
We adapt ONLY explicit posture-command conditioning, not its full gait/history,
random behavior sampling, action targets, reward composition or sim-to-real stack.
Our automatic scan-to-command rule is our adaptation, not a paper finding.

V13's actor sees88D compressed foothold hints but not the reward's explicit goal
or exact local ground reference. Hypothesis: explicit goals/feedback improve
learned posture/traversal. V13 failure does not prove missing-information causality.
Miki2022 recurrent perceptive teacher/student and PGTT2026 phase/spline/contact
reward are alternatives but change many more variables; not implemented here.

## Immutable boundary
Keep all89 v13+legacy frozen source/plan files, six reference models, previous
success/failure/smoke evidence and physical benchmark unchanged. Only NEW opt-in
v14 files and current documentation/TASK records. No dependencies, remote Git,
default-policy replacement, robot commands or terrain/selector tuning.

## Method and controlled contrast
- Append3 scalars AFTER the exact88D prefix: normalized commanded torso clearance,
  normalized measured torso clearance, and validity. Total91D. Compute using the
  SAME frozen PostureReward.geometry / PostureConfig as the reward, statelessly.
- Feature0=(target_height-.44)/.14; feature1=clip((body_clearance-.44)/.14,-8,8);
  feature2=valid. Invalid body/depth/motion => all3 exactly0, not flat with valid1.
  No family/difficulty labels, map IDs, future states, or special holdout logic.
- Both arms use the same91D observation configuration and same v13 reward weight1.
  `masked`: actor AND critic explicitly set final3 inputs tozero. `conditioned`:
  both receive them. Thus contrast concerns actor/critic command+feedback access,
  not just actor access and not direct proof of arbitrary command-following.
- Reuse PriorActorCritic/PriorPPO via a small additive subclass, anchored lambda.02,
  no normalization, same400/200/100ELU,8 torque actions and exact60D teacher slice.
- To preserve old88D arithmetic and state keys, first actor/critic affine layer is
  W*x88+b + U*command3, with U initialized EXACTLYzero. It is mathematically an
  augmented affine input, not a separate action policy/residual controller. Keep
  old weight/bias keys, append only command weights and a strict mode/schema code.
  Copy EVERY source actor/critic/teacher/buffer, resetting onlystd to.2 and Adam/
  iteration. Never use the old60D warm-start that erases learned depth columns.
- Require initial original88D policy/critic/teacher outputs exactly equal on CPU
  and same-device GPU prefix probes; use contiguous88D prefix in original affine.
  Both new arms' actual initial states/91D observations/RNG/trainable tensors must
  match, except declared mode code. Verify actual loaded state before first rollout.
- No new reward or target constants. No imposed gait/phase, new teacher, history,
  camera noise or speed controller. Existing v12 gate unchanged for hybrid tests.

## Training budget and inputs
Starting checkpoint for BOTH arms: original v10 anchored seed42 final749,
SHA fa1ec87ef2b89b698f250d91452685cef0a7f76e340e790650116138aee875c1.
Each250iterations x4096env x32steps=32,768,000transitions; total65,536,000.
Same fine-tune seed46, training geometry85, identical freshAdam1e-4/std.2. Check
real saved config equality except command mode and output names (including the
strictly validated per-mode initialization directory). Pre-materialize
geometry85 cache before both arms; pin bytes+initial root/joint/full91D+RNG hashes.
Choose ONLY final249, no best checkpoint/seed, reward-based earlystop or retries.
Initial teacher unchanged and all weights/optimizer/reward logs finite required.

## Evaluation
Fresh geometry86/reset58 and87/reset59. Warm cache before ANY scored process,
pin complete cache manifest to an exclusive evaluation-input record, bind its SHA
into every result/command, verify before/after each process and keep exact initial
pairing. Freeze final models and all sources BEFORE heldout rollouts.

Six controllers: `v13` (frozen v13 adaptive standalone), `masked`, `conditioned`,
`history_original` (v5+originalv10), `history_masked`, `history_conditioned`.
All tested in identical opt-in91D physical Eval task; legacy policies receive only
unchanged88D prefix, hybridv5 teacher only60D. Inputs, original prefix and outcomes
must match frozenv13 evaluator on nonholdout geometry51/reset24/35env first.
Metadata must explicitly say91D environment,88/91D selected expert,60Dteacher;
do not impersonate old88D schema. Reuse old tracker/gate/telemetry/audit primitives
with an explicit validated91D-to-base reporting projection, never edit old files.

Primary:16s mixed175env,12files/2100firstepisodes. Separate secondary:64s hardest
stones10env,12files/120firstepisodes. Total24files/2220firstepisodes. Strictone/six
13.1/53.1m, footprint1.1m,dt1/60, samefall/lane/world semantics and snapshots.
Track all family/level outcomes and flat separately, exact initial state91D plus
88Dprefix. Passive posture measurements BEFOREaction for active firstepisodes only.
Reuse v13 conditional body/target/foot/low-speed/coverage metrics without pooling
horizons. Separate initial-state command-input sweep (low/high normalizedgoal)
AFTERtraining on nonholdout51/24: no environment stepping, no checkpoint choice;
use deterministic action means, normalized goal endpoints exactly0/1, valid samples
only, hold prefix/feedback/validity fixed; verify masked/legacy invariance and
source observation/RNG unchanged. Record OOD input sensitivity only, NOT physical
command tracking or success evidence. Goals were not independently sampled in training.

## Predeclared decisions
Compare conditioned versus equal-budget masked, not only old pretrained models.
Primary actor: one/six no lower; fall/lane no higher;world0includingflat;flatfalls
no higher; flat mean per-episode speed STRICTLYhigher (samev13 criterion).
Primary hybrid: samecommon checks, flat speed/flatbehavior nonregression/expected
identity, at least one strict terrainone/six/fall/lane gain. Frozenv5 flat equality
is NOT attributed to new training. Separate secondary: one/six no lower,fall/lane
no higher,world0,and at least one strictgain. Secondary cannot overrideprimaryFAIL.
Report rough target error and flat body height separately as conditional
visited-state metrics, not matched-state causal proof. A nonzero action response
to counterfactual commands does not prove desired physical height tracking.
No universalbest, statisticalsignificance or real-camera/robot-safety claim.

## Verification / stop
CPU regression: feature scaling/validity, pose frames and postreset refresh; exact
prefix/state preservation, mode/schema/teacher/optimizer guards; zero U and masked
invariance/gradients; conditioned branch gradients and finite output; strict91D
metadata, no overwritten evidence, manifest/model/cache/source drift rejection.
64env2iteration smoke both arms on geometry51/reset24, then4096env capacity probe
on pre-materialized85. Nonholdout legacy evaluator/prefix parity and strict audits.
FullCPU+compile/diff checks, independent review, then frozen training/evaluation.
Failures preserved and fail closed; no silent retry or weakening equality after
holdout outcomes. Implementation+declared comparison+audit complete the request,
even if scientific improvement fails. GPU1 only, oneheavyGPUjob at a time.

## Pre-freeze development boundary notes
Historical geometry51 has30tiles with4old configurations per seed. The strict
cache checker rejected its ambiguous inventory before any training/scoring; retain
all old cache bytes. New85 has240unique tiles and is pinned, as86/87 must be.
The nonholdout35episode legacy test exactly reproduces initial88Dprefix/root/joints
and all episode outcomes/returns/routing. Passive height/foot metrics also match,
but velocity sums differ (maximum3.60054 summed m/s; one low-speed count). Extra
observation-time geometry reads are consistent with lazy derived root-link velocity
cache timing around reset; installed COM-velocity setter does not invalidate this
cache. Do not claim passive velocity telemetry byte-equivalence. Both91D arms and
all new evaluations share the same read ordering. Performance speed remains actual
distance/time. Retain both raw files and explicit differences; no scored rerun.
A nonexistent plural episode_returns audit key was corrected to actual episode_return
and the same raw development output re-audited. No frozen source or outcome changed.
