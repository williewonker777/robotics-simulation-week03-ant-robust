# v25 — teammate-inspired recovery and exploration transfer

Preregistered 2026-10-01 before v25 GPU learning. User requested porting useful
elements from Stick-0 and LimDaeKyung, actual training, fixed-condition comparison
against current models, and publication to the canonical Ant repository.

## Evidence and interpretation

- Stick pinned main `3cc718a4214f336fd4db7db5841fa86033b99d35`: same-parent,
  same600-iteration reward-package control, seed42, heldout4001/4002,
  survival920→969/1024, falls104→55, speed4.399→4.158m/s. Complete package, not
  component causality; smoothing-only predecessor regressed. [Pinned evidence](https://github.com/Stick-0/isaac-ant-rough-terrain/blob/3cc718a4214f336fd4db7db5841fa86033b99d35/docs/archive/reward_comparison.md).
- Lim pinned main `8d9eed1fe463f638d5a62528dbcc0a3656ddd52b`: same1000-iteration,
  five-seed boxes experiment with entropy0→.005, heldout mean return50.7008→62.1660,
  but falls19.96→21.28%. Our parent already uses.002, fixedLR1e-4, gamma.995 and
  teacher prior.02: transfer is **.002→.005**, not source reproduction.
  [Pinned evidence](https://github.com/LimDaeKyung/IsaacLab_RS/blob/8d9eed1fe463f638d5a62528dbcc0a3656ddd52b/docs/assignment1/RESULTS.md).
- Source metrics/tasks are not our strict lane benchmark. Do not compare return,
  survival percentages or training reward across repositories as performance ranking.
- Preserve source BSD-3-Clause notices, pinned attribution and adaptation notes.
  No teammate checkpoint, terrain importer, termination/evaluator or framework patch.

## Question, options and port boundary

Primary question: does the combined transferred recipe improve our fixed rough-lane
outcomes versus a same-budget continuation control? Secondary: recovery−control and
combined−recovery measure a reward-package effect and entropy increment **conditional
on recovery**. No entropy-only/interaction or individual recovery-term attribution.

Three arms × three seeds costs294,912,000main transitions. A six-run joint-only design
saves98,304,000 but cannot separate conditional changes; a12-run factorial adds
98,304,000 and identifies entropy-only/interaction. Choose the nine-run design as
bounded middle ground. One parent, one training geometry, three RNG realizations and
two heldout maps do not prove population-level superiority or safety.

Existing lane speed cap3m/s, uncapped flat progress, terminal penalty, lane keeping,
stone clearance and adaptive posture remain. Do not duplicate Stick's global target
speed/fall cost. Add a **Stick-inspired reward subset**, not its whole successful recipe:

```text
safe_height = min(0.48, existing_adaptive_target_height)
clearance_risk = clamp((safe_height − current_posture_clearance)
                       / (safe_height − 0.31), 0, 1)^2
tilt_risk = clamp((0.93 − upright_projection) / (0.93 − 0.5), 0, 1)^2
recovery_reward_rate = −2*clearance_risk −2*tilt_risk
                       −0.01*sum((action − previous_action)^2)
                       −0.025*sum(body_frame_angular_velocity_xy^2)
```

Reuse existing current-scan posture geometry; invalid clearance/target abstains from
clearance risk and logs abstention. The target cap avoids penalizing the existing
correct0.44m flat target. Scan clearance is above the local **maximum** valid ground,
whereas torso termination uses lane-ground averaging. Therefore0.31 is an imported
warning constant, not an identical measurement of the termination boundary. Tilt
0.93/.5 warns before the unchanged1.2rad overturn threshold; reward only.

Use unscaled action-manager action/prev_action, **not noisy last-action observation**,
not output effort or a divided-by-dt derivative. Body-frame roll/pitch angular velocity.
All four are continuous rates; RewardManager integrates once by step_dt=1/60. No new
action filtering, observation noise or sensor input. Nonfinite physical/action state
is a validation failure, not silently converted to successful/zero-cost data.
Compute passive diagnostics in every arm; control's additional applied rate is exactly0.

| Arm | Recovery contribution | PPO entropy |
|---|---:|---:|
| control |0|.002|
| recovery |1×above rate|.002|
| combined |1×above rate|.005|

## Frozen invariants and training

- Immutable parent v16control final249, SHA
  `1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc`.
- Same91D actor/critic,8D effort scale7.5,400/200/100ELU,
  conditioned/anchored/targets, no normalization, frozen-v5 training prior.02.
  All student layers learn; standalone inference is student-only.
- Same teacher SHA `889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e`.
- New initializer perseed61/62/63 from exact parent tensors; std reset.2 only,
  iteration0, fresh empty Adam1e-4, betas(.9,.999),eps1e-8, zero decay. Each seed's
  three arms use the same initializer. Teacher/all non-std tensors bit-identical.
- New training geometry130, original six rough families+flat/levels/reset distribution.
  Verify historical generator-config/scored-ledger inventory and cache identity before
  claiming130/131/132 unused. No heldout tuning or changed difficulty/termination.
- Perarm/perseed4096env×32step×250iteration=32,768,000transitions. Nine finals249 only;
  no intermediate/best checkpoint or favorable-seed selection. Order seed61control,
  recovery,combined; then same order seed62, then63. Single GPU job at a time cuda:1.
- Source/model/result/video bytes present at base `baee9d2` remain unchanged. New
  opt-in task/scripts/tests/manifests only; README/notice/publication manifest may be
  updated after validated results. No installed SDK/PPO/old scientific code changes.

## Before the first full run

1. CPU math boundary/monotonicity/unit tests, flat-target zero risk, invalid scan and
   malformed-input handling; exact control-zero and config/binding substitution tests.
   Independent implementation review and static checks.
2. Initialize common model/cache with zero scored episodes. Save complete original
   publication checksum inventory and source/runtime/dependency hashes.
3. Small preflight three arms seed61,256×32×2 (49,152transitions total).
4. Capacity preflight nine arm/seed cases,4096×32×2 (2,359,296transitions total).
   Record live/saved simulator/agent seed, initial root/joint/91Dobs/RNG/state hashes,
   same-device parent actor/critic/teacher outputs, empty Adam/std, full config and
   actual forwarded random episode-length assignment. Pair all three arms within seed.
5. Verify config after an exact allowlist: recovery enabled flag, entropy coefficient,
   declared output/task/experiment identities only. Preserve all true seeds/treatments.
6. Freeze scientific source/tests/plan, all initializers/cache/development evidence and
   each run's expected main initializer. Project **only max_iterations2→250** from
   each corresponding capacity record. Total development2,408,448transitions.
7. Main runner rechecks own expected initializer before learning. Audit actual entropy,
   prior coefficient, fixedLR/Adam groups and update/step counts; finite scalars/models,
   frozen teacher, actual8000policy steps/250PPOupdates/5000Adamsteps perrun.
   Save final model, original configs and hashes; freeze all9before heldout scoring.

## Fixed evaluation matrix

- Legacy reference endpoints: v5, v16_control, history_control, high53,
  history_high53. Resolve exactly as existing unseen evaluation, not latest/best;
  high53 is v22seed53final249 SHA5c86a9b6…, not v24capacity probe.
- All nine new models standalone and all nine in the **unchanged** v5 history gate.
  23controllers total. Gate uses depth history only, no family labels; history v5 is
  mixed-trained, not a pure-flat expert. Policy/SHA/teacher/buffer/mode bindings explicit.
- Heldout geometry/reset131/111 and132/112, same175env mixed7families×5levels×5starts.
  Percontroller350physical firstepisodes:300rough+50flat; two passive16/64windows of
  the **same** physical64s episode at1/60dt (3840step maximum). Main8,050physical
  episodes/16,100dependent-window observations, only350paired initial conditions.
- Use unchanged v16Eval task, strict final odometer distance float32>=13.1/53.1m
  AND no accumulated firstepisode fall/lane/world violation.53.1m is **six tiles**,
  not six metres; maximum distance/first hit/survival alone is not strict success.
- No virtual reset, policy/history/RNG mutation from passive snapshots. Pair actual
  geometry/config/initial physical state/full observations/RNG, not just seed labels.
  Later physical resets never count as new benchmark successes.
- Cache preparation has zero scored steps. Scored development uses geometry51/reset24,
  35env×64s, controllers v16_control,history_control,control61,recovery61,combined61,
  history_control61,history_recovery61,history_combined61, with seed61 common **initial**
  model for the six new endpoints:280physical/560dependent windows. Actor/value/teacher
  mean outputs match parent before learning (std reset does not affect means).
  Require exact development firstepisode parity for new standalone versus parent and
  new history versus history_control, plus explicit checkpoint/source binding.
  Reuse low-level immutable tracker/evaluator; no relabeling v22 training orchestration.
- Audit every raw array before aggregation, keep all seed/map/family/level rows,
  paired gains/losses and late successes/failures. Frozen legacy references appear
  once permap, not triplicated into independent training seeds.

## Preregistered verdicts and reporting

At each window, primary combined versus matched control requires aggregate strict
one/six nondecrease, rough falls/lane exits nonincrease,world0,flat falls and flat lane
exits nonincrease, standalone flat mean speed nondecrease, and at least one strict
rough improvement. Calculate a separate same-seed verdict with the same retention gate
on **both maps** and the combined two-map rows. An arm-level PASS requires **3/3 paired
seed PASS**, never pooled improvement overriding a failing seed. Report pooled counts
and every perseed verdict descriptively, without inferential significance from episodes.
History comparisons
apply unchanged gate and exact per-environment flat-v5 branch identity in addition.
Secondary recovery−control and combined−recovery use the same rules separately.
64s is a separate diagnostic and cannot rescue a failed16s primary gate.

Compare these gates with existing v16/high53/v5 descriptively under the same maps.
An improvement versus extra-training control need not beat the parent or the current
best history combination. Original60D stock Baseline42/Robust42 results and cross-team
returns are separate protocols; do not calculate misleading percentage gains.
Do not automatically replace default/submission policy, even if a candidate passes.

Training rewards differ; curves document learning/entropy/prior/regularization, not
performance ranking. Publish all9finals, configs, frozen hashes, raw firstepisode arrays,
summary, independent audit, exact reproduction commands, licenses/attribution and
README differences/results/tradeoffs. Optional predeclared qualitative16s obstacle
replay: combinedseed61 versus parent on131/111; keep resets/failures, no benchmark count.

## Pre-mortem, recovery and stop rules

1. Added stability slows required53.1m progress or fights posture: cap safe height at
   current target; retain original reward/termination, report speed and both horizons.
2. Larger entropy increases falls or breaks teacher preservation: fixed coefficient
   audit, perseed fall/lane metrics, exact frozen teacher; failure is retained evidence.
3. New wrapper silently changes seed/horizon/learning budget or resolves wrong model:
   positive/negative unit tests, same-device initial parity, actual horizon forwarding,
   exact manifest/source/checkpoint/config comparison and independent raw auditor.

Fix implementation/preflight errors before freeze and rerun affected proof. After
scientific freeze, no silent changes or outcome-conditioned tuning; disclose a new
revision/repair with old attempts retained. Recoverable infrastructure faults preserve
logs/checkpoints and retry only under an explicit recorded unchanged-input contract.
If no measured improvement, publish the negative result and keep old models/defaults.
Stop only after requested training/evaluation/documentation/push are validated, user
cancels, or no meaningful recovery path remains; do not claim a smoke run as training.
