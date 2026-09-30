# v22 — No-cost continuation training-seed sensitivity

Preregistered before any v22 GPU learning/scoring,2026-09-28. Direct bounded
replication in the established architecture; preserve all v0–v21 source/model bytes.

## Question and references

Does v21's no-cost continuation tradeoff repeat when only the continuation RNG seed
changes? Do NOT add/tune rewards or select an apparently favorable seed.
[Henderson et al., AAAI2018](https://ojs.aaai.org/index.php/AAAI/article/download/11694/11553)
reports sensitivity of deep RL results to training seeds.
[Agarwal et al., NeurIPS2021](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)
distinguishes independent training runs from evaluation episodes/tasks and cautions
about few-run uncertainty and framework/GPU nondeterminism. Our three-seed protocol
and exact-replay gate are engineering choices, NOT a paper-prescribed sufficient
sample size or a guarantee that equal seeds must reproduce every simulator state.

This is continuation-RNG sensitivity CONDITIONAL on one fixed parent and training
terrain, not initialization/distribution robustness or a general causal diagnosis.

## Fixed training

- Parent unchanged v16 control249 SHA
  `1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc`.
- Copy historical v21 init bytes into new `week03_ant_seed_continuation_v22/v22_init`;
  SHA `10d8485d6d18f1dd6c01724ff69fc6a2ced3bbcd3d7b0db1285b7097cb5c21bd`.
  Preserve infos,32non-std tensors equal parent,std0.2,empty Adam1e-4,iteration0,
  unchanged8tensor frozen-v5 teacher. No new random network initialization.
- Fresh no_cost full runs in EXACT order **seed51,seed52,seed53**. Each
  4096env×32steps×250iterations=32,768,000transitions; total98,304,000NEW main.
  Train CLI seed must reach both agent and simulator; never normalize it away.
- Same91D policy/value,8D actions,conditioned/anchored/targets,prior0.02,
  physics/rewards/terminations and geometry110 byte-identical cache as v21.
  Reuse the unchanged v21 zero-cost reward function: same sensors/geometry every
  call, RewardManager weight1, coefficient/scaled reward/dt penalty exactly0.
- Real episode-length randomization setter forwarding unchanged. Record full
  initialstate/obs/prefix/policy/RNG/config/contact audit, actualhorizon proof,
  rewardglobalclocks/raw/scaled/valid/dt, passivecontacts, complete scalars and Adam.
- Only function/task/experiment/loadrun/output identities may be explicitly projected
  to historical canonical format. The underlying reward function is unchanged.
  Keep requested seed and environment count explicit; reject other configuration drift.

## Compatibility, development and freeze

- Before ANY replay, pin v21 no_cost preflight/capacity model1 and full-model249,
  raw references and logs. Development model1 hashes are newly pinned historical
  evidence, NOT originally-v21-frozen checkpoint hashes. Final249 was frozen.
- Initialize trainingcache without scoring, prove exact v21 geometry110 bytes.
- Preflight in order51,52,53:256env×2iter,ramp metadata32. Capacity same order:
  4096env×2iter,ramp metadata4000. Six development runs835,584transitions, never selected.
- Seed51 in BOTH sizes must exactly reproduce historical no_cost initial proof,
  real horizons, full reward JSON/passive contact dict, checkpoint1 model/Adam/infos,
  and all33non-time scalar sequences. Five timing tags only are excluded from exact
  comparison: Perf/total_fps,Perf/collection time,Perf/learning_time,
  Train/mean_reward/time,Train/mean_episode_length/time. All38 tags remain finite/complete.
- Seeds52/53 have NO historical same-seed physical reference. Prove common policy/
  std/emptyAdam/fixed nonseed configuration and record distinct per-run state/RNG/
  horizon evidence. For this nonseed-only comparison, a temporary COPY may replace
  only env.seed and agent.seed with51; validate actual seed first and leave all
  persisted canonical/raw seed fields unchanged. This is not physical/RNG pairing.
  Do not claim equal physical states across seeds or environment counts.
- Capacity can serve each seed's MAIN initialization reference only by deriving a
  hash-pinned expected proof that changes the checked agent.max_iterations string
  exactly2→250. All other normalized fields, seed, initialstate/RNG and actual
  horizon must remain identical. Seed51 MAIN also matches original v21 MAIN proof.
- Freeze ALL training/evaluation definitions, references, ordered seeds/budgets,
  initialization,cache, development, derived MAIN expected proofs and comparison
  directions BEFORE seed51 main. No main work during an incomplete freeze.
- Seed51 FULL main is a positive control: require EXACT final249 model+Adam+infos,
  entire recorded reward/contact sequence and33non-time scalar sequences against
  frozen v21 no_cost main BEFORE starting seed52 or seed53. No tolerances/fallback
  to the old model if different. Preserve failure evidence and stop the branch.
- This proves the endpoint and recorded evidence, NOT every unrecorded simulator
  state, future-platform determinism or an independent fourth training realization.
- Fresh seed51 counts as one seed realization with historical v21 seed51, not two.
  All three final249 files must come from fresh v22 runs, even if51 bytes equal v21.
- New environment training total99,139,584=98,304,000main+835,584development.
  No historical trained model substitutes for these three new runs.

## Fixed fresh evaluation

- Controllers parent,seed51,seed52,seed53 and corresponding history_parent,
  history_seed51,history_seed52,history_seed53. History is unchanged v12 gate with
  frozen-v5 teacher and named expert. Parent evaluated once per condition, not
  duplicated/pseudoreplicated once per seed.
- Development51/24/35env:8runs280episodes; parent/seed51/history_parent/history_seed51
  reproduce corresponding v21 development physics/routing exactly. All8 initial/RNG
  proofs pair and4history flat branches equal. Initialization-only cache preparation.
- Fresh holdouts **115/75,116/76**; never retune/reuse old113/114 as new holdouts.
- Initialize480tiles without steps/scoring, then pin source/models/cache/ledger prefix.
- Primary16s175env/map/controller:16files2800episodes, each rough300+flat50.
- Separate64s10env/map/controller highest-level stones:16files160episodes,each20.
  Total32files2960FIRST episodes. Do not mix horizons or remove stalls/no-contact rows.
- Strict1/6tile13.1/53.1m plus nofirstfall/footprintlane/worldexit; same previousgates:
  roughone/six nonlower,falls/lane nonhigher,worldzeroincludingflat,flatfalls/lane
  nonhigher,flatmean episode speed nonlower,and atleastone strict roughimprovement.
  Secondary has noflatchecks. No defaultpromotion even if a gate passes.
- EXACT12 comparison directions: seed51/52/53 each vs parent and each corresponding
  history vs history_parent(6); seed52vs51,seed53vs51,seed53vs52 and corresponding
  history(6). Report EVERY pooled/permap/family/level/paired-six result and failedcheck.
- Report seed-specific values and descriptive min/max/ranges and parent deltas, not
  a duplicated-parent900episode pseudo-trial. Episodes/maps/history variants are NOT
  independent trainingreplicates. Three seeds do not establish significance/causality.

## Completion, verification and stop

One fixed three-seed diagnostic plus complete matrix only, regardless of outcome.
No reward/LR/model retuning, earliercheckpoint selection, map replacement or scored
retry. Any source/config/reference/cache/replay mismatch stops downstream execution.
Preserve failures; only separately recorded infrastructure development repair before
freeze can get a new attempt, never relaxed success thresholds.

CPU regression/available lint/type/AST/compile/whitespace before GPU. Independent
code review and raw arithmetic/ledger audit, plus separate CPU tensor/scalar checks.
Distinguish all training/dev/evaluation budgets; publish negative and mixed results.
One GPU-heavy job on cuda:1, leave unrelated processes untouched. No default policy,
newdependencies,commit,remoteGit/GitHub,or physicalrobot changes. TASK298's unrelated
older demo/publication scope stays separate from this bounded experiment.
