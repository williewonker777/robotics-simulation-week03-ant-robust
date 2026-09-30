# v23 — Paired 16s/64s windows from the same first episode

Preregistered before v23 GPU work, 2026-09-28. Direct bounded evaluation extension,
not a new policy/training architecture or a hyperparameter search. Preserve all
frozen v0–v22 source/model bytes, baseline defaults and existing dirty work.

## Question, evidence and alternatives

V22 primary used16s/mixed/175env while secondary used64s/highest-stones/10env.
Different horizons, scenario populations and starts prevent attributing their
contrast to duration alone. We will observe the SAME episode at16s and64s on
fresh maps, without additional learning, policy selection or gate retuning.

[Pardo et al., ICML2018, §§2–3](https://proceedings.mlr.press/v80/pardo18a/pardo18a.pdf)
distinguishes task deadlines from externally imposed cutoffs; its photo-finish
example illustrates that short-horizon performance need not persist. It concerns
learning/time observability, NOT a proof of our paired-evaluation estimator, a
v22 timeout bug, or authority to change frozen policies. [Agarwal et al., NeurIPS2021](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)
supports separating training runs from evaluation episodes and uncertainty-aware
interpretation. Our exact window choices and gates are engineering choices.

Alternative: a fixed global PPO LR1e-4→1e-5 intervention is plausible from PPO and
on-policy implementation studies, but changes learning before resolving this
measurement confound. Retaining learned initial std is a separate intervention.
Both are deferred; no new training, optimizer/LR/std/reward modification in v23.

## Frozen population and physical execution

- Four unchanged models from v22: parent(v16control249), seed51/52/53 final249.
  Every corresponding v12 history combination, total8controllers. Use ALL seeds;
  no seed52-only followup or checkpoint selection. Historical51/full replay remain
  one seed realization, not a fourth training replicate.
- Copy no model bytes and perform0new training/evaluation-learning transitions.
  Rehash v22 manifests,227legacy source files and22checkpoint paths, plus v5 teacher.
  Training ancestry is provenance, not a new or triplicated training budget.
- Fresh geometry/reset **117/77,118/78**. No prior scored use found locally.
- Each controller/map has175environments:7families×5levels×5starts, unchanged
  geometry, physics, observations91D/actions8D, teacher/history, dt1/60 and controls.
- Physical environment timeout is64s (3840control steps). Compute16s(960steps)
  and64s outcome windows from this ONE trajectory. Do NOT reset environment,
  policy history, RNG or physical counters at16s. Durations mean simulated time,
  not host wall time. No time observation or new task/reward is introduced.
- One bundle JSON per physical rollout:16bundles,32window records,
  **2,800 first episodes /5,600 repeated window observations**, NOT5,600independent
  episodes. Each controller/window has300rough+50flat, parent counted once.
  Highest-stones subset is10/controller/window, NOT the old20-episode secondary.

## Passive window instrumentation and scoring

- Retain frozen v22 action/sensor/environment-step/history-reset ordering. A full64s
  HorizonEpisodeTracker defines active first episodes. An additional16s tracker
  receives the same saved post-step evidence, with a VIRTUAL timeout at960.
  This timeout affects scoring only and must never reach env.step or gate.reset.
- If physical done and virtual cutoff coincide, use saved pre-auto-reset last_*
  terminal evidence and the real terminated flag. Otherwise use current position/
  lane/world as virtual terminal evidence. Early first episodes stay terminated;
  no reset episode substitutes for them. All original rows remain in denominators.
- Capture immutable JSON-safe raw outcomes and routing/contact/posture telemetry at
  each cutoff. An all-first-episodes-finished early exit records completed evidence
  for both windows, with early-capture reason; no surviving rows may be fabricated.
- Preserve exact existing STRICT scoring: window-terminal forward_distance>=13.1/
  53.1 AND no physical first-episode termination/footprint-lane/world violation by
  that window. It is final distance, NOT maximum-distance or mere threshold hit.
  A longer-window loss may reflect later failure/exit OR backward distance loss.
- Existing auxiliary distance snapshots remain8s for16s records,16s for64s records,
  correctly named distance_at_snapshot_m with explicit snapshot_seconds. They are
  NOT the two primary scoring endpoints. Old v22 outputs already name them explicitly.
- Keep real physical completion vs scoring censoring explicit per row. A16s
  virtual timeout is not a fall and event-free at64s is not indefinite safety.
- Snapshotting must preserve environment root/joint/episode counter/observation
  hashes, torch CPU/CUDA RNG hashes, policy mode and unchanged active history.
  Unit tests and real development controls verify passive behavior.

## Development and source/model/cache freeze

Before any scoring, pin old v22 development8controller files and their hashes.
Create the new source set, CPU regressions and independent implementation review
before GPU. No old source/model rewriting or unknown CLI overrides.

Development geometry51/reset24/35env:
1. Initialization-only cache preparation,0scored episodes.
2. Uninstrumented64s/mixed reference runs for parent and history_parent. This is
   the new evaluator with16s recording DISABLED and the same full64s legacy loop.
   This development-only mode is forbidden for holdouts.
3. Paired-window64s runs for all8controllers, fixed listed order. Their16s raw
   records must exactly match ALL35 existing PARITY_FIELDS of corresponding v22
   16s development files, including condition/snapshot semantics and telemetry.
   All8initial state/prefix/RNG proofs pair; each window's4history flat branches match.
4. Full64s parent/history_parent records exactly match uninstrumented64s references
   in all35 fields. No tolerance expansion. Early virtual snapshot must not change
   later trajectory evidence; record before/after passivity proofs.

Development totals10scored physical runs×35=350 executed first episodes
(8paired runs280episodes +2passivity-control replays70),630window observations.
These are development/replay executions, not independent training replicates.

Any parity/source/model/cache/passivity mismatch stops downstream execution.
Preserve failure evidence; only a separately reasoned infrastructure development
repair BEFORE freeze may use a new attempt; never relax success thresholds.
Freeze every new experimental definition, all model/reference hashes, comparison
order, budgets and development proofs before holdouts. Initialize fresh480tiles
without scoring, then pin cache and command-ledger prefix. One GPU-heavy jobcuda:1.

## Fixed comparisons and paired-duration diagnostics

- At EACH window, report ALL12 existing v22 directed controller comparisons:
  seed51/52/53vsparent; correspondinghistoryvs history_parent;52vs51,53vs51,53vs52
  andhistory equivalents. Primary16s;64s descriptive endurance cannot override it.
- Samegate BOTHwindows: rough1/6nonlower,falls/lane nonhigher,world0includingflat,
  flatfalls/lane nonhigher,flatmean episode speed nonlower,atleastone strict rough
  improvement. No automatic default promotion even if any or all gates pass.
- Report pooled/permap/family/level/family-level metrics, paired controller-six
  gains/losses and all failedchecks; rough-only subgroupgates noflatchecks.
- For eachcontroller and subgroup, pair SAME rows16→64. Report four-cell strictsix
  contingency (neither,late gain,late loss,both), and late-loss reason flags:
  newphysicaltermination,newlaneexit,newworldexit,finaldistancebelow53.1.
  Flags may overlap; they are not a causal attribution or exclusive partition.
  Include interval first-six-hit counts, distinguishing threshold hit from strict
  terminal success. Windows are dependent repeated observations of the same rows.
- Report perseed min/max/ranges and matching-parent deltas, never aggregate the
  parent3times or treat maps/history/windows as independent training repetitions.
- Do not compare new 64s mixed population directly to v22's20higheststones counts
  as if matched; within-v23 paired windows are the identified diagnostic.

## Verification, budget and stop

Unit tests: virtual cutoff/realdone collision, reset contamination, earlycomplete,
noenv/gate reset at16s, immutable snapshots/passivity/RNG, two-window bounds and
identities, strict final-vs-max distance, censoring, late gains/losses/overlap,
all12comparisons, seedranges/one-parent denominator, missing/tampered proofs,
forbiddenholdout uninstrumented mode, freeze/cache/ledger/retry integrity.

Expected GPU commands29 =1development preparation +2uninstrumented controls
+8paireddevelopment +2holdout preparation +16scored physical rollouts.
New training0;holdout16bundles/2,800firstepisodes/5,600windowobservations. No extra
training/checkpointselection/mapreplacement/scoredretry. Independent stdlib raw
arithmetic/provenance audit must distinguish window observations from episodes;
independent reviewer checks raw and report plus snapshot passivity and preserved
source/model bytes. Run full CPU/available lint,type,AST,compile,diff checks.

Stop after this single fixed study and complete reporting irrespective of result.
No significance/causal-safety/general robustness claim from3continuationseeds on
oneparent/terrain and2newmaps. No default/dependency/commit/remoteGit/GitHub/robot
changes. TASK298's older separate demo/publication scope remains separate.
