# v19 — Frozen-policy common-map rebaseline

Status: preregistered before new-map scoring, 2026-09-28.

## Question and scope

Does the accumulated v16 control policy, alone or with the unchanged v12 depth-history
gate, improve sustained traversal relative to the recommended v5 and original
v5/v10 history hybrid on the SAME unseen maps? Earlier v13–v18 comparisons used
different maps and cannot establish a cross-version ranking. This is a bounded,
evaluation-only checkpoint comparison, NOT a new training method or a causal
ablation of any single reward. New training transitions: **0**. Do not promote a
default, tune thresholds, select checkpoints, or publish remotely in this study.

## Fixed controllers

1. `v5`: frozen v5 portal-rehearsal round4 actor, evaluated through the byte-identical
   teacher inside the original v10 checkpoint; only its first 60 observations act.
2. `history_original`: unchanged v12 history gate combining that v5 actor and
   original v10 anchored final749, 88D expert input.
3. `v16_control`: frozen v16 no-slip-cost final249, 91D expert input, no switching.
4. `history_control`: unchanged history gate combining v5 and the same v16 model.

All use one unchanged v16 evaluation scene, physics, sensors, 91D observation
production, depth scan refresh order, time step and first-episode tracker. Validate
teacher tensors and checkpoint SHA; do not assume that a model label proves identity.
Action dimensions remain 8. Contact/posture telemetry is passive for every controller.

## Fixed data and run order

- Development only: geometry51/reset24, 35 environments, 16s, all four controllers.
  Compare v16 and original-history results with the stored v18 development results.
- New paired geometry/reset: **107/68, 108/69, 109/70**. No retuning on them.
- Primary: 175 environments/map, 16s, all 7 families × 5 levels × 5 repetitions.
  Per controller: rough450 + flat75 first episodes. Total primary: 2,100 episodes.
- Secondary: 10 environments/map, 64s, stepping stones level1.0 only.
  Per controller30 first episodes, secondary total120, kept separate from primary.
- Total:24 raw result files /2,220 scored first episodes, plus unscored preparation
  and development. One GPU-heavy job at a time on the idle GPU; preserve robot jobs.
- Prepare all 720 terrain tiles WITHOUT stepping before scoring. Pin cache file
  hashes, source/model/plan hashes, smoke evidence, pre-scoring ledger prefix.
- Score each map/scenario/controller once, controller-major declared order. Every
  initial full observation, 88D prefix, root/joints and CPU/CUDA RNG must match within
  a map/scenario. Require exact flat physical/routing arrays across v5 and hybrids.
- A validity failure stops scoring; keep failed logs/results. Do not silently rerun,
  replace a map, or use an outcome-based retry. Any new attempt needs a recorded
  reason and must not mix evidence across invalid attempts.

## Measures and decision rules

Reuse the strict definitions: one tile >=13.1m, six tiles >=53.1m, with no fall,
whole-footprint lane exit, or world exit over the FIRST episode. Report falls,
lane/world exits, completed first episodes, flat episode speed, family/level counts,
per-map results and paired six-tile gain/loss counts. Do not pool flat with rough,
pool16s/64s, or call survival or short progress a six-tile success.

Two predeclared comparisons: `v16_control` vs `v5`, and `history_control` vs
`history_original`. A descriptive primary gate requires no lower rough one/six
counts, no higher rough falls/lane exits, zero world exits including flat, no higher
flat falls/lane exits, flat mean episode speed >=reference, and at least one strict
rough improvement. Secondary uses the rough-only conditions and strict improvement.
Report pooled and per-map gates, and failures even when another metric improves.
These are descriptive gates, NOT statistical significance or automatic promotion.
Only three maps and fixed single-seed historical models do not establish general
algorithm superiority; no episode-independent confidence interval is claimed.

## Verification and stop condition

Regression tests before scoring; development parity and initial pairing; finite
actions/raw metrics; frozen sources/checkpoints/caches before/after each run;
independent raw-array audit after all24 files. Run full CPU suite, compileall,
available static checks and diff check. Record missing tools rather than install
dependencies. Preserve all v0–v18 files and current dirty worktree. Stop after the
bounded report, reproducible commands, hashes and limitations are recorded, even
if every gate fails. Existing unrelated demo/publication work stays out of scope.

## Reference boundary

Prior experiments cite Miki2022, Aractingi2023 and CaT2024; their particular Ant
adaptations and negative results remain in the versioned documents. This rebaseline
introduces none of their mechanisms and is not a reproduction claim. Primary-source
verification and RL evaluation guidance are recorded separately with the final report.

## Development validity repair, before any new-map scoring

The first development attempt encountered a cold `/tmp` geometry51 cache after a
machine restart: v5 generated240 tiles, then history_original loaded them. Root,
joints and CPU/CUDA RNG matched but scan observations differed. The second
history_original run exactly matched the historical v18 development reference.
The pairing guard stopped the attempt after two runs; neither result is selected
as scored study evidence. Both raw JSONs, logs and ledger entries are preserved.

Attempt02 adds explicit initialization-only `prepare_development` before all four
development runs, pins geometry51 cache hashes through them and checks source hashes
at start/end/freeze. No observation tolerance, model/parameter adjustment or new-map
retry is permitted. New maps107–109 remain unscored at this amendment. This repairs
cache-state comparability, the same prerequisite already specified for the holdouts.
