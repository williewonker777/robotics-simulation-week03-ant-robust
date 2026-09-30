# v11: depth-gated frozen v5/v10 experiment

## User request and stop condition
Test switching between the existing v5 on ordinary terrain and v10 anchored on
scan-detected challenging terrain. Complete implementation, regression/simulator
checks, frozen fresh-map comparisons, first-episode numerical audit, and a concise
result (positive or negative). No policy training, model replacement, or remote Git.

## Invariants
- Preserve all existing source/model hashes: add isolated v11 files only, plus docs.
- Same v10 Eval scene, unchanged 88D observation, 8 effort actions, physics,
  terrain geometry/difficulty, rewards, termination, and 13.1/53.1m strict metrics.
- Read only the causal yaw-aligned825-ray scan for routing. Existing analytic
  flat-plane ray completion is the same sensor model used by v10. No family/level,
  terrain generator maps, future trajectory or evaluation labels in selector.
- v5 = exact frozen teacher (60D prefix) inside each anchored v10 checkpoint;
  compare its tensors with the independently archived recommended v5 checkpoint.
- v10 anchored seeds42/43/44 are all retained; no best-seed selection.
- A stateful selector uses separate enter/exit thresholds, minimum dwell and
  bounded action crossfade. This is a tested heuristic, not a stability guarantee.
- Mask invalid/clipped scan rays as unknown; do not classify missing scan as safe
  flat. Hold current mode on inadequate coverage and report uncertainty. Reset
  controller state on episode reset, not on terrain portal wrapping.

## Development and fixed comparisons
Development geometry/reset51/24 only, distinct from holdouts. Up to three
predeclared threshold configurations on seed42; choose by strict terrain one-tile
success, then fewer falls/lane exits, then six-tile successes. Record every trial.
A direct hard-switch ablation may be used only as a separate diagnostic, not for
unreported selection. Parameter details will be fixed before development begins.

After development, freeze selected config, all source/plan hashes, all3 original
anchored checkpoint hashes and v5 hash before any holdout launches.
Fresh holdouts geometry/reset68/42 and69/43 (not previously used for geometry).
Compare frozen_v5 once/map, v10_anchored and hybrid each seed42/43/44/map:
- Primary175env, 16s,7families x5levels x5episodes =14JSON/2450firstepisodes.
- Separate64s stone1.0 diagnostic10env/map =14JSON/140firstepisodes. No pooling
  with primary, no retuning afterward. All comparisons share same scene/seed/batch.

## Extra evidence
Per-episode v10 duty, switch count/times, uncertain coverage, action disagreement,
action jump, falls within0.5s of switch; family and level only for reporting.
Every failure counted; no post-auto-reset evidence admitted. Measure baseline
zero/full duty. Use development fixed-placement scans to verify flat and stone
features, anticipation, both switch directions, hysteresis, and reset behavior.

Promotion: primary hybrid one/six rates >= both fixed baselines, terrain falls
and lane exits <=v5, world0, flat falls <=v5; separate64s stone six>v5 and falls
<=v5. Report every failed criterion and perseed tradeoffs; failed gate leavesv5.
Two new maps do not establish broad generalization; hundreds of environments are
not hundreds of independent maps. Ideal depth is not real RGB-D.

## Implementation and validation
New pure gate/controller module+tests; new isolated evaluator reuses
HorizonEpisodeTracker and existing prior registration, accepts16/64s; new runner,
summary/audit, docs. Tests cover exact v5/v10 action paths, sensor validity/geometry,
threshold hysteresis/dwell/fade/reset, threshold edge cases, first-episode switching
telemetry/terminal handling, hashes and non-overwrite guards. Fresh full tests,
compile, focusedGPUsmoke, independent code/evidence review, source/model hashes.
One GPU job at a time on currently idle GPU1; do not stop other workloads.

## Fixed development presets (declared before simulator trials)
- cautious: height q95-q05 enter0.12m, neighbor |dh| q95 enter0.08m.
- balanced: enter0.18m /0.12m.
- selective: enter0.27m /0.18m.
All use exit thresholds x0.6, ray AND eligible edge coverage>=0.90,
2consecutive entry frames, min dwell0.5s, continuous clear0.3s, linear alpha fade0.15s.
Enter ROI x[0,2],|y|<=1m; retentionROI x[-1.2,2],|y|<=1.1m.
Unknown clears confirmation; ongoing fade continues toward unchanged request.
Tie order after metrics: cautious, balanced, selective. Do not change presets after
seeing development metrics. Initial smoke35env may repair implementation bugs,
not select thresholds. Comparison video: predeclared seed42/geometry68/reset42,
stones1.0 10env, followenv0,16s; v5/v10/hybrid all retained, reset captions visible.

Architect advisory WATCH accepted with repairs: rename tracker snapshot/survival
fields for16s; pre-step firstepisode masks; terminal saved state; exact endpoints;
no oracle labels; invalid edge coverage; fixed selection andfreshholdouts.
Thresholds above are the leader's bounded predeclaration, not empirically validated.
