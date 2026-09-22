# v10 independent primary evidence review

**Evidence-integrity verdict: PASS**  
**Performance promotion verdict: FAIL — retain frozen v5**  
**Directional hypothesis verdict: PASS**

This review is based on an independent CPU audit of the frozen checkpoints,
saved configurations, command chronology, and all raw first-episode arrays. The
auditor imports none of the experiment runners or project summarizers. A passing
evidence audit establishes internal consistency and provenance; it does not turn
two held-out maps into independent terrain samples or establish statistical or
real-robot safety.

## Audit identity

- Auditor: `outputs/prior_v10_20260922/independent_audit.py`
- Auditor SHA-256: `3aeb89d8dfb14589deb698be78b7c07293f4b3f0900c02449f40861f4da0ed09`
- Result: `artifacts/terrain_demo/prior_v10/independent_audit.json`
- Result SHA-256: `2288464e1a3af5cd8d16c3825327b48d4bd8c7ffa9a67a2c35cbf1f73a4fa1b7`
- Result status: `pass`, mismatches: `0`
- Primary evidence: 14 distinct JSON files / 2,450 first episodes
- Scope: primary 16-second benchmark only. The separate 64-second diagnostic is
  not yet included in this review and cannot affect promotion.

## Provenance and protocol checks

- 53/53 v10 source, PLAN, and test hashes match the train-start and frozen maps.
- All 35 archived v9 source hashes and all 10 archived v9 model records still
  match. The original selected-v5 parent matches SHA-256
  `889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e`.
- Pinned RSL-RL 3.0.1 PPO provenance matches SHA-256
  `deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c`.
- Seven model files are path- and content-distinct: one zero-expanded v5
  reference and six final iteration-749 policies.
- Every final policy contains the expected typed mode/coefficient/reference-sigma
  buffers. Every frozen teacher actor tensor is bit-identical to the selected v5
  actor. The reference student/value networks reproduce the v5 base columns and
  have zero added columns.
- All six saved configurations cover exactly the declared two modes x three
  seed/geometry pairs. PPO, physics, sensing, observation, reward, termination,
  curriculum, robot, scanner, terrain and simulation settings are unchanged from
  the archived v9 targets arm except for the declared prior treatment and seeds.
- Free and anchored initial checkpoints are bit-identical for each training seed
  except for the two treatment buffers; optimizer state is empty and layout is
  identical.
- Command evidence contains exactly 13 allowed preparation/training commands and
  14 allowed primary evaluations, all with return code zero. All training ended
  before the freeze at `2026-09-22T07:03:57.926864+00:00`; all held-out
  evaluations began after the freeze. No extra training/evaluation command label
  was admitted.
- Every raw JSON has the declared task, checkpoint hash/path, geometry/reset pair,
  175 completed first episodes, 88D policy observations, 960-step maximum,
  60 Hz control, 7 families x 5 levels x 5 replicas, strict boolean boundary and
  terminal flags, consistent termination reasons, and finite/in-range episode
  evidence.
- Strict one-/six-tile outcomes were independently recomputed from float32
  13.1 m / 53.1 m thresholds and required no posture termination, lane exit, or
  world exit. Every stored total and every family/level cell matched.

## Independently recomputed primary totals

Terrain denominators exclude the 25 flat episodes per evaluation. Frozen v5 is
sampled once per condition rather than triplicated.

| Policy | One tile | Six tiles | Falls | Lane exits | World exits | Flat falls |
|---|---:|---:|---:|---:|---:|---:|
| frozen v5 | 260/300 (86.7%) | 132/300 (44.0%) | 30/300 (10.0%) | 0/300 (0.0%) | 0 | 4/50 (8.0%) |
| free, lambda=0 | 744/900 (82.7%) | 422/900 (46.9%) | 96/900 (10.7%) | 61/900 (6.8%) | 0 | 13/150 (8.7%) |
| anchored, lambda=.02 | 802/900 (89.1%) | 454/900 (50.4%) | 73/900 (8.1%) | 15/900 (1.7%) | 0 | 3/150 (2.0%) |

Against the matched free arm, anchoring produced +58 one-tile successes, +32
six-tile successes, 23 fewer falls, and 46 fewer lane exits at the same aggregate
denominator. These are descriptive counts on two maps, not uncertainty-adjusted
effect estimates.

## Paired training-seed evidence

| Seed | Policy | One | Six | Falls | Lane exits |
|---|---|---:|---:|---:|---:|
| 42 | free | 247/300 | 139/300 | 32/300 | 22/300 |
| 42 | anchored | 269/300 | 152/300 | 27/300 | 2/300 |
| 43 | free | 268/300 | 141/300 | 30/300 | 3/300 |
| 43 | anchored | 267/300 | 143/300 | 23/300 | 4/300 |
| 44 | free | 229/300 | 142/300 | 34/300 | 36/300 |
| 44 | anchored | 266/300 | 159/300 | 23/300 | 9/300 |

Falls improved in all three paired seeds. Falls and lane exits jointly improved in
seeds 42 and 44; seed 43 had one more anchored lane exit. Thus the predeclared
2/3 joint-safety requirement passes without hiding the dissenting seed.

## Difficult terrain checks

Across all six terrain families at difficulty 1.0:

| Policy | One | Six | Falls | Lane exits |
|---|---:|---:|---:|---:|
| frozen v5 | 42/60 | 1/60 | 12/60 | 0/60 |
| free | 151/180 | 24/180 | 26/180 | 3/180 |
| anchored | 150/180 | 13/180 | 21/180 | 5/180 |

For stepping stones specifically:

| Difficulty | Policy | One | Six | Falls | Lane exits |
|---|---|---:|---:|---:|---:|
| 0.8 | frozen v5 | 8/10 | 0/10 | 0/10 | 0/10 |
| 0.8 | free | 23/30 | 0/30 | 6/30 | 2/30 |
| 0.8 | anchored | 24/30 | 0/30 | 1/30 | 1/30 |
| 1.0 | frozen v5 | 4/10 | 0/10 | 1/10 | 0/10 |
| 1.0 | free | 24/30 | 0/30 | 4/30 | 2/30 |
| 1.0 | anchored | 20/30 | 0/30 | 3/30 | 3/30 |

No policy completed six tiles on 0.8 or 1.0 stepping stones within 16 seconds.
Anchoring substantially reduced 0.8-stone falls relative to the free arm, but its
1.0-stone one-tile count and lane count were worse than free. These retained
failures are why the later 64-second stone-only diagnostic is informative but
must remain non-gating.

## Gate verdicts

### Strict promotion: FAIL

Ten of eleven checks pass. The sole failure is:

- `lane_vs_frozen_v5`: anchored 15/900 is worse than frozen v5 0/300.

All other throughput, fall, flat-fall, world-exit, free-arm comparison, and paired
fall-majority checks pass. The gate is conjunctive, so v10 must not replace v5.

### Separate directional hypothesis: PASS

Anchored has strictly fewer aggregate falls and lane exits than free, joint fall
and lane improvement in 2/3 paired seeds, at least the frozen-v5 six-tile rate,
and zero world exits. This supports the limited hypothesis that the training-only
v5 mean prior reduces the safety regression of the unanchored 88D learner while
retaining terrain-informed traversal. It does not satisfy the stricter baseline
promotion requirement.

## Recommendation

Retain the frozen v5 policy as the recommended/default policy. Preserve all v10
models and raw evidence as an experimental result. Run the already predeclared
64-second stepping-stones diagnostic without training, selection, threshold
changes, or promotion use; audit it separately with `--horizon`.
