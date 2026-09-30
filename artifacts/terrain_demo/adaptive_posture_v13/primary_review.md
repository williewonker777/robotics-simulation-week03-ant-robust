# Independent v13 primary review

**Evidence integrity: PASS. Both predeclared primary improvement judgments: FAIL.**

This review covers only the 14 completed 16-second files / 2,450 first episodes.
The secondary 64-second batch is outside this review and is not pooled.

## Independent checks

- Recomputed strict float32-threshold one/six success, falls, lane/world exits,
  first-episode lengths, switch timing, float32 alpha/duty/target occupancy and
  censoring from per-environment arrays using the separate independent auditor,
  not the production summarizer's aggregates.
- Every file has 175 episodes in 35 family/level cells, five per cell. All seven
  controllers exactly match root/joint/full-observation initial hashes on each map.
- Verified 17 new frozen sources, 72 preserved legacy sources, four preserved
  legacy models, both new final249 models, 12 training-frozen sources, model/cache/
  evaluation-input links and all 14 successful primary command/log records.
- Training arms have identical initial policy tensors, state/observation hashes,
  RNG and normalized configuration; each warm start preserves the pinned learned
  depth weights with std reset to0.2. Final networks are finite and their frozen
  teachers exactly match original v5. Full training budget is32,768,000 per arm.
- Verified live training-cache240 tiles and holdout-cache480 tiles, including
  their complete pinned file inventories and no ambiguous expected tile seeds.
- Independently summed posture partitions, counts, validity/foot coverage and
  reward identities; flat speed retains falls and averages per-episode
  distance/(active steps/60), rather than pooling distance/time.

## Primary results

Each terrain column has n=300 per controller; flat falls have n=50.

| Controller | One | Six | Falls | Lane | Flat falls |
|---|---:|---:|---:|---:|---:|
| v5 | 257 | 132 | 33 | 1 | 3 |
| original | 254 | 135 | 40 | 4 | 1 |
| control | 262 | 139 | 37 | 0 | 2 |
| adaptive | 260 | 134 | 39 | 0 | 4 |
| history_original | 270 | 143 | 30 | 0 | 3 |
| history_control | 250 | 143 | 45 | 3 | 3 |
| history_adaptive | 262 | 142 | 34 | 4 | 3 |

World exits are zero, including flat ground.

- Adaptive actor versus continuation: one262→260, six139→134, falls37→39,
  flat falls2→4. **FAIL** despite faster flat motion.
- Adaptive hybrid versus continuation hybrid: one250→262, six143→142,
  falls45→34, lane3→4. **FAIL** because six-tile success decreases and lane exits
  increase. Mixed improvements do not satisfy the declared non-regression gate.

## Mechanism and coverage

Standalone flat speed rises10.57961→10.87023m/s (2.75%); visited valid
flat torso clearance falls0.461093→0.453785m (-7.31mm).
These observations support the narrower description "lower and faster on visited
flat states," not better overall traversal or increased safety: flat falls doubled.

In the depth-severity rough bin, torso clearance is0.475451→0.474214m and absolute
height-target error0.114557→0.115916m. The intended rough-ground body-raising/error
reduction mechanism is **not demonstrated** by these conditional means.
Valid body-posture sample coverage is99.9736%→99.9765%; known-motion coverage is100%
for both arms. Different visited states and episode durations still affect means.

All three history hybrids have exact per-flat-episode physical outcome, gate and
posture-array identity with fixed v5 on both maps. Their identical flat behavior
cannot be attributed to learning by the new rough actor.

## Limits

One starting actor/fine-tuning seed, two geometry/reset pairs and ideal depth.
Conditional posture means are sample-weighted visited-state diagnostics, not
matched-state causal estimates or statistical significance. The JSON's
posture-related active-step totals include flat episodes. Raw per-step depth is
not reconstructed; shared-cache hashes cannot rule out transient external
mutation. No universal-best, overall promotion or robot safety claim is approved.
