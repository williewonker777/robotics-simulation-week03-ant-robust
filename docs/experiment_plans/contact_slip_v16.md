# v16 — terrain-contact-conditioned stance-foot slip penalty

Continue TASK298 with one new paper-informed method after v15 directional cost failed
all four promotion comparisons. Direct scoped implementation with independent design,
code, training, and final raw-data review; no formal OMX goal/workflow mode. Stop
when this one preregistered experiment is trained and audited, even if it fails.
No coefficient/seed/checkpoint/map/score gate tuning after seeing fresh holdouts.

## Grounding and distinction
Miki et al. 2022 Supplement S7 uses a contact-set squared foot velocity slip
cost. Aractingi et al. 2023 also penalizes horizontal foot velocity during
contact. Our adaptation uses PhysX terrain-filtered normal-force threshold to
identify stance, world-XY kinematic foot-tip speed, and a bounded negative
cost. This is neither a friction-force measurement nor an exact contact-point
velocity, paper coefficient, paper reward mixture, or real robot proof.
Existing v13 foot clearance is scan/kinematic, not true contact detection.
V15 penalized torso lateral/yaw motion and did not improve sustained traversal;
v16 does NOT carry over that directional reward. A second simultaneous swing
clearance reward, learned expert gate, and terrain curriculum are deferred to
avoid confounding the one contact-slip term, not deemed ineffective.

## Baseline/selection and sensors
Source: frozen v15 **control** final249
`artifacts/terrain_demo/directional_stability_v15/runs/control/model_249.pt`,
SHA `4aa169eba098e7ab69e82958485879db2f34910c6cb4a6d0367f33efef3bc3ec`.
This source was chosen using v15's now-seen 96/97 maps as a stronger strict-six
baseline than v15 stable. Therefore those maps cannot be v16 fresh holdouts.
Preserve all 91D actor, critic, 60D v5 teacher, learned command U, conditioned
mode/schema and old rewards including adaptive_posture1. V15 directional reward
is NOT inherited. Reset only std0.2, fresh Adam1e-4/empty state, iteration0.

Both new arms use the SAME sensor-enabled config: robot asset spawner's
`activate_contact_sensors=True`, and four independent ContactSensorCfg objects,
one per named foot, filtering BOTH `/World/ground/terrain/mesh` and the live
collision child `/World/flatPlane/GroundPlane/CollisionPlane`, update_period0,
no air-time tracking or actor observation. The original foot name/order IDs are
checked; force_matrix_w must be [N,1,2,3] per foot and finite. Validate at live
USD stage that BOTH collision prims exist; combine their normal forces per foot.
The exact flat path was discovered in development-only live USD inspection
(`outputs/contact_slip_v16_20260923/collision_paths.json`) after an incorrect
flat filter produced a preserved PhysX failure log; no holdout seen.
No unfiltered-wrench or
incoming-joint-wrench fallback. Confirm scene config equals original v15-control
training task after removing exactly these four sensors, setting the reporter
flag back to false, and removing the one new reward. Same eval physics except
reporter+sensors, and unchanged 91D policy obs; all scored controllers use this
same v16 eval task. A development v15-control reference on geometry51/reset24
must reproduce the frozen v15 reference initial state/91D/prefix/RNG and raw
physical outcomes; if reporter changes them, record failure and resolve before
holdouts, not silently assert parity.

## One fixed reward
Let F_i be the combined **filtered mesh-or-plane normal contact force** from foot i; let
c_i = 1(||F_i|| > 2 N). Let p_i be the established world-frame distal foot tip,
v_i = link_linear_velocity_i + link_angular_velocity_i cross rotated_tip_offset_i.
Reward per environment before RewardManager weight/dt:

    r_slip = -(1/4) * sum_i c_i * min(||v_i,xy||² / (1 m/s)², 1)

Thus r_slip in [-1,0], control weight0, slip weight1. Four-foot denominator
preserves a consistent per-robot scale; no positive no-contact or standing
bonus, no arbitrary gait phase. Sensor missing/shape mismatch/terrain target
missing: hard failure. Nonfinite row: zero reward+invalid diagnostic; fail
preflight if invalid fraction exceeds 1%. Zero-filtered-foot-contact row:
zero reward but reported separately, NEVER equated to whole-body airborne
(underthreshold/torso contact are possible), never labeled good slip performance.
RewardManager applies weight*dt exactly once. This is a **contact-conditioned
foot-tip-speed proxy**, not direct tangential friction or true sole slip.
Normal force can classify side contacts with the filtered terrain mesh; report
that limitation. Threshold/scale fixed before full training; only sensor API
repair from failed development may change them, with failed evidence preserved.

## Equal-budget training
Control vs slip differ ONLY new reward weight0/1; contact reporter/sensors
are on in BOTH. Seed48, training geometry98, prewarmed byte-pinned terrain.
Both arms: 250iterations x4096env x32steps=32,768,000 transitions; total
65,536,000. Only final249; no seed, coefficient, intermediate checkpoint,
early-stop, or retry selection. Before learn: exact full91D/root/joints/prefix,
policy tensor, CPU+CUDA RNG, normalized saved YAML except treatment/output;
actual actor/critic/teacher outputs must equal source on same device; learned U
must NOT be zeroed. Sensor shape/finite/coverage proven for 64env2iter BOTH,
then4096env2iter BOTH before training freeze. Detect all-zero broken sensors
and reject calibration with implausible [0.02,0.98] terrain-contact fraction
in development rollouts. The reward manager skips weight-zero terms, so
capacity/preflight explicitly query the same passive contact geometry in BOTH
arms. Verify mesh-family AND flat-family contact coverage per foot; all-zero
on either family fails. Sensor update period zero is not proof of a fresh solved
contact sample. Training full 250-step logs/step sequence/all scalar
values/model+Adam tensors finite; teacher identity verified. Freeze new sources,
all prior123 sources/10 reference models and cache before training.

## Fresh matched evaluation
New holdouts geometry99/reset62, 100/reset63. Six controllers:
`v15_control` (frozen source reference), `control`, `slip`,
`history_original` (v5+original v10), `history_control`, `history_slip`.
Four model keys original, v15_control, control, slip. Original history expert88D;
all other experts conditioned91D; teacher60D. Reuse unchanged v12 depth-history
gate and original physical score formulas, with a NEW truthful v16 result schema
and explicit score projection after v16 metadata/physics/sensor validation.
No edit or impersonation of prior v11–v15 eval or summary files.

Primary16s mixed175env, six controllers x two maps=12 files/2100 first
episodes. Secondary64s hardest stepping_stones level4, 10env,
12 files/120 first episodes. Total24 files/2220, same13.1m/53.1m,
dt1/60, margin1.1m, eight/sixteen-second snapshots, fall/lane/world metrics.
All terrain families/levels and flat separate. Prewarm99/100 BEFORE scoring;
pin source/model/cache/plan/SHA and preholdout command-ledger prefix in each
command/result, pre/post audit, exact full initial pairing. Passive **pre-action**
first-episode valid filtered-foot-contact fraction, no-filtered-foot-contact
fraction, contacted foot-tip speed, bounded cost, speed-cap saturation and
invalid counts, plus unchanged posture telemetry. Mask initial after-reset and
immediately post-portal-wrap samples that have no co-timed solved contact; count
them as invalid diagnostics, never as zero-contact gait. Label sampled contact
as filtered foot contact, not whole-body airborne.
Contact metrics are conditional visited-state observations, not proof of
matched-state causal effect. No real camera/robot safety claim.

## Fixed decision criteria (same as v15)
Predeclared causal contrasts: `slip` vs equal-budget `control`, and
`history_slip` vs `history_control`. v15_control and history_original are
**descriptive references**, not post-result substitute comparators.
Primary actor: rough one/six not lower, falls/lane not higher, world0 including
flat, flat falls not higher and flat mean per-episode speed strictly higher.
Primary hybrid: rough common nonregression, flat nonregression and exact per-env
fixed-v5 raw flat identity, and at least one strict terrain improvement.
Separate secondary: one/six/falls/lane/world nonregression and at least one
strict improvement; never overrides primary failure. World exit0 inclflat.
A contact metric alone cannot override passage/fall criteria. One training
seed/two maps do not establish universal/statistical superiority.

## Verification / stop
CPU reward math (no contacts, one/multiple contacts, speed saturation,
world rotation/mirror, finite invalid, force threshold, bounds); sensor adapter
body ordering/filter shape, reset/portal, stationary and flight diagnostics;
full config/observation parity; shared checkpoint/Adam negative tests;
metadata/provenance/no-overwrite tests. 64env2iter and4096env2iter paired
GPU smokes on cuda:1 one job at a time using shared gpu.lock. Full CPU,
compile/AST/diff checks and independent review before freezing training;
final model smokes and independent training audit before freezing evaluation;
independent 24-file raw audit after scoring. Preserve failed development
evidence and old artifacts. No dependencies, default-controller changes,
commit, or remote Git without fresh authorization. Finish/report even if FAIL.

Primary links and adaptation caveats: artifacts/terrain_demo/contact_slip_v16/research.md.
