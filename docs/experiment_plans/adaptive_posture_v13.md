# v13 — depth-conditioned body posture and swing clearance

## User request and execution lane
Add ground-distance-dependent training shaping, retrain and test. User clarified:
raise the body for obstacles; lower it on clear ground to walk faster. Scoped direct
execution with bounded design/code review; no OMX runtime/goal workflow activated.
Existing 69 v11/v12 frozen sources, 3 video helper sources and all archived models
are immutable. New opt-in v13 files only. No new dependencies, remote Git or robot use.

## Hypothesis and alternatives
- Primary deliverable: a trained policy and matched evidence, NOT a guaranteed gain.
- Choose reward-only fine-tuning of the existing 88D v10 actor, plus a matched
  continuation control. Both consume the same unchanged depth-derived foothold
  observations. Reward can use full current scan, which is training-only privileged
  supervision; do not claim the actor sees all825 rays or an explicit height command.
- Alternative explicit body-height command/input expansion would change policy
  architecture and confound this reward-only test; defer it.
- Raising only feet misses the user's clarified body-posture intent. Changing just
  the selector cannot train posture and is not the chosen treatment.
- Keep existing recovery/velocity/energy/fall rewards in BOTH arms. New bounded
  terms supplement, not silently replace, the existing stone-only low-height cost.

## Proposed reward (engineering hypothesis, review before implementation)
Current ideal825-ray grid in torso-relative yaw frame, same sensor/config as v10.
No terrain family labels, global map, future state or evaluation outcomes in shaping.

1. Local ground reference: highest valid scan point under torso (x/y +-0.3m).
   Front profile ROI x0.3..1.5m,y+-0.6m. Require >=90% finite/unclipped coverage
   in both ROIs; unknown scan is NOT flat and earns neither new penalty nor bonus.
   Obstacle measure=max(front_max,local_ground)-min(front_min,local_ground),
   so a uniform downstep/gap mouth is NOT clear ground.
   Severity=clamp((measure-0.03)/0.12,0,1). All units metres.
2. Target torso clearance=0.44+0.14*severity (0.44m clear / up to0.58m rough).
   Existing fall threshold remains0.31m. Bounded two-sided body error
   min(((actual_clearance-target)/0.12)^2,1), not a perpetual high-body bonus.
3. Current distal capsule bottoms (radius0.08m), not ankle origins. Support is the
   maximum ray height in a +-0.15m x/y patch around each current tip. Require the
   entire patch inside the825-ray grid and all included rays valid (>=9 samples);
   unknown/out-of-grid feet excluded. Desired swing
   clearance0.04+0.08*severity m; bounded deficit charged only to forward swing
   relative to body (same kinematic velocity convention as existing stone reward).
   Swing factor=clamp(relative_forward_speed,0,2)/2, additionally zero when torso
   signed forward speed<=0. Foot deficit=min(clamp((target-clearance)/0.12,0,1)^2,1);
   average over four feet, invalid feet contribute0. This is a kinematic swing
   proxy, NOT contact sensing or a proof of stance exclusion.
   Combined cost=body_cost+0.25*foot_cost.
4. One new reward term = -combined_cost+0.5*flat_speed_bonus:
   flat_speed_bonus=(1-severity)*exp(-body_error_squared)*
   clamp(signed_forward_speed,0,6)/6*clamp(upright_dot,0,1).
   Positive bonus additionally requires upright_dot>=cos(1.2rad).
   Signed forward speed is root-link world linear velocity projected onto the
   existing lane-target direction, NOT norm/body-x. Body error is unbounded
   ((clearance-target)/0.12)^2 inside the Gaussian, bounded at1 for cost.
   No positive bonus while stationary/backward/invalid/overturned. Existing
   progress remains unchanged and its terrain3m/s cap remains unchanged.
   Control new term weight0; treatment weight1. Both share all other settings.
5. Recompute geometry once for the single new reward term, no temporal cache.
   Mask invalid inputs BEFORE arithmetic (never multiply0 byNaN). Reuse existing
   foothold_grid/encode_height_scan/FK constants. surface_z and feet_z share the
   torso-relative yaw frame: body_clearance=-local_ground; foot clearance=
   feet_z-0.08-support_z. Do not subtract world root height a second time.
   No positive stationary bonus does NOT prove camping is impossible; report
   first-episode fraction of time with signed forward speed<1m/s.

## Training contract (one bounded diagnostic, fixed before outcomes)
- Start BOTH arms from unchanged anchored v10 seed42 final749 checkpoint, SHA
  fa1ec87ef2b89b698f250d91452685cef0a7f76e340e790650116138aee875c1.
  Frozen teacher is original v5 SHA889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e.
- Preserve all actor/critic weights; reset Adam1e-4 state and exploration std0.2
  identically for both arms; iteration0. Save and compare initialization tensors.
- Fine-tuning seed45, geometry75. Two arms: continuation control and adaptive reward.
  4096env x32steps x250iterations=32,768,000 new transitions each;65,536,000 total.
  Existing anchored prior lambda0.02/network/observations/noise/randomization unchanged.
- Pre-materialize geometry75 cache BEFORE both training arms. Freeze cache input
  SHA and initial root/joint/full-observation hashes; stop on mismatch.
- Select ONLY final iteration249 for each arm; no best checkpoint/seed selection,
  early stopping based on return, threshold tuning or silent retry.
- Validate finite training/rewards, actual saved task/reward/PPO config and immutable
  teacher identity. Record training telemetry; reward return alone is not success.

## Evaluation contract
Fresh geometry78/reset52 and79/reset53, same conditions for every policy. Warm cache
before scoring; pin all240 tiles per geometry and verify before/after each process.
Require exact initial root/joint/full88D observation hashes for every paired cohort.

Seven controllers: v5, original v10, continuation actor, adaptive actor, original
v12 history hybrid, continuation history hybrid, adaptive history hybrid. All
hybrids retain SAME frozen v5, SAME v12 time windows/thresholds/dwell/fade. Only rough
expert weights differ. Thus fixed-v5 flat behavior CANNOT be attributed to new
training; standalone actors measure whether lowered flat posture/speed is learned.

- Primary:16s,175env (35family/level cells x5),14files/2450first episodes.
- Separate secondary:64s,hardest stones,10env,14files/140first episodes.
- Freeze BOTH final models and all new source/plan bytes before ANY holdout rollout.
- Reuse existing physical Eval task/rewards, HorizonEpisodeTracker and gates.
  New evaluator may load declared v13 checkpoints and record read-only posture
  metrics; never change scoring thresholds/termination/observation values.
  Verify non-holdout original-history evaluator parity before using new evaluator.
- Strict1/6tile13.1/53.1m; preserve first-episode falls/lane/world and censoring rules.
- Additional per-env sums/counts: valid torso clearance/target error and reward
  target, obstacle/clear-ground samples, forward speed, foot clearance/deficit;
  conditional metrics labelled on visited states, not matched-state causal effects.
  Flat family speed uses distance/active time with first-episode falls retained.
- Always report family/difficulty and flat separately. Do not pool16s and64s.

## Predeclared judgments
Reward-effect comparison = adaptive vs equal-budget continuation, not merely old
checkpoint vs more training. Report actor-alone and hybrid separately.
Practical primary common checks require one/six rates no lower, fall/lane rates no
higher, world0 including flat, flat fall rate no higher. Standalone ALSO requires
strictly higher flat mean episode speed (mean of distance/(steps*dt), retaining
falls). Hybrid instead requires non-regression/expected identity of fixed-v5 flat
behavior AND at least one strict terrain success/fall/lane gain; do not impose the
impossible strict flat-speed gain on a frozen-v5 flat branch. Report posture
mechanism (lower flat body, reduced rough target error) separately rather than
assuming it. Conditional clear/rough bins severity<=0.1/>=0.9, intermediate separate.
Secondary hard-stone judgment: one/six no lower, falls/lane no higher,world0, at
least one strict gain. Never use secondary to override primary failure.
One starting actor/fine-tuning seed/two maps: exploratory evidence, no universal
best, statistical significance or real-camera/robot safety claim.

## Tests and stop conditions
- CPU tests: flat/upstep/uniform downstep/gap mouth/slopes profiles; translation/
  vertical-reference invariance of profile severity and foot clearances;
  invalid/NaN/clipped scans; partial coverage; foot capsule/frame/edge correctness;
  no stationary/backward/overturned bonuses; swing-proxy/stance examples; bounded costs; target stays above fall bound.
- Config/launcher tests: zero-control parity, sole treatment differences, no original
  edits, exact warmstart and fresh optimizer, mismatch/overwrite rejection.
-35/64env2iteration GPU training+eval smoke on geometry51/reset24; non-holdout.
  Inspect actual rewards and posture/velocity values; do not tune from holdouts.
-4096env capacity/pre-materialization before frozen full training, no competing GPU.
- Full CPU suite, compile, diff/static checks, independent review, then train/eval.
- Fail closed on invalid tensors, mismatched initial inputs, source/cache drift,
  missing checkpoints/evidence. Preserve failures; no automatic identical retry.
- Completion means implementation and declared comparison finished/audited; an
  unsuccessful scientific outcome is still a completed experiment.
