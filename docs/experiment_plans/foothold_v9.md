# v9 scan-derived distal foothold hints (direct execution)

## Hypothesis / boundaries
v6-v8 encoded terrain directly or corrected torques; hard stones remain unsolved.
Test explicit plateau/endpoint proposals and a bounded current-support cost.
This is NOT a certified whole-capsule planner, IK controller, or safety proof.
Actual hard stones after heightfield integer rounding: levels.8/1.0 both width.5m,
gap.2m; heights vary. Whole .565685m capsule support is not inferred from its tip.

## Selected minimal design
- New isolated tasks/launcher preserve every old source/checkpoint hash.
- Same physical terrain,8effort actions,60oldfeatures,strict13.1/53.1m16s gates.
- Increase ideal yaw height scan resolution.2->.1m on same3.2x2.4m footprint:
  33x25=825rays. No new camera/dep/contact sensor. No camera realism claim.
- Plateau candidates require3x3 all-valid/unclipped neighbors with range<=.06m;
  borders invalid. Candidate validity is a heuristic, NOT support certification.
- Perfoot search within.45m of current FK distal center; preserve body quadrant,
  avoid low gap candidates when locally higher candidate surfaces exist.
  Conservative empirical endpoint neighborhood, not exact2DOF IK feasibility.
- Proposal nearest nominal current foot+.2m in body-yaw forward direction;
  no candidate => vector0/valid0. All candidates derived causally from sensor.
- Observation88D =old60 +4FKfeetXYZ(12) +4(target-relativeXYZ+valid)(16).
  Same400/200/100MLP actor/critic, sourcev5firstlayerextra28columns0, std.2.
  NoCNN, nofixed-base residual, no memory. PPOmatchesv7.
- Three arms: feet-only(last16masked), targets, guided(targets+supportcost).
  Identical prepared states except modebuffer, matching paired seeds/geometries.
- Guided training only: onstone family, nearest CURRENT local plateau provides
  support anchor (not the forward proposal). Bounded cost for foot below its
  top+.08 and for low foot farther than.1m from supported center, normalized.2m,
  squared/clipped[0,1], mean4feet, totalrange[0,2], weight−1. No velocity gate;
  static buried feet still penalized; high swing horizontal cost fades out.
  No forward-target tracking reward, latch/contact phase or positive standstill bonus.
- Actor/target planner never reads family/geometry truth. Guided reward uses the
  existing training-only stone mask, disclosed as privileged shaping.
- Planner uses noiseless ideal scan consistently in observation/reward, not two
  independent noise draws. Existing60D proprioception noise remains unchanged.

## Fixed experiment / alternative control
3modes xseeds42/43/44 pairedgeometry51/58/59,750iter x4096env x32steps.
Freeze all9final749 before freshgeometry64/reset38,65/reset39. Existingv5reference
in samev9scene,2files baseline +18trained =20primaryfiles/3500episodes.
No repeatedbaseline counts, no postholdout training/selection. Representative42.
Promotion guided requires noone/six/fall regression vsfrozenv5/feet/targets,
no flat-fall regression vsv5, zero worldexit, guidedone>targets in>=2/3seeds.
Hardstone.8/1.0 per-family breakdown mandatory. Existingv5retained onfail.
Depthperturbation and videos qualitative/diagnostic only, excludedfromprimary.

## Verification before long jobs
CPU math: plateau interiors vsedges/holes/misses, gridordering, nofakeboundary
clamping, body-side/displacementbounds, height/translation invariance, no-candidate
finite zeros, current-support cost0onflat, positiveforstationaryburied, swingfree,
finite/boundedgradless targetgeometry. Policy oldaction/value identity,modepersistence,
pairedinitialstates,disabledinputinvariance. Existing98tests preserved.
Simulatorprobe35env: scanordering825,88D; actualhardlane targetavailability and
edge/heightmask; FK/yaw/reset/wrapfreshness; loggedtargetvalid/error diagnostics.
64env2itertrain +35envsavedmodeleval; actualreward/action/terminationconfigchecks.
Independentarchitecture andcode review beforefixedbudgettraining.
Finalraw-array independentaudit,source/checkpointSHA,timefreeze,uncut16s video.
OneGPUjob atatime,existingfiles untouched,no newdeps/remote/commits/hardware.
Stop afterfixedexperiment+auditedcomparison; reportfailure honestly ifnotbetter.

## Pre-training architecture/probe corrections
- Architect review VIABLE/WATCH: endpoint proposals are not whole-capsule support;
  max height across3x3 patch determines candidate center Z. Permit .15m descending
  treads with .18m local-top band, exclude >=.25m pits. Highest observed ray is
  computed before vertical feasibility, preventing fallback to a reachable pit
  when the top is unreachable. XY search<=.45m, absolute Zdelta<=.4m.
- First simulator probe exposed wrong name-based quadrant assumptions (only foot0
  eligible). Corrected to actual Ant FK order ++,-+,--,+- and strengthened reset
  assertion. Fresh35envprobe now all4feet35/35, hardstone4/4, wrap error0.
  CPU regression tests use actual asset order, not conventional leg-name meaning.
- Target/support selection may change each step; only nearest CURRENT support
  penalized. No swing-phase-latching claim. Invalid candidates abstain (zero cost)
  and availability is recorded as a limitation, not interpreted as safety.

## Non-primary diagnostic fixed before holdout
After the frozen primary evaluation, rerun the existing300step availability probe
on traininggeometry51/reset24 with finalguidedseed42. Compare against the already
recorded zeroexpandedv5availability solely to expose abstention/zero-guidance
behavior; no diagnostic contributes to promotion or permits retuning.
