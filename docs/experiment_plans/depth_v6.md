# Depth-aware lane training v6

User requests terrain depth observations and actual training. Scope explicitly extends the old60D contract; retain v0-v5 tasks/checkpoints and course submission unchanged.

Mode: direct bounded implementation using existing scanner path, independent CPU-test/code-review assistance. No heavy OMX workflow is launched. One GPU job at a time, no remote Git/GitHub or new dependencies.

Design: yaw-stabilized downward raycast depth/height grid (not rendered RGB-D),143 rays (13x11,0.2m resolution,2.4x2m extent,xoffset0.8m,zoffset2m,maxrange4m). Coverage relativebody x[-.4,2],y[-1,1].143 clipped relativeheight values and143 validitybits appended to60coursefeatures =346D. Explicitfinite miss encoding. The static terrain mesh sensor is combined with analytic intersection of the existing infiniteflatplane atz=-50 for ALLrays withoutfamilylabels; nearestverticalhitwins. No globalterrainmap/family/level/targetpassedtopolicy. This remains idealizedvirtualheightperception, not physicallyrendered camera/occlusion equivalence.

Separate Depth-{Train,Eval,Demo}-v6 tasks reuse v5geometry, first60features,actionscale7.5,reward/terminationbenchmark. Train only inherits recovery shaping with stonebodytarget.55 weight-3,stall-2,fall-1800,margin1.1. Scene clone_in_fabric=False required by scanner view/reset path. Force sensor recompute at observation time to prevent stale data after lane-wrap/teleport. Train depth noise0.02m, no noise evaluation. Optional observation mask exists only for explicit evaluationablation; log mode.

Warmstart: zero-expand actor+critic first layers from retained portal_rehearsal4 by286columns; exactlypreserveinitialdeterministicactions foranydepthfeatures. Restorestd.20, clearoptimizer,iter0. Same400/200/100hiddenMLP.

Tests beforetraining: puremath finitebounds,closestmesh/planehit,maxrange,nohitmask,translationinvariance,nofamilyoracle; actualcheckpoint zeroexpansionexactactions. Simulator64env2iter smoke and sensorprobe:346Dfinite;nontrivialterrainvariance;flatplanevalid;scannerfollowsresetandwrap; warmstartoldactions preserved. v5regressiontests intact.

Run:4096env1500iterations fixedlr1e-4 gamma.995 entropy.002, stonesweight4flat2others1, save250. Screen checkpoint500/1000/1499 on fixed175env reset24 geometry51 againststored60D baseline. Test actualdepthdependence withzeroedscan/permutedscan (notjustnonzeroweights). If initialtrialregresses, one boundedstabilization stage or mixedterrain teacherrehearsal, preservingallfailures. Do notclaimdepthalonecausalimprovementwithout matchedcontrol.

Freeze selectedcandidate before fresh reset31/32 andgeometry56/57reset33; preservefixedbenchmark24/25/26comparability. Fresh paired60Dreference vsdepth. Gate:strict crossingsincrease, no greater totalterrainfalls on matchednewset,0worldexit. Report6tileandstonehighlevelsseparately. Include flatlane preservation; original60DcourseIDcannotload346D andmustnotpretendcompatibility.

Deliver checkpoints+params/hash, scannerdiagnostic snapshot, uncutactualwalkvideo, strictmetricJSON+independentaudit, docswithcameraidealization/observability/remainingfailures. Tests/lintavailablechecks;no universalterrainorhardwareclaim. No newremoteoperations/commits/dependencies.

## Matched reference clarification
RayCaster reset RNG and clone_in_fabric=False can change initial samples even under the same numeric seed. Therefore fair paired reference is the frozen zero-expanded346D copy of60D portal_rehearsal4, in the SAMEv6scene withdepthcolumnsallzero; actions mathematicallyignoredepth. Oldv5resultsremainhistorical, notguaranteedbit-identicalrollouts. Sensorprobe passed346Dfinite,143rays,flatplanevalid,firsttile stonevariance.118-.151,resetfresh,wrapequivalenceerror0.64env2itertrainingsmoke passed;57CPUtests passed.

## Initial full run launched
2026-09-21_20-26-11_depth143_recovery_20260921, GPU0,4096env1500it. Exactpath saved intraining_run.txt, console train_depth143_recovery.log. Native execsession8909; process662737 atlaunch. Reviewer found no sensor/benchmark blocker. Legacy checkpoint expander defaults(54extra,LR2e-4) intentionallyretained; v6explicit286/LR1e-4. Addedopt-in --reset-iteration withlegacyandv6regressions, fullsuite59tests. Actualwarmstartalreadyiter0 andfrozenhashinwarmstart_frozen.json.
