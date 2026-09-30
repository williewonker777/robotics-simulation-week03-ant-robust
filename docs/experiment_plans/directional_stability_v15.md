# v15 — directional velocity / yaw tracking regularization

Continue the paper-informed rough-terrain experiment. Direct scoped execution,
independent design and implementation review; no formal OMX goal/workflow mode.
V14 raises body clearance and reduces some falls but loses six-tile completion
and increases standalone lane exits. This is evidence of a trade-off, NOT proof
of its cause. Test one bounded stability intervention; no automatic tuning loop.

## Grounding / alternatives
Selected: orthogonal-velocity suppression (Miki2022 Supplement S7) and commanded
[vx,vy,yawrate] tracking (Solo12 Scientific Reports2023), with clipped proportional
heading-to-yaw command as documented by IsaacLab. Adaptation: planar lateral
velocity only (do not suppress vertical climbing), negative bounded regularizer,
existing forward progress reward retained instead of a fixed forward-speed command.
Not a reproduction of either paper's actor, sensors, reward mixture or deployment.
Contact-aware slip/swing costs require new contact sensing (joint incoming wrenches
are NOT ground-contact sensors). Learned expert arbitration changes action/policy
architecture. Both alternatives are deferred, not judged ineffective.

## Method, fixed before training
Preserve exact existing91D observations, CommandPriorActorCritic conditioned mode,
400/200/100ELU, torque actions, v5 teacher60D and anchored prior coefficient.02.
Add ONE training reward term, same config both arms; control weight0, stable weight1.
No observation, physical/randomization/termination, selector, target-height, sensor
or old reward changes. No existing/frozen file edits; all106sources/eightreference
models and all prior evidence are immutable. No new dependencies or remote Git.

Let d be existing lane target unit direction in worldXY, n=(-dy,dx), v root COM
worldXY velocity, e wrapped lane target heading minus torso yaw, w root angular
velocity about worldZ. Desired yawrate wc=clip(1.0*e,-1,1) rad/s.
Lateral cost=1-exp(-3*(v dot n)^2), yaw cost=1-exp(-(w-wc)^2).
New reward=-.75*lateral_cost-.25*yaw_cost, bounded[-1,0].
Unknown/nonfinite input, invalid direction or projectedbodyforwardXY norm<.1
=> reward0 and invalid diagnostics. Normalizevalid targetdirection explicitly.
Negative shiftingpaperpositiveexponentials changesepisode-length incentives; it
isnot anequivalent rewardtransformation. Monitorlow-speed/stall andfalls explicitly.
No positive standing bonus, forward-speed target/cap change, family/level labels,
future-state info, contact heuristic, or extra temporal state. Proportional yaw
feedback permits corrective turns rather than penalizing all yaw. WorldZ yawrate
is a bounded planar approximation on tilted bodies, not exact Euler yaw derivative.

## Initialization / matched budget
Both start v14 conditioned final249:
SHA844e6a3b9c4bee914ce65faa08322f892ad91f6dc520b5bc5167aac5fd543b41.
Preserve EVERY91D network/teacher/mode/schema/command-weight tensor; reset onlystd.2,
freshAdam1e-4 anditeration0. Learned command weights MUST NOT be zeroed.
Each250iterations*4096env*32steps=32,768,000transitions,total65,536,000.
Seed47, traininggeometry95 pre-materialized andbyte-pinned beforebotharms.
Actualloaded full91D/root/joints/prefix/network/CPU+CUDA RNG mustmatch; savedconfig
mustmatch except newrewardweight/outputnames. Beforelearn validateallsourcetensors,
freshAdam andsame-deviceinitialactor/critic/teacher exactoutput parity tosource.
Onlyfinal249 selected; no seed/coeff/checkpointselection, earlystopping or retries.
All250iterations/transitions/weights/Adam/TensorBoard scalars finite and complete.

## Fresh evaluation
Geometry96/reset60 and97/reset61, neverpreviousholdouts forselection.
Sixcontrollers: v14(reference), control, stable, history_original(v5+originalv10),
history_control, history_stable. Fourmodelinventory(original,v14,control,stable).
Originalhistoryexpert88D, others91D; frozenv5teacher60D. Reuseunchangedv14Eval task,
newtruthfulv15schema/metadata; unchangedv12historygate. Rawscoringreusewithvalidated
projection; never impersonate a changed old source or alter old evaluation files.
Primary16s mixed175env:12files/2100firstepisodes. Separate64s hardeststones10env:
12files/120firstepisodes. Total24files/2220. Same13.1/53.1m,dt1/60,footprint1.1m,
fall/lane/world semantics and8/16ssnapshots. Allfamily/levelcountsandflatseparate.
Warmcache96/97 beforeANYscoring; bindfreeze+cacheSHA toeveryprocess/result, verify
before/after and exactfullinitialpairing. Pinpreholdoutcommandledgerprefix too.
Passive pre-action firstepisode directional metrics: abs lateral speed, heading
error, yaw-rate tracking error, newreward, valid/active counts; keep posturemetrics.
These are conditional visited-state diagnostics, not matched-state causal effects.

## Criteria (unchanged v14)
Predeclared primary contrasts: stable vs equal-budget control; history_stable vs
history_control. v14 and history_original are descriptive reference baselines,
not alternative post-result comparators. This tests the combined lateral/yaw term,
not either component independently. Passive reward telemetry is unweighted and
unintegrated; RewardManager applies weight * dt exactly once during training.
Actorprimary: one/sixnotlower,falls/lanenothigher,world0includingflat;flatfalls
nothigher andflatmeanfirstepisodespeedSTRICTLYhigher. Hybridprimary:samecommon,
flatnonregression+exactper-envv5flatrawidentity,andone strictterrainimprovement.
Separate64s: one/sixnotlower,falls/lanenothigher,world0,andone strictgain.
SecondarycannotoverrideprimaryFAIL. No universalbest/statistical/realrobotsafetyclaim.

## Verification / stop
CPU math: alignedmotionzero, sidewaysdriftpenalty, correctiveyawbetterthanwrongyaw,
rotation/mirror invariance, wrappedangles, clipping/bounds/invalid handling.
Adapterstateless/reset/portal andCOMvslink velocity contract; nophysicalconfigchange.
Actualcheckpoint/Adam/configguardnegativecases, strictmetadata/provenance/evidence
nooverwrite. 64env2iterbotharms and4096env2iterpairedcapacity onprewarmed95; eval
legacyreference35envgeometry51/reset24 exactlymatchesv14finalsmoke outcomes/initial
91D/RNG/prefix. Useonlynew95 cacheinventory becausehistorical51containsduplicates.
Independentreview+fullCPU/compile/diff beforefreezingtraining; finalnewmodelsmokes
andreviewbeforeholdout. Failurespreserved; fixdevelopment/auditors beforeholdouts,
neverweakenphysicalcriteria/hashchecks afteroutcomes. GPU1only;oneheavyjobatatime.
Stopwhen declaredimplementation/training/comparison/auditcomplete, evenifnoimprovement.

Primarysourcesandscopedadaptationlimits: artifacts/terrain_demo/directional_stability_v15/research.md.
