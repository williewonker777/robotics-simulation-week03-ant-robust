# v10: training-only frozen-v5 Gaussian mean prior

Direct bounded experiment, not a safety proof. Preserve every old source/model.
User asked continue experiments afterv9. Aim: retain terrain-informed long traversal
while reducing fall/lane drift. Priorv8 fixed the v5 actor at inference and learned a
bounded residual; v10 instead trains the full88D student, with a teacher loss ONLY
on student-rollout training minibatches. Inference calls student alone.

## Evidence and hypothesis
v9targets six416/900 vsfeet283/900, butfalls91/900 vsfrozenv524/300,
lane66/900 vsv50. v9guided penalties increasedfalls105/900. Thus removefootshaping
and test a soft behavioral prior rather than more support reward or new sensor.
Neither forgetting nor its correction is proven by these aggregate scores.
Teacher is suboptimal onstones: excessiveprior may erase sensoradaptation.

## Fixed design (independent architecture review: VIABLE, 2026-09-22)
- Samev9targets88D/noiseless825ray/FK/proprio-noise,400/200/100ELU MLP,8effortactions.
- Samev5parent,warmstartold60weights+28zero columns,Adamreset,std.2.
- Frozen60Dteacher actor copiedbit-identical fromoriginalv5. Student consumesfull88;
  teacher consumes exactunchangedold60prefix. Teacherforwarddetached/eval.
- Auxiliary L=mean_batch,action((mu_student-mu_teacher)^2/(2*.2^2)).
  This is per-action KL between twoGaussians with FIXEDequalreference sigma.2,
  NOT KL to student's learned exploration Gaussian. StudentstdunchangedPPOlearnable.
- Loss=L_PPO+lambda*L. Twoarms free(lambda0),anchored(lambda.02),fixednotannealed.
  Numericlambda is a predeclared localhypothesis, notpaper-prescribed. LograwL,
  meanMSE,teacher-studentRMS,weightedprior,actionmeanabs>1fraction separatefromPPOloss/entropy.
- Samepolicyarchitecture/teacherstorage botharms. Modepersistedint64; lambda/fixedsigma typed scalar checkpoint buffers; strictload
  andtrainmodeguard; evalrestoresmode. Allteacherparamsfrozen; nofamilyprivilege
  inpriorloss. Existingrewards/terrain/terminations/strictbenchmarksunchanged.
- NewRSLPPOsubclass delegatesrollout/returns/storage toinstalled3.0.1; isolated
  minimalupdate loop for feedforward,singleGPU,fixedschedule,noRND/symmetry.
  Rejectunsupportedfeatures including desiredKL. PreserveBSDnotice. Pin upstream RSL-RL3.0.1 PPO source SHA256 deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c. Zerocoef numericalparity against
  installedPPO on seededidenticalCPUrollouts is a pretraininghardrequirement.
- No library edits/newdependencies/remotes/commits/hardware. Newopt-in task/launcher.

## Budget and freeze
2arms xseeds42/43/44,train geometries51/58/59,750iter x4096env x32steps each.
Bothstartfromsamev5, notselectedv9seed. New pairedfree controls avoid RNG/initialization
confounds from teacher construction vs reuseofoldv9trainedmodel.
Freeze all6final749 beforeunusedgeometry66/reset40,67/reset41. Includeonev5reference
percondition,notreplicated.7policies x2conditions x175env=14JSON/2450episodes;
terraindenoms300v5,900free,900anchored. Atmost16s/960steps,strict13.1/53.1m,
no terminal/lane/world; margin1.1. Allmodes/seeds retained, no postholdouttuning.

## Predeclared promotion gate
Anchored one/six rates>=v5 ANDfree; terrainfallrates<=v5 ANDfree; laneexitrate<=v5
ANDfree; flatfallrate<=v5; worldexit0; falls<free in>=2/3pairedtrainingseeds.
Must inspecthard.8/1.0stones andallfamily/level/seed tables. Ifnotallpass,retainv5.
Thisperformancegate is practical, not statistical significance proof.

## Separate directional hypothesis gate (does NOT authorize promotion)
Anchored terrain falls AND lane exits strictly lower than free in aggregate, with
BOTH lower jointly in at least2/3 paired training seeds; anchored six-tile rate>=v5;
world exits0. This tests the safety/throughput trade while preserving the stricter
promotion gate above. Report every failed check and never retune on these holdouts.

## Secondary horizon diagnostic (not promotion / noextraindependentmaps)
Afterfreeze/primaryaudit, same7policies and2geometries/resetseeds with64s/3840step
episodes,10environments all stepping_stones level4 (difficulty1.0).14JSON/140first
episodes:20v5,60free,60anchored. Both episode_length_s=64 and max_steps=3840.
Paired across policies, but the10env reset assignment is NOT the175env primary
assignment. Record first time-to13.1/53.1, distanceat16s (null if already terminated),
final distanceat64s/termination, maxdistance,falls,lane,world,fullsurvival. A firsthit
followed by a fall/exit is NOT strict success.53.1m/16srequires3.32m/s while v9hard
stones~1.2m/s, so0sixat16s doesnot establish inabilityatlongerhorizon. This is not
newtraining, newindependentmaps, or relaxedprimarymetric. Never combine diagnostic
counts with primary or tune/promote on them.

## Verification / presentation
CPUlossfinite/zero/scale/gradient/no-stdgradient,frozenteacher/inferenceindependence,
modepersist/mismatch,pairedinitialstates,oldv5identity,zerocoefPPOupdateparity;
existing151tests preserved.64env2iterand35envsave/loadeval plus4096capacity.
Actualsavedenv/PPOconfigs checked againstv9(nofootholdshaping),sourcehashpreservation,
independentpretrainreview,allfinaltimestamps/checkpointhashes,raw-arrayaudit.
Predeclarevideoanchoredseed42 vsv5 ongeometry66/reset40,stones1.0,uncut16s; failed
cases retained. Also predeclare an uncut64s video of this same pair; no seed selection. Verifydecode and
no statisticalaugmentation. Directstopafterthisfixedstudy+diagnostic+report.
