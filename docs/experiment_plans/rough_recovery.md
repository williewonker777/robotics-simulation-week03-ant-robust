# Stepping-stone recovery continuation (2026-09-21)

## Fixed boundary
Preserve selected traverse1499 and Eval-v5 terrain/seed/difficulty/termination. Keep 60D/8D, effort scale7.5 and MLP400/200/100. No remote operations or dependency changes. One heavy GPU process at a time.

## Evidence
25-environment gait diagnostic (5 per stone difficulty, seed24) uses an explicit first-episode mask. On difficulty0.8 after2s, 72.75% of foot samples have distal capsule bottoms below nominal stone top minus0.10m; torso medianz0.272m; 81.94% of samples have forward speed below0.3m/s. Example env3 remains at distance6.44m from3s to15s. Loaded USD geometry proves ankle origins are nottips: distal capsule centre offsets(+.4,+.4,0),(-.4,+.4,0),(-.4,-.4,0),(+.4,-.4,0),radius.08. Incoming wrench is not a contact sensor.

## Change
Separate Recovery-Train-v5 task adds bounded stone-only low-torso and forward-swing distal-foot-clearance costs. No observation/benchmark changes. Pure math tests first, 64env/2iteration smoke, then shallow-pit training -> evaluate original deep pits -> intermediate/full-depth continuation only when evidence warrants it. Low-torso cost saturates and has no high-jump bonus; clearance cost applies only to forward-relative foot swing, notstance. Portal included to prepare gait before the first gap.

## Initial trial
From traverse1499, reset optimizer/iteration and std.25; lr7.5e-5 fixed, gamma.995, entropy.002, 4096env1000iterations. Training-only depth-.10m, stonesweight4/flat2/others1, fall-1200, lane margin1.1; save250. Eval still-.30m.

## Gates
Unchanged175env seed24 benchmark plus100env courseID after each stage. Promote only after seeds25/26, heldout terrain52/53 and courseOOD; report difficulty.6/.8/1 separately. Benchmark strict containment/full16s criteria remain unchanged. Survival or greaterdistancealone is notsuccess. Preserve baseline if mixed-terrain safety or IDregresses materially.

## Independent review note
Shallow trial had already loaded the first implementation when review identified root COM vs link velocity subtraction in the swing gate. Corrected source to root_link_lin_vel_w for all subsequent runs (training randomizes torso COM). The first shallow trial remains an exploratory warmstart with the recorded original formula; evaluation is unaffected because it has no shaping. Reward intentionally targets distal-tip lift, not full-capsule collision clearance; monitor endpoint geometry and movement alongside strict crossing instead of interpreting reward decrease as success.

## Medium results / full-depth decision
medium999:105/150 strictsuccess, stones10/25, otherterrain regressions, ID153.70.
medium500:117/150 strictsuccess, stones10/25, fewerfalls/laneexits; diagnostic easy-stone median torsoz~.50m improved, buthard.8 staysstuck (rootz.226m). Chosen onlyasnextwarmstart, notdefault.
Full-depth1500: frommedium500std.25lr7.5e-5, stones6flat2others1, fall-1800, stall-2, stonebodyheight-3target.55. Defaultpitdepth-.30. Keepfootcostsame. Save250 forscreening. Originalselecteduntouched.

## Full-depth evidence / next selection
Full1000:129/150strictsuccess, stones20/25 (4,5,4,2,5 perlevel),19terrainfalls,0laneexits,ID152.2668. Full1499regressed120/150, full500116/150. Actualgait.8 rootz.387 vsbaseline.272, below-.1feet46.4%vs72.8%, notjustrewardgaming. Preservefull1000.
Separatekinematicdebug found swing-onlycost canbeevadedbystillfeet. Addedexplicit optional trap_weight(default0 preserveshistoricalcurriculum) with max(swingcost, belowtopcost*weight), teststationarygapvsnominalcontact/bounds. 36tests+64env2iterwarmstartsmokepass.
Stabilize800plan: initfromdeep1000std.10lr5e-5gamma.995entropy.001, stones4flat2others1, fall-2400,stall-2,stonebody-3target.55, trap_weight1. Noeval/baselinechanges.

## Post-selection regression / bounded rehearsal
Deep1000 fixed3seed: stones55/75 vs24/75, but terrain366/450 vs369/450 and falls67 vs32. Not promoted. Stability trial400/799 and parameter blends25/50/75 failed seed24 screen. Preserve baseline default.
Actor-only expert rehearsal trains one60D8D MLP from frozen traverse1499(nonstones) anddeep1000(stones), aggregating teacher/student rollout states. Labels use lane identity onlyduringtraining; no inference routing. Ungated round4 seed24:127/150 strict,15falls,stones15/25. Portal-gated repeat usesreference teacheroncommonflatentrance until3m, thenstoneteacheronstones; onebounded4roundtrialwithsameotherparameters. Choosefromtheseonseed24 beforeunusedreset28/29,geom54/55.
Promotion gate: strict one-tile success increases on new reset/geometry aggregate, terrain fall count no higher than old baseline on matching episodes; courseID>=80%oforiginalflat154.32, no newworldescape. If not met, retain originaldefaultandlabelstone-recoverycandidateexperimental. Keepnewholdoutresultsregardlessofoutcome. No furtherparameteradjustmentbasedonthesenewholdoutsinthisbranch.
