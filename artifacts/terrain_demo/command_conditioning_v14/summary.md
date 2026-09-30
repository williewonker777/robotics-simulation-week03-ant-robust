# v14 command conditioning: matched masked comparison

One starting actor/fine-tuning seed and two maps: exploratory, not universal or statistically significant. Conditioned is compared with equal-budget masked training; v13 and the original hybrid are baselines. The contrast changes actor AND critic command-plus-feedback access, not actor-only or goal-only causality. Action sensitivity does not establish arbitrary command tracking. 16s primary and 64s secondary remain separate; secondary cannot override primary failure. Posture means are conditional on visited valid states, not matched-state causal effects. Ideal ray depth and a kinematic swing proxy do not establish real-camera or robot safety.

## Primary: 16s mixed terrain

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v13 | 267/300 | 143/300 | 30/300 | 2/300 | 0 | 1/50 | 11.0841 |
| masked | 251/300 | 144/300 | 44/300 | 1/300 | 0 | 1/50 | 11.1450 |
| conditioned | 261/300 | 135/300 | 30/300 | 8/300 | 0 | 1/50 | 11.2795 |
| history_original | 272/300 | 146/300 | 25/300 | 2/300 | 0 | 3/50 | 9.4104 |
| history_masked | 263/300 | 145/300 | 31/300 | 3/300 | 0 | 3/50 | 9.4104 |
| history_conditioned | 273/300 | 135/300 | 24/300 | 3/300 | 0 | 3/50 | 9.4104 |

actor_conditioned_vs_masked: **FAIL**
- one_not_lower: PASS
- six_not_lower: FAIL
- falls_not_higher: PASS
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- flat_falls_not_higher: PASS
- flat_speed_strictly_higher: PASS

hybrid_conditioned_vs_masked: **FAIL**
- one_not_lower: PASS
- six_not_lower: FAIL
- falls_not_higher: PASS
- lane_not_higher: PASS
- world_zero_including_flat: PASS
- flat_falls_not_higher: PASS
- flat_speed_not_lower: PASS
- fixed_v5_flat_branch: PASS
- flat_lane_not_higher: PASS
- strict_terrain_gain: PASS
- fixed_v5_per_environment_flat_raw_identity: PASS

Visited-state posture mechanism (conditioned minus masked; not causal):
- actor: rough_target_error: -0.0093m, clear_body_clearance: -0.0009m, flat_family_body_clearance: 0.0011m
- hybrid: rough_target_error: -0.0058m, clear_body_clearance: -0.0004m, flat_family_body_clearance: 0.0000m

## Secondary: 64s hardest stones

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v13 | 9/20 | 6/20 | 9/20 | 2/20 | 0 | 0/0 | — |
| masked | 9/20 | 5/20 | 9/20 | 2/20 | 0 | 0/0 | — |
| conditioned | 12/20 | 9/20 | 6/20 | 2/20 | 0 | 0/0 | — |
| history_original | 8/20 | 6/20 | 9/20 | 4/20 | 0 | 0/0 | — |
| history_masked | 9/20 | 5/20 | 6/20 | 7/20 | 0 | 0/0 | — |
| history_conditioned | 7/20 | 2/20 | 10/20 | 3/20 | 0 | 0/0 | — |

actor_conditioned_vs_masked: **PASS**
- one_not_lower: PASS
- six_not_lower: PASS
- falls_not_higher: PASS
- lane_not_higher: PASS
- world_zero_including_flat: PASS
- strict_terrain_gain: PASS

hybrid_conditioned_vs_masked: **FAIL**
- one_not_lower: FAIL
- six_not_lower: FAIL
- falls_not_higher: FAIL
- lane_not_higher: PASS
- world_zero_including_flat: PASS
- strict_terrain_gain: PASS

Visited-state posture mechanism (conditioned minus masked; not causal):
- actor: rough_target_error: -0.0124m, clear_body_clearance: -0.0233m
- hybrid: rough_target_error: -0.0082m, clear_body_clearance: -0.0011m

Flat speed averages distance/(first-episode steps × dt), retaining falls. Frozen-v5 hybrid flat behavior is not evidence of learned posture.
See JSON for all baselines, family/level outcomes, valid-state body/foot metrics, low-speed occupancy and coverage.
