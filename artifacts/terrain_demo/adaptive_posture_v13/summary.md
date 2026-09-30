# v13 adaptive posture: matched continuation comparison

One starting actor/fine-tuning seed and two maps: exploratory, not universal or statistically significant. Adaptive is compared with equal-budget continuation; old v10 and v5 are baselines. 16s primary and 64s secondary remain separate; secondary cannot override primary failure. Posture means are conditional on visited valid states, not matched-state causal effects. Ideal ray depth and a kinematic swing proxy do not establish real-camera or robot safety.

## Primary: 16s mixed terrain

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 257/300 | 132/300 | 33/300 | 1/300 | 0 | 3/50 | 9.5833 |
| original | 254/300 | 135/300 | 40/300 | 4/300 | 0 | 1/50 | 10.6396 |
| control | 262/300 | 139/300 | 37/300 | 0/300 | 0 | 2/50 | 10.5796 |
| adaptive | 260/300 | 134/300 | 39/300 | 0/300 | 0 | 4/50 | 10.8702 |
| history_original | 270/300 | 143/300 | 30/300 | 0/300 | 0 | 3/50 | 9.5833 |
| history_control | 250/300 | 143/300 | 45/300 | 3/300 | 0 | 3/50 | 9.5833 |
| history_adaptive | 262/300 | 142/300 | 34/300 | 4/300 | 0 | 3/50 | 9.5833 |

actor_adaptive_vs_control: **FAIL**
- one_not_lower: FAIL
- six_not_lower: FAIL
- falls_not_higher: FAIL
- lane_not_higher: PASS
- world_zero_including_flat: PASS
- flat_falls_not_higher: FAIL
- flat_speed_strictly_higher: PASS

hybrid_adaptive_vs_control: **FAIL**
- one_not_lower: PASS
- six_not_lower: FAIL
- falls_not_higher: PASS
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- flat_falls_not_higher: PASS
- flat_speed_not_lower: PASS
- fixed_v5_flat_branch: PASS
- flat_lane_not_higher: PASS
- strict_terrain_gain: PASS

Visited-state posture mechanism (adaptive minus control; not causal):
- actor: rough_target_error: 0.0014m, clear_body_clearance: -0.0060m, flat_family_body_clearance: -0.0073m
- hybrid: rough_target_error: 0.0019m, clear_body_clearance: -0.0002m, flat_family_body_clearance: 0.0000m

## Secondary: 64s hardest stones

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 12/20 | 0/20 | 4/20 | 0/20 | 0 | 0/0 | — |
| original | 11/20 | 7/20 | 7/20 | 1/20 | 0 | 0/0 | — |
| control | 5/20 | 5/20 | 11/20 | 4/20 | 0 | 0/0 | — |
| adaptive | 12/20 | 10/20 | 6/20 | 2/20 | 0 | 0/0 | — |
| history_original | 12/20 | 9/20 | 7/20 | 1/20 | 0 | 0/0 | — |
| history_control | 12/20 | 7/20 | 6/20 | 4/20 | 0 | 0/0 | — |
| history_adaptive | 10/20 | 7/20 | 5/20 | 7/20 | 0 | 0/0 | — |

actor_adaptive_vs_control: **PASS**
- one_not_lower: PASS
- six_not_lower: PASS
- falls_not_higher: PASS
- lane_not_higher: PASS
- world_zero_including_flat: PASS
- strict_terrain_gain: PASS

hybrid_adaptive_vs_control: **FAIL**
- one_not_lower: FAIL
- six_not_lower: PASS
- falls_not_higher: PASS
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- strict_terrain_gain: PASS

Visited-state posture mechanism (adaptive minus control; not causal):
- actor: rough_target_error: -0.0067m, clear_body_clearance: -0.0010m
- hybrid: rough_target_error: -0.0156m, clear_body_clearance: -0.0063m

Flat speed averages distance/(first-episode steps × dt), retaining falls. Frozen-v5 hybrid flat behavior is not evidence of learned posture.
See JSON for all baselines, family/level outcomes, valid-state body/foot metrics, low-speed occupancy and coverage.
