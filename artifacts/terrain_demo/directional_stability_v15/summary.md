# v15 directional stability: matched reward comparison

One starting actor/fine-tuning seed and two maps: exploratory, not universal or statistically significant. Stable is compared with equal-budget control training; frozen v14 and original hybrid are baselines. Only the directional training reward differs; actor and critic retain identical conditioned 91D access. 16s primary and 64s secondary remain separate; secondary cannot override primary failure. Direction and posture means are conditional on visited valid states, not matched-state causal effects. Invalid direction samples are excluded, never counted as good tracking. Reward is unweighted and not dt-integrated. World-Z angular velocity is not Euler yaw derivative under tilt. Ideal ray depth and a kinematic swing proxy do not establish real-camera or robot safety.

## Primary: 16s mixed terrain

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v14 | 268/300 | 133/300 | 24/300 | 8/300 | 0 | 5/50 | 10.4593 |
| control | 268/300 | 160/300 | 30/300 | 2/300 | 0 | 3/50 | 9.9611 |
| stable | 275/300 | 158/300 | 21/300 | 3/300 | 0 | 3/50 | 10.0834 |
| history_original | 266/300 | 149/300 | 29/300 | 2/300 | 0 | 5/50 | 8.7177 |
| history_control | 267/300 | 155/300 | 32/300 | 1/300 | 0 | 5/50 | 8.7177 |
| history_stable | 270/300 | 149/300 | 26/300 | 2/300 | 0 | 5/50 | 8.7177 |

actor_stable_vs_control: **FAIL**
- one_not_lower: PASS
- six_not_lower: FAIL
- falls_not_higher: PASS
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- flat_falls_not_higher: PASS
- flat_speed_strictly_higher: PASS

hybrid_stable_vs_control: **FAIL**
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
- fixed_v5_per_environment_flat_raw_identity: PASS

Visited-state posture mechanism (stable minus control; not causal):
- actor: rough_target_error: 0.0009m, clear_body_clearance: 0.0022m, flat_family_body_clearance: 0.0015m
- hybrid: rough_target_error: -0.0005m, clear_body_clearance: 0.0011m, flat_family_body_clearance: 0.0000m

Visited-state directional diagnostics (stable minus control; not causal):
- actor: absolute_lateral_velocity: 0.0009, absolute_heading_error: -0.0023, absolute_yaw_error: 0.0716, reward: 0.0022
- hybrid: absolute_lateral_velocity: -0.0007, absolute_heading_error: 0.0051, absolute_yaw_error: 0.0555, reward: 0.0008

## Secondary: 64s hardest stones

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v14 | 4/20 | 3/20 | 12/20 | 4/20 | 0 | 0/0 | — |
| control | 10/20 | 9/20 | 4/20 | 5/20 | 0 | 0/0 | — |
| stable | 10/20 | 8/20 | 8/20 | 3/20 | 0 | 0/0 | — |
| history_original | 10/20 | 5/20 | 10/20 | 0/20 | 0 | 0/0 | — |
| history_control | 11/20 | 11/20 | 7/20 | 2/20 | 0 | 0/0 | — |
| history_stable | 6/20 | 6/20 | 11/20 | 2/20 | 0 | 0/0 | — |

actor_stable_vs_control: **FAIL**
- one_not_lower: PASS
- six_not_lower: FAIL
- falls_not_higher: FAIL
- lane_not_higher: PASS
- world_zero_including_flat: PASS
- strict_terrain_gain: PASS

hybrid_stable_vs_control: **FAIL**
- one_not_lower: FAIL
- six_not_lower: FAIL
- falls_not_higher: FAIL
- lane_not_higher: PASS
- world_zero_including_flat: PASS
- strict_terrain_gain: FAIL

Visited-state posture mechanism (stable minus control; not causal):
- actor: rough_target_error: 0.0069m, clear_body_clearance: 0.0111m
- hybrid: rough_target_error: 0.0106m, clear_body_clearance: 0.0008m

Visited-state directional diagnostics (stable minus control; not causal):
- actor: absolute_lateral_velocity: -0.0041, absolute_heading_error: 0.1006, absolute_yaw_error: 0.1846, reward: -0.0200
- hybrid: absolute_lateral_velocity: -0.0530, absolute_heading_error: -0.0383, absolute_yaw_error: -0.0732, reward: 0.0407

Flat speed averages distance/(first-episode steps × dt), retaining falls. Frozen-v5 hybrid flat behavior is not evidence of learned posture.
See JSON for all baselines, family/level outcomes, valid-state body/foot metrics, low-speed occupancy and coverage.
