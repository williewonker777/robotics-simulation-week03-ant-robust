# v16 contact-slip: matched reward comparison

One training seed and two fresh maps: exploratory, not universal or statistically significant. The only paired treatment is the contact-conditioned foot-tip-speed proxy, with both arms sensor-enabled. V15 control and original history hybrid are descriptive references, not substitute comparators. 16s primary and 64s secondary are separate; secondary cannot override primary failure. Contact/posture metrics are conditional on visited valid states, not matched-state causality. Filtered foot contact is not whole-body airborne, and speed is not exact contact-point slip. Ideal ray depth and simulation do not establish real-camera or real-robot safety.

## Primary: 16s mixed terrain

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v15_control | 277/300 | 148/300 | 18/300 | 3/300 | 0 | 3/50 | 10.0643 |
| control | 279/300 | 176/300 | 20/300 | 1/300 | 0 | 3/50 | 10.8072 |
| slip | 275/300 | 163/300 | 19/300 | 7/300 | 0 | 3/50 | 10.7073 |
| history_original | 277/300 | 152/300 | 21/300 | 4/300 | 0 | 5/50 | 9.1413 |
| history_control | 280/300 | 179/300 | 17/300 | 2/300 | 0 | 5/50 | 9.1413 |
| history_slip | 280/300 | 172/300 | 17/300 | 2/300 | 0 | 5/50 | 9.1413 |

actor_slip_vs_control: **FAIL**
- one_not_lower: FAIL
- six_not_lower: FAIL
- falls_not_higher: PASS
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- flat_falls_not_higher: PASS
- flat_speed_strictly_higher: FAIL

hybrid_slip_vs_control: **FAIL**
- one_not_lower: PASS
- six_not_lower: FAIL
- falls_not_higher: PASS
- lane_not_higher: PASS
- world_zero_including_flat: PASS
- flat_falls_not_higher: PASS
- flat_speed_not_lower: PASS
- fixed_v5_flat_branch: PASS
- flat_lane_not_higher: PASS
- strict_terrain_gain: FAIL
- fixed_v5_per_environment_flat_raw_identity: PASS

Contact mechanism (slip minus control; visited-state only, not causal):
- actor: bounded_cost: 0.0008, contact_fraction: 0.0027, contacted_tip_speed: -0.0142
- hybrid: bounded_cost: 0.0009, contact_fraction: 0.0026, contacted_tip_speed: -0.0003

## Secondary: 64s hardest stones

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v15_control | 11/20 | 11/20 | 4/20 | 5/20 | 0 | 0/0 | — |
| control | 10/20 | 8/20 | 7/20 | 3/20 | 0 | 0/0 | — |
| slip | 9/20 | 8/20 | 8/20 | 4/20 | 0 | 0/0 | — |
| history_original | 17/20 | 12/20 | 3/20 | 2/20 | 0 | 0/0 | — |
| history_control | 8/20 | 7/20 | 11/20 | 1/20 | 0 | 0/0 | — |
| history_slip | 6/20 | 6/20 | 9/20 | 4/20 | 0 | 0/0 | — |

actor_slip_vs_control: **FAIL**
- one_not_lower: FAIL
- six_not_lower: PASS
- falls_not_higher: FAIL
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- strict_terrain_gain: FAIL

hybrid_slip_vs_control: **FAIL**
- one_not_lower: FAIL
- six_not_lower: FAIL
- falls_not_higher: PASS
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- strict_terrain_gain: PASS

Contact mechanism (slip minus control; visited-state only, not causal):
- actor: bounded_cost: 0.0004, contact_fraction: -0.0423, contacted_tip_speed: 0.0128
- hybrid: bounded_cost: -0.0090, contact_fraction: 0.0131, contacted_tip_speed: -0.0531

Flat speed is distance/(first-episode steps × dt), retaining falls. See JSON for per-family/level scores and contact coverage, no-detected-foot-contact fraction, speed-cap saturation, posture metrics, provenance and exact flat-branch identity.
