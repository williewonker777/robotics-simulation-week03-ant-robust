# v17 learning-progress: matched sampling comparison

One training seed and two fresh maps: exploratory, not universal or statistically significant. The only paired treatment is signed learning-progress task sampling versus fixed-weight per-reset sampling. V16 static-startup control and original history hybrid are descriptive references, not substitute comparators. 16s primary and 64s secondary are separate; secondary cannot override primary failure. Contact/posture metrics are conditional on visited valid states, not matched-state causality. Filtered foot contact is not whole-body airborne, and speed is not exact contact-point slip. Ideal ray depth and simulation do not establish real-camera or real-robot safety.

## Primary: 16s mixed terrain

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v16_control | 278/300 | 179/300 | 19/300 | 3/300 | 0 | 2/50 | 11.0028 |
| fixed | 270/300 | 163/300 | 21/300 | 7/300 | 0 | 2/50 | 10.1889 |
| lp | 268/300 | 159/300 | 27/300 | 5/300 | 0 | 2/50 | 10.6663 |
| history_original | 268/300 | 147/300 | 30/300 | 0/300 | 0 | 7/50 | 8.9741 |
| history_fixed | 279/300 | 166/300 | 20/300 | 1/300 | 0 | 7/50 | 8.9741 |
| history_lp | 270/300 | 162/300 | 24/300 | 6/300 | 0 | 7/50 | 8.9741 |

actor_lp_vs_fixed: **FAIL**
- one_not_lower: FAIL
- six_not_lower: FAIL
- falls_not_higher: FAIL
- lane_not_higher: PASS
- world_zero_including_flat: PASS
- flat_falls_not_higher: PASS
- flat_speed_strictly_higher: PASS

hybrid_lp_vs_fixed: **FAIL**
- one_not_lower: FAIL
- six_not_lower: FAIL
- falls_not_higher: FAIL
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- flat_falls_not_higher: PASS
- flat_speed_not_lower: PASS
- fixed_v5_flat_branch: PASS
- flat_lane_not_higher: PASS
- strict_terrain_gain: FAIL
- fixed_v5_per_environment_flat_raw_identity: PASS

Passive contact diagnostics (lp minus fixed; visited-state only, not causal):
- actor: bounded_cost: 0.0034, contact_fraction: 0.0007, contacted_tip_speed: 0.0473
- hybrid: bounded_cost: 0.0017, contact_fraction: 0.0001, contacted_tip_speed: 0.0194

## Secondary: 64s hardest stones

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat mean episode speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v16_control | 9/20 | 9/20 | 9/20 | 2/20 | 0 | 0/0 | — |
| fixed | 5/20 | 3/20 | 10/20 | 7/20 | 0 | 0/0 | — |
| lp | 7/20 | 7/20 | 10/20 | 4/20 | 0 | 0/0 | — |
| history_original | 13/20 | 6/20 | 7/20 | 0/20 | 0 | 0/0 | — |
| history_fixed | 13/20 | 12/20 | 4/20 | 3/20 | 0 | 0/0 | — |
| history_lp | 8/20 | 8/20 | 10/20 | 5/20 | 0 | 0/0 | — |

actor_lp_vs_fixed: **PASS**
- one_not_lower: PASS
- six_not_lower: PASS
- falls_not_higher: PASS
- lane_not_higher: PASS
- world_zero_including_flat: PASS
- strict_terrain_gain: PASS

hybrid_lp_vs_fixed: **FAIL**
- one_not_lower: FAIL
- six_not_lower: FAIL
- falls_not_higher: FAIL
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- strict_terrain_gain: FAIL

Passive contact diagnostics (lp minus fixed; visited-state only, not causal):
- actor: bounded_cost: 0.0165, contact_fraction: -0.0312, contacted_tip_speed: 0.0815
- hybrid: bounded_cost: 0.0091, contact_fraction: -0.0252, contacted_tip_speed: 0.0586

Flat speed is distance/(first-episode steps × dt), retaining falls. See JSON for per-family/level scores and contact coverage, no-detected-foot-contact fraction, speed-cap saturation, posture metrics, provenance and exact flat-branch identity.
