# v18 depth-gated teacher style: paired result

Single training seed and two fresh maps: exploratory, not universal superiority. Only paired causal change is a depth-derived binary mask on the frozen-v5 teacher loss. Actor still receives 91D features, not the full scan. v16 and original history are descriptive. 64s hardest-stones evidence cannot override 16s mixed primary failure. Ideal simulator rays and visited-state contact telemetry do not demonstrate real-camera or robot safety.

## Primary: 16s mixed

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v16_control | 276/300 | 172/300 | 20/300 | 3/300 | 0 | 3/50 | 10.8763 |
| always | 280/300 | 162/300 | 19/300 | 1/300 | 0 | 4/50 | 10.3253 |
| gated | 273/300 | 154/300 | 18/300 | 9/300 | 0 | 4/50 | 10.5984 |
| history_original | 261/300 | 139/300 | 34/300 | 3/300 | 0 | 3/50 | 9.2060 |
| history_always | 279/300 | 163/300 | 20/300 | 1/300 | 0 | 3/50 | 9.2060 |
| history_gated | 271/300 | 156/300 | 26/300 | 3/300 | 0 | 3/50 | 9.2060 |

actor_gated_vs_always: **FAIL**
- one_not_lower: FAIL
- six_not_lower: FAIL
- falls_not_higher: PASS
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- flat_falls_not_higher: PASS
- flat_speed_strictly_higher: PASS

hybrid_gated_vs_always: **FAIL**
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

## Secondary: 64s hardest stones

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v16_control | 9/20 | 7/20 | 9/20 | 3/20 | 0 | 0/0 | — |
| always | 9/20 | 9/20 | 10/20 | 2/20 | 0 | 0/0 | — |
| gated | 6/20 | 6/20 | 7/20 | 8/20 | 0 | 0/0 | — |
| history_original | 10/20 | 7/20 | 7/20 | 2/20 | 0 | 0/0 | — |
| history_always | 7/20 | 6/20 | 11/20 | 2/20 | 0 | 0/0 | — |
| history_gated | 9/20 | 7/20 | 4/20 | 9/20 | 0 | 0/0 | — |

actor_gated_vs_always: **FAIL**
- one_not_lower: FAIL
- six_not_lower: FAIL
- falls_not_higher: PASS
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- strict_terrain_gain: PASS

hybrid_gated_vs_always: **FAIL**
- one_not_lower: PASS
- six_not_lower: PASS
- falls_not_higher: PASS
- lane_not_higher: FAIL
- world_zero_including_flat: PASS
- strict_terrain_gain: PASS

See summary.json and 24 raw JSON files for all family/level outcomes, paired initialization, routing and telemetry. No default-controller switch without both primary promotions.
