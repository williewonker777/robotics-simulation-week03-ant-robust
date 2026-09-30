# v11: depth-gated frozen v5/v10

Same two fresh maps; v5 once/map, each candidate3 frozen seeds. Not thousands of independent maps. No retraining. Ideal depth, not real RGB-D.

## 16s mixed terrain

| Policy | One tile | Six tiles | Falls | Lane | Flat falls | Mean v10 duty | Switches | Falls near switch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v5 | 263/300 | 133/300 | 23/300 | 0/300 | 4/50 | 0.000 | 0 | 0 |
| v10 | 814/900 | 438/900 | 67/900 | 11/900 | 13/150 | 1.000 | 0 | 0 |
| hybrid | 805/900 | 451/900 | 81/900 | 2/900 | 12/150 | 0.700 | 1654 | 1 |

## Separate64s stones1.0

| Policy | One tile | Six tiles | Falls | Lane | Flat falls | Mean v10 duty | Switches | Falls near switch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v5 | 7/20 | 1/20 | 7/20 | 2/20 | 0/0 | 0.000 | 0 | 0 |
| v10 | 25/60 | 15/60 | 26/60 | 8/60 | 0/0 | 1.000 | 0 | 0 |
| hybrid | 28/60 | 14/60 | 21/60 | 9/60 | 0/0 | 0.913 | 97 | 0 |

Promotion: **FAIL — retain v5**

- one_vs_v5: PASS
- one_vs_v10: FAIL
- six_vs_v5: PASS
- six_vs_v10: PASS
- falls_vs_v5: FAIL
- lane_vs_v5: FAIL
- world_zero: PASS
- flat_falls_vs_v5: PASS
- 64s_stone_six_vs_v5: PASS
- 64s_stone_falls_vs_v5: PASS

Falls near a switch are temporal associations, not causal attribution. See JSON for perseed and family/level results.
