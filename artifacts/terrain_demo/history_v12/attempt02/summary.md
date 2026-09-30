# v12: sustained depth-history selector

Diagnostic comparison: one frozen policy seed42 on two maps, not a universal-best claim. 16s and64s outcomes are separate. Ideal ray depth, no recurrent policy training. Switch evidence is checked for consistency; raw depth is not independently reconstructed.

## 16s mixed terrain

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat duty | Switches | Active seconds | Switches/100s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v5 | 263/300 | 132/300 | 26/300 | 1/300 | 0 | 4/50 | 0.000 | 0 | 4513.75 | 0.000 |
| v10 | 273/300 | 147/300 | 25/300 | 0/300 | 0 | 4/50 | 1.000 | 0 | 4535.92 | 0.000 |
| instant | 271/300 | 148/300 | 28/300 | 0/300 | 0 | 4/50 | 0.000 | 594 | 4495.42 | 13.213 |
| history | 269/300 | 146/300 | 28/300 | 0/300 | 0 | 4/50 | 0.000 | 460 | 4553.90 | 10.101 |

History improvement criteria: **FAIL**

- fewer_switches_per_active_time: PASS
- world_zero: PASS
- one_not_lower: FAIL
- six_not_lower: FAIL
- falls_not_higher: PASS
- lane_not_higher: PASS
- flat_falls_not_higher: PASS

### Per-family history minus instant (counts; negative crossing or positive failure is a regression)

| Family | Δone | Δsix | Δfalls | Δlane | Δworld incl.flat | Δflat falls |
|---|---:|---:|---:|---:|---:|---:|
| flat | +0 | +0 | +0 | +0 | +0 | +0 |
| obstacles | +2 | +0 | -2 | +0 | +0 | +0 |
| rough | +1 | -4 | -1 | +0 | +0 | +0 |
| slope | -2 | +2 | +2 | +0 | +0 | +0 |
| stairs | -2 | +1 | +2 | +0 | +0 | +0 |
| stepping_stones | -1 | +1 | -1 | +0 | +0 | +0 |
| waves | +0 | -2 | +0 | +0 | +0 | +0 |

## 64s stones, difficulty1.0

| Controller | One | Six | Falls | Lane | World incl.flat | Flat falls | Flat duty | Switches | Active seconds | Switches/100s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v5 | 11/20 | 0/20 | 5/20 | 1/20 | 0 | 0/0 | — | 0 | 1000.65 | 0.000 |
| v10 | 5/20 | 2/20 | 10/20 | 4/20 | 0 | 0/0 | — | 0 | 829.60 | 0.000 |
| instant | 10/20 | 5/20 | 10/20 | 1/20 | 0 | 0/0 | — | 29 | 806.08 | 3.598 |
| history | 10/20 | 5/20 | 9/20 | 0/20 | 0 | 0/0 | — | 28 | 920.52 | 3.042 |

History improvement criteria: **PASS**

- fewer_switches_per_active_time: PASS
- world_zero: PASS
- one_not_lower: PASS
- six_not_lower: PASS
- falls_not_higher: PASS
- lane_not_higher: PASS

### Per-family history minus instant (counts; negative crossing or positive failure is a regression)

| Family | Δone | Δsix | Δfalls | Δlane | Δworld incl.flat | Δflat falls |
|---|---:|---:|---:|---:|---:|---:|
| stepping_stones | +0 | +0 | -1 | -1 | +0 | +0 |

See JSON for complete per-family/level aggregates and source-file digests. No tuning on these holdouts.
