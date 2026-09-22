# Scan-derived foothold hints v9

9 runs: 750 iterations x4096 environments x32 steps; 3 paired seeds/mode.
Fresh geometry/reset64/38 and65/39. 20 evaluations /3500 first episodes, at most16s.

| Policy | One tile | Six tiles | Falls | Flat falls | Speed m/s |
|---|---:|---:|---:|---:|---:|
| frozen_v5 | 264/300 | 136/300 | 24/300 | 5/50 | 3.048 |
| feet | 734/900 | 283/900 | 99/900 | 10/150 | 2.864 |
| targets | 741/900 | 416/900 | 91/900 | 11/150 | 3.278 |
| guided | 763/900 | 424/900 | 105/900 | 12/150 | 3.226 |

## Highest difficulty 1.0

| Policy | One tile | Six tiles | Falls |
|---|---:|---:|---:|
| frozen_v5 | 44/60 | 2/60 | 12/60 |
| feet | 124/180 | 0/180 | 44/180 |
| targets | 137/180 | 16/180 | 36/180 |
| guided | 130/180 | 10/180 | 42/180 |

## Every training seed

| Seed | Policy | One tile | Six tiles | Falls |
|---|---|---:|---:|---:|
| 42 | feet | 240/300 | 86/300 | 34/300 |
| 42 | targets | 241/300 | 132/300 | 29/300 |
| 42 | guided | 252/300 | 139/300 | 45/300 |
| 43 | feet | 254/300 | 94/300 | 37/300 |
| 43 | targets | 265/300 | 145/300 | 28/300 |
| 43 | guided | 250/300 | 128/300 | 31/300 |
| 44 | feet | 240/300 | 103/300 | 28/300 |
| 44 | targets | 235/300 | 139/300 | 34/300 |
| 44 | guided | 261/300 | 157/300 | 29/300 |

Promotion gate: **FAIL**

Matched three-seed fixed-budget foothold-hint study. Frozen v5 is evaluated once per condition, not replicated. Two held-out terrain maps are not thousands of independent maps. The scan and distal-center heuristic are ideal simulation signals, not real RGB-D or certified foothold support.

See summary.json for every family/level and individual gate check.
