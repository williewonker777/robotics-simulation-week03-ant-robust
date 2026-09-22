# Frozen-base residual v8

6 new runs: 750 iterations x4096 environments x32 steps; 3 seeds/mode.
Fresh geometry/reset62/36 and63/37. 20 primary evaluations /3500 first episodes, at most16s.

| Policy | One tile | Six tiles | Falls | Flat falls | Speed m/s |
|---|---:|---:|---:|---:|---:|
| frozen_v5 | 259/300 | 136/300 | 32/300 | 3/50 | 3.042 |
| v7_footmap | 753/900 | 219/900 | 94/900 | 6/150 | 2.648 |
| residual_blind | 712/900 | 374/900 | 99/900 | 10/150 | 2.971 |
| residual_footmap | 747/900 | 370/900 | 96/900 | 10/150 | 2.972 |

## Highest difficulty1.0

| Policy | One tile | Six tiles | Falls |
|---|---:|---:|---:|
| frozen_v5 | 45/60 | 2/60 | 12/60 |
| v7_footmap | 132/180 | 0/180 | 32/180 |
| residual_blind | 106/180 | 3/180 | 43/180 |
| residual_footmap | 129/180 | 1/180 | 35/180 |

## Every seed

| Seed | Policy | One tile | Six tiles | Falls |
|---|---|---:|---:|---:|
| 42 | v7_footmap | 247/300 | 79/300 | 30/300 |
| 42 | residual_blind | 237/300 | 121/300 | 33/300 |
| 42 | residual_footmap | 245/300 | 118/300 | 32/300 |
| 43 | v7_footmap | 243/300 | 70/300 | 42/300 |
| 43 | residual_blind | 239/300 | 123/300 | 29/300 |
| 43 | residual_footmap | 248/300 | 123/300 | 31/300 |
| 44 | v7_footmap | 263/300 | 70/300 | 22/300 |
| 44 | residual_blind | 236/300 | 130/300 | 37/300 |
| 44 | residual_footmap | 254/300 | 129/300 | 33/300 |

Promotion gate: **FAIL**

Matched fixed-budget residual comparison, all three seeds. Frozen v5 is evaluated once per condition, not replicated. Two held-out terrain maps, not thousands of independent maps. Ideal raycasts, not real RGB-D. A frozen base/bounded mean is not a safety guarantee.

Depth zero/shuffle diagnostic episodes are not included in primary totals or promotion.
