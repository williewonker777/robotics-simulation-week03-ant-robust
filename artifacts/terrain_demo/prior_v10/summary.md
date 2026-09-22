# Frozen-v5 training-only mean prior v10

6 runs: 750 iterations x4096 environments x32 steps; 3 paired seeds/mode.
Fresh geometry/reset66/40 and67/41. 14 evaluations /2450 first episodes, at most16s.

| Policy | One tile | Six tiles | Falls | Lane exits | Flat falls | Speed m/s |
|---|---:|---:|---:|---:|---:|---:|
| frozen_v5 | 260/300 | 132/300 | 30/300 | 0/300 | 4/50 | 2.961 |
| free | 744/900 | 422/900 | 96/900 | 61/900 | 13/150 | 3.227 |
| anchored | 802/900 | 454/900 | 73/900 | 15/900 | 3/150 | 3.128 |

## Highest difficulty 1.0

| Policy | One tile | Six tiles | Falls |
|---|---:|---:|---:|
| frozen_v5 | 42/60 | 1/60 | 12/60 |
| free | 151/180 | 24/180 | 26/180 |
| anchored | 150/180 | 13/180 | 21/180 |

## Every training seed

| Seed | Policy | One tile | Six tiles | Falls |
|---|---|---:|---:|---:|
| 42 | free | 247/300 | 139/300 | 32/300 |
| 42 | anchored | 269/300 | 152/300 | 27/300 |
| 43 | free | 268/300 | 141/300 | 30/300 |
| 43 | anchored | 267/300 | 143/300 | 23/300 |
| 44 | free | 229/300 | 142/300 | 34/300 |
| 44 | anchored | 266/300 | 159/300 | 23/300 |

Promotion gate: **FAIL**

Directional gate: **PASS**

Matched three-seed fixed-budget training-only v5 mean-prior study. Frozen v5 is evaluated once per condition, not replicated. Two held-out terrain maps are not thousands of independent maps. The scan and distal-center heuristic are ideal simulation signals, not real RGB-D or certified foothold support.

See summary.json for every family/level and individual gate check.
