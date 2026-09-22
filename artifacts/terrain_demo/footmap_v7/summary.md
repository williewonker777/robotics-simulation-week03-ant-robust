# FootMap v7 — frozen, equal-budget comparison

750 iterations x4096 environments x32 steps per run; 3 training seeds per arm.
Held-out geometry/reset60/34 and61/35; first episode only, at most16s.

| Policy | One tile | Six tiles | Falls | Flat falls | Speed m/s |
|---|---:|---:|---:|---:|---:|
| frozen_v5 | 257/300 | 135/300 | 29/300 | 7/50 | 3.001 |
| blind | 746/900 | 326/900 | 87/900 | 17/150 | 2.912 |
| height | 755/900 | 144/900 | 108/900 | 13/150 | 2.433 |
| footmap | 760/900 | 216/900 | 85/900 | 16/150 | 2.627 |

## Highest difficulty1.0

| Policy | One tile | Six tiles | Falls |
|---|---:|---:|---:|
| frozen_v5 | 44/60 | 1/60 | 9/60 |
| blind | 136/180 | 1/180 | 25/180 |
| height | 133/180 | 0/180 | 39/180 |
| footmap | 137/180 | 0/180 | 28/180 |

## Every training seed

| Seed | Mode | One tile | Six tiles | Falls |
|---|---|---:|---:|---:|
| 42 | blind | 244/300 | 107/300 | 28/300 |
| 42 | height | 267/300 | 0/300 | 21/300 |
| 42 | footmap | 250/300 | 74/300 | 25/300 |
| 43 | blind | 253/300 | 126/300 | 41/300 |
| 43 | height | 250/300 | 84/300 | 39/300 |
| 43 | footmap | 251/300 | 81/300 | 32/300 |
| 44 | blind | 249/300 | 93/300 | 18/300 |
| 44 | height | 238/300 | 60/300 | 48/300 |
| 44 | footmap | 259/300 | 61/300 | 28/300 |

Promotion gate: **FAIL**

matched three-seed architecture/input study; ideal scan, not real depth. Reference evaluated once per condition, not replicated as independent data. Training seed is paired with training geometry; no broad-generalization guarantee.

See summary.json for every family/level and individual gate check.
