# Difficult-terrain comparison (frozen depth-v6 vs blind reference)

Existing fresh evaluation subgroup only: four generator/reset pairs51/31,51/32,56/33,57/33. No retraining or new quantitative rollouts.

| Family | Difficulty | Blind strict /20 | Depth strict /20 | Blind → depth falls | Blind → depth six-tile | Blind → depth lane exits |
|---|---:|---:|---:|---:|---:|---:|
| rough | 0.8 | 19 | 19 | 1 → 1 | 12 → 0 | 0 → 0 |
| rough | 1.0 | 20 | 20 | 0 → 0 | 3 → 0 | 0 → 0 |
| slope | 0.8 | 19 | 20 | 1 → 0 | 0 → 0 | 0 → 0 |
| slope | 1.0 | 15 | 13 | 5 → 7 | 0 → 0 | 0 → 0 |
| stairs | 0.8 | 19 | 16 | 1 → 4 | 0 → 0 | 0 → 0 |
| stairs | 1.0 | 19 | 18 | 1 → 1 | 0 → 0 | 0 → 1 |
| waves | 0.8 | 18 | 16 | 2 → 4 | 10 → 1 | 0 → 0 |
| waves | 1.0 | 11 | 9 | 8 → 11 | 0 → 0 | 1 → 0 |
| obstacles | 0.8 | 18 | 19 | 1 → 1 | 0 → 0 | 1 → 0 |
| obstacles | 1.0 | 16 | 15 | 4 → 5 | 0 → 0 | 0 → 0 |
| stepping_stones | 0.8 | 6 | 15 | 6 → 0 | 0 → 0 | 2 → 1 |
| stepping_stones | 1.0 | 10 | 10 | 1 → 4 | 0 → 0 | 0 → 3 |

| Group | Blind one | Depth one | Blind → depth falls | Blind → depth six |
|---|---:|---:|---:|---:|
| 0.8 | 99/120 | 105/120 | 12 → 10 | 22 → 1 |
| 1.0 | 91/120 | 85/120 | 19 → 28 | 3 → 0 |
| 0.8+1.0 | 190/240 | 190/240 | 31 → 38 | 25 → 1 |

One-tile13.1m / six-tile53.1m; any terminal/lane/world exit invalidates strict success. See JSON for denominators/provenance/limitations.
