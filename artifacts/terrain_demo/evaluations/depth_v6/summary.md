# v6 frozen paired evaluation

| Set | Policy | Strict one-tile | Six-tile | Terrain falls | Stones | Stone six-tile | Lane exits | Flat falls | Terrain speed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | blind_reference | 386/450 | 201 | 45 | 49/75 | 0 | 2 | 2/75 | 3.007 |
| fixed | selected | 401/450 | 84 | 34 | 61/75 | 0 | 9 | 3/75 | 2.617 |
| fresh | blind_reference | 524/600 | 274 | 46 | 62/100 | 2 | 5 | 11/100 | 3.027 |
| fresh | selected | 521/600 | 119 | 62 | 79/100 | 0 | 8 | 10/100 | 2.633 |
| ablations_selection_seed24 | actual | 135/150 | 28 | 12 | 22/25 | 0 | 2 | 1/25 | 2.567 |
| ablations_selection_seed24 | zero | 86/150 | 11 | 33 | 14/25 | 0 | 27 | 5/25 | 2.317 |
| ablations_selection_seed24 | shuffle | 106/150 | 2 | 35 | 14/25 | 0 | 4 | 1/25 | 2.209 |

Fresh gate: **FAIL**.

Fresh post-selection gate, not a matched-training causal depth experiment. Ablations are distribution shifts and establish reliance, not isolated causality. Repeated rollouts from one training seed do not measure training-seed uncertainty.
