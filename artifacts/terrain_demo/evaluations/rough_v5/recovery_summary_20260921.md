Cells: fall rate / mean forward speed (m/s); 3 available lane seed(s)

| Policy | flat | obstacles | rough | slope | stairs | stepping_stones | waves | terrain mean |
|---|---|---|---|---|---|---|---|---|
| selected_traverse1499 | 7% / 9.01 | 5% / 3.17 | 7% / 3.58 | 4% / 3.32 | 5% / 2.67 | 7% / 1.30 | 15% / 3.59 | 7% / 2.94 |
| selected_recovery_deep1000 | 13% / 10.16 | 19% / 2.95 | 5% / 3.23 | 8% / 3.00 | 20% / 2.49 | 13% / 1.60 | 24% / 3.35 | 15% / 2.77 |

Terrain traversal success (successful episodes / applicable episodes; distinct from survival):

| Policy | Family | one-tile clearance | full six-tile clearance | lane loops | lane exits | world exits |
|---|---|---:|---:|---:|---:|---:|
| selected_traverse1499 | obstacles | 70/75 (93.3%) | 43/75 (57.3%) | 37 | 1 | 0 |
| selected_traverse1499 | rough | 69/75 (92.0%) | 51/75 (68.0%) | 46 | 1 | 0 |
| selected_traverse1499 | slope | 72/75 (96.0%) | 37/75 (49.3%) | 31 | 0 | 0 |
| selected_traverse1499 | stairs | 70/75 (93.3%) | 21/75 (28.0%) | 16 | 1 | 0 |
| selected_traverse1499 | stepping_stones | 24/75 (32.0%) | 0/75 (0.0%) | 0 | 5 | 0 |
| selected_traverse1499 | waves | 64/75 (85.3%) | 45/75 (60.0%) | 41 | 0 | 0 |
| selected_recovery_deep1000 | obstacles | 57/75 (76.0%) | 23/75 (30.7%) | 18 | 4 | 0 |
| selected_recovery_deep1000 | rough | 71/75 (94.7%) | 37/75 (49.3%) | 31 | 0 | 0 |
| selected_recovery_deep1000 | slope | 68/75 (90.7%) | 25/75 (33.3%) | 25 | 1 | 0 |
| selected_recovery_deep1000 | stairs | 58/75 (77.3%) | 7/75 (9.3%) | 5 | 2 | 0 |
| selected_recovery_deep1000 | stepping_stones | 55/75 (73.3%) | 0/75 (0.0%) | 0 | 1 | 0 |
| selected_recovery_deep1000 | waves | 57/75 (76.0%) | 41/75 (54.7%) | 30 | 0 | 0 |

| Policy | lanes 100-env | ID | low friction | heavy | push |
|---|---:|---:|---:|---:|---:|
| selected_traverse1499 | 57.68 ± 33.70 | 134.80 ± 29.67 | 157.72 ± 28.36 | 142.79 ± 37.24 | 132.28 ± 34.53 |
| selected_recovery_deep1000 | 46.32 ± 42.81 | 152.27 ± 33.54 | 153.65 ± 36.15 | 148.50 ± 35.37 | 151.27 ± 33.22 |

selected_traverse1499: lane return per seed [55.39, 56.88, 59.2] (seed-mean std 1.57), overall fall rate 7.0%

selected_recovery_deep1000: lane return per seed [49.11, 49.48, 50.75] (seed-mean std 0.70), overall fall rate 14.7%
