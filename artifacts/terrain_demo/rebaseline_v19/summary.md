# v19 frozen-policy common-map rebaseline

Descriptive frozen-policy rebaseline on three paired maps, not statistical superiority or a causal reward ablation. Single-seed historical checkpoints and simulator-only sensors limit generalization. The separate 64s secondary cannot override the 16s primary. No automatic promotion; zero new training transitions.

## Primary: 16s mixed

### Pooled

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 375/450 | 209/450 | 49 | 5 | 0 | 3/0 | 9.4624 |
| history_original | 402/450 | 225/450 | 41 | 4 | 0 | 3/0 | 9.4624 |
| v16_control | 415/450 | 265/450 | 31 | 4 | 0 | 2/0 | 11.1483 |
| history_control | 407/450 | 265/450 | 39 | 2 | 0 | 3/0 | 9.4624 |

v16_control_vs_v5: **PASS**; paired six gains/losses 62/6; failed checks: none.

history_control_vs_history_original: **PASS**; paired six gains/losses 47/7; failed checks: none.

### geometry107_reset68

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 124/150 | 67/150 | 16 | 2 | 0 | 2/0 | 9.3865 |
| history_original | 130/150 | 72/150 | 17 | 1 | 0 | 2/0 | 9.3865 |
| v16_control | 139/150 | 91/150 | 8 | 3 | 0 | 0/0 | 11.4685 |
| history_control | 132/150 | 89/150 | 16 | 1 | 0 | 2/0 | 9.3865 |

v16_control_vs_v5: **FAIL**; paired six gains/losses 25/1; failed checks: lane_not_higher.

history_control_vs_history_original: **PASS**; paired six gains/losses 18/1; failed checks: none.

### geometry108_reset69

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 128/150 | 70/150 | 14 | 0 | 0 | 1/0 | 9.4336 |
| history_original | 138/150 | 76/150 | 11 | 0 | 0 | 1/0 | 9.4336 |
| v16_control | 139/150 | 88/150 | 11 | 0 | 0 | 2/0 | 10.6213 |
| history_control | 137/150 | 90/150 | 13 | 0 | 0 | 1/0 | 9.4336 |

v16_control_vs_v5: **FAIL**; paired six gains/losses 19/1; failed checks: flat_falls_not_higher.

history_control_vs_history_original: **FAIL**; paired six gains/losses 18/4; failed checks: one_not_lower, falls_not_higher.

### geometry109_reset70

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 123/150 | 72/150 | 19 | 3 | 0 | 0/0 | 9.5672 |
| history_original | 134/150 | 77/150 | 13 | 3 | 0 | 0/0 | 9.5672 |
| v16_control | 137/150 | 86/150 | 12 | 1 | 0 | 0/0 | 11.3551 |
| history_control | 138/150 | 86/150 | 10 | 1 | 0 | 0/0 | 9.5672 |

v16_control_vs_v5: **PASS**; paired six gains/losses 18/4; failed checks: none.

history_control_vs_history_original: **PASS**; paired six gains/losses 11/2; failed checks: none.

## Secondary: 64s stones

### Pooled

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 12/30 | 0/30 | 13 | 0 | 0 | 0/0 | — |
| history_original | 16/30 | 13/30 | 11 | 3 | 0 | 0/0 | — |
| v16_control | 16/30 | 16/30 | 12 | 4 | 0 | 0/0 | — |
| history_control | 16/30 | 14/30 | 11 | 3 | 0 | 0/0 | — |

v16_control_vs_v5: **FAIL**; paired six gains/losses 16/0; failed checks: lane_not_higher.

history_control_vs_history_original: **PASS**; paired six gains/losses 8/7; failed checks: none.

### geometry107_reset68

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 4/10 | 0/10 | 4 | 0 | 0 | 0/0 | — |
| history_original | 8/10 | 6/10 | 2 | 0 | 0 | 0/0 | — |
| v16_control | 6/10 | 6/10 | 3 | 2 | 0 | 0/0 | — |
| history_control | 8/10 | 7/10 | 0 | 2 | 0 | 0/0 | — |

v16_control_vs_v5: **FAIL**; paired six gains/losses 6/0; failed checks: lane_not_higher.

history_control_vs_history_original: **FAIL**; paired six gains/losses 3/2; failed checks: lane_not_higher.

### geometry108_reset69

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 4/10 | 0/10 | 4 | 0 | 0 | 0/0 | — |
| history_original | 4/10 | 3/10 | 5 | 1 | 0 | 0/0 | — |
| v16_control | 6/10 | 6/10 | 4 | 1 | 0 | 0/0 | — |
| history_control | 4/10 | 3/10 | 6 | 0 | 0 | 0/0 | — |

v16_control_vs_v5: **FAIL**; paired six gains/losses 6/0; failed checks: lane_not_higher.

history_control_vs_history_original: **FAIL**; paired six gains/losses 3/3; failed checks: falls_not_higher.

### geometry109_reset70

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 4/10 | 0/10 | 5 | 0 | 0 | 0/0 | — |
| history_original | 4/10 | 4/10 | 4 | 2 | 0 | 0/0 | — |
| v16_control | 4/10 | 4/10 | 5 | 1 | 0 | 0/0 | — |
| history_control | 4/10 | 4/10 | 5 | 1 | 0 | 0/0 | — |

v16_control_vs_v5: **FAIL**; paired six gains/losses 4/0; failed checks: lane_not_higher.

history_control_vs_history_original: **FAIL**; paired six gains/losses 2/2; failed checks: falls_not_higher.

See summary.json for family/level, contact/posture telemetry, paired outcomes and raw-file provenance.
