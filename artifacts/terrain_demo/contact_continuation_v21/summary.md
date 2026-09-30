# v21 matched no-cost continuation control

Matched no-cost continuation control with one new 32,768,000-transition arm; the two v20 cost arms are reused, not retrained. Historical two-iteration positive-control replays support compatibility, not identical full-run historical execution or statistical causality. Comparisons include no-cost versus parent, each cost arm versus no-cost and parent, and ramped versus immediate, with corresponding histories. Single seed/two maps; the separate 64s secondary cannot override the 16s primary. No automatic promotion or robot-safety claim.

## Primary: 16s mixed

### Pooled

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 272/300 | 169/300 | 27 | 0 | 0 | 2/0 | 11.1235 |
| no_cost | 276/300 | 164/300 | 18 | 5 | 0 | 3/0 | 10.4999 |
| immediate | 279/300 | 162/300 | 19 | 2 | 0 | 3/0 | 10.7125 |
| ramped | 280/300 | 165/300 | 16 | 5 | 0 | 3/0 | 10.8113 |
| history_parent | 274/300 | 175/300 | 21 | 5 | 0 | 5/0 | 8.9204 |
| history_no_cost | 282/300 | 168/300 | 16 | 1 | 0 | 5/0 | 8.9204 |
| history_immediate | 269/300 | 161/300 | 27 | 4 | 0 | 5/0 | 8.9204 |
| history_ramped | 276/300 | 161/300 | 23 | 1 | 0 | 5/0 | 8.9204 |

no_cost_vs_parent: **FAIL**; paired six gains/losses 15/20; failed checks: six_not_lower, lane_not_higher, flat_falls_not_higher, flat_speed_not_lower.

immediate_vs_no_cost: **FAIL**; paired six gains/losses 16/18; failed checks: six_not_lower, falls_not_higher.

ramped_vs_no_cost: **PASS**; paired six gains/losses 18/17; failed checks: none.

ramped_vs_immediate: **FAIL**; paired six gains/losses 16/13; failed checks: lane_not_higher.

history_no_cost_vs_history_parent: **FAIL**; paired six gains/losses 12/19; failed checks: six_not_lower.

history_immediate_vs_history_no_cost: **FAIL**; paired six gains/losses 14/21; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_ramped_vs_history_no_cost: **FAIL**; paired six gains/losses 11/18; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement.

history_ramped_vs_history_immediate: **PASS**; paired six gains/losses 20/20; failed checks: none.

immediate_vs_parent: **FAIL**; paired six gains/losses 9/16; failed checks: six_not_lower, lane_not_higher, flat_falls_not_higher, flat_speed_not_lower.

ramped_vs_parent: **FAIL**; paired six gains/losses 14/18; failed checks: six_not_lower, lane_not_higher, flat_falls_not_higher, flat_speed_not_lower.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 10/24; failed checks: one_not_lower, six_not_lower, falls_not_higher.

history_ramped_vs_history_parent: **FAIL**; paired six gains/losses 12/26; failed checks: six_not_lower, falls_not_higher.

### geometry113_reset73

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 134/150 | 85/150 | 15 | 0 | 0 | 1/0 | 11.0503 |
| no_cost | 136/150 | 81/150 | 11 | 3 | 0 | 1/0 | 10.6517 |
| immediate | 138/150 | 81/150 | 10 | 2 | 0 | 1/0 | 10.7645 |
| ramped | 138/150 | 79/150 | 9 | 4 | 0 | 2/0 | 10.7221 |
| history_parent | 134/150 | 87/150 | 13 | 3 | 0 | 3/0 | 8.7460 |
| history_no_cost | 137/150 | 80/150 | 11 | 1 | 0 | 3/0 | 8.7460 |
| history_immediate | 135/150 | 81/150 | 11 | 4 | 0 | 3/0 | 8.7460 |
| history_ramped | 137/150 | 80/150 | 12 | 1 | 0 | 3/0 | 8.7460 |

no_cost_vs_parent: **FAIL**; paired six gains/losses 7/11; failed checks: six_not_lower, lane_not_higher, flat_speed_not_lower.

immediate_vs_no_cost: **PASS**; paired six gains/losses 9/9; failed checks: none.

ramped_vs_no_cost: **FAIL**; paired six gains/losses 7/9; failed checks: six_not_lower, lane_not_higher, flat_falls_not_higher.

ramped_vs_immediate: **FAIL**; paired six gains/losses 6/8; failed checks: six_not_lower, lane_not_higher, flat_falls_not_higher, flat_speed_not_lower.

history_no_cost_vs_history_parent: **FAIL**; paired six gains/losses 4/11; failed checks: six_not_lower.

history_immediate_vs_history_no_cost: **FAIL**; paired six gains/losses 8/7; failed checks: one_not_lower, lane_not_higher.

history_ramped_vs_history_no_cost: **FAIL**; paired six gains/losses 6/6; failed checks: falls_not_higher, strict_rough_improvement.

history_ramped_vs_history_immediate: **FAIL**; paired six gains/losses 8/9; failed checks: six_not_lower, falls_not_higher.

immediate_vs_parent: **FAIL**; paired six gains/losses 5/9; failed checks: six_not_lower, lane_not_higher, flat_speed_not_lower.

ramped_vs_parent: **FAIL**; paired six gains/losses 6/12; failed checks: six_not_lower, lane_not_higher, flat_falls_not_higher, flat_speed_not_lower.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 5/11; failed checks: six_not_lower, lane_not_higher.

history_ramped_vs_history_parent: **FAIL**; paired six gains/losses 7/14; failed checks: six_not_lower.

### geometry114_reset74

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 138/150 | 84/150 | 12 | 0 | 0 | 1/0 | 11.1966 |
| no_cost | 140/150 | 83/150 | 7 | 2 | 0 | 2/0 | 10.3480 |
| immediate | 141/150 | 81/150 | 9 | 0 | 0 | 2/0 | 10.6606 |
| ramped | 142/150 | 86/150 | 7 | 1 | 0 | 1/0 | 10.9006 |
| history_parent | 140/150 | 88/150 | 8 | 2 | 0 | 2/0 | 9.0948 |
| history_no_cost | 145/150 | 88/150 | 5 | 0 | 0 | 2/0 | 9.0948 |
| history_immediate | 134/150 | 80/150 | 16 | 0 | 0 | 2/0 | 9.0948 |
| history_ramped | 139/150 | 81/150 | 11 | 0 | 0 | 2/0 | 9.0948 |

no_cost_vs_parent: **FAIL**; paired six gains/losses 8/9; failed checks: six_not_lower, lane_not_higher, flat_falls_not_higher, flat_speed_not_lower.

immediate_vs_no_cost: **FAIL**; paired six gains/losses 7/9; failed checks: six_not_lower, falls_not_higher.

ramped_vs_no_cost: **PASS**; paired six gains/losses 11/8; failed checks: none.

ramped_vs_immediate: **FAIL**; paired six gains/losses 10/5; failed checks: lane_not_higher.

history_no_cost_vs_history_parent: **PASS**; paired six gains/losses 8/8; failed checks: none.

history_immediate_vs_history_no_cost: **FAIL**; paired six gains/losses 6/14; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement.

history_ramped_vs_history_no_cost: **FAIL**; paired six gains/losses 5/12; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement.

history_ramped_vs_history_immediate: **PASS**; paired six gains/losses 12/11; failed checks: none.

immediate_vs_parent: **FAIL**; paired six gains/losses 4/7; failed checks: six_not_lower, flat_falls_not_higher, flat_speed_not_lower.

ramped_vs_parent: **FAIL**; paired six gains/losses 8/6; failed checks: lane_not_higher, flat_speed_not_lower.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 5/13; failed checks: one_not_lower, six_not_lower, falls_not_higher.

history_ramped_vs_history_parent: **FAIL**; paired six gains/losses 5/12; failed checks: one_not_lower, six_not_lower, falls_not_higher.

## Secondary: 64s stones

### Pooled

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 13/20 | 12/20 | 5 | 2 | 0 | 0/0 | — |
| no_cost | 8/20 | 8/20 | 8 | 4 | 0 | 0/0 | — |
| immediate | 5/20 | 4/20 | 13 | 3 | 0 | 0/0 | — |
| ramped | 8/20 | 8/20 | 8 | 5 | 0 | 0/0 | — |
| history_parent | 12/20 | 8/20 | 6 | 2 | 0 | 0/0 | — |
| history_no_cost | 9/20 | 7/20 | 9 | 3 | 0 | 0/0 | — |
| history_immediate | 7/20 | 6/20 | 11 | 4 | 0 | 0/0 | — |
| history_ramped | 8/20 | 6/20 | 8 | 4 | 0 | 0/0 | — |

no_cost_vs_parent: **FAIL**; paired six gains/losses 1/5; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

immediate_vs_no_cost: **FAIL**; paired six gains/losses 3/7; failed checks: one_not_lower, six_not_lower, falls_not_higher.

ramped_vs_no_cost: **FAIL**; paired six gains/losses 4/4; failed checks: lane_not_higher, strict_rough_improvement.

ramped_vs_immediate: **FAIL**; paired six gains/losses 6/2; failed checks: lane_not_higher.

history_no_cost_vs_history_parent: **FAIL**; paired six gains/losses 3/4; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_immediate_vs_history_no_cost: **FAIL**; paired six gains/losses 5/6; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_ramped_vs_history_no_cost: **FAIL**; paired six gains/losses 4/5; failed checks: one_not_lower, six_not_lower, lane_not_higher.

history_ramped_vs_history_immediate: **PASS**; paired six gains/losses 5/5; failed checks: none.

immediate_vs_parent: **FAIL**; paired six gains/losses 2/10; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

ramped_vs_parent: **FAIL**; paired six gains/losses 1/5; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 3/5; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_ramped_vs_history_parent: **FAIL**; paired six gains/losses 5/7; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

### geometry113_reset73

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 7/10 | 6/10 | 3 | 0 | 0 | 0/0 | — |
| no_cost | 5/10 | 5/10 | 4 | 1 | 0 | 0/0 | — |
| immediate | 1/10 | 1/10 | 8 | 1 | 0 | 0/0 | — |
| ramped | 4/10 | 4/10 | 3 | 4 | 0 | 0/0 | — |
| history_parent | 3/10 | 2/10 | 5 | 2 | 0 | 0/0 | — |
| history_no_cost | 5/10 | 3/10 | 5 | 0 | 0 | 0/0 | — |
| history_immediate | 3/10 | 3/10 | 5 | 3 | 0 | 0/0 | — |
| history_ramped | 5/10 | 3/10 | 4 | 2 | 0 | 0/0 | — |

no_cost_vs_parent: **FAIL**; paired six gains/losses 1/2; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

immediate_vs_no_cost: **FAIL**; paired six gains/losses 1/5; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement.

ramped_vs_no_cost: **FAIL**; paired six gains/losses 1/2; failed checks: one_not_lower, six_not_lower, lane_not_higher.

ramped_vs_immediate: **FAIL**; paired six gains/losses 4/1; failed checks: lane_not_higher.

history_no_cost_vs_history_parent: **PASS**; paired six gains/losses 3/2; failed checks: none.

history_immediate_vs_history_no_cost: **FAIL**; paired six gains/losses 3/3; failed checks: one_not_lower, lane_not_higher, strict_rough_improvement.

history_ramped_vs_history_no_cost: **FAIL**; paired six gains/losses 2/2; failed checks: lane_not_higher.

history_ramped_vs_history_immediate: **PASS**; paired six gains/losses 3/3; failed checks: none.

immediate_vs_parent: **FAIL**; paired six gains/losses 1/6; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

ramped_vs_parent: **FAIL**; paired six gains/losses 0/2; failed checks: one_not_lower, six_not_lower, lane_not_higher, strict_rough_improvement.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 1/0; failed checks: lane_not_higher.

history_ramped_vs_history_parent: **PASS**; paired six gains/losses 3/2; failed checks: none.

### geometry114_reset74

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 6/10 | 6/10 | 2 | 2 | 0 | 0/0 | — |
| no_cost | 3/10 | 3/10 | 4 | 3 | 0 | 0/0 | — |
| immediate | 4/10 | 3/10 | 5 | 2 | 0 | 0/0 | — |
| ramped | 4/10 | 4/10 | 5 | 1 | 0 | 0/0 | — |
| history_parent | 9/10 | 6/10 | 1 | 0 | 0 | 0/0 | — |
| history_no_cost | 4/10 | 4/10 | 4 | 3 | 0 | 0/0 | — |
| history_immediate | 4/10 | 3/10 | 6 | 1 | 0 | 0/0 | — |
| history_ramped | 3/10 | 3/10 | 4 | 2 | 0 | 0/0 | — |

no_cost_vs_parent: **FAIL**; paired six gains/losses 0/3; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

immediate_vs_no_cost: **FAIL**; paired six gains/losses 2/2; failed checks: falls_not_higher.

ramped_vs_no_cost: **FAIL**; paired six gains/losses 3/2; failed checks: falls_not_higher.

ramped_vs_immediate: **PASS**; paired six gains/losses 2/1; failed checks: none.

history_no_cost_vs_history_parent: **FAIL**; paired six gains/losses 0/2; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_immediate_vs_history_no_cost: **FAIL**; paired six gains/losses 2/3; failed checks: six_not_lower, falls_not_higher.

history_ramped_vs_history_no_cost: **FAIL**; paired six gains/losses 2/3; failed checks: one_not_lower, six_not_lower.

history_ramped_vs_history_immediate: **FAIL**; paired six gains/losses 2/2; failed checks: one_not_lower, lane_not_higher.

immediate_vs_parent: **FAIL**; paired six gains/losses 1/4; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement.

ramped_vs_parent: **FAIL**; paired six gains/losses 1/3; failed checks: one_not_lower, six_not_lower, falls_not_higher.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 2/5; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_ramped_vs_history_parent: **FAIL**; paired six gains/losses 2/5; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

See summary.json for family/level, contact/posture telemetry, paired outcomes and raw-file provenance.
