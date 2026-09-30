# v20 contact-cost curriculum: immediate versus ramped

Descriptive single-seed paired reward-curriculum experiment on two maps, not statistical superiority. The arms differ in coefficient timing AND coefficient-step mass; realized penalty depends on visited states. Both arms preserve the same raw contact reward, parent, sensors and inference. Comparisons against the frozen parent are required: beating a degraded immediate arm is insufficient. The separate 64s secondary cannot override the 16s primary. No automatic promotion or physical-robot safety claim.

## Primary: 16s mixed

### Pooled

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 275/300 | 178/300 | 23 | 2 | 0 | 3/0 | 10.8225 |
| immediate | 273/300 | 158/300 | 26 | 1 | 0 | 2/0 | 10.7322 |
| ramped | 273/300 | 164/300 | 21 | 6 | 0 | 3/0 | 10.7974 |
| history_parent | 279/300 | 172/300 | 19 | 1 | 0 | 2/0 | 9.3507 |
| history_immediate | 272/300 | 167/300 | 27 | 1 | 0 | 2/0 | 9.3507 |
| history_ramped | 276/300 | 172/300 | 22 | 2 | 0 | 2/0 | 9.3507 |

ramped_vs_immediate: **FAIL**; paired six gains/losses 26/20; failed checks: lane_not_higher, flat_falls_not_higher.

history_ramped_vs_history_immediate: **FAIL**; paired six gains/losses 17/12; failed checks: lane_not_higher.

immediate_vs_parent: **FAIL**; paired six gains/losses 9/29; failed checks: one_not_lower, six_not_lower, falls_not_higher, flat_speed_not_lower.

ramped_vs_parent: **FAIL**; paired six gains/losses 13/27; failed checks: one_not_lower, six_not_lower, lane_not_higher, flat_speed_not_lower.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 15/20; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement.

history_ramped_vs_history_parent: **FAIL**; paired six gains/losses 18/18; failed checks: one_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

### geometry111_reset71

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 141/150 | 93/150 | 7 | 2 | 0 | 2/0 | 10.6679 |
| immediate | 134/150 | 80/150 | 16 | 0 | 0 | 1/0 | 10.6173 |
| ramped | 141/150 | 83/150 | 8 | 1 | 0 | 2/0 | 10.7458 |
| history_parent | 139/150 | 84/150 | 11 | 0 | 0 | 1/0 | 9.3059 |
| history_immediate | 137/150 | 85/150 | 12 | 1 | 0 | 1/0 | 9.3059 |
| history_ramped | 140/150 | 87/150 | 10 | 1 | 0 | 1/0 | 9.3059 |

ramped_vs_immediate: **FAIL**; paired six gains/losses 12/9; failed checks: lane_not_higher, flat_falls_not_higher.

history_ramped_vs_history_immediate: **PASS**; paired six gains/losses 9/7; failed checks: none.

immediate_vs_parent: **FAIL**; paired six gains/losses 5/18; failed checks: one_not_lower, six_not_lower, falls_not_higher, flat_speed_not_lower.

ramped_vs_parent: **FAIL**; paired six gains/losses 4/14; failed checks: six_not_lower, falls_not_higher.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 11/10; failed checks: one_not_lower, falls_not_higher, lane_not_higher.

history_ramped_vs_history_parent: **FAIL**; paired six gains/losses 11/8; failed checks: lane_not_higher.

### geometry112_reset72

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 134/150 | 85/150 | 16 | 0 | 0 | 1/0 | 10.9771 |
| immediate | 139/150 | 78/150 | 10 | 1 | 0 | 1/0 | 10.8471 |
| ramped | 132/150 | 81/150 | 13 | 5 | 0 | 1/0 | 10.8490 |
| history_parent | 140/150 | 88/150 | 8 | 1 | 0 | 1/0 | 9.3954 |
| history_immediate | 135/150 | 82/150 | 15 | 0 | 0 | 1/0 | 9.3954 |
| history_ramped | 136/150 | 85/150 | 12 | 1 | 0 | 1/0 | 9.3954 |

ramped_vs_immediate: **FAIL**; paired six gains/losses 14/11; failed checks: one_not_lower, falls_not_higher, lane_not_higher.

history_ramped_vs_history_immediate: **FAIL**; paired six gains/losses 8/5; failed checks: lane_not_higher.

immediate_vs_parent: **FAIL**; paired six gains/losses 4/11; failed checks: six_not_lower, lane_not_higher, flat_speed_not_lower.

ramped_vs_parent: **FAIL**; paired six gains/losses 9/13; failed checks: one_not_lower, six_not_lower, lane_not_higher, flat_speed_not_lower.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 4/10; failed checks: one_not_lower, six_not_lower, falls_not_higher.

history_ramped_vs_history_parent: **FAIL**; paired six gains/losses 7/10; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement.

## Secondary: 64s stones

### Pooled

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 8/20 | 8/20 | 11 | 1 | 0 | 0/0 | — |
| immediate | 8/20 | 7/20 | 9 | 3 | 0 | 0/0 | — |
| ramped | 5/20 | 5/20 | 10 | 6 | 0 | 0/0 | — |
| history_parent | 8/20 | 7/20 | 11 | 3 | 0 | 0/0 | — |
| history_immediate | 8/20 | 5/20 | 12 | 1 | 0 | 0/0 | — |
| history_ramped | 5/20 | 5/20 | 11 | 9 | 0 | 0/0 | — |

ramped_vs_immediate: **FAIL**; paired six gains/losses 4/6; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_ramped_vs_history_immediate: **FAIL**; paired six gains/losses 4/4; failed checks: one_not_lower, lane_not_higher.

immediate_vs_parent: **FAIL**; paired six gains/losses 4/5; failed checks: six_not_lower, lane_not_higher.

ramped_vs_parent: **FAIL**; paired six gains/losses 2/5; failed checks: one_not_lower, six_not_lower, lane_not_higher.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 4/6; failed checks: six_not_lower, falls_not_higher.

history_ramped_vs_history_parent: **FAIL**; paired six gains/losses 5/7; failed checks: one_not_lower, six_not_lower, lane_not_higher, strict_rough_improvement.

### geometry111_reset71

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 4/10 | 4/10 | 6 | 0 | 0 | 0/0 | — |
| immediate | 4/10 | 4/10 | 3 | 3 | 0 | 0/0 | — |
| ramped | 4/10 | 4/10 | 3 | 3 | 0 | 0/0 | — |
| history_parent | 4/10 | 3/10 | 5 | 3 | 0 | 0/0 | — |
| history_immediate | 4/10 | 3/10 | 6 | 0 | 0 | 0/0 | — |
| history_ramped | 3/10 | 3/10 | 5 | 4 | 0 | 0/0 | — |

ramped_vs_immediate: **FAIL**; paired six gains/losses 3/3; failed checks: strict_rough_improvement.

history_ramped_vs_history_immediate: **FAIL**; paired six gains/losses 2/2; failed checks: one_not_lower, lane_not_higher.

immediate_vs_parent: **FAIL**; paired six gains/losses 3/3; failed checks: lane_not_higher.

ramped_vs_parent: **FAIL**; paired six gains/losses 2/2; failed checks: lane_not_higher.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 2/2; failed checks: falls_not_higher.

history_ramped_vs_history_parent: **FAIL**; paired six gains/losses 3/3; failed checks: one_not_lower, lane_not_higher, strict_rough_improvement.

### geometry112_reset72

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 4/10 | 4/10 | 5 | 1 | 0 | 0/0 | — |
| immediate | 4/10 | 3/10 | 6 | 0 | 0 | 0/0 | — |
| ramped | 1/10 | 1/10 | 7 | 3 | 0 | 0/0 | — |
| history_parent | 4/10 | 4/10 | 6 | 0 | 0 | 0/0 | — |
| history_immediate | 4/10 | 2/10 | 6 | 1 | 0 | 0/0 | — |
| history_ramped | 2/10 | 2/10 | 6 | 5 | 0 | 0/0 | — |

ramped_vs_immediate: **FAIL**; paired six gains/losses 1/3; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_ramped_vs_history_immediate: **FAIL**; paired six gains/losses 2/2; failed checks: one_not_lower, lane_not_higher, strict_rough_improvement.

immediate_vs_parent: **FAIL**; paired six gains/losses 1/2; failed checks: six_not_lower, falls_not_higher.

ramped_vs_parent: **FAIL**; paired six gains/losses 0/3; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_immediate_vs_history_parent: **FAIL**; paired six gains/losses 2/4; failed checks: six_not_lower, lane_not_higher, strict_rough_improvement.

history_ramped_vs_history_parent: **FAIL**; paired six gains/losses 2/4; failed checks: one_not_lower, six_not_lower, lane_not_higher, strict_rough_improvement.

See summary.json for family/level, contact/posture telemetry, paired outcomes and raw-file provenance.
