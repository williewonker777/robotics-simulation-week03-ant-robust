# v22 no-cost continuation: training-seed sensitivity

Three fresh continuation training seeds, conditional on one fixed parent and training terrain; no new network initialization or reward tuning. Fresh seed51 exactly replays historical v21 recorded evidence and endpoint; it is not an independent fourth realization. Parent is evaluated once per map/horizon and is never pooled three times. Per-seed extrema/ranges are descriptive, not significance, confidence intervals or general algorithm robustness from three seeds. All per-map failures remain visible; 64s secondary cannot override 16s primary. No automatic promotion.

## Primary: 16s mixed

### Pooled

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 274/300 | 178/300 | 21 | 4 | 0 | 3/0 | 10.7931 |
| seed51 | 284/300 | 165/300 | 12 | 4 | 0 | 4/0 | 10.3866 |
| seed52 | 276/300 | 164/300 | 18 | 5 | 0 | 3/0 | 10.5814 |
| seed53 | 273/300 | 171/300 | 21 | 5 | 0 | 3/0 | 10.7561 |
| history_parent | 277/300 | 177/300 | 21 | 2 | 0 | 4/0 | 9.0406 |
| history_seed51 | 277/300 | 172/300 | 23 | 0 | 0 | 4/0 | 9.0406 |
| history_seed52 | 276/300 | 168/300 | 21 | 1 | 0 | 4/0 | 9.0406 |
| history_seed53 | 277/300 | 169/300 | 21 | 2 | 0 | 4/0 | 9.0406 |

seed51_vs_parent: **FAIL**; paired six gains/losses 15/28; failed checks: six_not_lower, flat_falls_not_higher, flat_speed_not_lower.

seed52_vs_parent: **FAIL**; paired six gains/losses 15/29; failed checks: six_not_lower, lane_not_higher, flat_speed_not_lower.

seed53_vs_parent: **FAIL**; paired six gains/losses 18/25; failed checks: one_not_lower, six_not_lower, lane_not_higher, strict_rough_improvement, flat_speed_not_lower.

history_seed51_vs_history_parent: **FAIL**; paired six gains/losses 14/19; failed checks: six_not_lower, falls_not_higher.

history_seed52_vs_history_parent: **FAIL**; paired six gains/losses 12/21; failed checks: one_not_lower, six_not_lower.

history_seed53_vs_history_parent: **FAIL**; paired six gains/losses 11/19; failed checks: six_not_lower, strict_rough_improvement.

seed52_vs_seed51: **FAIL**; paired six gains/losses 17/18; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

seed53_vs_seed51: **FAIL**; paired six gains/losses 26/20; failed checks: one_not_lower, falls_not_higher, lane_not_higher.

seed53_vs_seed52: **FAIL**; paired six gains/losses 26/19; failed checks: one_not_lower, falls_not_higher.

history_seed52_vs_history_seed51: **FAIL**; paired six gains/losses 13/17; failed checks: one_not_lower, six_not_lower, lane_not_higher.

history_seed53_vs_history_seed51: **FAIL**; paired six gains/losses 10/13; failed checks: six_not_lower, lane_not_higher.

history_seed53_vs_history_seed52: **FAIL**; paired six gains/losses 13/12; failed checks: lane_not_higher.
actor one: three-seed min/max 273/284, range 11; parent 274 (counted once).
actor six: three-seed min/max 164/171, range 7; parent 178 (counted once).
actor falls: three-seed min/max 12/21, range 9; parent 21 (counted once).
actor lane: three-seed min/max 4/5, range 1; parent 4 (counted once).
actor flat_mean_episode_speed: three-seed min/max 10.386561755536597/10.756123099002014, range 0.369561343465417; parent 10.793139490037523 (counted once).
history one: three-seed min/max 276/277, range 1; parent 277 (counted once).
history six: three-seed min/max 168/172, range 4; parent 177 (counted once).
history falls: three-seed min/max 21/23, range 2; parent 21 (counted once).
history lane: three-seed min/max 0/2, range 2; parent 2 (counted once).
history flat_mean_episode_speed: three-seed min/max 9.040611622030369/9.040611622030369, range 0.0; parent 9.040611622030369 (counted once).

### geometry115_reset75

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 141/150 | 87/150 | 8 | 1 | 0 | 2/0 | 10.6666 |
| seed51 | 144/150 | 81/150 | 5 | 1 | 0 | 2/0 | 10.0772 |
| seed52 | 141/150 | 80/150 | 7 | 2 | 0 | 2/0 | 10.3256 |
| seed53 | 133/150 | 85/150 | 13 | 3 | 0 | 2/0 | 10.7851 |
| history_parent | 141/150 | 89/150 | 9 | 0 | 0 | 2/0 | 9.0986 |
| history_seed51 | 137/150 | 85/150 | 13 | 0 | 0 | 2/0 | 9.0986 |
| history_seed52 | 139/150 | 81/150 | 10 | 0 | 0 | 2/0 | 9.0986 |
| history_seed53 | 142/150 | 85/150 | 8 | 0 | 0 | 2/0 | 9.0986 |

seed51_vs_parent: **FAIL**; paired six gains/losses 7/13; failed checks: six_not_lower, flat_speed_not_lower.

seed52_vs_parent: **FAIL**; paired six gains/losses 6/13; failed checks: six_not_lower, lane_not_higher, flat_speed_not_lower.

seed53_vs_parent: **FAIL**; paired six gains/losses 9/11; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_seed51_vs_history_parent: **FAIL**; paired six gains/losses 4/8; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement.

history_seed52_vs_history_parent: **FAIL**; paired six gains/losses 4/12; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement.

history_seed53_vs_history_parent: **FAIL**; paired six gains/losses 4/8; failed checks: six_not_lower.

seed52_vs_seed51: **FAIL**; paired six gains/losses 6/7; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

seed53_vs_seed51: **FAIL**; paired six gains/losses 15/11; failed checks: one_not_lower, falls_not_higher, lane_not_higher.

seed53_vs_seed52: **FAIL**; paired six gains/losses 13/8; failed checks: one_not_lower, falls_not_higher, lane_not_higher.

history_seed52_vs_history_seed51: **FAIL**; paired six gains/losses 5/9; failed checks: six_not_lower.

history_seed53_vs_history_seed51: **PASS**; paired six gains/losses 6/6; failed checks: none.

history_seed53_vs_history_seed52: **PASS**; paired six gains/losses 7/3; failed checks: none.
actor one: three-seed min/max 133/144, range 11; parent 141 (counted once).
actor six: three-seed min/max 80/85, range 5; parent 87 (counted once).
actor falls: three-seed min/max 5/13, range 8; parent 8 (counted once).
actor lane: three-seed min/max 1/3, range 2; parent 1 (counted once).
actor flat_mean_episode_speed: three-seed min/max 10.077189967370359/10.78514102800224, range 0.7079510606318813; parent 10.66659790259473 (counted once).
history one: three-seed min/max 137/142, range 5; parent 141 (counted once).
history six: three-seed min/max 81/85, range 4; parent 89 (counted once).
history falls: three-seed min/max 8/13, range 5; parent 9 (counted once).
history lane: three-seed min/max 0/0, range 0; parent 0 (counted once).
history flat_mean_episode_speed: three-seed min/max 9.098635586019109/9.098635586019109, range 0.0; parent 9.098635586019109 (counted once).

### geometry116_reset76

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 133/150 | 91/150 | 13 | 3 | 0 | 1/0 | 10.9197 |
| seed51 | 140/150 | 84/150 | 7 | 3 | 0 | 2/0 | 10.6959 |
| seed52 | 135/150 | 84/150 | 11 | 3 | 0 | 1/0 | 10.8372 |
| seed53 | 140/150 | 86/150 | 8 | 2 | 0 | 1/0 | 10.7271 |
| history_parent | 136/150 | 88/150 | 12 | 2 | 0 | 2/0 | 8.9826 |
| history_seed51 | 140/150 | 87/150 | 10 | 0 | 0 | 2/0 | 8.9826 |
| history_seed52 | 137/150 | 87/150 | 11 | 1 | 0 | 2/0 | 8.9826 |
| history_seed53 | 135/150 | 84/150 | 13 | 2 | 0 | 2/0 | 8.9826 |

seed51_vs_parent: **FAIL**; paired six gains/losses 8/15; failed checks: six_not_lower, flat_falls_not_higher, flat_speed_not_lower.

seed52_vs_parent: **FAIL**; paired six gains/losses 9/16; failed checks: six_not_lower, flat_speed_not_lower.

seed53_vs_parent: **FAIL**; paired six gains/losses 9/14; failed checks: six_not_lower, flat_speed_not_lower.

history_seed51_vs_history_parent: **FAIL**; paired six gains/losses 10/11; failed checks: six_not_lower.

history_seed52_vs_history_parent: **FAIL**; paired six gains/losses 8/9; failed checks: six_not_lower.

history_seed53_vs_history_parent: **FAIL**; paired six gains/losses 7/11; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement.

seed52_vs_seed51: **FAIL**; paired six gains/losses 11/11; failed checks: one_not_lower, falls_not_higher, strict_rough_improvement.

seed53_vs_seed51: **FAIL**; paired six gains/losses 11/9; failed checks: falls_not_higher.

seed53_vs_seed52: **FAIL**; paired six gains/losses 13/11; failed checks: flat_speed_not_lower.

history_seed52_vs_history_seed51: **FAIL**; paired six gains/losses 8/8; failed checks: one_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_seed53_vs_history_seed51: **FAIL**; paired six gains/losses 4/7; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_seed53_vs_history_seed52: **FAIL**; paired six gains/losses 6/9; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.
actor one: three-seed min/max 135/140, range 5; parent 133 (counted once).
actor six: three-seed min/max 84/86, range 2; parent 91 (counted once).
actor falls: three-seed min/max 7/11, range 4; parent 13 (counted once).
actor lane: three-seed min/max 2/3, range 1; parent 3 (counted once).
actor flat_mean_episode_speed: three-seed min/max 10.695933543702838/10.8372241785947, range 0.14129063489186322; parent 10.919681077480316 (counted once).
history one: three-seed min/max 135/140, range 5; parent 136 (counted once).
history six: three-seed min/max 84/87, range 3; parent 88 (counted once).
history falls: three-seed min/max 10/13, range 3; parent 12 (counted once).
history lane: three-seed min/max 0/2, range 2; parent 2 (counted once).
history flat_mean_episode_speed: three-seed min/max 8.982587658041629/8.982587658041629, range 0.0; parent 8.982587658041629 (counted once).

## Secondary: 64s stones

### Pooled

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 7/20 | 7/20 | 10 | 4 | 0 | 0/0 | — |
| seed51 | 8/20 | 8/20 | 7 | 5 | 0 | 0/0 | — |
| seed52 | 17/20 | 17/20 | 2 | 1 | 0 | 0/0 | — |
| seed53 | 9/20 | 9/20 | 9 | 3 | 0 | 0/0 | — |
| history_parent | 7/20 | 7/20 | 8 | 6 | 0 | 0/0 | — |
| history_seed51 | 8/20 | 8/20 | 10 | 3 | 0 | 0/0 | — |
| history_seed52 | 5/20 | 4/20 | 7 | 8 | 0 | 0/0 | — |
| history_seed53 | 13/20 | 12/20 | 6 | 1 | 0 | 0/0 | — |

seed51_vs_parent: **FAIL**; paired six gains/losses 6/5; failed checks: lane_not_higher.

seed52_vs_parent: **PASS**; paired six gains/losses 10/0; failed checks: none.

seed53_vs_parent: **PASS**; paired six gains/losses 7/5; failed checks: none.

history_seed51_vs_history_parent: **FAIL**; paired six gains/losses 4/3; failed checks: falls_not_higher.

history_seed52_vs_history_parent: **FAIL**; paired six gains/losses 4/7; failed checks: one_not_lower, six_not_lower, lane_not_higher.

history_seed53_vs_history_parent: **PASS**; paired six gains/losses 6/1; failed checks: none.

seed52_vs_seed51: **PASS**; paired six gains/losses 11/2; failed checks: none.

seed53_vs_seed51: **FAIL**; paired six gains/losses 6/5; failed checks: falls_not_higher.

seed53_vs_seed52: **FAIL**; paired six gains/losses 2/10; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_seed52_vs_history_seed51: **FAIL**; paired six gains/losses 2/6; failed checks: one_not_lower, six_not_lower, lane_not_higher.

history_seed53_vs_history_seed51: **PASS**; paired six gains/losses 6/2; failed checks: none.

history_seed53_vs_history_seed52: **PASS**; paired six gains/losses 9/1; failed checks: none.
actor one: three-seed min/max 8/17, range 9; parent 7 (counted once).
actor six: three-seed min/max 8/17, range 9; parent 7 (counted once).
actor falls: three-seed min/max 2/9, range 7; parent 10 (counted once).
actor lane: three-seed min/max 1/5, range 4; parent 4 (counted once).
actor flat_mean_episode_speed: three-seed min/max None/None, range None; parent None (counted once).
history one: three-seed min/max 5/13, range 8; parent 7 (counted once).
history six: three-seed min/max 4/12, range 8; parent 7 (counted once).
history falls: three-seed min/max 6/10, range 4; parent 8 (counted once).
history lane: three-seed min/max 1/8, range 7; parent 6 (counted once).
history flat_mean_episode_speed: three-seed min/max None/None, range None; parent None (counted once).

### geometry115_reset75

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 4/10 | 4/10 | 4 | 3 | 0 | 0/0 | — |
| seed51 | 6/10 | 6/10 | 2 | 2 | 0 | 0/0 | — |
| seed52 | 9/10 | 9/10 | 0 | 1 | 0 | 0/0 | — |
| seed53 | 6/10 | 6/10 | 3 | 1 | 0 | 0/0 | — |
| history_parent | 2/10 | 2/10 | 6 | 3 | 0 | 0/0 | — |
| history_seed51 | 2/10 | 2/10 | 6 | 2 | 0 | 0/0 | — |
| history_seed52 | 1/10 | 1/10 | 5 | 4 | 0 | 0/0 | — |
| history_seed53 | 4/10 | 4/10 | 5 | 1 | 0 | 0/0 | — |

seed51_vs_parent: **PASS**; paired six gains/losses 4/2; failed checks: none.

seed52_vs_parent: **PASS**; paired six gains/losses 5/0; failed checks: none.

seed53_vs_parent: **PASS**; paired six gains/losses 5/3; failed checks: none.

history_seed51_vs_history_parent: **PASS**; paired six gains/losses 1/1; failed checks: none.

history_seed52_vs_history_parent: **FAIL**; paired six gains/losses 1/2; failed checks: one_not_lower, six_not_lower, lane_not_higher.

history_seed53_vs_history_parent: **PASS**; paired six gains/losses 2/0; failed checks: none.

seed52_vs_seed51: **PASS**; paired six gains/losses 4/1; failed checks: none.

seed53_vs_seed51: **FAIL**; paired six gains/losses 3/3; failed checks: falls_not_higher.

seed53_vs_seed52: **FAIL**; paired six gains/losses 1/4; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement.

history_seed52_vs_history_seed51: **FAIL**; paired six gains/losses 1/2; failed checks: one_not_lower, six_not_lower, lane_not_higher.

history_seed53_vs_history_seed51: **PASS**; paired six gains/losses 2/0; failed checks: none.

history_seed53_vs_history_seed52: **PASS**; paired six gains/losses 4/1; failed checks: none.
actor one: three-seed min/max 6/9, range 3; parent 4 (counted once).
actor six: three-seed min/max 6/9, range 3; parent 4 (counted once).
actor falls: three-seed min/max 0/3, range 3; parent 4 (counted once).
actor lane: three-seed min/max 1/2, range 1; parent 3 (counted once).
actor flat_mean_episode_speed: three-seed min/max None/None, range None; parent None (counted once).
history one: three-seed min/max 1/4, range 3; parent 2 (counted once).
history six: three-seed min/max 1/4, range 3; parent 2 (counted once).
history falls: three-seed min/max 5/6, range 1; parent 6 (counted once).
history lane: three-seed min/max 1/4, range 3; parent 3 (counted once).
history flat_mean_episode_speed: three-seed min/max None/None, range None; parent None (counted once).

### geometry116_reset76

| Controller | Rough one | Rough six | Falls | Lane | World incl.flat | Flat falls/lane | Flat speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 3/10 | 3/10 | 6 | 1 | 0 | 0/0 | — |
| seed51 | 2/10 | 2/10 | 5 | 3 | 0 | 0/0 | — |
| seed52 | 8/10 | 8/10 | 2 | 0 | 0 | 0/0 | — |
| seed53 | 3/10 | 3/10 | 6 | 2 | 0 | 0/0 | — |
| history_parent | 5/10 | 5/10 | 2 | 3 | 0 | 0/0 | — |
| history_seed51 | 6/10 | 6/10 | 4 | 1 | 0 | 0/0 | — |
| history_seed52 | 4/10 | 3/10 | 2 | 4 | 0 | 0/0 | — |
| history_seed53 | 9/10 | 8/10 | 1 | 0 | 0 | 0/0 | — |

seed51_vs_parent: **FAIL**; paired six gains/losses 2/3; failed checks: one_not_lower, six_not_lower, lane_not_higher.

seed52_vs_parent: **PASS**; paired six gains/losses 5/0; failed checks: none.

seed53_vs_parent: **FAIL**; paired six gains/losses 2/2; failed checks: lane_not_higher, strict_rough_improvement.

history_seed51_vs_history_parent: **FAIL**; paired six gains/losses 3/2; failed checks: falls_not_higher.

history_seed52_vs_history_parent: **FAIL**; paired six gains/losses 3/5; failed checks: one_not_lower, six_not_lower, lane_not_higher, strict_rough_improvement.

history_seed53_vs_history_parent: **PASS**; paired six gains/losses 4/1; failed checks: none.

seed52_vs_seed51: **PASS**; paired six gains/losses 7/1; failed checks: none.

seed53_vs_seed51: **FAIL**; paired six gains/losses 3/2; failed checks: falls_not_higher.

seed53_vs_seed52: **FAIL**; paired six gains/losses 1/6; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement.

history_seed52_vs_history_seed51: **FAIL**; paired six gains/losses 1/4; failed checks: one_not_lower, six_not_lower, lane_not_higher.

history_seed53_vs_history_seed51: **PASS**; paired six gains/losses 4/2; failed checks: none.

history_seed53_vs_history_seed52: **PASS**; paired six gains/losses 5/0; failed checks: none.
actor one: three-seed min/max 2/8, range 6; parent 3 (counted once).
actor six: three-seed min/max 2/8, range 6; parent 3 (counted once).
actor falls: three-seed min/max 2/6, range 4; parent 6 (counted once).
actor lane: three-seed min/max 0/3, range 3; parent 1 (counted once).
actor flat_mean_episode_speed: three-seed min/max None/None, range None; parent None (counted once).
history one: three-seed min/max 4/9, range 5; parent 5 (counted once).
history six: three-seed min/max 3/8, range 5; parent 5 (counted once).
history falls: three-seed min/max 1/4, range 3; parent 2 (counted once).
history lane: three-seed min/max 0/4, range 4; parent 3 (counted once).
history flat_mean_episode_speed: three-seed min/max None/None, range None; parent None (counted once).

See summary.json for family/level, contact/posture telemetry, paired outcomes and raw-file provenance.
