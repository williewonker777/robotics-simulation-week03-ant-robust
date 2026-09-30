# v23 paired 16s/64s first-episode windows

2,800 first episodes, 5,600 dependent window observations; three continuation seeds on one parent and two maps. Parent counted once. No significance, causal safety, general robustness or automatic promotion; 64s cannot override 16s.

## 16s — primary

### pooled

seed51_vs_parent: FAIL; failed checks: six_not_lower, flat_speed_not_lower; paired six: {'n': 300, 'gains': 14, 'losses': 22, 'both_success': 147, 'both_failure': 117}.
seed52_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 9, 'losses': 28, 'both_success': 141, 'both_failure': 122}.
seed53_vs_parent: FAIL; failed checks: six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 14, 'losses': 23, 'both_success': 146, 'both_failure': 117}.
history_seed51_vs_history_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement; paired six: {'n': 300, 'gains': 12, 'losses': 19, 'both_success': 155, 'both_failure': 114}.
history_seed52_vs_history_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 300, 'gains': 7, 'losses': 26, 'both_success': 148, 'both_failure': 119}.
history_seed53_vs_history_parent: FAIL; failed checks: six_not_lower; paired six: {'n': 300, 'gains': 15, 'losses': 19, 'both_success': 155, 'both_failure': 111}.
seed52_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 300, 'gains': 11, 'losses': 22, 'both_success': 139, 'both_failure': 128}.
seed53_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 300, 'gains': 20, 'losses': 21, 'both_success': 140, 'both_failure': 119}.
seed53_vs_seed52: FAIL; failed checks: lane_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 24, 'losses': 14, 'both_success': 136, 'both_failure': 126}.
history_seed52_vs_history_seed51: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 300, 'gains': 10, 'losses': 22, 'both_success': 145, 'both_failure': 123}.
history_seed53_vs_history_seed51: PASS; failed checks: none; paired six: {'n': 300, 'gains': 20, 'losses': 17, 'both_success': 150, 'both_failure': 113}.
history_seed53_vs_history_seed52: PASS; failed checks: none; paired six: {'n': 300, 'gains': 29, 'losses': 14, 'both_success': 141, 'both_failure': 116}.
### geometry117_reset77

seed51_vs_parent: FAIL; failed checks: six_not_lower, flat_speed_not_lower; paired six: {'n': 150, 'gains': 9, 'losses': 16, 'both_success': 71, 'both_failure': 54}.
seed52_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 4, 'losses': 17, 'both_success': 70, 'both_failure': 59}.
seed53_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_speed_not_lower; paired six: {'n': 150, 'gains': 6, 'losses': 14, 'both_success': 73, 'both_failure': 57}.
history_seed51_vs_history_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 8, 'losses': 11, 'both_success': 75, 'both_failure': 56}.
history_seed52_vs_history_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 4, 'losses': 14, 'both_success': 72, 'both_failure': 60}.
history_seed53_vs_history_parent: FAIL; failed checks: lane_not_higher; paired six: {'n': 150, 'gains': 8, 'losses': 8, 'both_success': 78, 'both_failure': 56}.
seed52_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 6, 'losses': 12, 'both_success': 68, 'both_failure': 64}.
seed53_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 11, 'losses': 12, 'both_success': 68, 'both_failure': 59}.
seed53_vs_seed52: FAIL; failed checks: one_not_lower, falls_not_higher, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 12, 'losses': 7, 'both_success': 67, 'both_failure': 64}.
history_seed52_vs_history_seed51: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 6, 'losses': 13, 'both_success': 70, 'both_failure': 61}.
history_seed53_vs_history_seed51: PASS; failed checks: none; paired six: {'n': 150, 'gains': 12, 'losses': 9, 'both_success': 74, 'both_failure': 55}.
history_seed53_vs_history_seed52: PASS; failed checks: none; paired six: {'n': 150, 'gains': 16, 'losses': 6, 'both_success': 70, 'both_failure': 58}.
### geometry118_reset78

seed51_vs_parent: FAIL; failed checks: six_not_lower, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 5, 'losses': 6, 'both_success': 76, 'both_failure': 63}.
seed52_vs_parent: FAIL; failed checks: six_not_lower, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 5, 'losses': 11, 'both_success': 71, 'both_failure': 63}.
seed53_vs_parent: FAIL; failed checks: six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 8, 'losses': 9, 'both_success': 73, 'both_failure': 60}.
history_seed51_vs_history_parent: FAIL; failed checks: six_not_lower, falls_not_higher; paired six: {'n': 150, 'gains': 4, 'losses': 8, 'both_success': 80, 'both_failure': 58}.
history_seed52_vs_history_parent: FAIL; failed checks: six_not_lower; paired six: {'n': 150, 'gains': 3, 'losses': 12, 'both_success': 76, 'both_failure': 59}.
history_seed53_vs_history_parent: FAIL; failed checks: six_not_lower; paired six: {'n': 150, 'gains': 7, 'losses': 11, 'both_success': 77, 'both_failure': 55}.
seed52_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher; paired six: {'n': 150, 'gains': 5, 'losses': 10, 'both_success': 71, 'both_failure': 64}.
seed53_vs_seed51: FAIL; failed checks: lane_not_higher; paired six: {'n': 150, 'gains': 9, 'losses': 9, 'both_success': 72, 'both_failure': 60}.
seed53_vs_seed52: FAIL; failed checks: lane_not_higher; paired six: {'n': 150, 'gains': 12, 'losses': 7, 'both_success': 69, 'both_failure': 62}.
history_seed52_vs_history_seed51: FAIL; failed checks: six_not_lower; paired six: {'n': 150, 'gains': 4, 'losses': 9, 'both_success': 75, 'both_failure': 62}.
history_seed53_vs_history_seed51: PASS; failed checks: none; paired six: {'n': 150, 'gains': 8, 'losses': 8, 'both_success': 76, 'both_failure': 58}.
history_seed53_vs_history_seed52: PASS; failed checks: none; paired six: {'n': 150, 'gains': 13, 'losses': 8, 'both_success': 71, 'both_failure': 58}.
## 64s — secondary descriptive

### pooled

seed51_vs_parent: FAIL; failed checks: lane_not_higher, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 46, 'losses': 45, 'both_success': 180, 'both_failure': 29}.
seed52_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 300, 'gains': 41, 'losses': 43, 'both_success': 182, 'both_failure': 34}.
seed53_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 41, 'losses': 52, 'both_success': 173, 'both_failure': 34}.
history_seed51_vs_history_parent: PASS; failed checks: none; paired six: {'n': 300, 'gains': 42, 'losses': 21, 'both_success': 199, 'both_failure': 38}.
history_seed52_vs_history_parent: PASS; failed checks: none; paired six: {'n': 300, 'gains': 39, 'losses': 32, 'both_success': 188, 'both_failure': 41}.
history_seed53_vs_history_parent: PASS; failed checks: none; paired six: {'n': 300, 'gains': 44, 'losses': 25, 'both_success': 195, 'both_failure': 36}.
seed52_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement; paired six: {'n': 300, 'gains': 41, 'losses': 44, 'both_success': 182, 'both_failure': 33}.
seed53_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 300, 'gains': 38, 'losses': 50, 'both_success': 176, 'both_failure': 36}.
seed53_vs_seed52: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, strict_rough_improvement, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 42, 'losses': 51, 'both_success': 172, 'both_failure': 35}.
history_seed52_vs_history_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 300, 'gains': 24, 'losses': 38, 'both_success': 203, 'both_failure': 35}.
history_seed53_vs_history_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 300, 'gains': 28, 'losses': 30, 'both_success': 211, 'both_failure': 31}.
history_seed53_vs_history_seed52: PASS; failed checks: none; paired six: {'n': 300, 'gains': 41, 'losses': 29, 'both_success': 198, 'both_failure': 32}.
### geometry117_reset77

seed51_vs_parent: FAIL; failed checks: lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 29, 'losses': 24, 'both_success': 86, 'both_failure': 11}.
seed52_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 23, 'losses': 24, 'both_success': 86, 'both_failure': 17}.
seed53_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_speed_not_lower; paired six: {'n': 150, 'gains': 18, 'losses': 29, 'both_success': 81, 'both_failure': 22}.
history_seed51_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 19, 'losses': 11, 'both_success': 98, 'both_failure': 22}.
history_seed52_vs_history_parent: FAIL; failed checks: lane_not_higher; paired six: {'n': 150, 'gains': 22, 'losses': 16, 'both_success': 93, 'both_failure': 19}.
history_seed53_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 20, 'losses': 15, 'both_success': 94, 'both_failure': 21}.
seed52_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 18, 'losses': 24, 'both_success': 91, 'both_failure': 17}.
seed53_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_speed_not_lower; paired six: {'n': 150, 'gains': 16, 'losses': 32, 'both_success': 83, 'both_failure': 19}.
seed53_vs_seed52: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 19, 'losses': 29, 'both_success': 80, 'both_failure': 22}.
history_seed52_vs_history_seed51: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 14, 'losses': 16, 'both_success': 101, 'both_failure': 19}.
history_seed53_vs_history_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 14, 'losses': 17, 'both_success': 100, 'both_failure': 19}.
history_seed53_vs_history_seed52: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher; paired six: {'n': 150, 'gains': 18, 'losses': 19, 'both_success': 96, 'both_failure': 17}.
### geometry118_reset78

seed51_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 17, 'losses': 21, 'both_success': 94, 'both_failure': 18}.
seed52_vs_parent: FAIL; failed checks: six_not_lower, lane_not_higher, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 18, 'losses': 19, 'both_success': 96, 'both_failure': 17}.
seed53_vs_parent: FAIL; failed checks: lane_not_higher, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 23, 'losses': 23, 'both_success': 92, 'both_failure': 12}.
history_seed51_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 23, 'losses': 10, 'both_success': 101, 'both_failure': 16}.
history_seed52_vs_history_parent: FAIL; failed checks: falls_not_higher; paired six: {'n': 150, 'gains': 17, 'losses': 16, 'both_success': 95, 'both_failure': 22}.
history_seed53_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 24, 'losses': 10, 'both_success': 101, 'both_failure': 15}.
seed52_vs_seed51: FAIL; failed checks: falls_not_higher; paired six: {'n': 150, 'gains': 23, 'losses': 20, 'both_success': 91, 'both_failure': 16}.
seed53_vs_seed51: FAIL; failed checks: lane_not_higher; paired six: {'n': 150, 'gains': 22, 'losses': 18, 'both_success': 93, 'both_failure': 17}.
seed53_vs_seed52: FAIL; failed checks: lane_not_higher; paired six: {'n': 150, 'gains': 23, 'losses': 22, 'both_success': 92, 'both_failure': 13}.
history_seed52_vs_history_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 10, 'losses': 22, 'both_success': 102, 'both_failure': 16}.
history_seed53_vs_history_seed51: PASS; failed checks: none; paired six: {'n': 150, 'gains': 14, 'losses': 13, 'both_success': 111, 'both_failure': 12}.
history_seed53_vs_history_seed52: PASS; failed checks: none; paired six: {'n': 150, 'gains': 23, 'losses': 10, 'both_success': 102, 'both_failure': 15}.

All controller metrics, family/level/cell gates, per-seed ranges and matching-parent deltas, overlapping late-loss flags and interval first-hit counts are in summary.json.
