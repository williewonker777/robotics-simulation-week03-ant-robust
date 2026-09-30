# v24 fixed global LR: paired 16s/64s first-episode windows

4,900 first episodes, 9,800 dependent window observations; matched high/low global-LR arms, three continuation seeds on one parent and two maps. Parent counted once. No significance, causal safety, general robustness or automatic promotion; 64s cannot override 16s.

## 16s — primary

### pooled

low51_vs_seed51: FAIL; failed checks: one_not_lower, falls_not_higher, lane_not_higher; paired six: {'n': 300, 'gains': 25, 'losses': 18, 'both_success': 143, 'both_failure': 114}.
low52_vs_seed52: FAIL; failed checks: falls_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 24, 'losses': 17, 'both_success': 133, 'both_failure': 126}.
low53_vs_seed53: FAIL; failed checks: flat_falls_not_higher; paired six: {'n': 300, 'gains': 23, 'losses': 14, 'both_success': 143, 'both_failure': 120}.
history_low51_vs_history_seed51: FAIL; failed checks: one_not_lower, lane_not_higher; paired six: {'n': 300, 'gains': 18, 'losses': 18, 'both_success': 149, 'both_failure': 115}.
history_low52_vs_history_seed52: FAIL; failed checks: one_not_lower, lane_not_higher; paired six: {'n': 300, 'gains': 20, 'losses': 13, 'both_success': 144, 'both_failure': 123}.
history_low53_vs_history_seed53: FAIL; failed checks: lane_not_higher; paired six: {'n': 300, 'gains': 21, 'losses': 15, 'both_success': 150, 'both_failure': 114}.
low51_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_speed_not_lower; paired six: {'n': 300, 'gains': 16, 'losses': 19, 'both_success': 152, 'both_failure': 113}.
low52_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 14, 'losses': 28, 'both_success': 143, 'both_failure': 115}.
low53_vs_parent: FAIL; failed checks: six_not_lower, lane_not_higher, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 14, 'losses': 19, 'both_success': 152, 'both_failure': 115}.
history_low51_vs_history_parent: FAIL; failed checks: lane_not_higher; paired six: {'n': 300, 'gains': 19, 'losses': 18, 'both_success': 148, 'both_failure': 115}.
history_low52_vs_history_parent: FAIL; failed checks: six_not_lower, lane_not_higher; paired six: {'n': 300, 'gains': 18, 'losses': 20, 'both_success': 146, 'both_failure': 116}.
history_low53_vs_history_parent: FAIL; failed checks: lane_not_higher; paired six: {'n': 300, 'gains': 18, 'losses': 13, 'both_success': 153, 'both_failure': 116}.
seed51_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 17, 'losses': 27, 'both_success': 144, 'both_failure': 112}.
seed52_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 9, 'losses': 30, 'both_success': 141, 'both_failure': 120}.
seed53_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, strict_rough_improvement, flat_speed_not_lower; paired six: {'n': 300, 'gains': 11, 'losses': 25, 'both_success': 146, 'both_failure': 118}.
history_seed51_vs_history_parent: PASS; failed checks: none; paired six: {'n': 300, 'gains': 19, 'losses': 18, 'both_success': 148, 'both_failure': 115}.
history_seed52_vs_history_parent: FAIL; failed checks: six_not_lower; paired six: {'n': 300, 'gains': 14, 'losses': 23, 'both_success': 143, 'both_failure': 120}.
history_seed53_vs_history_parent: FAIL; failed checks: six_not_lower; paired six: {'n': 300, 'gains': 18, 'losses': 19, 'both_success': 147, 'both_failure': 116}.
### geometry119_reset79

low51_vs_seed51: FAIL; failed checks: one_not_lower, falls_not_higher, lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 10, 'losses': 9, 'both_success': 76, 'both_failure': 55}.
low52_vs_seed52: FAIL; failed checks: one_not_lower, falls_not_higher, lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 11, 'losses': 10, 'both_success': 68, 'both_failure': 61}.
low53_vs_seed53: FAIL; failed checks: flat_falls_not_higher; paired six: {'n': 150, 'gains': 11, 'losses': 7, 'both_success': 72, 'both_failure': 60}.
history_low51_vs_history_seed51: FAIL; failed checks: one_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 10, 'losses': 8, 'both_success': 76, 'both_failure': 56}.
history_low52_vs_history_seed52: FAIL; failed checks: one_not_lower, falls_not_higher, lane_not_higher; paired six: {'n': 150, 'gains': 11, 'losses': 8, 'both_success': 72, 'both_failure': 59}.
history_low53_vs_history_seed53: FAIL; failed checks: six_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 7, 'losses': 8, 'both_success': 82, 'both_failure': 53}.
low51_vs_parent: FAIL; failed checks: six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 9, 'losses': 11, 'both_success': 77, 'both_failure': 53}.
low52_vs_parent: FAIL; failed checks: six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 7, 'losses': 16, 'both_success': 72, 'both_failure': 55}.
low53_vs_parent: FAIL; failed checks: six_not_lower, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 6, 'losses': 11, 'both_success': 77, 'both_failure': 56}.
history_low51_vs_history_parent: FAIL; failed checks: six_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 8, 'losses': 10, 'both_success': 78, 'both_failure': 54}.
history_low52_vs_history_parent: FAIL; failed checks: six_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 6, 'losses': 11, 'both_success': 77, 'both_failure': 56}.
history_low53_vs_history_parent: FAIL; failed checks: lane_not_higher; paired six: {'n': 150, 'gains': 7, 'losses': 6, 'both_success': 82, 'both_failure': 55}.
seed51_vs_parent: FAIL; failed checks: six_not_lower, flat_speed_not_lower; paired six: {'n': 150, 'gains': 9, 'losses': 12, 'both_success': 76, 'both_failure': 53}.
seed52_vs_parent: FAIL; failed checks: six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 6, 'losses': 16, 'both_success': 72, 'both_failure': 56}.
seed53_vs_parent: FAIL; failed checks: six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 6, 'losses': 15, 'both_success': 73, 'both_failure': 56}.
history_seed51_vs_history_parent: FAIL; failed checks: six_not_lower; paired six: {'n': 150, 'gains': 9, 'losses': 13, 'both_success': 75, 'both_failure': 53}.
history_seed52_vs_history_parent: FAIL; failed checks: six_not_lower; paired six: {'n': 150, 'gains': 6, 'losses': 14, 'both_success': 74, 'both_failure': 56}.
history_seed53_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 11, 'losses': 9, 'both_success': 79, 'both_failure': 51}.
### geometry120_reset80

low51_vs_seed51: FAIL; failed checks: one_not_lower; paired six: {'n': 150, 'gains': 15, 'losses': 9, 'both_success': 67, 'both_failure': 59}.
low52_vs_seed52: FAIL; failed checks: falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 13, 'losses': 7, 'both_success': 65, 'both_failure': 65}.
low53_vs_seed53: PASS; failed checks: none; paired six: {'n': 150, 'gains': 12, 'losses': 7, 'both_success': 71, 'both_failure': 60}.
history_low51_vs_history_seed51: FAIL; failed checks: six_not_lower; paired six: {'n': 150, 'gains': 8, 'losses': 10, 'both_success': 73, 'both_failure': 59}.
history_low52_vs_history_seed52: PASS; failed checks: none; paired six: {'n': 150, 'gains': 9, 'losses': 5, 'both_success': 72, 'both_failure': 64}.
history_low53_vs_history_seed53: PASS; failed checks: none; paired six: {'n': 150, 'gains': 14, 'losses': 7, 'both_success': 68, 'both_failure': 61}.
low51_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_speed_not_lower; paired six: {'n': 150, 'gains': 7, 'losses': 8, 'both_success': 75, 'both_failure': 60}.
low52_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 7, 'losses': 12, 'both_success': 71, 'both_failure': 60}.
low53_vs_parent: FAIL; failed checks: lane_not_higher; paired six: {'n': 150, 'gains': 8, 'losses': 8, 'both_success': 75, 'both_failure': 59}.
history_low51_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 11, 'losses': 8, 'both_success': 70, 'both_failure': 61}.
history_low52_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 12, 'losses': 9, 'both_success': 69, 'both_failure': 60}.
history_low53_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 11, 'losses': 7, 'both_success': 71, 'both_failure': 61}.
seed51_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_speed_not_lower; paired six: {'n': 150, 'gains': 8, 'losses': 15, 'both_success': 68, 'both_failure': 59}.
seed52_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 3, 'losses': 14, 'both_success': 69, 'both_failure': 64}.
seed53_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 5, 'losses': 10, 'both_success': 73, 'both_failure': 62}.
history_seed51_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 10, 'losses': 5, 'both_success': 73, 'both_failure': 62}.
history_seed52_vs_history_parent: FAIL; failed checks: six_not_lower; paired six: {'n': 150, 'gains': 8, 'losses': 9, 'both_success': 69, 'both_failure': 64}.
history_seed53_vs_history_parent: FAIL; failed checks: six_not_lower; paired six: {'n': 150, 'gains': 7, 'losses': 10, 'both_success': 68, 'both_failure': 65}.
## 64s — secondary descriptive

### pooled

low51_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_falls_not_higher; paired six: {'n': 300, 'gains': 44, 'losses': 55, 'both_success': 169, 'both_failure': 32}.
low52_vs_seed52: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_speed_not_lower; paired six: {'n': 300, 'gains': 39, 'losses': 54, 'both_success': 178, 'both_failure': 29}.
low53_vs_seed53: FAIL; failed checks: flat_falls_not_higher; paired six: {'n': 300, 'gains': 58, 'losses': 34, 'both_success': 170, 'both_failure': 38}.
history_low51_vs_history_seed51: FAIL; failed checks: one_not_lower, lane_not_higher; paired six: {'n': 300, 'gains': 35, 'losses': 34, 'both_success': 191, 'both_failure': 40}.
history_low52_vs_history_seed52: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 300, 'gains': 34, 'losses': 37, 'both_success': 192, 'both_failure': 37}.
history_low53_vs_history_seed53: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, strict_rough_improvement; paired six: {'n': 300, 'gains': 30, 'losses': 37, 'both_success': 197, 'both_failure': 36}.
low51_vs_parent: FAIL; failed checks: lane_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 44, 'losses': 40, 'both_success': 169, 'both_failure': 47}.
low52_vs_parent: FAIL; failed checks: lane_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 49, 'losses': 41, 'both_success': 168, 'both_failure': 42}.
low53_vs_parent: FAIL; failed checks: flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 54, 'losses': 35, 'both_success': 174, 'both_failure': 37}.
history_low51_vs_history_parent: FAIL; failed checks: lane_not_higher; paired six: {'n': 300, 'gains': 36, 'losses': 30, 'both_success': 190, 'both_failure': 44}.
history_low52_vs_history_parent: FAIL; failed checks: lane_not_higher; paired six: {'n': 300, 'gains': 42, 'losses': 36, 'both_success': 184, 'both_failure': 38}.
history_low53_vs_history_parent: FAIL; failed checks: lane_not_higher; paired six: {'n': 300, 'gains': 44, 'losses': 37, 'both_success': 183, 'both_failure': 36}.
seed51_vs_parent: FAIL; failed checks: flat_speed_not_lower; paired six: {'n': 300, 'gains': 52, 'losses': 37, 'both_success': 172, 'both_failure': 39}.
seed52_vs_parent: FAIL; failed checks: flat_speed_not_lower; paired six: {'n': 300, 'gains': 54, 'losses': 31, 'both_success': 178, 'both_failure': 37}.
seed53_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 300, 'gains': 47, 'losses': 52, 'both_success': 157, 'both_failure': 44}.
history_seed51_vs_history_parent: PASS; failed checks: none; paired six: {'n': 300, 'gains': 40, 'losses': 35, 'both_success': 185, 'both_failure': 40}.
history_seed52_vs_history_parent: PASS; failed checks: none; paired six: {'n': 300, 'gains': 34, 'losses': 25, 'both_success': 195, 'both_failure': 46}.
history_seed53_vs_history_parent: PASS; failed checks: none; paired six: {'n': 300, 'gains': 45, 'losses': 31, 'both_success': 189, 'both_failure': 35}.
### geometry119_reset79

low51_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 22, 'losses': 28, 'both_success': 89, 'both_failure': 11}.
low52_vs_seed52: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_speed_not_lower; paired six: {'n': 150, 'gains': 20, 'losses': 32, 'both_success': 85, 'both_failure': 13}.
low53_vs_seed53: FAIL; failed checks: flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 26, 'losses': 16, 'both_success': 89, 'both_failure': 19}.
history_low51_vs_history_seed51: FAIL; failed checks: one_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 19, 'losses': 18, 'both_success': 97, 'both_failure': 16}.
history_low52_vs_history_seed52: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 16, 'losses': 20, 'both_success': 96, 'both_failure': 18}.
history_low53_vs_history_seed53: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 13, 'losses': 18, 'both_success': 103, 'both_failure': 16}.
low51_vs_parent: FAIL; failed checks: flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 27, 'losses': 17, 'both_success': 84, 'both_failure': 22}.
low52_vs_parent: FAIL; failed checks: lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 25, 'losses': 21, 'both_success': 80, 'both_failure': 24}.
low53_vs_parent: FAIL; failed checks: flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 28, 'losses': 14, 'both_success': 87, 'both_failure': 21}.
history_low51_vs_history_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 16, 'losses': 17, 'both_success': 100, 'both_failure': 17}.
history_low52_vs_history_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 17, 'losses': 22, 'both_success': 95, 'both_failure': 16}.
history_low53_vs_history_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 18, 'losses': 19, 'both_success': 98, 'both_failure': 15}.
seed51_vs_parent: FAIL; failed checks: flat_falls_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 31, 'losses': 15, 'both_success': 86, 'both_failure': 18}.
seed52_vs_parent: FAIL; failed checks: flat_speed_not_lower; paired six: {'n': 150, 'gains': 32, 'losses': 16, 'both_success': 85, 'both_failure': 17}.
seed53_vs_parent: FAIL; failed checks: lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 29, 'losses': 25, 'both_success': 76, 'both_failure': 20}.
history_seed51_vs_history_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher; paired six: {'n': 150, 'gains': 19, 'losses': 21, 'both_success': 96, 'both_failure': 14}.
history_seed52_vs_history_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 15, 'losses': 16, 'both_success': 101, 'both_failure': 18}.
history_seed53_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 22, 'losses': 18, 'both_success': 99, 'both_failure': 11}.
### geometry120_reset80

low51_vs_seed51: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 22, 'losses': 27, 'both_success': 80, 'both_failure': 21}.
low52_vs_seed52: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 19, 'losses': 22, 'both_success': 93, 'both_failure': 16}.
low53_vs_seed53: PASS; failed checks: none; paired six: {'n': 150, 'gains': 32, 'losses': 18, 'both_success': 81, 'both_failure': 19}.
history_low51_vs_history_seed51: FAIL; failed checks: one_not_lower, lane_not_higher; paired six: {'n': 150, 'gains': 16, 'losses': 16, 'both_success': 94, 'both_failure': 24}.
history_low52_vs_history_seed52: PASS; failed checks: none; paired six: {'n': 150, 'gains': 18, 'losses': 17, 'both_success': 96, 'both_failure': 19}.
history_low53_vs_history_seed53: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, strict_rough_improvement; paired six: {'n': 150, 'gains': 17, 'losses': 19, 'both_success': 94, 'both_failure': 20}.
low51_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, falls_not_higher, lane_not_higher, strict_rough_improvement, flat_speed_not_lower; paired six: {'n': 150, 'gains': 17, 'losses': 23, 'both_success': 85, 'both_failure': 25}.
low52_vs_parent: FAIL; failed checks: lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 24, 'losses': 20, 'both_success': 88, 'both_failure': 18}.
low53_vs_parent: FAIL; failed checks: lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 26, 'losses': 21, 'both_success': 87, 'both_failure': 16}.
history_low51_vs_history_parent: FAIL; failed checks: lane_not_higher; paired six: {'n': 150, 'gains': 20, 'losses': 13, 'both_success': 90, 'both_failure': 27}.
history_low52_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 25, 'losses': 14, 'both_success': 89, 'both_failure': 22}.
history_low53_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 26, 'losses': 18, 'both_success': 85, 'both_failure': 21}.
seed51_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 21, 'losses': 22, 'both_success': 86, 'both_failure': 21}.
seed52_vs_parent: FAIL; failed checks: flat_speed_not_lower; paired six: {'n': 150, 'gains': 22, 'losses': 15, 'both_success': 93, 'both_failure': 20}.
seed53_vs_parent: FAIL; failed checks: one_not_lower, six_not_lower, lane_not_higher, flat_speed_not_lower; paired six: {'n': 150, 'gains': 18, 'losses': 27, 'both_success': 81, 'both_failure': 24}.
history_seed51_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 21, 'losses': 14, 'both_success': 89, 'both_failure': 26}.
history_seed52_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 19, 'losses': 9, 'both_success': 94, 'both_failure': 28}.
history_seed53_vs_history_parent: PASS; failed checks: none; paired six: {'n': 150, 'gains': 23, 'losses': 13, 'both_success': 90, 'both_failure': 24}.

All controller metrics, family/level/cell gates, per-seed ranges and matching-parent deltas, overlapping late-loss flags and interval first-hit counts are in summary.json.
