# Separate64s stone1.0 diagnostic

Non-gating64s stone1.0-only diagnostic; two maps, not140 independent maps; never combine with primary.

| Policy | Strict1tile | Strict6tiles | Falls | Lane | World | Full64survival |
|---|---:|---:|---:|---:|---:|---:|
| frozen_v5 | 13/20 | 0/20 | 5/20 | 0/20 | 0/20 | 15/20 |
| free | 19/60 | 14/60 | 29/60 | 12/60 | 0/60 | 31/60 |
| anchored | 23/60 | 15/60 | 31/60 | 5/60 | 0/60 | 29/60 |

## Conditional times and censored distances

Hit-time statistics are conditional on reaching distance, even if later falling/exiting; not unconditional traversal time.

Distanceat16s is null if the first episode ended at/before16s. Final distance is at64s OR earlier termination.

### frozen_v5
- first_hit_one: observed15/20; mean=13.032222222222222; median=12.3
- first_hit_six: observed0/20; mean=None; median=None
- distance16: observed17/20; mean=14.190395916209502; median=14.674798011779785
- final_distance: observed20/20; mean=20.836239099502563; median=21.910184860229492
- maximum_distance: observed20/20; mean=20.954207944869996; median=22.218098640441895

### free
- first_hit_one: observed46/60; mean=9.986956521739131; median=10.016666666666666
- first_hit_six: observed23/60; mean=51.66884057971014; median=52.199999999999996
- distance16: observed47/60; mean=18.33226307402266; median=17.852195739746094
- final_distance: observed60/60; mean=41.73087445894877; median=42.15825271606445
- maximum_distance: observed60/60; mean=41.78796230951945; median=42.1634521484375

### anchored
- first_hit_one: observed48/60; mean=10.799652777777778; median=10.341666666666667
- first_hit_six: observed21/60; mean=56.719047619047615; median=57.18333333333333
- distance16: observed49/60; mean=16.58693348631567; median=17.463077545166016
- final_distance: observed60/60; mean=39.81793084144592; median=43.51162910461426
- maximum_distance: observed60/60; mean=39.90923601786296; median=43.52122116088867
