# v15 independent final holdout audit

**Integrity verdict: CLEAR; performance promotion: FAIL.** Results below are
raw-array recomputations, not a universal claim.

- Inventory: 24 files / 2,220 first episodes; 17 v15 + 106 legacy source
  hashes, 8 references, and 2 final models all match.
- Per-scenario-map full root/joint/91D-prefix/CPU+CUDA-RNG pairing is exact
  across all six controllers; all three hybrid flat branches are byte-identical
  on both mixed maps.

## Primary 16s mixed maps

- Actor control: one 315/350, six 207/350, falls 33, lane 2, world 0;
  stable: one 322/350, six 205/350, falls 24, lane 3, world 0. **FAIL**
  (six lower: 205<207; lane higher: 3>2).
- Hybrid control: one 312/350, six 200/350, falls 37, lane 1, world 0;
  stable: one 315/350, six 194/350, falls 31, lane 2, world 0. **FAIL**
  (six lower: 194<200; lane higher: 2>1).
- The actor flat-speed criterion passes (10.0834 > 9.9611 m/s); hybrid flat
  raw identity, zero world exits, and strict terrain gain pass, but cannot
  offset the failed non-regressions.

## Secondary 64s hardest stepping stones

- Actor control: one 10/20, six 9/20, falls 4, lane 5, world 0; stable:
  one 10/20, six 8/20, falls 8, lane 3, world 0. **FAIL** (six lower:
  8<9; falls higher: 8>4).
- Hybrid control: one 11/20, six 11/20, falls 7, lane 2, world 0; stable:
  one 6/20, six 6/20, falls 11, lane 2, world 0. **FAIL** (one/six lower:
  6<11 and 6<11; falls higher: 11>7).

## Limitations

One training seed and two maps are exploratory, not universal or statistically
significant. The 64s result cannot promote a failed 16s primary result.
Direction and posture diagnostics are conditional visited-state telemetry, not
matched-state causal evidence; ideal depth does not establish real-camera or
robot safety.
