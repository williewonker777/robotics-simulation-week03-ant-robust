# v17 independent raw-evidence audit — 2026-09-23

Read-only verifier independently recomputed reported first-episode results from the
24 holdout JSON files, rather than accepting the generated Markdown verdict.

- Inventory: 12 mixed/16s files = 2,100 episodes and 12 stones/64s files = 120
  episodes. Every file has its declared controller, geometry/reset pair, 175/10
  environments, horizon, holdout phase, and family×level assignment.
- Every raw-file SHA-256 matches `summary.json`. The six controllers have exact
  initial state, 88D observation prefix, and CPU/CUDA RNG equality within each
  map/scenario. All reported controller×family×level aggregates independently
  recompute from raw first-episode arrays (mixed 210 entries; stones 6 entries).
- Primary `lp` vs `fixed`: 1-tile **268 vs 270**, 6-tile **159 vs 163**,
  falls **27 vs 21**, lane exits **5 vs 7** per 300 rough episodes; flat falls
  **2 vs 2** per 50 and mean speed **10.6663 vs 10.1889 m/s**. Actor **FAIL**.
  `history_lp` vs `history_fixed`: **270/162/24/6** vs **279/166/20/1**
  (1-tile/6-tile/fall/lane per 300); hybrid **FAIL**. Its fixed-v5 flat branch
  has exact environment-by-environment raw identity.
- Secondary stones `lp` vs `fixed`: **7/7/10/4** vs **5/3/10/7**
  (1-tile/6-tile/fall/lane per 20), actor **PASS** on this separate horizon;
  it cannot override primary failure. History: **8/8/10/5** vs
  **13/12/4/3**, **FAIL**.
- All 16 v17 frozen sources, 143 prior sources, and 12 prior models still match
  `frozen.json`; no pinned v16 source or model mutation detected. Training,
  final checkpoints, evaluation input/cache manifest, and the pre-holdout command
  ledger prefix (17 lines, 17,492 bytes) validate. All scoring followed the
  pinned input preparation.

The audit verifies recorded data and provenance, not an independent simulator
replay. One training seed and two fresh maps cannot establish universal or
statistically significant performance. The preregistered primary actor and
hybrid promotion decisions are **both FAIL**; no default switch is justified.
