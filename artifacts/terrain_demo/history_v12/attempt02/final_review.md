# Independent v12 attempt02 final review

**Verdict: APPROVE the evidence and reporting, not an overall controller promotion.**

No blocking findings. The 16-second primary improvement contract fails; the
separate 64-second secondary contract passes. Neither result is pooled with the
other or with the excluded first attempt.

## Independently verified

- Exactly 16 expected evaluation files / 1,480 first episodes: 1,400 primary and
  80 secondary. Family/difficulty inventories are balanced within each condition.
- Exact root-state, joint-position, joint-velocity and full-observation hashes
  match all four controllers for each map/condition. Primary hashes also match
  the predeclared cached initial-only probes.
- All 69 frozen source files and four model files remain byte-identical. Both
  predeclared maps retain all 480 cache tiles / 1,440 pinned files; no duplicate
  expected tile seed was found in the live cache.
- All 16 serial command records report exit 0 and the exact intended conditions;
  no attempt02 FAILED marker exists. The original two files / 350 episodes and
  their failure evidence retain their pinned hashes and are not aggregated.
- Every backend and event-sidecar digest matches; original numerical arrays are
  unchanged by the adapter. All 1,480 episode boundaries, strict float32-distance
  outcomes, switch timing, target occupancy, float32 crossfade alpha sums and
  duty fractions were independently reconstructed and checked.
- All 1,111 switch events have consistent first-episode identities and spatial
  evidence. History events satisfy recorded-sample window/vote/confirmation
  thresholds and dwell requirements.
- All final JSON groups, family groups, family/level groups and exact-rational
  decision gates match independent raw-array recomputation. README.md,
  docs/HISTORY_V12.md and the v12 section of docs/EXPERIMENT_HISTORY.md distinguish
  the historical failed attempt, completed amended batch and scoped conclusions.

## Results, history versus instant

| Separate condition | One | Six | Falls | Lane exits | Switches / 100 active seconds |
|---|---:|---:|---:|---:|---:|
| 16s mixed terrain, n=300 each | 271 → 269 | 148 → 146 | 28 → 28 | 0 → 0 | 13.21346 → 10.10123 |
| 64s stones, n=20 each | 10 → 10 | 5 → 5 | 10 → 9 | 1 → 0 | 3.59764 → 3.04177 |

Primary switching decreases 23.55%, but both success counts decrease by two:
**primary FAIL**. Flat falls are 4/50 for each controller; world exits are zero.
Secondary criteria **PASS**, without overturning the primary failure.

## Review boundaries

Raw-array verification uses the independent v11 audit routines plus separate
v12 event/provenance/summary checks; it does not import the production gate,
evaluator or summarizer. The JSON report pins that independent auditor and the
reviewed summary. No GPU jobs, remote operations or experiment-source edits were
performed for this review.

This is one actor seed on two maps, with only 20 secondary episodes per
controller. No statistical significance, universal-best, causal fall-prevention
or safety guarantee is established. Recorded switch features are internally
consistent, but every timestep of raw depth is not independently reconstructible.
Shared-cache pre/post checks cannot eliminate transient external mutation.

The code/test implementation reviews and the leader's fresh CPU verification
are complementary evidence; this final slice independently verifies completed
experimental data and reporting rather than rerunning the entire test suite.
