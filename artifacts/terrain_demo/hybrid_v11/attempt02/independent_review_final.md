# Final independent review — APPROVE evidence integrity

The code-reviewer independently reran the pinned attempt02 auditor after all
numerical evaluations and media commands: exit0, no mismatches.

- 33 evaluation files / 3,465 first episodes, including875 development episodes.
- Heldouts alone: 28 files / 2,590 first episodes, with16s and64s kept separate.
- 39 successful command records (five development,28 heldouts,three renders,
  three full decodes); original aborted attempt's three extra runs excluded.
- 63 frozen source/plan files, two recovery overlays, four checkpoint files,
  three exact teacher identities, original aborted evidence all verified.
- No `FAILED.json`; no original holdout result mixed into attempt02.
- README and HYBRID_V11 claims match raw metrics and disclose the retry,
  denominator differences, ideal depth, limited maps and causal limitations.

The scientific promotion verdict remains **FAIL**, not a software test failure:
`one_vs_v10`, `falls_vs_v5`, and `lane_vs_v5` fail. Both separate64s gates pass.
Retain v5; six-tile progress improvement does not establish overall robustness.

No blocking code/evidence/reporting issue was found. Remaining limitations:
two maps; ideal raycasting; unresolved initial-observation mismatch cause in
the excluded attempt; no raw per-step sensor/action trace for independent
reconstruction of feature extrema or action RMS. The successful operational
retry does not establish that the simulator startup issue was fixed.
