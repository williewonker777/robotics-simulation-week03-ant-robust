# Independent raw-data audit

Generated: 2026-10-01T01:10:40.872601+00:00

## Verdict

**PASS — raw numeric recomputation completed for all 10 bundles; see failed/inconclusive checks below.

## Raw-derived rough-terrain strict six-tile results (300 episodes/window)

| window | v5 | v16 control | history v16 | high53 | history high53 |
|---|---:|---:|---:|---:|---:|
| 16s | 135 | 177 | 167 | 162 | 166 |
| 64s | 214 | 224 | 225 | 223 | 236 |

Strict rule independently applied: IEEE-754 `float32(final_distance) >= float32(13.1/53.1)` **and** no posture termination, lane exit, or world exit.

## Findings

- 16s rough leader: **v16_control** (177/300 strict six-tile).
- 64s rough leader: **history_high53** (236/300 strict six-tile).
- 64s rough high53 history versus high53 solo: 49 paired gains, 36 regressions; net +13 strict six-tile successes.
- 64s rough v16 history versus v16 solo: 42 paired gains, 41 regressions; net +1.
- Family leaders vary; use `by_family` in the JSON rather than making a global routing claim.

## Evidence checks

- **PASS** `complete_raw_set`
- **PASS** `strict_rule_recomputed`
- **PASS** `family_counts`
- **PASS** `paired_initial_conditions_and_physical_config`
- **PASS** `single_rollout_window_integrity_no_reset_at_960`
- **PASS** `flat_v5_equivalence`
- **PASS** `checkpoint_and_teacher_identity`
- **PASS** `novelty_declaration`
- **PASS** `preexisting_file_freeze`
- **PASS** `execution_source_freeze`
- **PASS** `historical_selection_claims_from_historical_raw`
- **PASS** `generated_summary_compare_after_raw_audit`

## Limits

- Layouts are new, but terrain families are known-generator families; this is not novel-family/OOD proof.
- 350 paired initial conditions across two maps per controller/window are not 1,750 independent statistical examples.
- This audit did not assess the four GUI demo videos; media validation is a separate requirement.
- Historical selection values were recomputed from preserved historical raw JSON; they do not establish transfer beyond their historical layouts.
