## Code Review Summary

**Files Reviewed:** 2  
**Total Issues:** 2

### By Severity
- CRITICAL: 0
- HIGH: 1
- MEDIUM: 1
- LOW: 0

### Issues

**[HIGH] Silent omission of historical evidence links**  
File: `scripts/verify_teammate_publication_v25.py:160-162`  
Issue: missing/empty `evidence_sha256` defaults to no checks; unknown references are silently skipped. Independently reproduced PASS with `missing-required-proof.json` and `../../outside.json` in the summary mapping (updating only the public checksum). This is a fail-open provenance path, not an acceptable cache compatibility fallback.  
Fix: require a nonempty mapping with at least freeze + all 46 enriched + all 46 raw entries. Validate each path and digest and require every referenced structured JSON in the public inventory with its historical original SHA. Root intends to publish cache-preparation JSON as well, so no partial-exemption path is needed. Preserve all scientific SHA values. Add missing/empty/unsafe/omitted-link negative tests.

**[MEDIUM] Filename identity not bound to recorded identity**  
File: `scripts/verify_teammate_publication_v25.py:141-154`  
Issue: the expected filename is constructed, but only its model path is bound. Two maps of the same controller can exchange complete raw/enriched files while the independent auditor still sees a complete correctly keyed matrix; public filenames then mislabel maps.  
Fix: explicitly compare the record's controller/geometry/reset tuple with the filename tuple before aggregation; test swapped-map records.

### Verification
- Current 8 helper tests PASS under `python3 -S`; both files AST-parse.
- Independent temporary-checkout integration with the **actual frozen auditor**, source20/base6 bytes, existing checkpoint bytes and synthetic complete arrays PASS: 123 public files, 46 records, 8050 physical / 16100 dependent observations. This is **synthetic plumbing evidence, not actual experiment evidence**.
- Frozen science20 hashes still match; no source/artifact edits, GPU, remote or Git operations by reviewer.
- Reviewed file hashes and exact evidence are saved in `publication-helper-review.json`.

### Recommendation
**REQUEST CHANGES** for the fail-open evidence-link path. Root has accepted the minimal no-exemption repair; these reviewed bytes do not contain that repair yet. Re-review after stable fixes.

Nonblocking trust boundary: `--root` contains Python code that is imported. Use a trusted reviewed repository revision; self-supplied hashes provide consistency, not independent authenticity. The helper correctly distinguishes public recorded-array verification from private runtime/cache/training replay.
