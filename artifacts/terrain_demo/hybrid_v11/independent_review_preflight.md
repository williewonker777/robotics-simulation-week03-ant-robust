# Independent review before evaluation

- Gate, telemetry, evaluator, runner and summary reviewed separately from authors.
- A flat-world-exit promotion omission was repaired before development and covered
  by a regression. Final CPU suite: 269 passing tests.
- Independent auditor imports no project summary/controller code. It reconstructs
  float32 blending from switch events, first-episode metrics and promotion rates.
- Exact task/family guards added before final testing; malformed evidence rejected.
- Development audit: 875 first episodes, five files, 63 source hashes (53 archived),
  four models and three exact v5-teacher identities. Zero mismatches; cautious selected.

## Operational retry review

Original holdout attempt failed exact initial-observation pairing after three files.
The cause is unconfirmed. All three are preserved and excluded, including files
whose individual episode arrays were otherwise valid. No performance comparison
from that incomplete batch is accepted.

The separate code-reviewer approved one entire same-condition batch retry after
reviewing `run_attempt02.py`, `independent_audit_attempt02.py` and the pinned recovery
amendment. Independent pre-retry audit again passed all 875 development episodes,
original source/model identities, both overlay hashes, three aborted JSON hashes,
and the eight-line original command ledger. Only artifact/log destinations change.

Exact initial-state and observation hashes remain mandatory; no input normalization,
model/gate change, condition change, threshold relaxation or outcome-based retry is
allowed. The original 63-file freeze and the two-file recovery layer are distinct.

The auditor does not itself reject the orchestration `FAILED.json` marker. The root
completion check must explicitly assert that marker is absent in addition to audit
success. A failed retry must not be reported as a completed comparative benchmark.

Limitations: scalar action RMS and feature extrema are range/consistency checked,
not independently reconstructed from per-step traces (those traces are not stored).
Switch-associated falls are not causal attribution; two maps are not thousands of
independent maps; ideal raycasting is not a real RGB-D sensor.
