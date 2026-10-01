# v25 pre-main verification — final

## Verdict

**APPROVE_FOR_FROZEN_MAIN_TRAINING_ONLY**

- Frozen plan SHA-256: `160aeab578ad0bb2de2e58dea1eeef4f01b521c6f806b9afe20c6ea25e525216`
- Training-freeze SHA-256: `af2d613ac97af833bd5c5c03357f657705e520de33c733c546876fb82d515408`
- 77/77 independent checks passed.

## Evidence

- Revision-2 CPU validation: 2,276 passing tests, zero failures/errors/skips; its 20-file source map equals the training freeze.
- Fresh preflight and capacity independent audits: 49,152 and 2,359,296 transitions; 1,909 old immutable bytes pass.
- Development: 8 frozen controllers, 280 physical first episodes / 560 dependent windows, exact per-environment parity. A newly rerun raw audit agrees with both recorded development summaries.
- Nine expected-main snapshots hash-match the freeze and equal the capacity initial state except the declared `max_iterations: 2 → 250`; seed/arm, 91D state, fresh optimizer, recovery enablement, fixed PPO/prior values and entropy arm values were checked.
- The attempt-1 adapter failure is retained and records zero scored episodes and zero main learning. No main process existed at this check.

## Scope

This permits only the unchanged frozen nine-run main learning plan (294,912,000 transitions). It does **not** prove performance, held-out evaluation, documentation, or publication. Any frozen source/artifact/initializer change invalidates this approval.
