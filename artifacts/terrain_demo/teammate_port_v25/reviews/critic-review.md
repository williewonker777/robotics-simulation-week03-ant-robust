# Independent critic design review

**OKAY / APPROVE** — no issues found requiring a plan revision.

Bound to plan SHA `160aeab578ad0bb2de2e58dea1eeef4f01b521c6f806b9afe20c6ea25e525216`. This is design quality review, not implementation validation, a performance guarantee, or host-issued Ralplan authority.

## Evidence and summary

- **Clarity:** exact reward rates, target-capped clearance thresholds, three arms, three seeds, parent/teacher identity, budgets, controller matrix, and reporting decisions are explicit. Cached primary referenced reports and both licenses were read; source manifests independently match 34 Stick and 165 Lim files. Parent checkpoint SHA independently matches the plan.
- **Verifiability:** nine fixed final249 checkpoints, actual configuration/seed/RNG/initial-state/optimizer/horizon proof, passive paired 16/64-second first episodes, independent raw audit, and source/model freeze prevent obvious substitution and favorable-selection routes. Arithmetic independently checks 294,912,000 main and 2,408,448 development transitions; 8,050 physical episodes and 16,100 dependent windows.
- **Completeness:** source transfer, compatibility adaptation, meaningful full learning, same-budget control, historical endpoints, raw evidence, attribution, README/publication, negative results, and stop/recovery rules are covered.
- **Big picture:** the experiment answers whether a compatible adaptation improves this repository, without equating teammate survival/return to local strict lane success or promising a new best model.
- **Principle/option consistency:** PASS. The three-arm design supports recovery-package and entropy-conditional-on-recovery contrasts, explicitly not entropy-only effects or interactions. Six-, nine-, and twelve-run alternatives state the real information/budget tradeoff.
- **Risk/verification rigor:** PASS. Both heldout maps and combined rows must satisfy same-seed retention, and all three paired seeds must pass; pooled gains cannot overrule failures. Long horizon cannot rescue the short primary. Frozen defaults and original bytes remain.
- **Deliberate additions:** PASS. Premortem addresses slowed traversal, increased falls, and silent harness/config/model substitutions. Expanded tests and full-capacity preflights precede freeze and main learning.

## Representative implementation-path checks

1. **Reward adapter:** inspected `posture_v13_cfg.py`, `posture_math.py`, and cached Stick `recovery_mdp.py`. Existing geometry supplies target height in [0.44, 0.58], local-maximum clearance and validity; capping at 0.48 leaves denominator positive and does not penalize the flat 0.44 target. Missing-scan abstention is intentionally a documented departure from Stick. Four new continuous terms can be additive without touching sensors, original terminations or evaluation.
2. **Continuation runner:** inspected `scripts/seed_continuation_v22.py`, `seed_study_v22.py`, continuation task configuration, and `prior_ppo.py`. Existing initialization, same-device parity, fresh Adam, forwarded episode-length setter, and audit patterns support the proposed proof. A new opt-in harness must preserve actual new seeds/geometry rather than mask them through historical v22 guards. Existing PriorPPO supports the declared entropy coefficient without a framework edit.
3. **Paired evaluator:** inspected `evaluate_unseen_terrain.py`, `contact_v16_cfg.py`, and `history_gate.py`. Fixed legacy resolver, v16 evaluation, unchanged history gate and low-level paired trackers support 5 legacy + 18 new endpoints. Policy construction can consume different RNG for 88D and 91D models, so the already-required actual pairing proof must catch/control that; never waive it based only on seed labels.

## Stop condition and handoff

Proceed with implementation and preregistered preflight. Do not start main learning until independent implementation review, CPU/static checks, actual initialization/config/seed/cache/budget evidence and scientific freeze pass. A smoke run is not completion. Retain failures and negative outcomes; publish only measured results with original-byte and remote-publication verification.
