# Independent training implementation and integration review

**APPROVE training**, not a performance result. The code reviewer independently
ran 76 focused CPU tests and confirmed exact 4096-environment paired initial
root/joints/full88D observations, policy, RNG and normalized saved YAML.

A HIGH finding (actual learner could omit/wrongly resume the source) was repaired:
before any learn rollout every policy tensor must equal the pinned v10 source,
except std reset to0.2; exact fresh Adam settings are required. Seven negative
regressions cover incorrect weights/std/teacher/keys/LR/optimizer state/class.
The reviewer independently approved the repair with22 training tests.

The new evaluator matches all38 original list arrays—including episode_return—
and initial/condition/gateconfig,41 explicit present fields, against the unchanged
v12 nonholdout reference. Earlier evidence contained an absent plural key checked
with dict.get; preserved and superseded by evaluator_parity_corrected.json.

MEDIUM evaluation-provenance finding: pin the prepared cache-manifest digest across
primary/horizon phases. Before training-source freeze, an exclusive evaluation
inputs record, parent/child checks, per-result/command digest and mutation test
were added. The summary/provenance review remains separate before any holdout.
