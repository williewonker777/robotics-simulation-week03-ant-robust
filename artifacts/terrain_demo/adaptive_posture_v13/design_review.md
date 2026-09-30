# Independent architecture review

Verdict after repairs: **APPROVE / CLEAR for implementation**, not a performance claim.

The reviewer required two repairs before coding: uniform downsteps/gap mouths must
not count as clear ground, and strict flat-speed gain applies only to standalone
actors (fixed-v5 flat hybrid must instead preserve/non-regress). Both were applied.
The final plan specifies one bounded additive term, torso-relative geometry, valid
local foot support, signed lane speed, angular velocity tip kinematics, orientation
boundary, unknown masking and a kinematic—not contact-certified—swing proxy.

Matched continuation/control, exact warm starts, final-checkpoint-only selection,
unchanged evaluation physics, two fresh maps and separate horizons were approved.
The compressed88D actor inputs and lower-posture/faster-speed hypothesis remain
limitations to test, not assumptions of success. “Target” means reward target,
not a command delivered to the actor. Original recovery penalties remain present.

Required validation: regression tests, actual saved-config parity, zero-control
parity, GPU training/evaluation smoke and non-holdout evaluator parity before freeze.
No files, GPU tasks, dependencies, models or remotes were changed by the reviewer.
