# v18 preregistered plan — depth-gated teacher style

## Hypothesis and reference
CaT (Chane-Sane et al., 2024, https://arxiv.org/html/2403.18765v1)
applies style constraints on flat ground and relaxes them on rough terrain.
This experiment is **inspired by that terrain-conditional style principle**, not
an implementation of CaT's stochastic terminations. The specific Ant hypothesis
is that freezing the v5 action-mean teacher on rough scans limits improvement,
while keeping it on flat scans prevents the regressions observed in the v10
unconstrained-prior arm. Efficacy is unknown until fresh-map evaluation.

## Fixed source, one treatment
Both arms start from the exact frozen v16 **control** final249 checkpoint, SHA-256
`1a937bfa9b33449b6a088471b6202ab4db9d8231131e9f12509f6fe40db737fc`.
Every tensor except scalar exploration std (reset to .2) remains identical;
optimizer is fresh Adam1e-4 with empty state and iteration0. Both arms use the
same v16 contact-sensor training scene with contact-slip reward weight0, same
v13 posture reward, same 91D actor/critic input, 8D action, fixed v5 teacher,
PPO hyperparameters, physical dynamics, terrain/task sampling and seed. No
v17 reset sampler, direction reward, new actor input, or inference gate change.

The only intervention is the teacher auxiliary loss mask. `always` retains the
v5 mean target on every sample. `gated` retains it when a **train-only** 825-ray
terrain-scan detector says flat or unknown, and substitutes the detached current
student mean on rough samples. The latter makes the existing coefficient.02
prior loss and its gradient exactly zero on those samples; PPO/reward terms are
unchanged. Both arms compute/store the detector in a separate `[N,1]` observation
group; policy and critic groups remain exactly `policy:[N,91]`. The detector
never uses privileged terrain-family labels. It uses the existing yaw-aligned
forward scan ROI (`0<=x<=1.5m`, `|y|<=0.8m`), excludes invalid/clipped rays,
and calls rough when at least 20% of ROI rays are valid and either relative
height standard deviation exceeds `.035m` or coverage is below `.90`. Unknown
or fewer valid rays retain the teacher, not a false rough classification.
The threshold is a predeclared engineering adaptation, not a paper constant.
Binary mask only: no soft-interpolation loss/gradient mismatch.

## Training, development, and stop
Seed50, training geometry104. Shared checkpoint and paired initial state/RNG
audit. Exactly 4,096 environments ×32 policy steps ×250 PPO iterations per arm
=32,768,000 transitions each, one GPU arm at a time on cuda:1 under the shared
lock. Final model249 only; no intermediate selection. Before full training:
pure CPU mask/loss/gradient/checkpoint/inference and config tests, a short
paired 256-environment development smoke to verify both detector states and
family-wise coverage, and 4,096-environment capacity check. Freeze training
source hashes and paired initialization before the full runs. Log actual
per-family mask occupancy, PPO scalar finiteness and exact transition budget.
If the detector never marks both flat and rough samples appropriately, fix
against **training geometry only** and amend this plan before training, never
against holdouts. If mechanism is invalid, stop without claiming a comparison.

## Fresh, untouched evaluation
Use geometry105/reset66 and geometry106/reset67, not v17's maps102/103.
Prewarm cache without scoring, freeze source/model/cache/command-prefix hashes,
then use the unchanged v16 sensor-enabled evaluation physics and v12 history
depth gate. Controllers: `v16_control` descriptive reference, `always`,
`gated`, `history_original` (v5+v10 descriptive), `history_always`,
`history_gated`. Primary: 16s mixed, 175 environments per map/controller
(300 rough +50 flat first episodes across maps per controller). Secondary:
64s level4 stepping stones, 10 environments per map/controller (20 first
episodes each). All 24 files and raw outcomes, not a chosen map/checkpoint.
Exact initial observations, source prefix and RNG must match across controllers
within a map/scenario. No deployment/default switch unless both primary actor
and hybrid comparisons pass the v17 nonregression gates: rough one/six passage
not lower, falls/lane/world exits not higher, flat falls not higher; actor
flat speed strictly higher; hybrid flat v5-branch raw identity and at least
one strict rough improvement. Secondary cannot override primary failure.
Report uncertainty: one fine-tune seed and two maps do not establish universal
superiority. Stop after this one preregistered paired experiment and raw audit,
PASS or FAIL. Do not retune after fresh scores. No push or remote Git operation.
