# v8 frozen-base residual experiment (direct execution)

## Scope and hypothesis
Prior v7 spatial-input fine-tuning degraded six-tile completion from 45% to 24%.
Keep the successful 60D v5 actor frozen; train only a bounded corrective actor,
terrain CNN, critic and action exploration. This preserves base parameters and
initial actions, NOT closed-loop safety or performance. No reward/physics change.

## Alternatives and selection
- Larger feed-forward/depth model: already tested, long-traversal regression remains.
- Recurrent memory: useful for partial sensing, but the current idealized map is not
  primarily occlusion-limited and it adds sequence/reset complexity; defer.
- Frozen-base additive residual: selected because it directly constrains policy drift.

## Exact contract
- Reuse unchanged v7 514D observation, 221 ideal height rays + validity + 4 foot XYZ.
- Reuse v7 six-channel spatial encoder and separate critic. New actor mean is
  frozen_v5(raw[:60]) + 0.5*tanh(residual_MLP([raw[:60], CNN(map)])).
- Residual head last layer zero initialized. Base actor requires_grad=False.
- Bound applies to deterministic mean difference at the SAME observation. Gaussian
  training exploration and changed subsequent trajectories are not bounded by it.
- No training terrain label or runtime teacher routing. Four legs, eight efforts.
- Two modes: blind residual (all spatial channels zero) and footmap residual.
  Same initialization for each paired seed except persistent input-mode buffer.
- 3 training seeds 42/43/44, geometry51/58/59, 750 iterations x4096env x32steps.
  Same v7 PPO hyperparameters and reward/termination/action/curriculum contracts.
- Reserve fresh final geometry62/reset36 and63/reset37, 175 balanced first episodes
  per policy/condition, 16s, strict 13.1m/53.1m with footprint margin1.1m.
- Freeze all six final749 checkpoints before any fresh evaluation. No best-iteration
  or seed selection. Reference frozen v5 and all three v7 footmap final policies
  reevaluated in the same scene/conditions. Do not repeat baseline samples.
- Zero/shuffle depth at inference for representative seed42 is diagnostic, not an
  extra independently trained arm. Both done after freezing; no tuning afterwards.
- Promotion requires no aggregate one/six/fall regression vs v5, blind residual and
  v7 footmap; zero world exit; no flat-fall regression vs v5; more strict one-tile
  successes than blind in >=2/3 training-seed pairs. Fail means keep original v5.

## Verification and stop conditions
CPU regression tests before long jobs: exact initial action equivalence, frozen
base stays bit-identical after optimizer steps, nonzero residual/CNN gradients,
mode/scale strict state roundtrip and malformed-state rejection, ablation independence,
finite outputs, same-observation residual bound, inherited v7 tests unchanged.
64env2iteration GPU smoke and35envsaved-model evaluation. Read actual dumped config;
verify training contract equals v7 and old checkpoint hashes unchanged.
Then one GPU job at a time, archive source hashes/configs/checkpoint hashes/timing.
Independently recompute every raw result and summarize pooled and per-seed outcomes.
Record uncut16s stone1.0 videos for predeclared seed42 vs v5, geometry62/reset36.
Stop after completed fixed-budget experiment, audited metrics and artifacts; do not
claim difficult-terrain solution or real depth-camera capability if gate fails.
No dependencies, commits, remote Git or hardware actuation.

## Review clarifications (before training)
Critic = unchanged v7 mode-filtered separate CNN32 + MLP400/200/100,
original v5 critic zero-column warm-start. Initial paired state and values match.
Primary audit =20 JSONs/3500episodes. Depth zero/shuffle diagnostics retain actual
foot maps; they are OOD sensitivity tests, not isolated train-time ablations.
Mean bound .5 becomes3.75 effort at action scale7.5; not inherently safe.
Scale0 is accepted only to permit disabled-residual identity diagnostics; all
planned training uses strict .5. Config match is enforced by v8 training launcher.
Separate opt-in launcher/registrations preserve all archived v7 source hashes.
