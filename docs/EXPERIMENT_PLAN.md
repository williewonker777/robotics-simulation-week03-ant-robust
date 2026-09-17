# Experiment plan

## Question

Does moderate domain randomization improve PPO policy robustness on unseen Ant
physics without changing the observation/action interface or training budget?

## Hypothesis

The robust policy will have lower or similar in-distribution return than the
baseline but higher mean return, longer episodes, and fewer catastrophic falls
under unseen low-friction, payload, and impulse conditions.

## Controlled variables

- PPO implementation and hyperparameters: unchanged course `AntPPORunnerCfg`
- observations/actions: 60D state / 8D joint effort
- environments per run: 4096
- rollout length: 32 steps/environment
- PPO iterations: 1000
- training seeds: 42, 43, 44 for baseline and robust
- evaluation: seed 24, 100 environments, first episode, at most 960 steps

## Independent variable

| Variant | Training distribution |
|---|---|
| baseline | unmodified `Isaac-Ant-v0` |
| friction | contact friction and restitution buckets only |
| robust | friction + torso mass/COM + reset state + state noise + interval pushes |

The friction-only run is an ablation used to distinguish one simple physical
randomization from the combined robust design. It is run with seed 42 because
the primary statistical comparison is baseline versus robust over three seeds.

## Public holdouts

These are deliberately outside the robust training range and are not claimed to
represent the private grading environment.

| Scenario | Change |
|---|---|
| ID | original course task |
| low friction | static 0.30, dynamic 0.25 |
| heavy | torso mass ×1.30 and off-center COM (+0.04, -0.03, 0.0) m |
| push | random x/y velocity impulses up to 0.80 m/s every 3–5 s |

## Metrics

1. 100-environment episode return mean and population standard deviation.
2. Mean episode length and catastrophic early termination evidence.
3. TensorBoard learning curve and final convergence region.
4. Qualitative 960-step rollout video for baseline and robust checkpoints.

## Decision rule

The hypothesis is supported only if robust improves the pooled OOD return or
survival on more than one holdout without a severe ID collapse. One favorable
video or a single best checkpoint is insufficient.

## Known limitations

- The actual hidden environment is released only during presentation.
- Three training seeds estimate initialization sensitivity but do not establish
  broad statistical significance.
- Holdouts change one physical factor at a time and do not cover terrain shape,
  actuator latency, or correlated multi-factor shifts.
