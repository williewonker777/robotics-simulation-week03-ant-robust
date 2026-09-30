# v19 reference grounding — 2026-09-28

Primary texts checked for the user's request to continue using papers. This study
is an evaluation-only checkpoint rebaseline, not a reproduction of these papers.

| Reference | Verified principle | Boundary in this repository |
|---|---|---|
| [Chane-Sane et al., *CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning* (2024), §IV–V](https://arxiv.org/html/2403.18765v1) | Uses height-scan variance to activate style constraints on flat terrain and deactivate them elsewhere; compares against always-on style constraints. Tables II–III use four training seeds. | v18 gated an existing frozen-v5 auxiliary action loss, not CaT's stochastic-termination mechanism. One training seed is not equivalent to the paper's evidence. |
| [Miki et al., *Learning robust perceptive locomotion for quadrupedal robots in the wild* (2022), Supplement S7–S9](https://arxiv.org/html/2201.08117v1) | Contact-conditioned foot-velocity cost; privileged teacher to noisy-perception recurrent student with learned belief/feature gating. | Supports the motivation for slip costs and perception robustness. It does not directly justify this repository's heuristic switch between two fixed policies. |
| [Aractingi et al., *Controlling the Solo12 quadruped robot with deep reinforcement learning* (2023)](https://arxiv.org/html/2309.16683v1), [published article](https://doi.org/10.1038/s41598-023-38259-7) | Binary foot contact times horizontal foot-speed squared for slip cost; penalty curriculum; joint-position outputs with PD torque control. | v16 uses a world-XY distal-foot kinematic proxy and Ant torque actions. These are adaptations, not matching control interfaces or actual sole-slip measurement. The older local research note's abbreviated title is corrected here without changing frozen historical evidence. |
| [Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice* (NeurIPS 2021)](https://papers.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) | Few-run point estimates can be unreliable; emphasizes uncertainty-aware reporting and robust aggregates. | Motivates honest uncertainty boundaries, not the exact three-map/16s/64s settings. More evaluation episodes do not add independent training seeds. |

## Application judgment, not a quotation or prescribed paper protocol

Comparing fixed policies on common geometry/reset/horizon reduces the cross-version
map confound. We retain per-map, family/level and paired outcomes, separate16s from
64s, and avoid episode-independent confidence intervals or algorithm-superiority
claims. Three maps do not precisely characterize map generalization. If these
results guide later training, these maps become development data; final evaluation
needs new untouched maps.

One possible later experiment is a preregistered immediate-versus-ramped contact
penalty comparison motivated by Aractingi's penalty curriculum. Keep initial
checkpoint, final coefficient, total budget and evaluation fixed, and measure
stationary/airborne exploitation along with strict traversal. This is a hypothesis,
not a claim that penalty onset caused the prior failures, and is outside v19.
