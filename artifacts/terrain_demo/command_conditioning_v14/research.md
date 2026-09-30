# Primary literature consulted — 2026-09-22

## Selected: explicit behavior-command conditioning
[Walk These Ways, Margolis & Agrawal, CoRL2022/PMLR2023](https://proceedings.mlr.press/v205/margolis23a.html),
[paper §§3.1–3.2/Table1](https://proceedings.mlr.press/v205/margolis23a/margolis23a.pdf).
Body-height and footswing/gait behavior parameters are policy inputs and define
tracking objectives. The paper uses histories and randomized behavior sampling,
with human selection of behavior at deployment. It does not demonstrate our
automatic terrain-to-height rule. We test only command/feedback input access;
the Ant, torque actions, frozen-v5 prior and existing reward are our adaptation.

## Alternatives considered, not implemented
- [Miki et al.2022, primary full text](https://arxiv.org/html/2201.08117v1):
  privileged teacher, recurrent noisy-perception student, belief reconstruction
  and feature gating. This is not a switch between two independently trained experts.
  A faithful adaptation needs a larger teacher/student/history study.
- [PGTT, version2](https://arxiv.org/html/2510.18348v2),
  [version metadata](https://arxiv.org/abs/2510.18348v2): phase-visible actor and
  terrain-adaptive swing trajectories/rewards with contact penalties. This would
  add gait timing/contact supervision, whereas our bounded study isolates inputs.
  2026v2 publication status and quadruped platforms do not establish Ant performance.

## Inference, not established cause
V13's failure to raise the torso on mixed terrain might reflect insufficient
explicit task information, but its foothold hints contain correlated information.
This experiment tests command+feedback access, not a proof that partial
observability caused previous failures. No new dependency or paper code is copied.
