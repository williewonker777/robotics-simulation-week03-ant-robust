# v15 primary references — 2026-09-23

- Miki et al., *Learning robust perceptive locomotion for quadrupedal robots in the wild* (2022), Supplement S7: orthogonalvelocity reward exp(-3||v_o||²), coefficient.75; projection requiresunitdesired direction. https://arxiv.org/html/2201.08117v1
- Aractingi et al., *Controlling quadruped robots with reinforcement learning* (Scientific Reports13,11945,2023), Commandvelocitytracking: c_vel exp(-||Vcmd-[vx,vy,wz]||²), c_vel6.0(Table3). https://www.nature.com/articles/s41598-023-38259-7 ; https://www.nature.com/articles/s41598-023-38259-7/tables/3
- IsaacLab v2.2.0 UniformVelocityCommandCfg official API: heading_command andheading_control_stiffness(default1.0) convertheadingerror intoangularvelocitycommand. https://isaac-sim.github.io/IsaacLab/v2.2.0/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.commands.commands_cfg.UniformVelocityCommandCfg

Ouradaptation: planar-onlylateralprojection, clippedheading-derivedworldZyawtarget,
weights.75/.25 andnegativeboundedregularizers; forwardreward retained. Thisisnot
fullpaperreproduction. Negative-shiftingpositiveexponentials changesepisode-length
incentives; monitorstall/low-speed/falls andretainunchangedstrictperformancegates.
WorldZangularvelocity differsfromEulerheadingderivative ontiltedbodies. Nearvertical
projectedbodyheading invalidatesdiagnostics. Sidewayssuppression aloneisnotcentering;
existinglane targetdirection alreadyturnstowardthecenter. V14didnotprovethefailurecause.

Deferred: contact-basedslip/swingneedsrealcontactreporting, notjointincomingwrenches.
Frozen-expertlearnedmultiplicativearbitration inYu2026 requiresnewgate/actionsemantics
andisnotjustaverageactions: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1697159/full
