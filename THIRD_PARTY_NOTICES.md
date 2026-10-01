# Third-party notices

## RSL-RL 3.0.1

`src/week03_ant/prior_ppo.py` adapts the upstream PPO implementation.
The experiment pins the upstream file SHA-256 in `docs/PRIOR_V10.md`;
the following is the installed package license, reproduced verbatim.
This repository does not redistribute the simulator, robot assets, or the
course Python environment. Their respective licenses continue to apply.
Isaac Lab-derived files retain their upstream copyright headers.

```text
Copyright (c) 2025, ETH Zurich
Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

See licenses/dependencies for license information of dependencies of this package.
```

## Teammate-inspired v25 adaptation

The opt-in v25 recovery reward adapts concepts from
[Stick-0/isaac-ant-rough-terrain](https://github.com/Stick-0/isaac-ant-rough-terrain/tree/3cc718a4214f336fd4db7db5841fa86033b99d35),
pinned at `3cc718a4214f336fd4db7db5841fa86033b99d35`.
Its continuous clearance/upright/action-change/angular-sway penalties are adapted to
this repository's existing terrain-relative posture measurements. It is a subset,
not a reproduction of that repository's complete recovery package.

The conditional entropy experiment is informed by
[LimDaeKyung/IsaacLab_RS](https://github.com/LimDaeKyung/IsaacLab_RS/tree/8d9eed1fe463f638d5a62528dbcc0a3656ddd52b),
pinned at `8d9eed1fe463f638d5a62528dbcc0a3656ddd52b`.
This repository changes its existing coefficient from 0.002 to 0.005; it does not
redistribute a teammate checkpoint or reproduce their boxes-only task.

Both pinned repositories include the same BSD-3-Clause license with the
2022–2025 Isaac Lab Project Developers copyright. The full original copyright,
conditions, and disclaimer are retained verbatim in
[licenses/teammate_ant_BSD-3-Clause.txt](licenses/teammate_ant_BSD-3-Clause.txt).
See [v25 research and limitations](artifacts/terrain_demo/teammate_port_v25/research.md)
for source-specific evidence and boundaries. Names are attribution, not endorsement.
