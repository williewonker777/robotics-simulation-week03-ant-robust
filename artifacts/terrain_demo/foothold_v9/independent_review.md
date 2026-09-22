# Independent v9 final audit

**Raw-array integrity:** PASS — 0 mismatches

The standalone CPU auditor imported neither project summarizer and recomputed every strict crossing 
from the 20 raw JSON files (3,500 first episodes).

| Policy | One tile | Six tiles | Falls | Lane | World | Flat falls |
|---|---:|---:|---:|---:|---:|---:|
| frozen_v5 | 264/300 | 136/300 | 24/300 | 0 | 0 | 5/50 |
| feet | 734/900 | 283/900 | 99/900 | 47 | 0 | 10/150 |
| targets | 741/900 | 416/900 | 91/900 | 66 | 0 | 11/150 |
| guided | 763/900 | 424/900 | 105/900 | 30 | 0 | 12/150 |

Promotion gate: **FAIL**

- Sources: 35/35 current; legacy v7/v8 and parent hashes unchanged.
- Checkpoints: 10/10 hashes, modes, 88D shapes and iterations verified.
- Saved training configs: 9/9 operationally equal outside declared arm/seed/geometry fields and checked run provenance.
- Commands: 43 successful records; all training/preparation finished before freeze and all 20 primary evaluations started after freeze.
- Every raw family/level cell contains five first episodes per condition; pooled, per-seed, per-family/level and official summary/gates agree.
- Auditor correction: After the first final run, changed the family_breakdown check from JSON key order to set equality; the writer serializes that mapping alphabetically, while lane_family_names preserves semantic lane order.

Standalone CPU audit; imports neither experiment summarizer. Strict crossings were recomputed from raw per-environment float32 distances, actual geometry thresholds, and terminal/lane/world flags.
