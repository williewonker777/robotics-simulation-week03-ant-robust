# Independent primary validation — PASS

The separate code-reviewer independently validated all 14 primary files with the
pinned independent auditor's row/aggregation implementation, not the producer's
summary functions. Scope: 16-second first episodes only; 64-second results pending
at the time of this review.

- 2,450 first episodes; exact file and family/difficulty multiplicities.
- Identical root/joint/observation initialization hashes across policies per map.
- 63 frozen source/plan hashes, four checkpoint hashes, three teacher identities.
- Recovery amendment and isolated destinations intact; 14 primary commands exit0.
- No terminal `FAILED.json` marker. No evidence-integrity blocker.

| Policy | Terrain N | One tile | Six tiles | Falls | Lane exits |
|---|---:|---:|---:|---:|---:|
| v5 | 300 | 263 | 133 | 23 | 0 |
| v10 | 900 | 814 | 438 | 67 | 11 |
| Hybrid | 900 | 805 | 451 | 81 | 2 |

Hybrid mean terrain v10 blend duty: 69.99%; flat duty: 0%. Flat falls12/150,
same rate as v5's4/50. All world-exit counts zero.

Primary promotion already fails one-tile versus v10, falls versus v5, and lane
exits versus v5. Better six-tile progress does not constitute overall promotion.
