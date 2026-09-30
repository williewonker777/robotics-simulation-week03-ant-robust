# v15 independent pre-holdout training review

**Verdict: CLEAR.** This review recomputed the frozen 106 legacy source hashes, 12 v15 training source hashes, and eight reference-model hashes; it is not a performance result.

- Shared v14-conditioned 91D warm start is byte-identical for every tensor except `std`, reset to 0.2; both learned command weights are preserved and Adam is fresh.
- Both actual learner start audits have exact root/joint/full-91D/prefix/policy/CPU-RNG/CUDA-RNG pairing; only directional reward weight differs (control 0, stable 1).
- Raw text logs and TensorBoard events independently reproduce 250 iterations, 32,768,000 transitions, and finite scalar values for every recorded tag in each arm. Final model and Adam tensors are finite; frozen v5 teacher identity holds.
- Geometry-95 cache exactly matches its frozen 240-tile manifest; the unchanged-v14 geometry-51 evaluator parity fields remain exact.
- Three final GPU checkpoint inference smokes are present and exactly paired at initialization.

No holdout performance was inspected or claimed.
