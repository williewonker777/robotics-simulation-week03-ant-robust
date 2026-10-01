# v25 post-main outcome verification

## Verdict

**PASS_MEASURED_PIPELINE__NO_PROMOTION**

The recorded 9-run learning and 46-cell raw holdout pipeline is internally consistent and source-bound. It does **not** establish improvement: all six preregistered 16 s arm contrasts fail (0/3 seeds).

## Direct verification

- 737/737 checks passed.
- Current 20-file scientific source map and immutable base evaluator hash match the frozen maps; preflight/capacity audit records retain 1,909 unchanged original files.
- All 9 final checkpoints are iteration 249, each with 32,768,000 transitions; planned main total is 294,912,000. Development retry evidence records 4,816,896 transitions and overall total 299,728,896.
- 46 enriched files exactly preserve their 46 original raw outputs; the adapter records only `CONTROLLERS`, `SCHEMA`, and `resolve_model` substitutions, zero scoring training transitions, and the unchanged base evaluator.
- Raw recomputation found 8,050 physical first episodes, 16,100 window observations, 350 paired conditions, and exact pairing within both geometries.
- Independently recomputed all 92 posthoc `episode_return` aggregates (23 controllers × 2 windows × rough/flat) from raw float values; means and population standard deviations match `evaluation_returns.json` within 1e-12.
- 9 runs × 6 training tags × 250 finite values and 9 × 8,000 schedule rows are present; their artifacts explicitly prohibit performance ranking/model selection.

## Measured result

- Primary 16 s: recovery/control, combined/recovery, combined/control and all three history contrasts are **FAIL**, each 0/3 seed passes.
- Diagnostic 64 s recovery/control is **FAIL**, 1/3 seed passes.
- Combined vs control (all three seeds, rough aggregate): at 16 s, six-tile 460→489, falls 63→62, lane failures 18→20; at 64 s, six-tile 672→650, falls 162→159, lane failures 72→94. These are descriptive tradeoffs, not a passing improvement result.

## Scope / gaps

- Read-only artifact verification; no GPU re-execution was performed.
- No README/publication/media/Git/GitHub work was reviewed or approved.
