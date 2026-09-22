# v9 pre-training independent code review

**Review scope:** 17 files (plan; foothold math/policy/task/config; launcher/checkpoint/probe; fixed experiment runner/auditor; viewer/video helpers; four focused test files)  
**Code/spec/security recommendation:** **COMMENT — no training blocker found**  
**Architecture status inherited from the separate architecture lane:** **WATCH** (endpoint heuristic, not certified foothold planning)

## Severity summary

- CRITICAL: 0
- HIGH: 0
- MEDIUM: 0
- LOW: 0
- Explicit experimental limitations: 4 (below; not implementation defects against the frozen v9 plan)

## Stage 1 — specification and root-cause guard

The implementation matches `outputs/foothold_v9_20260922/PLAN.md`:

- `src/week03_ant/foothold_math.py:13-17` represents the scanner's yaw-frame footprint including the configured +0.4 m x offset (`[-1.2, 2.0] x [-1.2, 1.2]`, 33x25).
- `src/week03_ant/foothold_math.py:20-31` rejects borders, missing/clipped neighbors, and patches whose 3x3 height range exceeds 0.06 m.
- `src/week03_ant/foothold_math.py:51-66` uses the 3x3 maximum support height, actual Ant distal order `++,-+,--,+-`, 0.45 m XY reach, 0.4 m vertical reach, and the 0.18 m local-top band.
- `src/week03_ant/foothold_math.py:68-79` deterministically separates the forward proposal from the nearest current-support anchor and explicitly abstains when no candidate exists.
- `src/week03_ant/tasks/foothold_v9_cfg.py:28-44` derives geometry from the forced-current ideal ray scan in the same root-relative/yaw frame as distal FK; clipped ranges are invalid rather than silently treated as safe.
- `src/week03_ant/tasks/foothold_v9_cfg.py:51-63` applies the bounded current-support cost only as disclosed training-time stepping-stone shaping. The family label does not enter proposal observations or the actor.
- Isaac Lab's step order computes reward before reset and interval events, then observations after reset and `lane_wrap`; both reward and observation paths force ray recomputation (`foothold_v9_cfg.py:30,57`). This satisfies pre-reset reward and post-reset/wrap freshness.
- `src/week03_ant/foothold_policy.py:18-77` persists a strict scalar mode code, masks the final 16 dimensions for both actor and critic in `feet`, restores checkpoint mode in evaluation, and rejects a training-config mismatch.
- `src/week03_ant/foothold_policy.py:87-106` and `scripts/prepare_foothold_checkpoint.py:13-46` preserve every 60D actor/critic layer, zero only the 28 added first-layer columns, set std=0.2, reset Adam and iteration state, record provenance, and refuse overwrite.
- `scripts/foothold_v9.py:12-31` fail-closes the mode/reward contract: guided=-1, feet/targets=0, and strict config matching cannot be disabled.
- `scripts/run_foothold_experiment.py:173-240` freezes all 9 final-749 policies before any holdout evaluation; `:243-274` uses only predeclared geometry/reset 64/38 and 65/39. Saved configs, source hashes, checkpoint hashes, architecture, mode, iteration, and reference identity are checked.
- `scripts/summarize_foothold.py:18-33,51-161` independently recomputes raw first-episode outcomes, enforces 20 files / 3,500 episodes and balanced cells, checks every family/level count, and applies the predeclared pooled, flat, world-exit, and per-seed gates.

No broad exception, silent default, alternate execution path, or swallowed failure was introduced. Candidate abstention is an explicit experimental contract and is reported diagnostically; it does not hide a software failure.

## Stage 2 — quality, safety, and performance

No code-quality or security defect requiring a change was found. Commands use argument vectors without `shell=True`; the experiment refuses evidence overwrite; there are no credentials, remote writes, or hidden dependency changes.

### Explicit limitations (WATCH, not blockers)

1. **Endpoint support is not whole-capsule support or IK feasibility** — `foothold_math.py:34-39,55-66`. A distal center can be acceptable while the long capsule or joint path is not. Keep the current disclosure; a future certified planner would need support-polygon/capsule collision and joint-reach checks.
2. **No candidate means zero hint and zero shaping** — `foothold_math.py:74-79,94-101`. Guidance disappears in states outside the heuristic envelope, including some severe traps. The probe correctly reports availability instead of calling abstention safe; retain per-family invalid-candidate statistics in the final report.
3. **Guided shaping is privileged and stone-specific** — `foothold_v9_cfg.py:58-63`. This is valid for the controlled simulation arm but limits claims about a sensor-only reward or real deployment. Preserve the disclosure and do not present it as an actor input.
4. **Guided training recomputes 825 rays for reward and again for observation** — `foothold_v9_cfg.py:30,54-57`. This is deliberately freshness-first rather than cached. The measured 4096-env smoke/profile was stable (~95.7k steps/s, ~1.37 s/iteration), so it is not a fixed-budget blocker.

## Verification evidence

- `../run-python -m pytest -q tests`: **151 passed**.
- Focused foothold tests: **53 passed** after the runner/auditor appeared.
- `py_compile`: all reviewed v9 Python sources and scripts passed.
- A Python LSP/type checker is not installed in this environment (`pyright`, `basedpyright`, `pylsp`, `mypy`, and `ruff` unavailable); therefore the formal verdict is COMMENT rather than merge-ready APPROVE. Runtime imports, compile checks, and focused/full tests are the available substitutes.
- Frozen old source maps independently checked: v7 **9/9** and v8 **17/17**, zero hash mismatches.
- v9 source freeze map resolves **35** files (including transitive scan/evaluation/task/CLI and video/viewer sources) and rejects mutation.
- Saved smoke configuration: **17/17** physical/action/event/termination/reward/PPO/mode checks passed.
- Simulator evidence read independently: 88D finite observations, 825 rays, all 4 feet available at reset and static hard stones, wrap error 0, reset freshness true; 64-env train and 35-env saved-model evaluation passed.
- 300-step diagnostic discloses hard-1.0 per-foot availability (0.847, 0.573, 0.627, 0.610), so absence is material and visible rather than suppressed.

## Recommendation

**No pre-training blocker. Proceed with the frozen 3-mode x 3-seed experiment exactly once.** Keep the four WATCH limitations in the final interpretation, do not retune on held-out results, and retain frozen v5 if the predeclared promotion gate fails.
