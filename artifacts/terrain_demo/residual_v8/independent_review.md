# Independent v8 review

Native reviewer: /root/review_rough_safety (read-only), 2026-09-22.

## Before training: APPROVE

- Policy is a registered self-contained frozen base plus bounded deterministic
  mean correction. PPO sampling/mean/log-prob/inference agree.
- Strict mode/scale loading, training mismatch rejection and re-freezing tested.
- Staged gradient tests retain exact zero-init; paired initial states/values match.
- CPU98/98 tests and AST checks passed independently.
- Initial review requested a v8-specific predeclared summarizer/gate and reference
  zero-head validation. Both were implemented and re-reviewed before training.
- Final review found no blocking implementation or compatibility issue.
- No Python LSP/type checker is installed; no new tooling was installed.
- This approves pretrain integrity, NOT performance promotion. Frozen final
  checkpoint and raw-metric review must be recorded after all evaluation.

## Completed training seed42 pair: CPU audit PASS

-17/17 archived source hashes unchanged; original v5 SHA unchanged.
- Prepared blind/footmap states differ only by mode buffer; optimizer states match.
- All8 actor.base tensors match initialization and original v5 actor exactly.
- Both finals iter749, finite tensors, limit0.5, correct mode0/2; SHA manifests match.
- Both logs report98,304,000 transitions. Actual env/PPO YAMLs equal v7 except
  intended class/limit/config guard and log/run/load names.
- No holdout/GPU evaluation was performed by the reviewer at this stage.

## All frozen checkpoints: CPU integrity audit PASS

-10 unique frozen policy paths/hashes: v5 reference1, original v7 footmaps3,
  and final v8 policies6. Every SHA matches; legacy v7 records unchanged.
-17/17 source hashes unchanged. Six v8 finals iter749/finite/limit0.5; embedded
  bases exactly match initialization and original v5. Total589,824,000 transitions.
- All3 prepared seed pairs differ only mode; optimizer states identical.
- Full paired env/PPO contracts equal and match v7 except intended policy fields
  and logging/run/load names. Reference residual head is exactly zero.
- This review did not evaluate performance or run any GPU job.

## Independent final raw-array audit: PASS; promotion FAIL

The reviewer used a standalone CPU script and imported neither summarizer.
-20 primary JSONs x175 =3500 first episodes; balanced7families x5levels x5cases.
- Recomputed float32 thresholds13.100000381469727/53.099998474121094 with no
  terminal/lane/world exit; all strict arrays, counters, per-seed,120 pooled
  family-level cells, hard levels, and flat-fall counts match. Mismatches0.
- v5:259/300 one,136six,32falls,3lane,0world,flat3/50.
- v7:753/900 one,219six,94falls,15lane,0world,flat6/150.
- Blind residual:712/900 one,374six,99falls,38lane,0world,flat10/150.
- Terrain residual:747/900 one,370six,96falls,25lane,0world,flat10/150.
- Gate FAIL exactly: one/six vs v5; one/falls vs v7; six vs blind; flatfalls vs v5.
- Higheststones:8/10,16/30,8/30,12/30 one-tile respectively; every six-tile0.
-17 source,10 checkpoint,20 raw result SHAs match. Models predate freeze and
  all primary evaluations postdate freeze; no training/preparation after freeze.
- Four seed42 zero/shuffle diagnostic files are identity/condition matched and
  excluded from every primary denominator/gate. No metric inflation/blocker.
