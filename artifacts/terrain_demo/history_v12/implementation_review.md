# v12 implementation review before paired evaluation

- Reviewer: independent native code-reviewer (`review_publication_safety`); author and reviewer separated.
- Verdict: APPROVE final repairs; no remaining blocking implementation issue reported.
- Independently rerun 70 history tests passed (28 gate,12 adapter,30 summary); full suite343 passed.
- Frozen original63 source/PLAN files and4 models preserved;6 additive source/test files frozen separately.

## Defects found and repaired before holdout

1. RecordedGate.reset attempted to mutate inference-created state outside inference mode. Real-gate red regression confirmed it; reset now owns its inference context.
2. Isaac SimulationApp.close terminated the process before runpy caller finalization. First development smoke's backend and log remain under ignored outputs; no accepted final result was claimed. Parent/worker boundary now requires exit0 AND invocation-matched raw backend and a once-only pre-shutdown event sidecar. The second smoke produced a valid versioned final result,35 firstepisodes audited beforefreeze.
3. Summary event consistency initially accepted impossible zero duty after a v10 switch. Float32 crossfade, float64 alpha sum, duty and target occupancy are now reconstructed and checked; contradictory fixtures corrected, rejection regression added.

## Verified contracts

Recorded-sample temporal voting, relevant-ROI unknown handling, independent episode reset, no portal reset, exact fixed-policy action endpoints, terminal-step event inclusion, post-reset exclusion, callback once, ordinary-exception import restoration, exclusive-new evidence writes, source/controller/config/model provenance, exact initial observation/state pairing, exposure-normalized switching, separate16/64second outcome criteria and flatworld accounting.

## Limitations

Switch snapshots establish consistency with recorded features and thresholds; they do not independently reconstruct every historical depth observation. History and confirmation windows form a temporal persistence intervention, not a trained recurrent actor and not proof that memory alone causally improves navigation. GPU holdout outcomes are a separate result, not this implementation approval.
