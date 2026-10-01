# v25 final documentation verification

## Verdict

**PASS** — 132/132 checks passed.

## Passed evidence

- All five requested documents have resolvable local Markdown links.
- README and v25 documentation numeric controller, return, pooled-arm, gate, diagnostic, media and attribution claims were checked against the recorded CSV/JSON/manifest evidence.
- Documentation correctly preserves evaluation-reward immutability, new-layout/same-family limitation, history-not-expert limitation, posthoc-return restriction, n=1/non-cherry-picked media restriction, and no-promotion conclusion.

## Reproduction evidence

- `python3 -S scripts/verify_teammate_publication_v25.py` passed on CPython 3.12.3: 484 public files, 46 records, 23 controllers, 8,050 physical episodes, and 16,100 dependent windows.
- `sha256sum -c artifacts/PUBLICATION_SHA256SUMS` passed for all 2,390 listed files.

## Scope

- Read-only verification; no GPU, training, scoring, Git, remote operation, or documentation edit was performed.
