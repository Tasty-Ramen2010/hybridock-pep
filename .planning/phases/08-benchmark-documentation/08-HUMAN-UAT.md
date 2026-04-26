---
status: partial
phase: 08-benchmark-documentation
source: [08-VERIFICATION.md]
started: 2026-04-26T21:00:00Z
updated: 2026-04-26T21:00:00Z
---

## Current Test

Awaiting human sign-off on two RTX-machine items.

## Tests

### 1. Benchmark accuracy on RTX 5070 (SC1)
expected: `hybridock-pep benchmark --test-csv data/test_complexes.csv` produces Pearson r >= 0.55 (hybrid vs exp. pKd) and delta improvement >= 0.10 over Vina-alone on the 10 held-out complexes
result: [pending — RTX machine required]

### 2. pip-licenses actual output committed (SC4)
expected: Run pip-licenses on both conda envs on RTX machine; replace [PENDING] header in docs/licenses.txt with actual tool output and commit
result: [pending — RTX machine required]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
