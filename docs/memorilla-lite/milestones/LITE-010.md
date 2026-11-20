# LITE-010 — Hierarchical memory scaling

## Objective
Evaluate full attention versus Memorilla-H routing at measured corpus sizes.

## Scientific rationale
Hierarchy adds a discrete bottleneck and is not equivalent to full-store Memorilla.

## Dependencies
LITE-005; LITE-006; LITE-009.

## Implementation steps
Test 1k/10k/25k/100k/synthetic 1M; compare missed-evidence rate, accuracy, latency, storage.

## Tests
Deterministic corpus generation, recall and source-resolution tests.

## Acceptance criteria
Adopt hierarchy only if measured tradeoff is documented and acceptable.

## Resource limits
Synthetic/public data; bounded storage and runtime.

## Rollback
Drop cluster indexes and rebuild full-store mode.

## Completion reporting
Report crossover and nomenclature clearly.

## Status
Not started. Owner authorization required before execution.
