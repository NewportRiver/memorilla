# LITE-005 — Conventional retrieval baseline

## Objective
Establish cosine/top-k control metrics.

## Scientific rationale
Conventional retrieval is a baseline, not equivalent Memorilla functionality.

## Dependencies
LITE-003; LITE-004.

## Implementation steps
Create mission-disjoint fixtures; measure exact facts, associations, latency, evidence resolution, unsupported claims.

## Tests
Deterministic replay and baseline metric suite.

## Acceptance criteria
Baseline metrics frozen before learned module work.

## Resource limits
Synthetic/public fixtures only; no holdout.

## Rollback
Delete benchmark outputs; preserve manifests.

## Completion reporting
Report baseline table and known failure modes.

## Status
Not started. Owner authorization required before execution.
