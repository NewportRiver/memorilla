# LITE-012 — Experimental APEX integration

## Objective
Prepare an isolated research adapter without production changes.

## Scientific rationale
APEX evidence and policy boundaries must remain authoritative.

## Dependencies
LITE-011; explicit owner authorization.

## Implementation steps
Implement only isolated ingest/query/resolve/health/invalidate adapter; require exact evidence resolution and fallback.

## Tests
Integration contract, authorization, rollback, no-trade/no-tool tests.

## Acceptance criteria
Owner-approved isolated experiment only; no production promotion.

## Resource limits
No brokerage, cloud, or strategy action; preserve APEX state.

## Rollback
Disable/remove adapter and restore prior branch.

## Completion reporting
Report scope, hashes, evidence behavior, and promotion decision.

## Status
Not started. Owner authorization required before execution.
