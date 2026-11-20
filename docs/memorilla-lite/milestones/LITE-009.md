# LITE-009 — Android resource optimization

## Objective
Measure and bound RSS, latency, battery, thermal, and process survival.

## Scientific rationale
Android thermal/process limits can invalidate desktop-feasible workloads.

## Dependencies
LITE-002 through LITE-008 as applicable.

## Implementation steps
Run repeated fixed workloads; collect p50/p95, RSS, battery, thermal and OOM evidence; optimize only measured bottlenecks.

## Tests
Endurance, fallback, thermal and storage tests.

## Acceptance criteria
Accepted envelope has zero process death/OOM and deterministic fallback.

## Resource limits
No unattended long jobs; stop at severe thermal pressure.

## Rollback
Revert optimization and remove caches.

## Completion reporting
Report before/after measurements and accepted envelope.

## Status
Not started. Owner authorization required before execution.
