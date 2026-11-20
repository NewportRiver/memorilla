# LITE-011 — Cognitive performance evaluation

## Objective
Evaluate association, exact facts, repeated failures, unsupported claims, cost and fallback.

## Scientific rationale
Infrastructure correctness is distinct from improved autonomous reasoning.

## Dependencies
LITE-005 through LITE-010.

## Implementation steps
Freeze metrics; run sealed evaluation splits; compare Mode A/B/C/D where available.

## Tests
Leakage, provenance, metric confidence, statistical repeatability.

## Acceptance criteria
All safety gates pass; improvements are evidence-backed or marked absent.

## Resource limits
No sealed holdout access beyond authorized evaluation protocol.

## Rollback
Remove evaluation artifacts except sanitized reports.

## Completion reporting
Report verified, estimated, blocked, and unresolved results.

## Status
Not started. Owner authorization required before execution.
