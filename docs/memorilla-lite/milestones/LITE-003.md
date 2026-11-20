# LITE-003 — Lightweight semantic encoder

## Objective
Benchmark 384-D encoder candidates on sanitized synthetic fixtures.

## Scientific rationale
MiniLM/BGE candidates preserve a small embedding dimension but require new memory training.

## Dependencies
LITE-001; LITE-002.

## Implementation steps
Pin model revisions/licenses; encode fixtures; compare latency, RSS, dimension, determinism.

## Tests
Hash inputs/outputs; repeatability and resource tests.

## Acceptance criteria
Candidate chosen by measured accuracy/resource report.

## Resource limits
No private APEX history; bounded fixture size.

## Rollback
Remove derived model/cache; retain manifest.

## Completion reporting
Report candidate decision and rejected alternatives.

## Status
Not started. Owner authorization required before execution.
