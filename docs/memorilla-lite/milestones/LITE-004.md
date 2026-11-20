# LITE-004 — Persistent embedding storage

## Objective
Build a disposable hash-linked vector layer over authoritative artifacts.

## Scientific rationale
Derived vectors must never become a second source of truth.

## Dependencies
LITE-003.

## Implementation steps
Define schema, source hash, encoder revision, quantization metadata, atomic rebuild/invalidation.

## Tests
Corruption, source mismatch, rebuild, idempotency, recovery tests.

## Acceptance criteria
Every vector resolves to a valid source artifact or fails closed.

## Resource limits
Do not alter authoritative SQLite; cap storage.

## Rollback
Drop/rebuild derived store.

## Completion reporting
Report schema, hashes, overhead, and failures.

## Status
Completed as a synthetic/public-fixture disposable SQLite vector bank. Schema version 1, evidence-linked provenance, idempotent ingestion, restart recovery, source/model invalidation, corruption rejection, bounded iteration, and streaming attention compatibility are implemented. Full bounded regression: 25 passed, 1 warning. INT8 mean cosine and mean top-five overlap targets passed, but minimum top-five overlap was 80%, below the suggested 95% per-query target; Float16/Float32 remain safer ranking formats. No production APEX integration. LITE-005 requires separate owner authorization.
