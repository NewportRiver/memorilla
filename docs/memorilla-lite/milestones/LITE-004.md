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
Not started. Owner authorization required before execution.
