# LITE-002 — Minimal runtime installation

## Objective
Prove a bounded ARM64 runtime for encoder experiments.

## Scientific rationale
ONNX Runtime is the recommended first mobile runtime; Termux has Android-specific constraints.

## Dependencies
LITE-001.

## Implementation steps
Record package versions; install only approved minimal packages after owner authorization; run import and CPU smoke tests.

## Tests
Offline import, deterministic tensor test, package inventory.

## Acceptance criteria
Runtime passes CPU smoke test without APEX changes.

## Resource limits
No model or large data download; stop on storage/RAM risk.

## Rollback
Uninstall only added packages or revert isolated environment.

## Completion reporting
Report exact packages, versions, sizes, and failures.

## Status
Not started. Owner authorization required before execution.
