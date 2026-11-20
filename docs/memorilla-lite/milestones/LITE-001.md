# LITE-001 — Android hardware and runtime audit

## Objective
Measure actual Android/Termux and Architect1 facts before any model work.

## Scientific rationale
The report explicitly marks RAM/GPU/Termux facts as requiring local verification.

## Dependencies
None; documentation-only.

## Implementation steps
Run uname/getprop/meminfo/df/nproc and Architect1 free/nvidia-smi/python; hash a sanitized manifest.

## Tests
Review commands offline; verify manifest hash and no secrets.

## Acceptance criteria
Facts, versions, RAM/storage/GPU/thermal availability are recorded and reproducible.

## Resource limits
No model downloads; bounded read-only commands; preserve storage floor.

## Rollback
Delete only the audit manifest; source repo unchanged.

## Completion reporting
Report measured, unavailable, and assumed fields separately.

## Status
Not started. Owner authorization required before execution.
