# LITE-007 — Memory module training

## Objective
Train the compact module on synthetic and authorized mission-disjoint data.

## Scientific rationale
Upstream checkpoint dimensions cannot be reused; new endpoint mapping needs new training.

## Dependencies
LITE-006; owner authorization.

## Implementation steps
Use synthetic associative tasks, optional bounded distillation, then APEX-format fixtures; maintain provenance.

## Tests
Split leakage, checkpoint hash, reproducibility, loss/metric and rollback tests.

## Acceptance criteria
Training artifact reproducible and no holdout/private leakage.

## Resource limits
No cloud expense; bounded Architect1 budget; no production APEX.

## Rollback
Delete checkpoints and reset isolated run.

## Completion reporting
Report data/license manifest, metrics, costs, and limitations.

## Status
Not started. Owner authorization required before execution.
