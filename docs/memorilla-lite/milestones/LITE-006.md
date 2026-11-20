# LITE-006 — Query-conditioned neural attention

## Objective
Implement compact learned slots over stored embeddings.

## Scientific rationale
The scientific objective is query-conditioned collection-wide latent memory, not discrete RAG.

## Dependencies
LITE-004; LITE-005.

## Implementation steps
Prototype 8 slots/384-D/4 heads; compare full-store attention; export isolated model.

## Tests
Shape, gradient, deterministic inference, association and exact-fact tests.

## Acceptance criteria
Passes model/schema/provenance tests and improves association without safety regression.

## Resource limits
CPU-first; bounded corpus; no production integration.

## Rollback
Remove experimental model and code branch.

## Completion reporting
Report architecture, training status, metrics, and unresolved questions.

## Status
Not started. Owner authorization required before execution.
