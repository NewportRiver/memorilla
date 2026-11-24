# LITE-008H Architecture Decision

Proceed only with a diagnostic 16-coefficient fixed-basis continuous embedding experiment. Do not integrate it with MemoryModule, SQLite, retrieval, or production checkpoints.

Diagnostic representation:

`z(theta) = theta B`, where `theta ∈ R^16` and fixed `B ∈ R^(16×576)`.

This tests whether direct decoder feedback can move a continuous slot toward a factual answer without beginning unrestricted 221,760-parameter optimization. It is not a deployable memory architecture and has no claim of generalization.

The textual capability prerequisite passed candidate ranking, but the oracle SPSA diagnostic failed to improve over its initial margin or budget-matched random search. Therefore no generalized decoder-likelihood bridge is authorized by the evidence from this milestone.
