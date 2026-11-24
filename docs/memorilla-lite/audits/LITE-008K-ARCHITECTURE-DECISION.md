# LITE-008K Architecture Decision

The scientifically preferred architecture remains a standalone Float32 bridge:

`frozen MiniLM 384-D record representation -> compact bridge -> dedicated 576-D decoder memory position`

Candidate architectures were not comparatively trained because the mandatory numerical and counterfactual-oracle prerequisites failed. No per-entity lookup table, complete MemoryModule, production checkpoint, or associative-memory integration was introduced.

The smallest candidate remains a dense linear projection (`384 x 576 + 576 = 221,760` parameters), followed by possible low-rank or small nonlinear alternatives only after the oracle and gradient gates pass. This remains a design decision, not a generalized training result.
