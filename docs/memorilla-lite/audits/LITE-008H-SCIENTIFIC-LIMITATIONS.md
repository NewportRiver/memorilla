# LITE-008H Scientific Limitations

1. The oracle uses one training example and one competing value; it tests influence, not generalization.
2. The 16-dimensional basis is intentionally restrictive and cannot represent the full bridge.
3. SPSA had high estimator variance: its final margin was below its initial margin and below random search.
4. Textual candidate ranking succeeded, but unrestricted one-token greedy generation remained dominated by a repeated decoder token and was not a reliable answer generator.
5. Values are shared across partitions by design; entity and fact identities are disjoint. This does not remove all possible lexical shortcuts.
6. Multi-token candidate scoring and natural completion formats were not promoted to a full optimization run after the oracle prerequisite failed.
7. The H script recorded capability per-call telemetry but did not retain aggregate oracle RSS; this is reported unavailable rather than inferred.
8. No decoder gradients, decoder fine-tuning, complete MemoryModule training, production retrieval, SQLite mutation, GPU/cloud inference, or protected holdout access occurred.
9. Gate D, Gate E, and Gate F remain unpassed.
