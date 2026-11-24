# LITE-008J Scientific Limitations

1. The full-model result is a mechanism qualification, not semantic-memory training. Only one external embedding and one synthetic factual target were optimized.
2. The optimization reduced correct-answer loss, but the zero embedding had a larger correct-over-alternate margin than the optimized embedding (`0.75` versus `0.25` nats). This does not support a memory-specific semantic claim.
3. The BF16 directional finite-difference check was limited: relative error `0.3604` at epsilon `0.5`. The result is reported as not passed, not hidden or relabeled.
4. The counterfactual diagnostic was not executed because the first per-example diagnostic qualified the backward mechanism but did not justify a broader semantic claim or additional budget.
5. Quantized transfer showed candidate-ranking agreement for one optimized embedding only. It does not establish general training/inference compatibility.
6. The forward equivalence control used the same token embedding tensor submitted through `inputs_embeds`; it validates the interface and causal path, not tokenizer semantic independence.
7. No unrestricted generation, multi-example optimization, MemoryModule training, activation-checkpoint comparison, or decoder fine-tuning was performed.
8. No protected final holdout was accessed. Existing production retrieval, SQLite evidence, MiniLM, and LITE-007 checkpoints were not modified.
9. Gate D, Gate E, and Gate F remain unpassed.
