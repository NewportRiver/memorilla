# LITE-008N GO / NO-GO Decision

## Decision: CONDITIONAL GO for owner review

### Passed

- Reduced two-layer conventional FP32 equivalence passed exactly.
- Complete 30-layer layer-wise FP32 reference executed under the unchanged resource envelope.
- All official transformer operations were included: grouped-query attention, rotary embeddings, causal masking, residuals, RMS normalization, gated feed-forward layers, final normalization and tied vocabulary projection.
- Ordinary token-ID and equivalent external-embedding behavior matched exactly.
- External memory gradients were finite and nonzero.
- Isolated negative FP32 and negative BF16 directions reduced correct-answer NLL across three bounded examples.
- Protected artifacts and holdout boundaries were preserved.

### Partial / failed gate

The full BF16 gradient comparison did not meet every retained criterion. Cosine (`0.9952204`) and norm ratio (`1.0040144`) passed, but coordinate sign agreement was `0.9704861`, below `0.98`. The threshold was not changed.

### Stop boundary

Do not claim unrestricted BF16 full-model qualification. Do not start generalized bridge training, oracle requalification, complete associative-memory training, decoder fine-tuning or protected-holdout evaluation. Stop at owner review.

The numerical method is ready for further owner-authorized investigation, but the BF16 comparison remains a documented partial threshold failure.
