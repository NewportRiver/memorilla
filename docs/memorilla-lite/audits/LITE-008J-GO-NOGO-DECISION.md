# LITE-008J GO / NO-GO Decision

## Decision: CONDITIONAL GO for full-model differentiable mechanism qualification

The authorized full-model diagnostic passed the primary mechanism and resource requirements:

- Official SmolLM2-135M loaded locally in BFloat16.
- Parameter count: `134,515,008`.
- Forward token-ID versus equivalent external-embedding logits matched exactly: max difference `0.0`.
- Frozen decoder parameters were verified.
- No decoder parameter gradients were allocated.
- External embedding gradient was finite and nonzero with norm `6.0855`.
- Answer-only loss backpropagated through the full decoder into the external 576-dimensional embedding.
- 32 embedding-only updates reduced correct-answer loss from `9.25` to `4.9375` at the final update.
- Peak training RSS: `622,157,824` bytes, below the temporary 1 GiB ceiling.
- Minimum available memory: `1,037,422,592` bytes, above the 512 MiB reserve.
- Process exited normally within `100.73 s`.

Qualification limits:

- BF16 directional finite-difference verification was not passed; relative error was `0.3604`.
- Zero initialization produced a larger correct-over-alternate margin than the optimized embedding.
- Counterfactual optimization was not executed.
- Q4 transfer agreed on candidate preference for one per-example embedding, but compatibility is not generalized.

Therefore this is a CONDITIONAL GO for the original differentiable decoder-feedback mechanism, not a semantic-memory success. Generalized bridge training, complete associative-memory training, production integration, and Gate D/E/F claims remain blocked pending owner review.
