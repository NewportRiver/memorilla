# LITE-007B NumPy Training Feasibility

Status: research and verification complete; no optimizer update performed.

## Architecture retained

The existing NumPy implementation preserves query-conditioned learned slots, masked memory-summary initialization, multi-head cross-attention, residual paths, layer normalization, feed-forward refinement, fixed 128-D latent slots, and the 64-D evidence-scoring projections. Its state dictionary is already compatible with the existing NumPy export format.

The NumPy implementation is inference-only today: it has forward primitives but no backward graph or optimizer implementation.

## Candidate stages

### A — scoring head only

Train `score_slot_w`, `score_slot_b`, `score_mem_w`, and `score_mem_b` while keeping all associative-memory state fixed. This is feasible with a small manual gradient implementation because normalized cosine-style scoring and the pairwise logistic objective are local to the projected latent slots and memory vectors. It is not full associative-memory training and must be reported as scoring-head adaptation only.

Expected state: 2 x 384 x 64 weights plus two 64-D biases, approximately 196,736 parameters. Float32 parameters, gradients, and AdamW moments together require approximately 3.0 MiB before runtime overhead.

### B — scoring head plus selected memory projections

Add `mem_k_*` and `mem_v_*` to the trainable set. This remains feasible in principle, but gradients must pass through masked summary initialization and attention keys/values. Manual backpropagation is substantially more complex because the same projections affect both summary slots and attention paths. This is the most credible lightweight associative adaptation stage, but it still is not complete-architecture training.

### C — complete architecture

Manual backpropagation would be required through layer normalization, GELU, softmax with masks, multi-head reshape/transposes, residual connections, summary initialization, output projections, and the ranking objective. It is mathematically possible but high-risk and not justified under the current Android envelope. A new framework is prohibited, and no implementation was started.

## Complexity and memory

For one query and N memory records, forward attention is O(S*N*D) per block, with S=8 and D=128 latent width (or 32 per head). A manual backward pass has the same asymptotic order but requires retaining or recomputing intermediates. Recompute is preferable for memory; retained activations are preferable for speed. The existing 429,312-parameter full model has only about 6.55 MiB of raw Float32 parameters, gradients, and AdamW moments, but runtime overhead previously reached 1.58 GB RSS, so raw tensor arithmetic is not a safe Android capacity estimate.

## Gradient verification

The existing PyTorch gradient audit is the reference result: the ranking loss produced nonzero gradients through the repaired memory projections; `out_w` and `out_b` were zero for the initial latent-ranking objective because the scoring path does not depend on the output projection. No optimizer step was performed.

A finite-difference check for a NumPy implementation should use a deterministic one-example objective, perturb one scalar at a time, and compare `(f(theta+h)-f(theta-h))/(2h)` against the manual derivative with a tolerance appropriate to Float32. A PyTorch-reference comparison should load identical NumPy state, compare forward outputs, then compare selected manual gradients against PyTorch autograd gradients. These checks are prerequisites for Stage B or C.

No manual gradient primitive was added in this recovery because the repository has no existing backward contract, and implementing one without a verified reference would create unvalidated training semantics.

## Optimizers

Plain SGD requires only gradients and one learning-rate scalar. AdamW requires two persistent Float32 moment tensors per trainable parameter plus step/bias-correction state. Both would need atomic state serialization and restart tests. Optimizer code must remain separate from inference serialization; an exported trained model must not be mistaken for an optimizer checkpoint.

## Recommendation

If training is later authorized, begin with an offline, finite-difference-verified Stage-A scoring-head prototype on a tiny synthetic fixture. Do not call it associative-memory training. Proceed to Stage B only after PyTorch-reference gradient parity and atomic checkpoint tests pass. Do not attempt Stage C on Android under the current RSS ceiling.
