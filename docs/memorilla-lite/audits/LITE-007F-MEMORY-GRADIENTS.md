# LITE-007F Memory Projection Gradients

The exact trainable tensors are:

- `mem_k_w`: `(384, 128)`
- `mem_k_b`: `(128,)`
- `mem_v_w`: `(384, 128)`
- `mem_v_b`: `(128,)`

They are used only in masked memory-summary initialization:

`summary = mean((memory @ mem_k_w + mem_k_b) + (memory @ mem_v_w + mem_v_b))`

The summary is added to the learned slots after query conditioning. Backpropagation then traverses the complete frozen architecture: query-conditioned slots, layer normalization, scaled multi-head dot-product attention, masked softmax, attention aggregation, residual connections, second layer normalization, GELU feed-forward refinement, and the evidence-ranking score head. Frozen downstream tensors still propagate derivatives; only the four selected projection tensors receive optimizer updates.

The NumPy implementation caches only forward intermediates required for reverse propagation. It does not construct a full Jacobian and does not add an autodiff dependency. Invalid and padded records are masked before summary accumulation and attention normalization.

Acceptance gates:

- Finite differences: three seeds and 2/4/6-record collections; maximum absolute error `3.23e-05`, within the predeclared `1e-3` tolerance.
- Isolated PyTorch autograd: loss absolute error `0`; maximum projection gradient absolute error `5.01e-09`.
- Covered multiple positives, hard negatives, masks, padded records, finite values, and deterministic seeds.
- A small projection-only SGD step reduced the diagnostic objective.

This is projection-only associative-memory adaptation. It is not full architecture training.
