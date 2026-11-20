# LITE-007 Training Audit

Status: preparation complete; Android optimizer pilot blocked before update 1 by the resource envelope. No sustained training occurred.

## Architecture and objective

The existing 429,312-parameter architecture remains in use: 384 input, 128 latent, 8 slots, 4 heads, one cross-attention block, 256 feed-forward width, 384 output, and 64-D scoring projections. The previously declared top-level memory key/value projections were found disconnected from forward execution and were wired into masked memory-summary slot initialization in both NumPy and PyTorch without changing parameter count.

The initial objective is multi-positive pairwise logistic ranking over authoritative positive evidence, ordinary negatives, semantically similar hard negatives, and independent multi-evidence examples. Coverage, diversity, and abstention losses remain separate zero-weight experimental variants and were not silently added.

The MiniLM encoder is frozen. The training loader consumes precomputed 384-D Float32 embeddings and never initializes the encoder concurrently with the optimizer.

## Gradient audit

The actual ranking loss was executed before the pilot guard. After wiring the memory projections, only `out_w` and `out_b` had zero observed gradients. They are the output projection and are not on the initial latent evidence-ranking path. Their zero status is expected for this objective, but they are explicitly tracked as an output-path limitation rather than claimed to be trained retrieval capacity. The previously disconnected memory projections now receive gradients.

## Android pilot

- Optimizer: AdamW
- Learning rate: 0.0003
- Weight decay: 0.01
- Gradient clipping: 1.0
- Batch size: 1
- Maximum updates: 8
- Maximum duration: 30 seconds
- Numerical threads: 2
- Actual optimizer updates: 0
- Stop reason: available RAM fell below 768 MiB during guarded initialization
- Peak RSS: 1,581,723,648 bytes in the guarded pilot process
- Status: blocked; no checkpoint was produced

The earlier standalone forward/backward smoke remains a separate feasibility result and reached 236,376,064 bytes RSS. It did not perform optimizer updates and is not training evidence.

## Environment

No authorized Architect1 connection or configuration was discoverable. Android has no GPU and is authorized only for the tiny feasibility pilot, not sustained training. The final holdout is unavailable beyond its descriptor, so final holdout evaluation is blocked independently.
