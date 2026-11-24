# LITE-008N Differentiation Method Comparison

## A. Conventional Float32 execution with reduced temporary allocations

Mathematically direct and easiest to inspect, but complete Float32 parameter residency exceeded the existing RSS ceiling. Reduced temporaries alone did not solve the parameter-storage boundary. Not selected.

## B. Activation checkpointing with frozen decoder parameters

Mathematically equivalent if the checkpointed function preserves masks, positions, rotary embeddings, residuals and all layer operations. It saves activations but still requires all Float32 parameters to remain available during recomputation unless combined with layer-wise materialization. Installed PyTorch supports checkpointing, but the combined implementation is more complex than necessary for this milestone. Not selected as the primary method.

## C. Sequential layer-wise Float32 conversion and recomputation

Selected. Official BF16 weights are retained; one layer is copied to Float32 at a time. Forward outputs are detached at layer boundaries. Reverse execution recreates one Float32 layer at a time and computes a vector-Jacobian product with respect to its input. This is mathematically equivalent to the corresponding frozen-layer composition, adds recomputation, and avoids decoder parameter gradients. It passed a conventional two-layer equivalence test exactly and completed all 30 layers.

## D. Safetensors-backed parameter materialization

Useful for loading-time allocation and verified in LITE-008M through `low_cpu_mem_usage=True`, but safetensors mapping alone does not provide a backward path or prevent autograd from retaining converted parameters. It is a storage prerequisite, not a complete differentiation method.

## E. Manual layer-wise vector-Jacobian products

Selected together with C. It explicitly propagates gradients toward the external memory embedding without constructing a complete decoder parameter-gradient graph. It requires careful handling of causal masks, rotary embeddings, grouped-query attention, residual paths and gated feed-forward operations. The installed LlamaDecoderLayer contract supports the required arguments.

## F. Reduced-memory answer-position vocabulary projection

The implementation computes and retains logits only at the answer position rather than retaining sequence-wide logits. The vocabulary projection remains the official full tied projection; no sparse or approximate vocabulary operation is introduced. This reduces logits retention without changing the objective.

## Failure and recovery

Each run is monitored for the existing RSS, available-memory and timeout limits. A terminated run is not counted as numerical success. Detached layer boundaries and case-level ledger records preserve recoverability without inferring optimizer updates.
