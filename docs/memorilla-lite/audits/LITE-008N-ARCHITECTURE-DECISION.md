# LITE-008N Architecture Decision

## Selected architecture

Use manual layer-wise FP32 vector-Jacobian products over the official 30-layer SmolLM2-135M decoder:

1. Load official safetensors in BF16 with `low_cpu_mem_usage=True`.
2. Create the dedicated external memory position followed by ordinary query tokens.
3. Construct the official causal mask, position IDs and rotary position embeddings.
4. Convert only the current frozen decoder layer to Float32.
5. Execute its complete decoder computation, including grouped-query attention, rotary embeddings, residual connections, RMS normalization and gated feed-forward computation.
6. Store detached hidden boundaries only.
7. During reverse execution, recreate each layer and compute its input VJP.
8. Compute final tied vocabulary projection only at the answer position.
9. Return the external memory-position gradient; never calculate decoder parameter gradients.

## Rationale

This is the smallest method that is mathematically defensible under the existing Android envelope. Two-layer conventional equivalence passed exactly, and all 30 layers completed below the RSS ceiling. Checkpointing alone would not solve Float32 parameter residency; safetensors mapping alone would not provide differentiation.

The method is a Float32-computation reference using official BF16-stored weights converted per layer. It is not a claim that the serialized artifact contains an independent original Float32 checkpoint.
