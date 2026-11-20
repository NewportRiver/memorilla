# LITE-007G Native Differentiation

LITE-007F verified reverse-mode differentiation only for the four memory-summary projection tensors. The implementation correctly propagates through the frozen downstream stack, but it does not yet expose derivatives for every connected tensor.

Verified:

- `mem_k_w`, `mem_k_b`, `mem_v_w`, `mem_v_b`
- Corrected ranking loss, score-head normalization, latent slots, layer normalization, attention softmax, attention aggregation, residual paths and GELU are covered as upstream derivative operations.

Unsupported but connected:

- `slots`
- `query_w`, `query_b`
- `b0.q_w`, `b0.q_b`
- `b0.k_w`, `b0.k_b`
- `b0.v_w`, `b0.v_b`
- `b0.attn_out_w`, `b0.attn_out_b`
- `b0.ln1_g`, `b0.ln1_b`, `b0.ln2_g`, `b0.ln2_b`
- `b0.ff1_w`, `b0.ff1_b`, `b0.ff2_w`, `b0.ff2_b`

Disconnected from the present ranking objective:

- `out_w`, `out_b`

`out_w/out_b` are intentionally not assigned fabricated gradients. They require an objective that consumes output-space vectors or a decoder/inference contract. The complete differentiation curriculum therefore remains staged rather than claiming full-model training.

No optimizer execution was performed for LITE-007G. The prior LITE-007F projection pilot remains the latest authorized associative update.
