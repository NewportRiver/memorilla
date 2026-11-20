# LITE-007G Parameter Inventory

Source: `lite/neural/numpy_model.py`, `NeuralConfig`, architecture version `lite006-associative-v1`, seed `20261120`.

| Group | Exact parameters | Current gradient status | Current optimizer status | Notes |
|---|---|---|---|---|
| Learned slots | `slots` | not implemented in native reverse mode | frozen | Connected through slot initialization and attention path |
| Query conditioning | `query_w`, `query_b` | not implemented | frozen | Connected through initial slot state |
| Memory summary | `mem_k_w`, `mem_k_b`, `mem_v_w`, `mem_v_b` | verified in LITE-007F | projection-only pilot authorized/executed | Only projection tensors currently supported by native associative gradients |
| Attention query | `b0.q_w`, `b0.q_b` | not implemented | frozen | Connected through normalized slots |
| Attention key/value | `b0.k_w`, `b0.k_b`, `b0.v_w`, `b0.v_b` | not implemented | frozen | Connected through memory attention |
| Attention output | `b0.attn_out_w`, `b0.attn_out_b` | not implemented | frozen | Connected through residual attention output |
| Layer normalization | `b0.ln1_g`, `b0.ln1_b`, `b0.ln2_g`, `b0.ln2_b` | not implemented | frozen | Connected through both block normalization paths |
| Feed-forward | `b0.ff1_w`, `b0.ff1_b`, `b0.ff2_w`, `b0.ff2_b` | not implemented | frozen | Connected through GELU residual refinement |
| Output projection | `out_w`, `out_b` | disconnected from current ranking objective | frozen | Current ranking consumes latent slots before output projection; requires output-space objective or decoder integration |
| Scoring head | `score_slot_w`, `score_slot_b`, `score_mem_w`, `score_mem_b` | verified in LITE-007C/E | scoring-head pilot executed | Ranking objective directly consumes this head |

Inventory count: 29 tensors, 429,312 Float32 parameters.

The native differentiation implementation must not fabricate gradients for disconnected `out_w/out_b`. The next supported expansion should add learned slots and query conditioning, then attention/FFN tensors in separate verified subsets.
