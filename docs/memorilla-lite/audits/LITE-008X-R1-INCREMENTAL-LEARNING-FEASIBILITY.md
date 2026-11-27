# LITE-008X-R1 incremental learning feasibility

New addendum; [decision](LITE-008X-R1-GO-NOGO-DECISION.md). **Analytical estimates only. No incremental update, inverse, solve, refit or optimizer was executed.** Exact feature identity recovery remains blocked; no spectrum or effective rank is asserted.

## Sufficient-statistic proposal

For independently authorized future fixed-feature ridge learning with feature dimension d=384 and decoder slot width h=576, maintain `S = sum(x x^T)` and `T = sum(x y^T)` in float64 and a sample-count/provenance ledger. Add a sample with a rank-one S contribution and outer-product T contribution; replacement/deletion needs subtracting the exact prior contribution and an auditable sample ledger. Refresh W only in a separately authorized learning phase, not in evaluation. Regularization is frozen independently, not chosen by development latent outcomes. FP32 inference W is immutable between published versions.

Storage estimates calculated by `python -m experiments.lite008x_r1_audit`, persisted in the immutable-audit JSON:

| Component | Formula | Bytes |
|---|---|---:|
| Dense weights, 221184 parameters | 4*d*h | 884736 |
| S float64 | 8*d*d | 1179648 |
| T float64 | 8*d*h | 1769472 |
| S+T+W, excluding workspace | sum | 3833856 |
| 1000 FP32 features | 4*1000*d | 1536000 |
| 1000 one-slot FP32 teachers | 4*1000*h | 2304000 |
| Example structured d=768 S+T+W | 8*d*d+8*d*h+4*d*h | 10027008 |

These are payload-only lower bounds. Factorization copies, alignment, language/runtime, encoder, indices, text, metadata, WAL, temporary publication and allocator fragmentation are **unknown**, not zero. Float64 accumulation improves rounding but does not repair ill-conditioning, feature collisions, distribution shift or insufficient relational capacity. Incremental ridge can exactly maintain its chosen statistics yet still learn the wrong objective; teacher interpolation is not decoder functional success.

## Deployment boundaries

The 256 MiB production cap is 268435456 bytes. Small S/T/W storage alone fits, but does **not** certify whole-process RSS. Decoder vocabulary embedding alone (49152 by 576) is 56623104 bytes BF16 or 113246208 bytes FP32; all remaining layers/activation/workspace/runtime costs are additional. Existing scientific reference routes are not production qualification. Production decoder availability and end-to-end latency/RSS remain unknown.

SQLite design is a proposal only: content-hash keyed sample provenance, versioned S/T snapshots and W artifact pointer; atomic transaction publishes a complete version, readers pin immutable snapshots, checksums verified after crash recovery, no mixed encoder versions. Do not load/update the production DB to test this proposal. Maintain full rollback provenance and distinguish metadata transactions from scientific tensor updates. Zero SQLite access in this addendum.

Original-principle tradeoff: algebraic incremental updates avoid iterative optimizer steps and can be compact, but require valid targets and periodic authorized matrix computation; they do not by themselves satisfy query-conditioned retrieval, semantic binding, decoder usefulness or continual-learning claims. Prefer retrieval/fallback baselines until content-bound and functional evidence justifies complexity. No authorization inferred from the low payload estimate.
