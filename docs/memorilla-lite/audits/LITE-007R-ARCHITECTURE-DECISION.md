# LITE-007R Architecture Decision

## Decision

Retain the existing hybrid/conventional retrieval path as the operational Memorilla-Lite architecture. Do not replace it with the trained associative memory ranker. Keep latent memory conditioning as a separate research objective rather than a retrieval implementation requirement.

## Candidate assessment

| Candidate | Accuracy evidence | Multi-evidence completeness | Latency | Peak RSS | Complexity | Android compatibility | Decision |
|---|---|---|---|---|---|---|---|
| Existing hybrid retrieval | Strong conventional artifact; required-only recall/completeness `1.0` on 77 queries | `1.0` required-only in LITE-006 | `5.443 ms` mean complete pipeline | Historical exact hybrid RSS unavailable | Low/moderate; existing path | Proven in existing Android-first scope | Operational baseline |
| Direct cosine | Historical LITE-007J size-stratified evidence; not identical independent endpoint | Complete@5 `0.225–0.4917` by collection size | Not reported in comparable artifact | Not reported | Low | Existing conventional path compatible | Control/reference |
| Direct bilinear ranking | Hit@1 `0.8861`, MRR `0.9351`, Complete@5 `0.3361` on LITE-007L independent corpus | Better than associative but not a product storage architecture | Not isolated in LITE-007L | Training telemetry exists; comparable inference latency unavailable | Low parameterized ranker, but learned checkpoint lifecycle required | NumPy/Android feasibility demonstrated | Strong research control, not default product path |
| Associative memory | Hit@1 `0.1222`, MRR `0.2446`, Complete@5 `0.1333` | Weak selective routing; near-uniform attention | LITE-007Q sweep runtime about 13 s for 360-query diagnostic process; per-query latency not isolated | `86,132 KiB` peak in Q process; LITE-007L training peak `127,680,512` bytes | Higher; slots, attention, gradients, checkpoints | CPU feasible, but scientific benefit absent | Do not replace baseline |

Unavailable measurements are deliberately marked unavailable; measurements from different corpus/protocols are not merged into a false ranking.

## Why attention temperature did not justify Gate B

LITE-007Q changed the attention logit spread monotonically:

- Scale `0.5`: spread `0.02079`, entropy `2.772587`
- Scale `1.0`: spread `0.04158`, entropy `2.772582`
- Scale `2.0`: spread `0.08316`, entropy `2.772562`
- Scale `4.0`: spread `0.16633`, entropy `2.772480`

Yet every frozen scale produced Hit@1 `0.1222`, with enrichment approximately `0.995–0.999` and essentially unchanged negative margins. The intervention altered sharpness but did not create evidence-selective routing or improve scores. Because Gate B would add training exposure and checkpoint confounding, no pilot is recommended from this evidence.

## Three separate objectives

1. Persistent evidence storage and retrieval: store, index, rank, cite, and recover evidence reliably. This is the product baseline and is not dependent on latent conditioning.
2. Learned query-conditioned evidence ranking: learn a score function over stored evidence. Direct bilinear ranking is the stronger research control; associative ranking failed to replace it.
3. Compact latent memory conditioning for language models: test whether a compact representation can condition a language model without material context expansion. This is a separate capability question and remains untested by the retrieval experiments.

Failure on objective 2 does not disprove objective 3.

## LITE-008 disposition

LITE-008 is not approved for implementation by this closeout. The feasibility requirements document defines the smallest future experiment and acceptance gates. Owner review is required before any code or training begins.
