# LITE-006 Evaluation Audit

## Scope

This milestone implemented and validated a compact trainable-style neural memory architecture. It did not perform full training, optimizer updates, checkpoint selection, external inference, or holdout evaluation.

## LITE-005 audit findings

The original LITE-005 benchmark is preserved unchanged. A standalone rerun found that `SemanticRetriever` called `MiniLMEncoder` as a callable object even though the verified encoder contract is `.encode(text)`. The controller converted this failure into lexical fallback. Original semantic and hybrid metrics therefore represented lexical fallback, not semantic retrieval.

The correction is implemented as a narrow interface compatibility fix. Corrected artifacts are separate:

- `LITE-005-BENCHMARK-CORRECTED.json`
- `LITE-005-CORRECTION.json`
- `LITE-005-CORRECTION.md`

Corrected exact per-query ranking equality is 9/120 lexical-semantic, 9/120 lexical-hybrid, and 11/120 semantic-hybrid. Corrected semantic query encoding averages 3.889 ms and complete pipeline latency averages 5.013 ms. The original aggregate denominator is also corrected separately by excluding policy-ineligible/no-required queries from required-evidence recall.

The apparent context expansion is four newline separators joining five verified excerpts, not additional source text. The checked-in holdout remains only a descriptor; no independent sealed labels are present.

## Architecture validation

The implementation is `lite/neural` and contains:

- `NeuralConfig` with 384/128/8/4/1/256/384 configuration;
- deterministic seeded learned slot initialization;
- query conditioning;
- masked 4-head cross-attention;
- residual LayerNorm and tanh-GELU FFN refinement;
- fixed 8×384 output;
- full and two-pass streamed NumPy attention;
- compact slot-to-evidence scoring;
- prepared multi-positive ranking and diversity objectives;
- policy-gated authoritative evidence ranking;
- safe checksummed NPZ/JSON serialization;
- lazy PyTorch parity/reference implementation.

Exact parameter count is `429,312`. Raw Float32 weight storage is `1,717,248` bytes. The architecture is random/untrained.

## PyTorch validation

The isolated subprocess used PyTorch `2.11.0`, two numerical threads, a 30-second CPU limit, no optimizer, and no checkpoint. Results:

```text
output:                  [1, 8, 384]
parameter count:         429,312
finite output:           true
NumPy max absolute diff: 1.56e-6
NumPy/PyTorch allclose:  true
nonzero gradient tensors: 21 / 29
peak RSS:                225,116,160 bytes in the final recorded run
```

The initial parity discrepancy was traced to exact versus tanh-approximate GELU. Both implementations now use the same tanh approximation.

## Resource validation

After making PyTorch lazy-loaded, the NumPy benchmark remained inside the approved envelope:

| Records | Full cold | Full warm median | Streamed | SQLite cold/warm | Evidence verification cold/warm | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2.99 ms | 2.24 ms | 2.88 ms | 7.64 / 6.84 ms | 0.024 / 0.005 ms | 39,923,712 B |
| 1,000 | 12.18 ms | 11.39 ms | 16.00 ms | 71.58 / 69.83 ms | 0.022 / 0.005 ms | 54,558,720 B |
| 5,000 | 55.24 ms | 54.71 ms | 73.89 ms | 397.16 / 396.65 ms | 0.033 / 0.006 ms | 101,720,064 B |

The first benchmark attempt reached 277,356,544 bytes because importing PyTorch eagerly polluted the NumPy process baseline. That run is retained as a development observation but is not the accepted measurement. Lazy loading removed the unnecessary footprint without relaxing limits. The final rerun with cold/warm database measurements reached 101,720,064 bytes.

Full/streamed maximum absolute disagreement was below `6.5e-10` at all measured sizes. Streamed attention used two collection passes. SQLite traversal and evidence verification were measured separately from matrix computation.

## Policy tests

The neural scorer cannot authorize evidence. Scope and temporal filtering occur before ranking; returned IDs pass the existing resolver; empty collections abstain. Tests cover private/future distractors, valid lower-ranked evidence, empty memory, masks, invalid tensors, permutation invariance, serialization corruption and compatibility rejection.

## Interpretation boundary

No cognitive superiority, learned associative effectiveness, or training success claim is made. Random weights demonstrate shape, numerical, policy, serialization, and gradient compatibility only. LITE-007 requires separately authorized bounded training and sealed evaluation.
