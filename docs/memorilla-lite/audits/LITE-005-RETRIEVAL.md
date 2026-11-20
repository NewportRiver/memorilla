# LITE-005 — Evidence-Verified Retrieval and Cognitive Evaluation Baseline

Status: completed bounded development baseline. No learned memory, external reranker, production APEX data, private mission history, new model, package, or dependency was used.

## Architecture

```text
authoritative synthetic source catalog
  -> SQLite FTS5 lexical index
  -> pinned MiniLM query encoder
  -> disposable LITE-004 derived Float32/Float16/INT8 vectors
  -> streamed cosine candidate scan
  -> optional RRF hybrid fusion
  -> exact source resolver and hash verification
  -> verified evidence packet or explicit abstention/fallback
```

Implemented modules:

```text
lite/retrieval/__init__.py
lite/retrieval/controller.py
lite/retrieval/cosine.py
lite/retrieval/lexical.py
lite/retrieval/hybrid.py
lite/retrieval/evidence.py
lite/retrieval/fallback.py
benchmarks/lite/retrieval_benchmark.py
benchmarks/lite/cognitive_evaluation.py
benchmarks/lite/fixtures/development_sources.json
benchmarks/lite/fixtures/development_queries.json
benchmarks/lite/fixtures/EVALUATION_PROTOCOL.json
benchmarks/lite/fixtures/holdout_manifest.json
```

The retrieval controller returns evidence ID, source revision, source hash, excerpt, collection, scope, timestamp, method, score/rank, verification status, omission reasons, fallback state, and query telemetry. It never treats a similarity score as proof that an event occurred.

## Dataset

```text
Scenarios:  12
Sources:    60
Queries:    120
Categories: 10, exactly 12 each
Seed:       20261120
Query hash: d66b545c650b9b0eab793922cfb516099b54c8d6917bc68cdedd0e3dd6ccf169
```

Categories:

```text
exact, paraphrase, failure, correction, association,
contradiction, negation, superseded, malicious, missing
```

Sources and answer labels are separate. The sealed holdout label set is external and absent from Git. Scenario grouping prevents query/source leakage across the development partition.

## Development results

The complete frozen benchmark is in `LITE-005-BENCHMARK.json`. At top-5:

| Method | Recall@5 | Complete support | MRR | Median complete latency |
|---|---:|---:|---:|---:|
| Lexical Float16 | 0.6417 | 0.6583 | 0.5958 | 0.3545 ms |
| Semantic Float16 | 0.6417 | 0.6583 | 0.5958 | 0.3802 ms |
| Hybrid Float16 | 0.6417 | 0.6583 | 0.5958 | 0.7348 ms |
| Semantic Float32 | 0.6417 | 0.6583 | 0.5958 | 0.3881 ms |
| Semantic INT8 | 0.6417 | 0.6583 | 0.5958 | 0.3730 ms |

All three baseline methods produced identical aggregate rankings on this templated fixture. This is a limitation of fixture diversity, not evidence that lexical, semantic, and hybrid retrieval are generally equivalent.

The category-level table is preserved in the benchmark JSON. Contradiction, association, negation, superseded, missing, and malicious categories are tracked independently rather than hidden in an aggregate score.

## Evidence and fallback

Focused tests verify:

- Exact authoritative source resolution.
- Source-hash mismatch rejection.
- Deleted/missing evidence rejection.
- Access-scope rejection.
- Historical cutoff filtering before ranking.
- Model/preprocessing compatibility filtering.
- Corrupt/unavailable derived bank fallback to lexical retrieval.
- Explicit access-scope-required abstention.
- Deterministic top-k validation and tie behavior.
- Evidence packet reference and duplicate accounting.

No unauthorized scope violation or temporal leakage was observed in the development benchmark. A closed semantic store successfully used lexical fallback while preserving evidence verification.

## Representation comparison

| Representation | Mean reconstructed cosine | Minimum reconstructed cosine | Eligible-query top-5 overlap with Float32 | Required-evidence recall delta |
|---|---:|---:|---:|---:|
| Float16 | 1.00000002 | 0.99999976 | 100% | 0.0 |
| INT8 | 0.99996055 | 0.99994558 | 100% | 0.0 |

The all-query minimum overlap is 0% for both formats because the private unauthorized scenario has no eligible candidates under the public scope. This is an availability/policy result, not a quantization ranking result. On 111 queries with eligible candidates, both formats matched Float32 top-five rankings. Float16 is therefore the provisional operational default. INT8 remains experimental because the larger benchmark and future less-templated data may expose ranking changes; its lower cosine reconstruction must not be ignored.

## Attention comparison

The existing untrained NumPy attention reference was exercised with identical query and vector dimensions:

| Records | Attention time | Passes | Output | Peak RSS observation |
|---:|---:|---:|---|---:|
| 100 | 28.52 ms | 2 | 8 × 384 | 97,648,640 B |
| 1,000 | 53.97 ms | 2 | 8 × 384 | 99,643,392 B |
| 5,000 | 173.03 ms | 2 | 8 × 384 | 105,844,736 B |

The slots are untrained and were not used as a semantic answer or evidence ranker. Conventional retrieval remains the scientific control.

## Evidence-packet accounting

Median development query:

```text
Raw eligible collection text: 728 characters
Verified retrieved text:      732 characters
Verified references:          5
Median character reduction:   -0.55%
```

The negative reduction is expected for this small, repetitive fixture and demonstrates why context reduction must be measured rather than assumed. Counts are characters/bytes, not model tokens.

## Resource and restrictions

```text
Peak observed RSS: 105,844,736 bytes
Maximum allowed RSS: 268,435,456 bytes
New storage request: 100 MiB
Configured storage ceiling: 128 MiB
```

The ONNX Runtime Android warning from LITE-003 remains inherited. No package repair, model download, dependency change, production access, external inference, brokerage activity, or sustained endurance test occurred.

## Limitations and blockers

- Overall Recall@5 is 0.6417; this is a baseline measurement, not a success claim.
- The templated fixture makes lexical/semantic/hybrid rankings indistinguishable.
- The sealed holdout has not been evaluated because labels remain owner-controlled and external.
- The development benchmark is not sufficient to establish generalization or tail latency.
- Context reduction is not reasoning improvement.
- Conventional retrieval is not learned associative intelligence.

## Recommended LITE-006

Design a learned-memory architecture only after owner approval, using this baseline as the control. Preserve exact evidence verification, scope/temporal filters, deterministic lexical fallback, Float16 operational storage, sealed scenario-disjoint holdouts, and category-specific reporting. Learned slots must be evaluated for evidence recall and unsupported claims against the same control; no claim of cognitive superiority should be accepted from slot similarity or context compression alone.
