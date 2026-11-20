# LITE-005 — Evidence-Verified Retrieval and Cognitive Evaluation Baseline

## Status

Completed as a bounded conventional retrieval control on synthetic/public fixtures. Commit and remote verification are pending final regression/release checks.

## Frozen baseline

```text
Baseline commit: 4ac8b967e28dea666e9ae2f0c53a00adbe77307b
Branch:          feature/memorilla-lite-android
Encoder:         pinned LITE-003 MiniLM, 384-D
Default format:  Float16
Reference:       Float32
Experimental:    INT8
RRF constant:    60
Top-k:           1, 3, 5, 10
Seed:            20261120
```

## Dataset

```text
Sources:         60
Scenarios:       12
Development:     120 labeled queries
Categories:      10, 12 queries each
Query hash:      d66b545c650b9b0eab793922cfb516099b54c8d6917bc68cdedd0e3dd6ccf169
Sealed holdout:  external; labels not in repository
```

Source and answer-label files are separate. The dataset includes exact facts, paraphrases, failures, corrections, multi-record associations, contradictions, negation, superseded history, missing evidence, unauthorized scope, and malicious historical text.

## Implemented

- SQLite FTS5 lexical retrieval.
- Streamed semantic cosine retrieval over LITE-004 vectors.
- Deterministic reciprocal-rank-fusion hybrid retrieval.
- Collection, scope, model, preprocessing, and historical cutoff eligibility filtering.
- Exact authoritative evidence resolution and source-hash verification.
- Evidence-level result packet formatting.
- Lexical fallback for missing, stale, corrupt, incompatible, or resource-unavailable semantic banks.
- Explicit abstention and omission reasons.
- Query-level telemetry and bounded top-k validation.
- Float32/Float16/INT8 representation comparison.
- Attention reference resource comparison at 100/1,000/5,000 records.

## Development metrics

Top-5 development results:

| Method | Recall@5 | Complete support | MRR | Median latency |
|---|---:|---:|---:|---:|
| Lexical Float16 | 0.6417 | 0.6583 | 0.5958 | 0.3545 ms |
| Semantic Float16 | 0.6417 | 0.6583 | 0.5958 | 0.3802 ms |
| Hybrid Float16 | 0.6417 | 0.6583 | 0.5958 | 0.7348 ms |
| Semantic Float32 | 0.6417 | 0.6583 | 0.5958 | 0.3881 ms |
| Semantic INT8 | 0.6417 | 0.6583 | 0.5958 | 0.3730 ms |

The templated fixture produced identical aggregate rankings for all methods. This is disclosed as a fixture-diversity limitation.

Representation results:

```text
Float16 mean cosine: 1.00000002
Float16 minimum:     0.99999976
INT8 mean cosine:    0.99996055
INT8 minimum:        0.99994558
Eligible-query top-5 overlap with Float32: 100% for Float16 and INT8
Required-evidence recall delta:             0.0 for Float16 and INT8
```

The all-query minimum overlap is zero only for the unauthorized private scenario with no eligible candidates. Float16 is the provisional operational default. INT8 remains experimental.

Resource measurements:

```text
Peak RSS: 105,844,736 bytes
Attention: 28.52 ms / 53.97 ms / 173.03 ms at 100 / 1,000 / 5,000 records
Attention output: 8 × 384, finite, two passes
```

## Acceptance and limitations

Passed:

- Lexical, semantic, and hybrid operation.
- Evidence verification and source-hash enforcement.
- Zero observed scope violations.
- Zero observed temporal leakage.
- Missing/corrupt/incompatible semantic fallback.
- Deterministic ranking and bounded top-k.
- Float16 and INT8 evaluation against Float32.
- Development dataset integrity and category reporting.
- Attention reference preservation.
- Full resource envelope and previous regressions.

Limitations:

- Recall@5 is 0.6417; poor retrieval is reported, not concealed.
- No sealed holdout evaluation has occurred.
- Fixture phrasing is too templated to strongly differentiate retrieval methods.
- Median context reduction is -0.55%; context compression is not demonstrated.
- No learned memory, training, external inference, or cognitive-superiority claim is made.

## Rollback

Delete temporary benchmark databases and generated cache outputs; preserve committed fixture manifests and audit provenance. The derived store remains disposable.

## Next milestone

LITE-006 may design a learned-memory architecture only after separate owner authorization. It must retain this conventional retrieval control, exact evidence verification, sealed holdouts, and category-specific failure accounting.
