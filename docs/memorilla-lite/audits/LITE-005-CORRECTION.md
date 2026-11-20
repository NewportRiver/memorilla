# LITE-005 Correction Audit

The original LITE-005 artifacts remain unchanged. This audit records corrections discovered before LITE-006 implementation.

## Interface defect

`SemanticRetriever` called the configured encoder as `self.encoder(query)`. The verified LITE-003 `MiniLMEncoder` exposes `encode(text)` and is not callable. `RetrievalController` caught the resulting `TypeError` and silently invoked lexical fallback for every semantic and hybrid query.

Consequences:

- Original semantic and hybrid LITE-005 rankings were lexical fallback rankings.
- Original semantic query-encoding latency was absent.
- Original semantic/hybrid accuracy and quantization comparisons must not be interpreted as real semantic measurements.
- The original benchmark JSON is preserved unchanged for historical provenance.

The corrected interface accepts `.encode(text)` while retaining callable encoders for test doubles.

## Corrected per-query ranking audit

The corrected audit reran all 120 development queries with authoritative evidence verification:

| Comparison | Identical rankings |
|---|---:|
| Lexical vs semantic | 9 / 120 (7.5%) |
| Lexical vs hybrid | 9 / 120 (7.5%) |
| Semantic vs hybrid | 11 / 120 (9.2%) |

This disproves the original apparent method equivalence.

## Corrected timing

| Method | Mean query encoding | Mean complete pipeline |
|---|---:|---:|
| Lexical | 0 ms | 0.467 ms |
| Semantic Float16 | 3.889 ms | 5.013 ms |
| Hybrid Float16 | 0 ms in the controller telemetry path; semantic work is included in complete pipeline | 5.443 ms |

The hybrid implementation does not expose a separate query-encoding field, so its complete-pipeline measurement is authoritative for end-to-end timing. It includes lexical retrieval, semantic encoding/scanning, fusion, and evidence verification.

## Denominator correction

The original aggregate metrics averaged required-evidence recall over all queries, assigning zero to queries with no required evidence or no policy-eligible required evidence. The corrected audit separates the 77 queries with eligible required evidence:

```text
Required-evidence queries:        77
Corrected required-only Recall@5: 1.0
Corrected complete support:       1.0
Policy-expected abstentions:      43
Correct abstentions:               9
```

The abstention result is a fixture/controller protocol finding, not a learned-memory result. Future evaluation must explicitly label policy-ineligible, missing-evidence, and genuine no-answer cases before scoring.

## Context accounting

The original packet count exceeded raw eligible source characters in 111 cases by exactly four characters. `EvidencePacket` joins five excerpts with four newline separators. The packet did not contain additional source content; the discrepancy is formatting overhead. Future reports must separately report source characters, separators, and metadata bytes.

## Holdout custody

The checked-in holdout artifact remains only a descriptor manifest and hash. No independent sealed label file exists in the repository or workspace. This is correctly treated as a custody blocker; no holdout claim is made.

Artifacts:

- `LITE-005-BENCHMARK.json`: original historical result, unchanged.
- `LITE-005-BENCHMARK-CORRECTED.json`: rerun after encoder-interface correction.
- `LITE-005-CORRECTION.json`: per-query and denominator audit.
