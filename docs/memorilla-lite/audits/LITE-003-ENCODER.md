# LITE-003 — Lightweight Android Semantic Encoder Audit

Status: completed with a verified single ARM64 quantized MiniLM artifact and a bounded native Termux ONNX execution path. Package-manager hygiene has a documented blocker described below.

## Resource gate

Preflight before installation and inference:

```text
Available RAM:       1,378,140,160 bytes
Home storage free:   6,829,686,784 bytes
Minimum RAM:         1,073,741,824 bytes
Abort threshold:     805,306,368 bytes
Minimum storage:     5 GiB
```

The gate passed. No endurance testing was performed because thermal and battery telemetry remain unavailable.

## Native runtime

Installed through the native Termux ARM64 package repository:

```text
onnxruntime:        1.27.0
python-onnxruntime: 1.27.0
libre2:             2025-11-05-1
protobuf (apt):     2:33.1-1
```

The package metadata reported 11.0 MB of archives and 54.0 MB additional installed disk usage. No package upgrades were performed. The installation exposed a pre-existing dpkg condition involving nine unconfigured packages. `xdg-utils` was already in the pending configuration set; apt attempted its post-install action, which invoked CPAN and failed. The requested ONNX packages themselves are installed and import successfully, but the unrelated dpkg state remains a system blocker that should be repaired separately by owner-directed maintenance, not by this milestone.

The package post-install also added Python dependencies:

```text
flatbuffers 25.12.19
protobuf    7.36.2
```

The ONNX Runtime import reports:

```text
Version:   1.27.0
Providers: NnapiExecutionProvider, XnnpackExecutionProvider, CPUExecutionProvider
Selected:  CPUExecutionProvider
Device:    CPU
```

It emits this warning:

```text
Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
```

This warning is retained as a compatibility limitation. The native Termux build nevertheless executed the selected model successfully on ARM64. This is operational evidence, not upstream Android support certification.

## Selected encoder and provenance

Exactly one model artifact was acquired:

```text
Repository: sentence-transformers/all-MiniLM-L6-v2
Revision:   1110a243fdf4706b3f48f1d95db1a4f5529b4d41
Artifact:   onnx/model_qint8_arm64.onnx
Bytes:      23,026,053
SHA-256:    4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474
License:    Apache-2.0 according to pinned model-card metadata
```

The pinned revision exposes no standalone `LICENSE` file through the repository path inspected. The audit therefore records the model-card license metadata and does not claim a local license file. Upstream Memorilla’s Apache-2.0 source license remains separately preserved.

Tokenizer/config assets were acquired only for this artifact and are listed with hashes in `LITE-003-MODEL-MANIFEST.json`. Model weights and tokenizer assets remain under the external cache and are excluded from Git.

## Model contract

Verified from the ONNX session and pinned configuration:

```text
Architecture:           BERT/MiniLM
Hidden dimension:       384
Layers:                 6
Attention heads:        12
ONNX inputs:            input_ids, attention_mask, token_type_ids
ONNX output:            last_hidden_state [batch, sequence, 384]
Position capacity:      512
Sentence-transformer:   configured max sequence length 256
Bounded implementation: 128 tokens
Pooling:                attention-mask mean pooling
Normalization:          L2 normalization
Remote code:            disabled/not used
```

The tokenizer preserves `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, attention masks, token type IDs, truncation, and the pinned vocabulary. The implementation is dependency-free BasicTokenizer plus WordPiece and does not install Transformers or sentence-transformers.

## Encoder interface

Implemented in `lite/encoder/onnx_encoder.py`:

- Initialization and model checksum verification.
- Runtime/provider identity reporting.
- Single-document encoding.
- Bounded batch encoding.
- CPU provider selection.
- Maximum two numerical threads.
- 128-token bounded input.
- Mean pooling using the attention mask.
- L2 normalization.
- Finite-value and dimension checks.
- Explicit cleanup through `close()`.
- Resource telemetry capture.

All successful embeddings are float32, finite, exactly 384 dimensions, and unit-normalized.

## Encoder benchmark

Initialization:

```text
290.27 ms
```

| Requested words | Actual tokens | Truncated | Encoding latency | Peak RSS | Norm | Repeat error |
|---:|---:|---|---:|---:|---:|---:|
| 16 | 18 | no | 4.68–6.24 ms | 96,120,832 bytes | 0.99999994 | 0.0 |
| 32 | 34 | no | 7.17–7.53 ms | 96,120,832 bytes | 1.0 | 0.0 |
| 64 | 66 | no | 12.76–12.79 ms | 96,120,832 bytes | 1.0 | 0.0 |
| 128 | 128 | yes | 24.05–24.39 ms | 96,120,832 bytes | 1.0 | 0.0 |

Available RAM remained approximately 1.35 GiB during the benchmark. Swap remained available and no endurance test was run. Peak RSS stayed below the 256 MiB observation ceiling and below the 192 MiB target.

## Semantic evaluation

The evaluation used ten synthetic/public-style text fixtures, including paraphrases, unrelated text, repeated failure, contradiction, negation, similar wording with different meaning, and an intentionally long truncated input. Provisional acceptance was defined before evaluation as: every labeled relevant pair must appear in the top two for its query. This is a retrieval heuristic, not factual-equivalence validation.

### Query: mission outcome

Expected relevant: `mission_outcome`, `mission_paraphrase`

Complete ranking:

```text
1 mission_paraphrase   0.59186113  PASS relevant
2 mission_outcome      0.53227425  PASS relevant
3 long_truncated       0.47056514
4 similar_different    0.37217432
5 contradiction         0.22635680
6 repeated_failure     0.15747955
7 technical_problem    0.14999123
8 technical_paraphrase 0.13849682
9 negation             0.13660893
10 unrelated_weather  -0.07038357
```

### Query: technical failure

Expected relevant: `technical_problem`, `technical_paraphrase`

Complete ranking:

```text
1 technical_problem    0.76442540  PASS relevant
2 technical_paraphrase 0.52092540  PASS relevant
3 repeated_failure     0.43965489
4 contradiction        0.36512190
5 mission_paraphrase   0.25145298
6 mission_outcome      0.24467055
7 negation             0.18829042
8 similar_different    0.18807159
9 long_truncated       0.09932083
10 unrelated_weather  -0.05312932
```

### Query: negation

Expected relevant: `negation`

Complete ranking:

```text
1 negation             0.52088314  PASS relevant
2 contradiction        0.44379488
3 technical_problem    0.33619934
4 repeated_failure     0.26199806
5 mission_paraphrase   0.17796440
6 technical_paraphrase 0.15457395
7 mission_outcome      0.12834203
8 similar_different    0.11022699
9 long_truncated      -0.02670636
10 unrelated_weather  -0.03184438
```

Provisional result: 3/3 query criteria passed. The contradiction ranked second for the negation query, demonstrating that embedding similarity does not resolve logical polarity or factual contradiction. No claim of factual equivalence is made.

## Attention compatibility

The real 384-dimensional embeddings were connected to the existing NumPy reference attention implementation.

Verified:

```text
Corpus embeddings:       [3, 384]
Query embedding:         [384]
Attention output:        [8, 384]
Chunked output:           [8, 384]
Numerical stability:      finite
Chunked agreement:        max error below 1e-5
```

No attention weights were trained. This connection demonstrates interface and numerical compatibility only; it does not demonstrate learned associative memory, Stanford performance, or cognitive intelligence.

## Acceptance and blockers

Passed:

- Native ARM64 ONNX Runtime executes the selected artifact.
- Exactly one encoder artifact was acquired and checksum-verified.
- Tokenization, special tokens, padding, truncation, and pooling are implemented.
- Actual text produces finite 384-dimensional embeddings.
- Semantic provisional criteria pass for all three queries.
- Encoder-to-attention interface passes.
- Resource limits remained satisfied.
- Existing LITE-002 and new encoder tests pass.
- Model provenance and license metadata are documented.
- Model cache and temporary artifacts remain outside Git.

Blockers/limitations:

- ONNX Runtime officially warns that Android is unsupported upstream, despite native Termux execution succeeding.
- The Termux dpkg state contains pre-existing unconfigured packages; the ONNX install attempted unrelated `xdg-utils` configuration and failed through CPAN. No upgrades occurred, but package-manager repair is required before unrelated package operations.
- Thermal and battery telemetry remain unavailable, so endurance acceptance is not established.
- No trained semantic adaptation or domain-specific evaluation was performed.
- No production APEX integration was performed.

## Recommended LITE-004

Implement persistent embedding storage over the existing authoritative evidence boundary:

1. Store artifact ID, source hash, encoder revision, model checksum, vector dimension, dtype, and vector bytes.
2. Keep vectors disposable and rebuildable.
3. Reject source-hash or model-revision mismatches.
4. Preserve exact evidence resolution and provenance.
5. Add SQLite BLOB persistence and migration/idempotency tests.
6. Use the 384-D normalized encoder output without changing authoritative APEX records.
7. Keep storage bounded and measure per-vector overhead.
8. Require separate owner authorization before implementation.
