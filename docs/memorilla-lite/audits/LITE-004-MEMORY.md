# LITE-004 — Persistent Cognitive Memory Bank Audit

Status: completed as a bounded synthetic/public-fixture derived memory bank. The implementation is disposable and rebuildable; original evidence remains external and authoritative. No production APEX database or private mission data was accessed.

## Baseline and controls

```text
Baseline: f8ac4989571809ba9463b95e7c20fdcd9d733815
Branch:   feature/memorilla-lite-android
Storage:  SQLite only
New dependencies: none
Model downloads:  none
Package repair:   none
Training:         none
```

Resource preflight passed before implementation benchmarks:

```text
Available RAM:      1,534,119,936 bytes
Storage free:       6,727,188,480 bytes
Minimum RAM:        1 GiB
Abort RAM:          768 MiB
Target RSS:         192 MiB
Observed ceiling:   256 MiB
Storage budget:     128 MiB
Threads:            2 maximum
Concurrent tests:   1
Timeout:            30 seconds
```

Temporary databases, WAL/SHM files, fixtures, and caches were created outside the repository and removed after each benchmark.

## Architecture

The derived memory bank is a SQLite cache over evidence that must be resolvable through an injected authoritative evidence resolver:

```text
source record + source hash + scope
        -> contract/idempotency key
        -> encoder
        -> float32 / float16 / INT8 vector bytes
        -> checksum + provenance row in derived SQLite
        -> resolver verification before retrieval
        -> bounded iterator
        -> streaming global-normalized attention
```

The bank never stores a copy of the original document. It stores the evidence identity, source hash, provenance, revision, scope, timestamps, collection/chunk boundaries, encoder contract, vector metadata, vector bytes, checksum, and validity status.

Implemented modules:

```text
lite/memory/schema.py
lite/memory/provenance.py
lite/memory/quantization.py
lite/memory/store.py
lite/memory/iterator.py
```

## Schema

Schema version: `1`.

The `embeddings` table includes:

```text
memory_id
idempotency_key
evidence_id
source_hash
provenance
source_revision
access_scope
source_timestamp
ingestion_timestamp_ns
collection_id
chunk_id
chunk_start
chunk_end
model_identity
encoder_revision
artifact_sha256
tokenizer_identity
preprocessing_identity
dimension
dtype
quantization_method
quantization_scale
vector_bytes
vector_checksum
validity_status
invalid_reason
```

Validity states are:

```text
valid
invalid
superseded
corrupt
```

SQLite is configured with:

```text
WAL journal mode
FULL synchronous mode
foreign key enforcement
parameterized statements
```

The idempotency key includes source content hash, evidence identity, access scope, collection/chunk identity, model identity, encoder revision, artifact checksum, tokenizer identity, preprocessing identity, and dimensionality.

## Quantization formats

Float32:

```text
little-endian IEEE-754 binary32
4 bytes per dimension
scale: none
```

Float16:

```text
little-endian IEEE-754 binary16
2 bytes per dimension
scale: none
```

Symmetric INT8:

```text
signed int8, little-endian byte representation
scale = max(abs(vector)) / 127
minimum positive float32 scale for zero vectors
reconstruction = int8_value * scale
1 byte per dimension
```

All formats use SHA-256 over the serialized vector bytes. Decode rejects malformed lengths, checksum mismatches, invalid scales, unsupported types, NaN, infinity, and non-1-D vectors.

## Encode-once and invalidation behavior

Verified behavior:

- Unchanged evidence and identical encoder contract return the existing row without calling the encoder.
- Distinct scopes remain distinct in the idempotency key and retrieval filters.
- Changed source content creates a new source revision and supersedes the prior compatible source row.
- Changed model, artifact, tokenizer, or preprocessing contract does not silently reuse an incompatible vector.
- Explicit incompatible-model invalidation marks incompatible valid rows invalid.
- Missing evidence, deleted evidence, source-hash mismatch, and cross-scope access fail closed.
- Retrieval verifies the authoritative evidence resolver before returning a vector.
- Encoder failures occur before the write transaction and leave no partial row.
- SQLite WAL and transactional writes support close/reopen recovery.

## Actual MiniLM ingestion benchmark — Group A

Fixtures: 100 unique synthetic public text records encoded by the LITE-003 MiniLM encoder.

```text
Encoder:       all-MiniLM-L6-v2 ARM64 quantized ONNX
Dimension:     384
Input bound:   128 tokens
```

| Format | Ingestion time | Duplicate time for 100 records | DB size | Raw vector bytes | Mean cosine | Minimum cosine | Mean top-5 overlap | Minimum top-5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Float32 | 645.48 ms | 5.10 ms | 483,328 B | 153,600 B | 1.00000005 | 1.00000000 | 100% | 100% |
| Float16 | 643.74 ms | 5.04 ms | 212,992 B | 76,800 B | 1.00000003 | 0.99999964 | 100% | 100% |
| INT8 | 649.06 ms | 5.11 ms | 176,128 B | 38,400 B | 0.99997665 | 0.99997056 | 96% | 80% |

The duplicate pass skipped all 100 existing records. The approximately 5 ms duplicate latency is database/idempotency work; no encoder call was required for those duplicates. The benchmark’s separate measurement pass performed additional encoder calls only to compare reconstructed vectors against fresh float32 references.

The suggested INT8 mean cosine target of 0.995 passed. The suggested top-five overlap target of 95% passed on the mean but failed on the minimum per-query overlap: one of five queries reached 80%. This is disclosed as a partial accuracy result; INT8 is not certified as universally ranking-preserving.

## Synthetic numerical storage benchmark — Group B

Synthetic INT8 vectors, one transaction per record, bounded iteration in chunks of 512:

| Records | Insert time | Iterate time | SQLite files | Raw vector bytes | Metadata/index overhead | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 75.88 ms | 8.43 ms | 176,128 B | 38,400 B | 137,728 B | 96,985,088 B |
| 1,000 | 522.56 ms | 76.30 ms | 1,433,600 B | 384,000 B | 1,049,600 B | 96,985,088 B |
| 5,000 | 2,797.24 ms | 410.38 ms | 7,057,408 B | 1,920,000 B | 5,137,408 B | 96,985,088 B |
| 10,000 | 5,135.37 ms | 827.07 ms | 14,069,760 B | 3,840,000 B | 10,229,760 B | 96,985,088 B |
| 25,000 | 13,214.01 ms | 2,113.99 ms | 35,221,504 B | 9,600,000 B | 25,621,504 B | 96,985,088 B |

All requested synthetic sizes, including 25,000, stayed within the 128 MiB storage budget and observed RSS ceiling. The benchmark is storage throughput only and must not be confused with 25,000 actual semantic encoder inferences.

## Evidence integrity and recovery tests

Passed failure and recovery cases:

- Original evidence missing.
- Source hash mismatch.
- Model revision mismatch.
- Tokenizer/preprocessing mismatch.
- Invalid evidence identity.
- Cross-scope access attempt.
- Corrupt vector bytes.
- Deleted source record after ingestion.
- Superseded source revision.
- Duplicate ingestion.
- Transaction rollback after encoder failure.
- Database close/reopen persistence.
- Vector checksum verification after restart.
- Explicit source invalidation.
- Incompatible model invalidation.
- Bounded iteration at 128, 256, and 512 records.
- Deterministic ordering by `memory_id`.

A corrupt row is never returned as validated vector data. A missing or unverifiable source is never returned as validated historical knowledge.

## Streaming attention compatibility

The existing attention reference was extended with a restartable two-pass `attention_stream` function. It computes global normalization without materializing the entire persistent collection in one call.

For 300 persisted vectors:

```text
Stored vectors:       [300, 384]
Iterator chunks:      128 records
Attention output:     [8, 384]
Finite output:        yes
Query-dependent:      yes
Deterministic:        yes
Stream/reference max: below 2e-5
```

No attention weights were trained. This validates persistence, iteration, and numerical compatibility only; it does not demonstrate learned associative memory.

## Test results

```text
25 passed, 1 warning in 2.53s
```

The warning is the previously documented ONNX Runtime Android support warning from LITE-003. All LITE-002 and LITE-003 regression tests continue to pass.

## Acceptance status

Passed:

- Persistent SQLite vector storage.
- Duplicate ingestion avoidance.
- Evidence-linked provenance.
- Source/model/preprocessing invalidation.
- Scope-preserving retrieval.
- Restart persistence.
- Float32, Float16, and INT8 representation.
- Bounded 128/256/512 iteration.
- Streaming attention compatibility.
- Resource limits.
- Synthetic corruption and recovery tests.
- No production service or APEX database changes.

Partial / limitations:

- INT8 mean cosine target passed.
- INT8 mean top-five overlap passed.
- INT8 minimum top-five overlap was 80%, below the suggested 95% per-query target. Float16 or Float32 remains the safer ranking format until a larger labeled benchmark is performed.
- SQLite metadata/index overhead is substantial relative to raw INT8 payload at small collection sizes.
- WAL and journal overhead is measured in the temporary benchmark but should be rechecked under the eventual deployment lifecycle.
- Thermal and battery endurance remain unmeasured.
- No trained attention or learned cognitive behavior was evaluated.

## Recommended LITE-005

Implement a bounded query/retrieval control layer over the derived bank:

1. Add explicit query encoding and scope-aware candidate retrieval.
2. Compare Float32, Float16, and INT8 retrieval against labeled synthetic evidence.
3. Require exact source resolution before returning any historical claim.
4. Preserve deterministic fallback when the bank is unavailable or invalid.
5. Measure top-k/cosine baseline against streamed attention at 100/1,000/5,000 records.
6. Keep the bank disposable and rebuildable from authoritative fixtures.
7. Do not introduce learned attention weights without a new authorization and training/evaluation plan.
