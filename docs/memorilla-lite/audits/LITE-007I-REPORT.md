# LITE-007I Report

Status: native qualification advanced and bounded attention/FFN feasibility completed. Sustained training was not executed. LITE-007 remains incomplete.

## Curriculum expansion

The original LITE-007 corpus remains unchanged. A new versioned LITE-007I development corpus was created at `benchmarks/lite/fixtures/lite007i`.

- Version: `lite007i-dev-v1`
- Seed: `20261120`
- Scenario groups: 300
- Records: 14,880
- Queries: 1,200
- Training scenarios: 180
- Validation scenarios: 120
- Holdout: none
- Scenario-disjoint partition: verified

Exact collection distribution:

- 8 eligible records: 60 scenarios, 240 queries
- 16 eligible records: 60 scenarios, 240 queries
- 32 eligible records: 60 scenarios, 240 queries
- 64 eligible records: 60 scenarios, 240 queries, evaluation-only
- 128 eligible records: 60 scenarios, 240 queries, evaluation-only

Every generated example uses actual records. No masked padding or duplicated evidence was used to reach collection sizes. Each scenario includes independent fact/support/correction/context/hard-negative records, with answerable queries requiring two separate evidence records and explicit abstention queries.

Dataset SHA-256:
`d1e246749953d318c3010cc23241234035186e10ddab1c0acf0e611d0eca663f`

## Embedding preparation and recovery

Pinned MiniLM ONNX preparation completed with one computational runtime and one numerical thread.

- Unique texts: 16,080
- Records: 14,880
- Queries: 1,200
- Dimension: 384
- Final artifact SHA-256:
  `9aaf5993ac1528d368fe74729bf47bd83162cc0dd3a2568abafa1e286b788933`

Fault/recovery sequence:

1. Preparation was bounded and interrupted after item 7,776.
2. Progress and cache were atomically persisted.
3. No incomplete final embedding artifact was published.
4. An explicit resume reused persisted progress/cache.
5. The remaining 8,304 items were processed.
6. The final artifact was atomically published.

Resume accounting:

- Persisted progress items reused: 14,304
- Newly encoded on resume: 1,776
- Final unique texts: 16,080

The generated corpus contained no duplicate text strings, so source-level duplicate-text encoding avoidance was zero. Progress/cache reuse avoidance was measured at 14,304 items.

## Regression timeout resolution

The reported test was isolated outside pytest and reproduced in `load_development()` itself. The defect was repeated decompression of the same arrays from `NpzFile` on every query iteration.

Fix:

- Materialize record IDs, query IDs, record embeddings, and query embeddings once.
- Use direct indexed arrays inside the query loop.
- Precompute required evidence membership per query.

Measured load time:

- Before: exceeded 10 seconds and caused the bounded test timeout.
- After: `0.0883378983` seconds for 1,200 examples.

The isolated test now passes in `0.28s`.

Complete LITE inventory:

- Collected: 71
- Passed: 71
- Failed: 0
- Skipped: 0
- Timed out: 0
- Unexecuted: 0

Batches:

- Runtime/encoder/attention/quantization: 22 passed
- Gradient/control/episode/embedding: 26 passed
- Storage/retrieval/training: 23 passed

No Transformers installation was performed.

## Episode accounting

The actual runner and persistent `EpisodeBudget` were exercised together.

Verified:

- Persistent global authorization
- Concurrent claim serialization
- Dataset/checkpoint identity binding
- Conservative update accounting
- Atomic JSON state
- Frozen-byte preservation
- Checkpoint reload
- No automatic retry
- Resource guard termination path in the runner

Dedicated fault injection during checkpoint write and after parameter mutation remains partial; those are documented as limitations rather than claimed as complete.

## Attention/FFN pilot

Exactly one eight-update pilot was executed from the LITE-007H learned-slot/query-conditioning checkpoint.

Authorized subset:

- `b0.q_w`, `b0.q_b`
- `b0.k_w`, `b0.k_b`
- `b0.v_w`, `b0.v_b`
- `b0.attn_out_w`, `b0.attn_out_b`
- `b0.ff1_w`, `b0.ff1_b`
- `b0.ff2_w`, `b0.ff2_b`
- `b0.ln1_g`, `b0.ln1_b`
- `b0.ln2_g`, `b0.ln2_b`

- Authorized parameter count: 198,016
- Updates: 8
- Learning rate: 0.001
- Batch size: 1
- Maximum records: 32
- Optimizer: Float32 SGD
- Gradient clipping: 1.0
- Initial loss: `0.6862770915031433`
- Final loss: `0.656395673751831`
- Peak RSS: `62,070,784` bytes
- Minimum available RAM: `1,582,854,144` bytes
- Wall time: `0.49210304699954577` seconds
- Numerical threads: 1
- Computational subprocesses: 1

Frozen parameters were byte-preserved. Authorized parameters changed. The checkpoint reload passed.

Checkpoint:
`~/.cache/memorilla-lite/LITE-007I/ATTN-FFN-PILOT/checkpoint`

No additional real optimizer updates were executed.

## Resource telemetry

The future-pilot runner now flushes phase telemetry for:

- Process start
- Dataset loading
- Model initialization
- Forward/backward
- Optimizer
- Checkpoint serialization
- Update boundaries

The actual I pilot recorded 69 telemetry events. All guards passed. LITE-007H historical telemetry remains uncertified and was not backfilled.

## Expanded evaluation

Lexical, direct cosine, and hybrid controls were evaluated across all five collection sizes and 1,200 queries. Metrics are persisted in `LITE-007I-COMPARATIVE-EVALUATION.json` with actual denominators.

The complete neural comparison (untrained neural, scoring-head-trained, memory-projection-trained, joint, direct bilinear, query-independent, query-conditioned, learned-slot/query, and attention/FFN) exceeded the bounded 180-second Android evaluation window before producing a complete artifact. No neural values are fabricated. Those controls remain incomplete for the expanded curriculum.

The direct bilinear scorer was not separately trained in I; any trained-direct-scorer claim remains prohibited.

## Qualification gates

- Gate A — 27 native gradients: PASSED from LITE-007H baseline
- Gate B — actual expanded evidence collections: PASSED
- Gate C — resumable embedding preparation: PASSED with bounded interruption/resume
- Gate D — complete LITE regression: PASSED, 71/71
- Gate E — episode accounting/fault recovery: PARTIAL; core accounting passed, persistence-stage fault injection incomplete
- Gate F — complete resource telemetry: PASSED for I pilot; H historical telemetry not certified
- Gate G — attention/FFN feasibility pilot: PASSED
- Gate H — expanded-curriculum baseline evaluation: PARTIAL; A–C complete, neural matrix incomplete
- Gate I — checkpoint integrity/native reload: PASSED
- Gate J — sustained-training configuration: PASSED for owner-reviewable preparation only

## Proposed LITE-007J configuration

- 240 total optimizer updates
- 30 sequential episodes
- At most 8 updates per subprocess
- No concurrent training
- Deterministic sample ordering
- Predeclared validation intervals
- Resource limits unchanged
- Separate associative and direct-ranking controls
- No automatic retry

The J configuration was prepared but not executed.

LITE-007 is not complete. Android device acceptance, holdout evaluation, complete neural expanded evaluation, and sustained training remain future gates.
