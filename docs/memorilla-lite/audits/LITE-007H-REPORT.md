# LITE-007H Report

Status: operational implementation advanced and bounded learned-slot/query-conditioning pilot executed. LITE-007 remains incomplete and sustained multi-episode training was not performed.

## Native differentiation

All 27 tensors connected to the current ranking objective now have explicit native reverse-mode gradients:

- learned slots
- query conditioning
- memory summary projections
- attention query/key/value/output projections
- layer-normalization scale/bias parameters
- feed-forward weights/biases
- scoring-head parameters

`out_w` and `out_b` remain intentionally unsupported because the ranking objective consumes latent slots before output projection.

Finite differences across all 27 tensors passed with maximum absolute error `1.77868e-05` against a `1e-03` tolerance. An isolated PyTorch reference using identical NumPy-exported weights and inputs passed with loss absolute error `5.96e-08` and maximum gradient absolute error `2.56e-08` across all 27 tensors.

## Curriculum and embeddings

The actual deterministic development corpus already present in the repository satisfies the requested baseline:

- 300 scenario groups
- 1,500 records
- 1,200 queries
- 240 train scenarios / 60 validation scenarios
- 960 train queries / 240 validation queries
- 240 validation queries
- Scenario-disjoint split
- 10 scenario families
- Explicit multi-hop, multi-evidence, insufficient-evidence, and access-policy categories
- Holdout excluded

The pinned embedding artifact contains:

- 1,500 record embeddings
- 1,200 query embeddings
- 384 dimensions
- Float32 values
- Existing pinned MiniLM encoder contract
- Content SHA-256 `ee90683bfef5bafdebe005647c513dd319596f288edb6a0492ff4060fbac3746`

Prepared lazy artifacts load successfully:

- Train: 960 examples, 715 pairwise-eligible
- Validation: 240 examples, 175 pairwise-eligible

The corpus was independently verified for scenario partitioning, temporal eligibility, evidence identities, hard-negative eligibility, and holdout exclusion. A separate corpus-wide near-duplicate scan and resumable embedding preparation were not independently executed in H and remain limitations.

## Controls

Implemented:

- Lexical retrieval
- Hybrid lexical/semantic reciprocal-rank fusion
- Separately trainable direct bilinear query/evidence scorer
- Query-independent slot ablation

Previously evaluated:

- Direct cosine
- Untrained neural memory
- Scoring-head-only
- Memory-projection-only
- Combined scoring/projection
- Query-conditioned associative memory

The baseline validation results remain those recorded in LITE-007F. Ranking-only controls do not claim abstention capability without an explicit decision policy.

## Episode integration

The operational runner now binds:

- Dataset content identity
- Checkpoint identity
- Persistent global update budget
- Atomic checkpoint writes
- Conservative update accounting before optimizer mutation
- Persistent episode status
- Exclusive budget locking for concurrent processes
- RAM/RSS/wall-time guards in the operational runner
- Flushed phase telemetry for future episodes

Fault-injection tests cover persistent budget recovery, identity mismatch, conservative interrupted accounting, atomic state, and concurrent claims. Checkpoint reload equivalence passed for the actual pilot checkpoint.

## Authorized feasibility pilot

Exactly one eight-update episode was executed on the newly supported subset:

- Subset: `slots`, `query_w`, `query_b`
- Updates: `8`
- Initial loss: `0.6864654421806335`
- Final observed loss: `0.6558041572570801`
- Frozen parameters: verified
- Slot parameters changed: yes
- Query-conditioning parameters changed: yes
- Output projection remained frozen: yes
- Checkpoint reload inference equality: passed
- Checkpoint: `~/.cache/memorilla-lite/LITE-007H-SLOTS-QUERY-PILOT/checkpoint`

The pilot was completed before the final resource telemetry instrumentation was added. No RSS, RAM, or wall values are fabricated; the resource telemetry gate for that episode remains not passed. No second subset pilot was executed.

## Regression

Bounded batches:

- Gradient/control/attention/orchestration batch: `29 passed in 0.91s`
- Curriculum/training/integrity batch: `10 passed in 0.32s`
- Full collection: 69 tests collected
- Full collection command exceeded the bounded timeout
- First exact timeout identity: `tests/lite/test_training_abstention.py::test_policy_and_insufficient_examples_are_distinct`

Compilation passed. The complete inherited suite is not certified. Transformers was not installed and no tests were silently excluded.

## Acceptance gates

- Gate A — Complete native gradients: PASSED
- Gate B — Actual generated curriculum: PASSED for existing deterministic corpus; independent duplicate scan remains pending
- Gate C — Prepared embeddings/lazy loading: PASSED for existing artifacts; resumable preparation not independently exercised
- Gate D — Operational controls: IMPLEMENTED; only prior controls fully evaluated
- Gate E — Episode controller integration/recovery: PASSED for bounded tests; checkpoint transition and resource fault coverage remain partial
- Gate F — Authorized pilot: PASSED for eight updates and frozen parameters
- Gate G — Checkpoint persistence/reload: PASSED
- Gate H — Cumulative regression: NOT PASSED; full suite timed out

## Remaining limitations

- No sustained or multi-episode training
- No second parameter-subset pilot
- Resource telemetry was added after the pilot and does not certify that historical episode
- Full suite timeout remains unresolved
- Near-duplicate scan not independently persisted
- Resumable embedding preparation remains incomplete
- Missing full evaluation of newly implemented lexical, hybrid, direct-trained, and query-independent controls
- No Android device acceptance
- No holdout evaluation

LITE-007 remains incomplete.
