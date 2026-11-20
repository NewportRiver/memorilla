# LITE-007E Dataset Loader Audit

## Root cause

LITE-007D called `run_pilot -> load_development(root, split='train')`. The original loader eagerly parsed the 1,708,510-byte `dataset.json`, opened the 3,855,431-byte compressed `embeddings.npz`, built full record/query/scenario dictionaries, decoded the complete embedding arrays, and assembled all 960 train examples with `np.stack` before model initialization. It also materialized validation data in other training paths.

Bounded phase measurement showed the archive itself was not the 30-second bottleneck: metadata parsing was 15.2 ms, NPZ open 28.0 ms, record embedding decode 45.7 ms in a fresh process. The expensive boundary was eager example assembly and its full-object materialization inside the prior loader path; LITE-007D had no completion marker before the 30-second wall limit.

The source contains 300 scenarios, 1,500 records, 1,200 queries, 240 train scenarios, 60 validation scenarios, 960 train queries, and 240 validation queries. Embeddings are Float32, 384-dimensional, precomputed by the pinned MiniLM contract. Each scenario has five records; eligibility applies access scope and historical cutoff before positive labels are assigned.

## Optimized architecture

`prepared_dataset.py` performs a separately bounded preparation pass over the verified development source. It writes an indexed, train-only, deterministic NumPy artifact outside Git:

- Query embeddings: `(960, 384)` Float32
- Memory embeddings: `(960, 5, 384)` Float32
- Valid and positive masks
- Query, scenario, evidence, and category identities
- Source dataset, manifest, embedding, and encoder-manifest hashes
- Atomic artifact and metadata replacement
- Flushed resumable progress markers

`PreparedTrainingSet` verifies the content hash, partition, holdout flag, and count before opening the artifact. Training retrieves a single indexed example and never materializes the complete training corpus. Eligibility remains fail-closed and the pilot uses only examples with at least one positive and one negative eligible record.

## Measurements

Preparation of all 960 train queries: 0.296 seconds.

Single-process indexed retrieval measurements:

- 1 example: 19.4 ms
- 8 examples: 137.7 ms
- 32 examples: 517.5 ms
- 100 examples: 1,536.6 ms

The 1-example target is met. The 8-example target is met. The 32/100 measurements are diagnostic batch totals, not pilot requirements.

Pilot dataset-loading phase: approximately 59.6 ms.

## Equivalence and integrity

An independently assembled source-path comparison covered 100 deterministic train examples and passed for identities, scenario membership, eligibility, labels, and exact embedding values. Corruption and missing-artifact rejection passed. Wrong-partition metadata rejection passed. Targeted loader and recovery tests passed.

The preparation artifact contains no holdout material and is bound to the existing encoder contract and source hashes. No source files or generated weights were added to Git.

## Remaining limitation

The original eager loader remains available for legacy/reference paths. Native scoring-head training now requires the prepared train artifact; future preparation interruption recovery should resume from flushed progress before committing a final artifact. No memory-projection training was performed.
