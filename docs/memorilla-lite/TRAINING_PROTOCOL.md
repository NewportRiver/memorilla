# LITE-007 Training Protocol Preparation

This document prepares a bounded training experiment authorized separately by LITE-007. It authorized no training in LITE-006; Android remains limited to the guarded eight-update feasibility pilot.

## Preferred environment

Architect1 is the preferred training environment, subject to live hardware and memory verification. Android inference architecture is not blocked on Architect1 availability. Any Android training run requires separate authorization.

## Android/Termux operator environment

All LITE-007 runtime operations, dataset inspection, dependency checks, and regression commands must use the verified native Android/Termux Python environment for this checkout. Before any operational command, record the runtime home, Python interpreter, repository root, available memory, available storage, numerical thread count, and active process count. Do not silently initialize a replacement environment or database when an existing artifact is expected. Abort rather than continue if the interpreter or repository root is not the intended one. Computational runs must use one numerical thread, one subprocess, a 30-second wall limit, a 256 MiB RSS ceiling, and the guarded RAM thresholds recorded in the recovery checkpoint. Training and optimizer updates remain separately owner-authorized; offline tests and artifact checks do not authorize training.

The inherited upstream test suite may require `transformers` through `tests/conftest.py` and `tests/helpers.py`; this is not a requirement of the LITE precomputed-embedding training package. Run LITE checks in explicit sequential batches with `--confcutdir=tests/lite`, record timeouts as unexecuted, and never convert collection failures into passes by silently excluding tests. Do not install the upstream dependency solely to make an Android-native LITE check appear complete.

LITE-007C's native NumPy path must not import PyTorch. The scoring-head pilot may train only `score_slot_w`, `score_slot_b`, `score_mem_w`, and `score_mem_b`; all other architecture state is frozen and byte-checked. Use plain SGD before considering AdamW. A pilot that fails the RAM/RSS/time guards is a blocked run with zero updates, not a failed training result and must not be retried automatically.

The corrected ranking objective is `softplus(margin + negative_score - positive_score)`. Every future pilot must perform directional and finite-difference checks before launch. Native startup and dataset-loading phases must be separately flushed to telemetry; a timeout in either phase is an unexecuted pilot, not a zero-loss or training result.

The native pilot uses a deterministic, content-addressed, train-only prepared artifact when eager development loading cannot meet the wall limit. The artifact must include source dataset/embedding/encoder hashes, scenario and partition identities, eligibility masks, and atomic metadata. Training retrieves one example at a time and must preserve the original positive-and-negative usable-example filter; abstention examples are not optimization examples.

LITE-007F projection adaptation may update only `mem_k_w`, `mem_k_b`, `mem_v_w`, and `mem_v_b` after finite-difference and isolated autograd parity checks. The score head and all other tensors remain frozen. Projection-only loss reduction is an optimization observation, not a cognition or generalization claim; paired untrained, scoring-head, projection, combined, and direct retrieval controls are required before interpretation.

LITE-007G adds no optimizer episode. Future episodes must use a persistent identity-bound global authorization budget across fresh subprocesses, atomic checkpoint metadata, deterministic ordering, and fail-closed recovery. Collection levels above 32 records are evaluation-only until separately authorized.

LITE-007H operational episodes use the persistent budget with exclusive locking, conservative pre-update accounting, atomic checkpoints, and phase telemetry. A newly supported subset may receive at most eight updates per explicit episode; no sustained or multi-episode training is authorized by this milestone.

## Dataset requirements

- Synthetic/public evidence only until separately authorized.
- Development, training, validation, and sealed-test scenarios must be disjoint.
- Positive labels are authoritative evidence IDs, never generated answer text.
- Future and unauthorized records are excluded before label construction.
- Malicious embedded instructions are data content and never execution directives.
- Sealed labels remain external owner-controlled artifacts; only a descriptor hash may be checked in.
- Preserve LITE-005 original and corrected benchmarks as immutable controls.
- Use a more diverse expanded development set with paraphrase, low lexical overlap, distractors, temporal ordering, negation, supersession, multi-document association, and no-answer cases.

## Initial objective

Start with multi-positive pairwise logistic evidence ranking:

```text
positive IDs: authoritative evidence required by the query
negative IDs: valid same-scope/same-collection non-positive records
score: maximum projected slot-to-evidence similarity
loss: mean softplus(margin + negative_score - positive_score)
```

Optional follow-up terms:

- hard-negative weighting;
- coverage loss for multiple required evidence IDs;
- slot diversity penalty;
- calibrated evidence-selection abstention.

No answer labels are passed as inference features. The encoder remains frozen.

## Training controls

- Deterministic seed and recorded dataset hashes.
- Fixed train/validation scenario split.
- Sealed holdout released only by owner after model freeze.
- No hyperparameter search against sealed labels.
- Fixed checkpoint naming containing architecture and dataset hashes.
- Early stopping on validation complete-support and unsupported-return rate.
- Reject checkpoints with evidence-policy violations even if retrieval accuracy improves.
- Compare against LITE-005 original and corrected lexical/semantic/hybrid controls.
- Report per-category and policy-stratified metrics, not only aggregate recall.

## Resource estimate to measure before authorization

- Peak RSS for model, minibatch, optimizer, and checkpoint simultaneously.
- Numerical thread count and process concurrency.
- Dataset and checkpoint storage including temporary optimizer state.
- CPU time per batch and projected epoch time.
- Android export/inference memory after removing training-only state.

If the approved Android envelope cannot support backward validation, the result is a NumPy/reference-only architecture milestone and training remains blocked.

## Ablations

At minimum, compare:

1. frozen conventional retrieval control;
2. query-conditioned slots without FFN;
3. query-conditioned slots without diversity penalty;
4. one versus multiple positives;
5. full attention versus any future routed attention;
6. Float32/Float16 inference representation.

No ablation may use the sealed holdout for tuning.

## Evidence acceptance

Every returned evidence ID must resolve through the authoritative resolver, match source hash and revision, satisfy collection/scope/as-of filters, and carry a verification status. Unsupported neural candidates count as failures regardless of score or attention weight.

## Android export

Export only versioned, non-executable tensor data compatible with the frozen encoder contract. Validate tensor shapes, dtypes, checksums, architecture version, seed, training status, and encoder revision before loading. Randomly initialized or partially trained weights must remain explicitly labeled.
