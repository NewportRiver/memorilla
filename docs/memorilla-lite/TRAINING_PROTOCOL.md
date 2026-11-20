# LITE-007 Training Protocol Preparation

This document prepares a bounded training experiment authorized separately by LITE-007. It authorized no training in LITE-006; Android remains limited to the guarded eight-update feasibility pilot.

## Preferred environment

Architect1 is the preferred training environment, subject to live hardware and memory verification. Android inference architecture is not blocked on Architect1 availability. Any Android training run requires separate authorization.

## Android/Termux operator environment

All LITE-007 runtime operations, dataset inspection, dependency checks, and regression commands must use the verified native Android/Termux Python environment for this checkout. Before any operational command, record the runtime home, Python interpreter, repository root, available memory, available storage, numerical thread count, and active process count. Do not silently initialize a replacement environment or database when an existing artifact is expected. Abort rather than continue if the interpreter or repository root is not the intended one. Computational runs must use one numerical thread, one subprocess, a 30-second wall limit, a 256 MiB RSS ceiling, and the guarded RAM thresholds recorded in the recovery checkpoint. Training and optimizer updates remain separately owner-authorized; offline tests and artifact checks do not authorize training.

The inherited upstream test suite may require `transformers` through `tests/conftest.py` and `tests/helpers.py`; this is not a requirement of the LITE precomputed-embedding training package. Run LITE checks in explicit sequential batches with `--confcutdir=tests/lite`, record timeouts as unexecuted, and never convert collection failures into passes by silently excluding tests. Do not install the upstream dependency solely to make an Android-native LITE check appear complete.

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
loss: mean softplus(margin - positive_score + negative_score)
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
