# LITE-007G Report — Native Differentiation and Associative Curriculum

Status: bounded audit, differentiation inventory, curriculum specification, evaluation-control registry, and resumable orchestration completed. No LITE-007G optimizer update was executed.

## 1. Parameter coverage

The architecture contains 29 named tensors and 429,312 Float32 parameters. LITE-007C/E verified the four scoring-head gradients. LITE-007F verified `mem_k_w`, `mem_k_b`, `mem_v_w`, and `mem_v_b` through the frozen downstream architecture.

Connected but unsupported native trainable groups are learned slots, query conditioning, attention query/key/value/output projections, layer-normalization parameters, and feed-forward parameters. `out_w` and `out_b` are disconnected from the current ranking objective because ranking consumes latent slots before the output projection. No fabricated gradients were added for them; they require an output-space objective or decoder integration.

## 2. Mathematical verification

The preserved LITE-007F verification remains valid:

- Finite-difference maximum absolute error: `3.2231e-05`, tolerance `1e-03`.
- Isolated PyTorch maximum absolute error: `5.006e-09`, tolerance `1e-05`.
- Multiple seeds, collection sizes, masks, padding, multiple positives, hard negatives, empty eligibility, nonfinite rejection, and descent were covered.

No new unsupported parameter group was presented as verified in LITE-007G.

## 3. LITE-007F scientific audit

Direct cosine retrieval substantially outperformed the neural controls on the existing synthetic fixture because the frozen query embeddings already expose direct semantic correspondence, while the neural path compresses evidence through random projections, attention, residual refinement, and a max-slot score head. This is a fixture/architecture bottleneck hypothesis, not evidence that the architecture cannot learn.

Eight updates are exposure probes, not meaningful training coverage. The existing scenarios contain at most five eligible records, so Recall@5 saturates at `1.0` for every reported associative control. An eight-slot architecture is therefore not meaningfully challenged by collection cardinality or slot competition. The prior ranking report also omitted abstention accuracy, unsupported retrieval, temporal correctness, and policy-violation denominators; these are now explicit required metrics.

The direct cosine control is not the same as a trained direct query-to-evidence scorer. Control H remains pending and must be trained separately rather than substituted with cosine.

## 4. Curriculum

A separate development-only curriculum specification defines collection levels of 8, 16, 32, 64, and 128 records. Levels 64 and 128 are evaluation-only initially; the current 32-record training limit remains unchanged. Required families include cross-record relationships, multi-support facts, corrections/supersession, multi-step association, hard negatives, insufficient evidence, temporal dependencies, cross-document associations, malicious inert content, and failed-approach lessons.

Scenario-disjoint partitions, near-duplicate checks, answer-leakage checks, historical cutoffs, evidence identity checks, source hashes, encoder identity, and holdout exclusion are required. No new curriculum payload or holdout data was generated in this mission.

## 5. Evaluation controls

Implemented and actually evaluated: direct cosine, untrained neural memory, scoring-head-only, memory-projection-only, combined scoring/projection, and query-conditioned associative memory. Pending: lexical, hybrid lexical-semantic, separately trained direct scorer, and query-independent slot ablation.

The control registry distinguishes implemented, evaluated, and pending states.

## 6. Orchestration and recovery

`lite.training.orchestration.EpisodeBudget` now provides an atomic persistent global update budget bound to dataset and checkpoint identities. It supports fresh subprocess resume, deterministic episode accounting, fail-closed identity mismatch handling, atomic JSON replacement, and no automatic retry. Tests cover persistence, budget exhaustion, identity mismatch, and temporary-file cleanup.

No model training episode was run. Optimizer updates in LITE-007G: `0`.

## 7. Resources

All prior limits remain unchanged:

- Minimum available RAM: 1,280 MiB
- Abort RAM: 1,024 MiB
- Maximum RSS: 256 MiB; target below 128 MiB
- Maximum wall time: 30 seconds
- Maximum records per training example: 32
- One numerical thread and one computational subprocess
- No automatic retry after exhaustion

## 8. Verification

Targeted isolated suite: `16 passed in 0.89s`.

Compilation and `git diff --check` passed. Full inherited LITE regression remains incomplete because upstream fixtures require unavailable `transformers`; it was not installed and no tests were silently excluded.

## 9. Checkpoints

All LITE-007E and LITE-007F checkpoints and historical reports remain preserved. No LITE-007G generated weights were published or created. No holdout data, external inference, private mission data, production integration, or brokerage activity was used.

## 10. Remaining limitations

- Full native differentiation is not complete.
- Learned slots, query conditioning, attention, normalization, and FFN parameter gradients remain unsupported.
- Output projection remains disconnected from the ranking objective.
- Curriculum payload generation and evaluation remain future authorized work.
- Several scientific controls remain pending.
- No sustained associative-memory training was performed.
- No Android trained-checkpoint acceptance or holdout evaluation exists.

LITE-007 remains incomplete.
