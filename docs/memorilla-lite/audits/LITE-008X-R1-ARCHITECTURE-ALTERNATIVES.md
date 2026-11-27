# LITE-008X-R1 architecture alternatives

New addendum; [decision](LITE-008X-R1-GO-NOGO-DECISION.md). **Proposals, not approved experiments.** No architecture training or prospective latent acceptance. “Original principles” here means the supplied constraints: compact/local, incremental, query-relevant memory with immutable auditable evaluation; the absent original R1 directive cannot supply additional principles.

Dimensions from retained encoder metadata/local decoder config: d=384, h=576. Calculated payload estimates are in the immutable-audit JSON. No alternative is certified under 256 MiB whole-process RSS; workspace and runtime measurements are absent.

## A — Broader, content-bound teachers

- **Problem/evidence:** X familiar pairs repeat; five selected W-R3 pairs offer narrow lexical/entity coverage. Textual relevance confound and missing raw rows mean failure attribution is incomplete.
- **Benefit:** independently balanced teachers might cover new lexical/template/entity/relational conditions. **Failure:** more examples can repeat the same shortcut or encode inconsistent decoder targets; no demonstrated generalization gain.
- **Parameters/storage/workspace:** retain 221184 dense parameters (884736 FP32 bytes); 1000 single-slot teachers require 2304000 payload bytes, excluding metadata/checkpoints and features. Generation workspace unknown and decoder-dependent.
- **Decoder/optimizer:** qualified decoder required for generation/functional testing; teacher optimization is expressly unauthorized now. No automatic budget inferred.
- **Incremental/SQLite/256 MiB/principles:** add provenance-indexed examples and versioned artifacts only in future; do not touch SQLite now. Teacher storage is small but scientific decoder memory not production-certified. Broadening evidence preserves small inference mapping but increases acquisition cost and may violate intended cheap updates.

## B — Float64 incremental ridge S/T, FP32 weights

- **Problem/evidence:** X recomputes fitted projections inside evaluation and weakly joins identities; incremental published statistics would separate learning from frozen inference.
- **Benefit:** deterministic auditable sample contributions and explicit versioned W. **Failure:** conditioning/feature collisions, target drift, relational incapacity, or decoder objective mismatch persist; rank is not measured.
- **Parameters/storage/workspace:** 221184 parameters; S+T+W payload 3833856 bytes. Factorization workspace unknown, plus copies/runtime.
- **Decoder/optimizer:** no gradient optimizer required for algebraic statistics; target creation/functional evaluation may require decoder. Any future matrix solve/refit requires separate authorization; zero here.
- **Incremental/SQLite/256 MiB/principles:** natural sufficient-statistic updates; atomic versioned snapshots and provenance only by future authorization. Payload fits cap; process RSS unqualified. Preserves compact noniterative update principle, not proof of useful memory.

## C — Query-weighted lexical/cosine retrieval

- **Problem/evidence:** X unrelated-B confound and weak query-conditioned binding suggest relevance needs its own baseline before more latent training.
- **Benefit:** explicit auditable evidence selection, easy incremental insertion/deletion and separate lexical/entity constraints. **Failure:** lexical overlap shortcuts, synonym failure, wrong-entity ties, and multi-hop binding errors; retrieval does not itself make a useful latent slot.
- **Parameters/storage/workspace:** fixed lexical/cosine mixture can have zero learned parameters; 1000 d=384 FP32 vectors cost 1536000 bytes plus text/index/metadata. Streaming scan workspace O(d); exact overhead unknown.
- **Decoder/optimizer:** retrieval alone needs no decoder or optimizer; answer generation still does. Freeze weights on text-only calibration if later authorized; no latent tuning.
- **Incremental/SQLite/256 MiB/principles:** index one record at a time with versioned encoder identity; SQLite schema/index proposal only, no access. Plausibly lower memory, not certified. Strong compact/query-relevance baseline; not evidence of learned semantic latent memory.

## D — Structured relational features

- **Problem/evidence:** D swaps assignments but queries only first entity; bag-like semantic embeddings may lose who-has-what, not yet causally established.
- **Benefit:** encode entity-role-value associations explicitly; test both queries on same-word swaps. **Failure:** parser brittleness, unseen schema, collisions, and hand-engineering rather than general semantics.
- **Parameters/storage/workspace:** illustrative d=768 dense FP32 W costs 1769472 bytes; float64 S/T plus W 10027008 bytes. Actual feature schema/dimension and parser workspace unknown; estimates are not chosen architecture.
- **Decoder/optimizer:** parser features need no decoder; trained mapping/teacher targets require separately authorized learning, and any ridge refit prohibited here.
- **Incremental/SQLite/256 MiB/principles:** append typed facts with content/version identities, atomically invalidate changed relations; future SQLite representation only. Payload small, process unqualified. Makes relational binding explicit but sacrifices unconstrained text flexibility.

## E — Decoder-functional low-rank refinement

- **Problem/evidence:** reconstructing teacher slot coordinates may not preserve answer margins; existing FP32/BF16 paths are unmatched. No functional benefit established.
- **Benefit:** optimize a rank-k residual for decoder behavior rather than vector MSE. **Failure:** overfitting calibration, latent threshold leakage, unstable gradients and extra decoder cost; low rank may cap capacity.
- **Parameters/storage/workspace:** illustrative k=8 gives k*(d+h)=7680 parameters, 30720 FP32 bytes (additional to base if residual). Gradients/optimizer states/activations and decoder workspace unknown; not included in payload.
- **Decoder/optimizer:** decoder forwards/backprop and optimizer required; **not authorized**. No updates or gradient experiment in this addendum.
- **Incremental/SQLite/256 MiB/principles:** future versioned low-rank artifacts with replay/forgetting controls; no SQLite mutation now. Small parameter count does not make decoder refinement production-feasible. Trades cheap algebraic updates for task alignment; diverges from no-optimizer incremental simplicity.

## F — Confidence-based fallback / abstention

- **Problem/evidence:** missing textual competence detail and irrelevant-evidence sensitivity make unconditional latent use unsafe scientifically.
- **Benefit:** route uncertain retrieval/binding to ordinary evidence or abstention; preserve negative cases. **Failure:** poorly calibrated confidence, lexical shortcuts and silent distribution shift; a fallback can conceal latent failure if denominators are merged.
- **Parameters/storage/workspace:** fixed calibrated thresholds can use zero learned model parameters and a few scalars; exact policy table/storage unknown. Reuses chosen backend; no independent memory saving guaranteed.
- **Decoder/optimizer:** abstention/routing need not invoke decoder; textual fallback may. Calibrate separately on ordinary-text partition, never choose thresholds by latent success. No optimizer now.
- **Incremental/SQLite/256 MiB/principles:** version confidence policy with encoder/data identities; track fallback and latent denominators separately, future SQLite logging only. Whole-process cap depends on fallback backend. Preserves conservative/local use but is not proof latent memory succeeded.

## Recommendation

Prioritize C plus F as explicit non-learning baselines, with D as a falsifiable relational probe after benchmark and identity repair. B is an engineering storage/update option, not a remedy established by X. A/E require new teacher/optimizer authorization. None should be launched before original scope, raw provenance, resource monitoring and matched textual calibration are resolved.
