# LITE-005 Evaluation Protocol

Status: frozen bounded development protocol. This is a conventional retrieval control group, not learned associative memory.

## Provenance and partitions

- Repository: NewportRiver/memorilla
- Branch: `feature/memorilla-lite-android`
- Baseline: `4ac8b967e28dea6669eae2f0c53a00adbe77307b`
- Fixture seed: `20261120`
- Development sources: 60 records, 12 scenarios
- Development queries: 120 queries, 10 categories, 12 per category
- Development query canonical SHA-256: `d66b545c650b9b0eab793922cfb516099b54c8d6917bc68cdedd0e3dd6ccf169`
- Sealed holdout descriptor SHA-256: `f6917c335f9b8c1ee29e312de2350fb1472d42c8d8181a0bf3c4feeada0d654c`

Development sources and query labels are separate files. The sealed holdout labels are external and are not stored in this repository.

## Frozen retrieval configuration

- Lexical: SQLite FTS5, lowercase alphanumeric query normalization, OR term matching, deterministic BM25 ordering, evidence ID tie-break.
- Semantic: exact cosine similarity over streamed valid vectors, Float16 operational representation, bounded top-k, memory ID/evidence ID deterministic tie-break.
- Hybrid: reciprocal rank fusion, `rrf_k=60`, equal lexical/semantic contribution, no external reranker.
- Top-k: 1, 3, 5, 10; benchmark reports the frozen top-5 run.
- Encoder: pinned LITE-003 MiniLM contract, 384 dimensions, attention-mask mean pooling, L2 normalization, CPU, two numerical threads.
- Eligibility filters precede ranking: collection, access scope, model/preprocessing contract, and historical cutoff.
- Every returned result requires authoritative source resolution and source-hash verification.
- Semantic failure, stale/incompatible bank, corruption, or zero compatible candidates falls back to lexical retrieval when scope and authoritative fixtures are available.
- Missing/unauthorized/temporally unavailable evidence produces explicit omission or abstention; no historical claim is fabricated.

## Metrics

For each method and category:

- Recall@1/3/5/10
- Precision@k
- Mean reciprocal rank
- Multi-evidence coverage
- Complete supporting-evidence retrieval rate
- Unsupported/invalid return rate
- Scope violation rate
- Temporal leakage rate
- Expected-abstention behavior
- Representation ranking agreement

Relevance and evidence verification are separate outcomes. Cosine similarity is never treated as factual proof.

## Holdout protection

No parameter, fusion weight, representation choice, or acceptance decision may be tuned with sealed labels. The checked-in holdout manifest contains only descriptor metadata and a hash. Final holdout evaluation requires owner-controlled release of the external labels.

## Resource envelope

The existing LITE limits remain frozen: minimum available RAM 1 GiB, abort threshold 768 MiB, target RSS 192 MiB, maximum observed RSS 256 MiB, two numerical threads, one benchmark process, 30-second individual timeout, 128 MiB new storage, and 5 GiB minimum free storage.

## Interpretation boundaries

Context reduction is reported in characters and bytes, not model tokens. Attention output is a resource comparison only; untrained slots are not a semantic answer, evidence ranker, or cognitive result. No external inference, learned attention training, private data, production APEX access, or endurance testing is authorized.
