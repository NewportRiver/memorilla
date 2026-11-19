# Memorilla × APEX Trader — Development Plan
Version 0.1 | 2026-11-19 | Newport River Company | Research only

## Mission
Determine whether learned latent memory improves autonomous reasoning, historical recall, and avoidance of repeated failed actions without sacrificing evidence fidelity or resource discipline. APEX is the first consumer; keep interfaces reusable. Do not modify production APEX, enable brokerage actions, or advance strategy development in this phase.

## Upstream baseline
Preserve upstream main and attribution. Record exact fork commit, dependency versions, checkpoint hashes and dataset licenses in a reproducibility manifest. See upstream README, docs/data.md, docs/evaluation.md, profiling/README.md and https://arxiv.org/abs/2610.21375. Released checkpoints use Qwen3-8B decoder and Qwen3-Embedding-4B encoder; the example requires an 80GB GPU and approximately 25GB downloads. The 16 latent tokens are not text tokens usable in Claude/OpenAI APIs. Upstream reports 18.6% to 24.2% TextWorld success; this is not trading evidence.

## Architecture and boundaries
APEX SQLite and recoverable content-addressed artifacts remain authoritative. Export immutable, sanitized snapshots with record IDs, timestamps, mission IDs, content hash, schema version, provenance and access scope. Derived embeddings/cache are rebuildable, not a second source of truth. Exclude credentials, account identifiers, holdout data and untrusted executable content. No external memory service.
Two modes: (A) native latent-memory inference with matched local decoder; (B) optional local decoder bridge returning bounded textual hypotheses to existing APEX provider. Mode B adds inference and is a research exception requiring owner approval before integration. Keep current APEX cognitive corrective baseline unchanged. No latent output may authorize a tool, trade, or risk-policy change. Exact facts must be retrieved and verified from original records; model-generated citations are untrusted until resolved.
Proposed interface: ingest(snapshot_manifest), query(question, scope, as_of, budget), resolve(evidence_id), health(), invalidate(snapshot_id). Return hypotheses, verified evidence references, unresolved claims, checkpoint identity, latency, resource usage and status. Do not claim latent attention provides faithful citations. On failure, stale cache, budget exhaustion or incompatible schema use current APEX retrieval without retry storms. Invalidate derived cache on source deletion or revision.

## Phases and acceptance gates
### M0 — Source and security audit
Inspect model loading, cache, input embedding injection, data preprocessing, tests, dependency advisories and licenses. Pin trusted revisions; disable remote code, avoid unsafe deserialization, isolate process and network. Run upstream CPU tests, lint and record failures. Gate: reproducible manifest and no unresolved critical security blockers.
### M1 — Hardware feasibility
Architect1 target: RTX 4070 Super 12GB VRAM, 128GB RAM, Linux/WSL2 environment subject to verification. Measure available VRAM under existing load; no assumption that model fits. Start with tiny synthetic corpus, CPU tests, then supported evaluation backend with SDPA where applicable. Test sequential encoder/decoder loading, CPU offload or quantization only where compatible; measure recall degradation. Do not substitute Qwen 9B or 30B without compatibility evidence. Cache disabled initially. Stop on OOM, thermal/resource pressure or budget breach; defer to approved larger GPU if necessary. No automatic cloud spending. Gate: repeatable checkpoint inference and measured peak VRAM/RAM, disk, latency, cold/warm behavior. Android remains an optional client only.
### M2 — Evidence corpus and benchmark
Export read-only APEX mission histories, tool results, learnings, skill evaluations and failures; preserve full artifacts locally. Build labeled questions for exact facts, cross-record relationships, repeated failure detection, contradictory records, temporal reasoning, missing evidence and adversarial tool text. Use upstream parquet columns collection_id, question, answer, documents, with external provenance manifest. Split by mission and time; no future data or holdout access by runtime/training. Deduplicate cross-split. Include synthetic cases before private records. Gate: all cited records resolve and hash-verify; leakage tests pass.
### M3 — Comparative evaluation
Compare current APEX retrieval, no memory, matched local decoder text RAG, matched decoder full context where feasible, native Memorilla, and optional bridge. Use matched queries, seeds, budgets and model settings where possible; disclose confounds. Evaluate exact fact accuracy, association accuracy, unsupported claims, abstention, evidence resolution, repeated failed actions, task completion, total tokens across all models, end-to-end p50/p95 latency, peak RAM/VRAM, cache rebuild costs, calls and estimated cost. Compare end-to-end APEX outcomes separately from local model comparisons. Report sample sizes, uncertainty and per-case failures.
Provisional promotion thresholds (owner to approve before evaluation): 100% cited artifact resolution, zero policy bypass or leakage, exact fact accuracy no worse than baseline by >2 percentage points, and either >=20% fewer repeat failures or >=20% lower total inference cost/latency without material task-success regression. Freeze metrics before holdout. No promotion on inconclusive results.
### M4 — Shadow integration
Propose a separate APEX research branch research/memorilla-latent-memory after approval. Implement feature flag default OFF, read-only adapter, bounded timeout, circuit breaker, resource telemetry and deterministic fallback. Observe outputs without changing decisions; compare with historical baseline. All model inputs/outputs sanitized and tracked via existing evidence architecture. Test restart, cache invalidation, empty/corrupt corpus, injection, timeout, OOM and provider outage. Gate: reproducible benchmark and rollback drill; owner approval.
### M5 — Controlled adaptation (optional)
Use released checkpoint first. Only if measured deficiencies justify it, evaluate upstream single-task warm-start using mission-disjoint training/validation data. No holdout tuning, no automatic retraining on self-generated conclusions, no unverified lessons. Document training budget and checkpoint lineage. RL is deferred until supervised benchmark demonstrates need.
### M6 — Release readiness
Owner reviews benchmark, security, hardware impact and integration diff. Keep conventional retrieval intact. Document opt-in configuration, uninstall/rollback, cache lifecycle and licenses. No trading authorization or strategy promotion implied.

## Proposed fork layout
docs/newportriver/APEX_INTEGRATION_PLAN.md (this plan)
integrations/apex/ (future adapter and contracts)
benchmarks/apex/ (future synthetic fixtures, harness and metrics)
tests/integration/ (future boundary, failure and leakage tests)
Do not commit private mission exports, credentials, checkpoints or raw telemetry. Preserve upstream README and main.

## Future trading research boundary
Alpaca is the canonical market-data source; adapt imported skills to the approved pipeline and validate candle synchronization, timestamps and leakage. No new backtests or strategy optimization are authorized by this memory research plan.

## Immediate developer assignment
Execute M0 and M1 only. Deliver source audit, pinned manifest, CPU test report, hardware feasibility report, measured blockers and recommended next step. Do not train, deploy, modify APEX production, call brokerage services, or spend on cloud resources. Report actual evidence rather than anticipated success.
