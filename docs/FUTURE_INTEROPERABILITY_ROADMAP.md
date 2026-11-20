# Memorilla: Future Interoperability Roadmap (Deferred)

**Status:** Design intent only — NOT authorized for implementation during current standalone qualification.
**Recorded:** 2026-11-20
**Guiding decision:** Prove the mobile-first standalone Memorilla core before expanding scope.

## North star
Deliver useful, persistent AI memory on resource-constrained Android devices, operating locally without mandatory desktop GPUs, cloud inference, or server infrastructure. Maintain Memorilla as an independent engine that could eventually be installed from Git and operated through an agent skill.

## Current priority and gate
Continue the existing LITE research and qualification milestones without changing experimental architecture or introducing new dependencies. The LITE-007K scheduler correction and 81 passing tests establish a repaired test path, **not** a demonstrated memory advantage or Android performance qualification.

Before interoperability work begins, require reproducible, versioned evidence for:
- Memory retention and retrieval accuracy, including after process/device restart;
- Comparison against no-memory and conventional retrieval baselines under equivalent conditions;
- Correctness, provenance, staleness, and leakage/isolation behavior;
- Android RAM, CPU, storage, latency, battery, and offline operation within explicitly defined budgets;
- Stable standalone behavior and documented failure modes.

Define numerical pass thresholds and evaluation protocol before executing the qualification. Do not claim success solely from unit-test counts.

## Deferred integration vision
Once the standalone gate passes, design a minimal versioned interface for save, search/recall, update, delete/forget, namespaces, metadata/provenance, and health/capability checks. Build optional adapters, preserving native standalone operation:
1. **Hermes and other local-model runtimes:** Git-installable plugin plus agent skill exposing explicit memory operations; optional automatic memory management.
2. **MemPalace:** Non-destructive bridge to existing memories; assess actual APIs and storage semantics before design.
3. **SQLite-backed memories:** Integrate with existing application schemas through adapters; do not assume all SQLite memories share a schema.
4. **PostgreSQL-backed memories:** Optional adapter with explicit credentials, access controls, and migration boundaries; remote hosting must not become a requirement for core operation.
5. **APEX Trader and other projects:** Isolated namespaces, source provenance, and evidence-aware recall; retrieval does not authorize trades or bypass risk controls. Alpaca remains APEX's canonical market-data source.

## Future acceptance tests
- Installation and rollback documented for Android/Termux and other supported platforms.
- Existing memories remain intact; migrations are opt-in and reversible.
- Agent/project isolation, permission boundaries, and delete semantics are verified.
- No sensitive information leaks between namespaces or into logs.
- Adapter failures cannot corrupt standalone storage; core works offline.
- Benchmark standalone, each existing memory tool, and combined configurations on identical hardware and task sets.
- Publish latency, memory, energy, storage, accuracy, and regression results, including negative findings.

## Explicit non-goals now
No adapter code, database migration, new plugin or skill packaging, new dependencies, cloud service, APEX integration, trading logic changes, optimizer runs, or changes to existing LITE qualification limits as part of this documentation-only decision.

## For future contributors
This document preserves a potential direction, **not** a commitment or an approved implementation task. Open a separate design proposal/branch after standalone evidence is reviewed. Preserve the research baseline; document interface contracts and risks before implementation. Contributions should not imply that interoperability or on-device gains have already been proven.
