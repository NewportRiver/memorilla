# Memorilla-Lite Android Development Plan

## Scope

This plan is documentation and preparation only. It does not authorize implementation, training, large downloads, APEX production changes, brokerage activity, or cloud expense.

## Sequence and parallelism

The critical sequence is LITE-001 -> LITE-002 -> LITE-003 -> LITE-004 -> LITE-005 -> LITE-006 -> LITE-007 -> LITE-008 -> LITE-009 -> LITE-010 -> LITE-011 -> LITE-012.

Safe parallel work after LITE-001 can include documentation/license manifests and independent benchmark-fixture design. LITE-003 encoder benchmarking and LITE-005 conventional retrieval fixture design can proceed independently after runtime audit. LITE-008 soft-token work must wait for a pinned decoder and LITE-006/LITE-007 evidence. LITE-012 must wait for all safety and benchmark gates.

## Decision gates

- Gate A: device/runtime facts are measured and hashed.
- Gate B: minimal runtime works without production APEX changes.
- Gate C: baseline evidence resolution is deterministic.
- Gate D: learned attention materially improves association without worsening exact-fact or unsupported-claim gates.
- Gate E: soft-token integration is deterministic on desktop and Android.
- Gate F: thermal/resource envelope and fallback pass.

## First executable task

LITE-001: run the documented Android ARM64 and Architect1 audit, capture versions, memory, storage, CPU/GPU, thermal/battery capability, hash the manifest, and report only measured facts.

## Rollback policy

All mobile artifacts, derived stores, model manifests, and experimental code remain isolated from APEX production. Roll back by deleting disposable derived artifacts and reverting the mobile branch commit; never rewrite authoritative APEX records.

## References

- `research/memorilla.txt` — byte-preserved imported report.
- `research/SHA256SUMS` — source integrity.
- Upstream `README.md`, `LICENSE`, and `docs/` — implementation and attribution baseline.
