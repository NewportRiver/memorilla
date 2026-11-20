# Memorilla-Lite Android Research Baseline

Mission: MEMORILLA-LITE-001
Repository: NewportRiver/memorilla
Target branch: `feature/memorilla-lite-android`
Platform: Android ARM64 / Termux

## Provenance

The original report was imported byte-for-byte from `/storage/emulated/0/Download/Apextrader/memorilla.txt`.

- Size: 28,640 bytes
- SHA-256: `771571b065a279d87506fa0b5bfc374772d8e605305f5c7517e0ce4575b8a864`
- Alternate Termux path produced the same size and hash.
- The report contains no detected credential values, private keys, API tokens, account identifiers, or private system data.
- Citation markers such as `turn224823view0` remain source-provided identifiers; they are not silently converted into canonical citations.

## Verified versus proposed

**Verified from the imported report and upstream repository documentation:** Memorilla uses frozen document/decoder endpoints, query-conditioned learned memory slots, collection-wide cross-attention, and latent memory tokens. The upstream README documents Qwen3-8B/Qwen3-Embedding-4B, 16 memory tokens, CUDA/H100-oriented training, Apache-2.0 code licensing, and the published benchmark claims. The current upstream branch is preserved; this document does not replace Stanford attribution or upstream implementation.

**Verified Android-relevant findings reported by the research:** Termux has Android-specific process/resource constraints; ONNX Runtime is a practical first runtime for an encoder and compact memory module; `llama.cpp` exposes raw embedding mechanisms, but Qwen3-0.6B mixed token/embedding integration on the target handset remains unverified.

**Assumptions or engineering hypotheses:** 384-dimensional MiniLM/BGE candidates, 8 slots, 4 heads, 1–2 compact refinement blocks, 0.6B decoder, provisional RSS/latency envelopes, and the 10k–25k full-attention crossover are proposed targets, not measured acceptance facts.

## Scientific foundation

The learned system is not equivalent to conventional top-k RAG. It persists document embeddings, computes a query embedding, conditions learned slots, attends over stored memories, and projects the resulting slots toward a decoder or controlled bridge. Hierarchical routing is named `Memorilla-H` in the report because it introduces a discrete bottleneck and must be evaluated against full-store attention.

## Recommended Android architecture

```text
APEX-authoritative sanitized SQLite/artifacts
  -> immutable snapshot
  -> 384-D frozen encoder (ONNX Runtime / ARM CPU)
  -> disposable INT8/FP16 derived vector store
  -> 8-slot query-conditioned compact attention module
  -> evidence resolver and exact artifact IDs
  -> optional bounded text bridge or experimental Qwen3-0.6B soft-token path
```

The derived memory layer is disposable and invalidated by source-hash mismatch. APEX authoritative evidence remains the truth layer. Memory output cannot authorize tools, policy, trading, or claims without resolution.

## Runtime and model candidates

- ONNX Runtime: first choice for encoder and compact attention prototype.
- `llama.cpp`: decoder and raw-embedding experiment only.
- ExecuTorch: later native Android option after algorithm stabilization.
- ncnn: optional optimization path after profiling.
- Candidate encoders: all-MiniLM-L6-v2 and BGE-small-en-v1.5; licensing and model revision must be pinned.
- Candidate local decoder: Qwen3-0.6B, only after a desktop and device raw-embedding proof.

## Training and quantization

Stanford’s released checkpoint cannot be resized from its Qwen3-Embedding-4B/Qwen3-8B dimensions. A new compact memory module is required. The proposed sequence is synthetic associative pretraining, carefully bounded distillation, APEX-format supervised adaptation on mission-disjoint data, and only then matched-decoder latent training. No training or large model download is authorized by this import mission.

Quantize derived vectors only; retain authoritative evidence unchanged. Compare FP32, FP16, and per-vector INT8 with rebuild and accuracy measurements.

## Resource constraints and acceptance

The report identifies Android RAM, thermal throttling, process termination, battery, decoder RAM, and collection-wide attention bandwidth as the constraints. Proposed provisional engineering envelopes are not device measurements. Required future measurements include RSS, latency p50/p95, thermal state, battery, OOM/process death, exact-fact accuracy, cross-record association, unsupported-claim rate, evidence-resolution rate, and deterministic fallback.

Hard safety gates from the report: evidence resolution 100%; policy/tool/trading authorization from memory 0; holdout leakage 0; OOM/process death 0 in the accepted envelope; deterministic fallback after memory failure.

## Research gaps and limitations

- Physical Galaxy A37 RAM SKU is unverified until local audit.
- Architect1 GPU/RAM configuration is unverified until `nvidia-smi` audit.
- Qwen3-0.6B raw embedding integration on Android is unverified.
- Proposed compact dimensions and performance envelopes are not measured.
- Citation identifiers in the imported report require later canonical bibliography resolution.
- Dataset licenses remain separate from Apache-2.0 code licensing.
- True learned latent injection must not be represented as conventional RAG.

## Development posture

This branch establishes documentation only. No production APEX modification, model training, large download, cloud cost, brokerage activity, or strategy operation is authorized. The first executable engineering task after owner authorization is LITE-001: run and hash the Android/Architect1 hardware and runtime audit, without installing models or changing APEX.

## Attribution

The imported report attributes the underlying scientific work to Stanford Memorilla. The upstream implementation and `LICENSE` remain authoritative for upstream code attribution and Apache-2.0 terms. This Android plan is a Newport River development proposal and does not claim to be Stanford’s implementation.
