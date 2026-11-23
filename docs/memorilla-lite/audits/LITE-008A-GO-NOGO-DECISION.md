# LITE-008A GO / NO-GO Decision

## Decision: NO-GO

The requested local latent-memory conditioning feasibility experiment cannot be executed or credibly demonstrated at this baseline under the stated restrictions.

## Evidence

1. The only repository decoder integration is `memorilla.llm.MemoryLLM` / `memorilla.vllm.MemoryVLLM`.
2. That integration targets `Qwen/Qwen3-8B` by default, requires a decoder/tokenizer, and injects memory tokens into decoder input embeddings.
3. The vLLM path explicitly constructs a vLLM engine with prompt embeddings and is GPU-oriented.
4. No local decoder artifact was found in the inspected model caches.
5. Current module inspection found `torch` and `onnxruntime`, but not `transformers`, `vllm`, or `safetensors`.
6. The `tests/test_vllm.py` fake engine only records requests and returns fixed text. It cannot evaluate language-model conditioning.
7. The existing `lite/neural` checkpoint emits retrieval/ranking representations, not decoder-compatible hidden-width tokens and not generated language.
8. The pinned MiniLM encoder is available and verified, but an encoder alone does not provide a decoder.

## Arm assessment

- Arm A, conventional retrieval context: retrieval and evidence serialization exist, but no executable local decoder is available.
- Arm B, compact latent memory: latent retrieval representations exist, but no compatible decoder-conditioning consumer exists in the Android-lite stack.
- Arm C, no evidence: no executable local decoder exists for the matched control.

The experiment must not be simulated with the fake engine or reported as successful conditioning.

## What would change the decision

A future proposal could become CONDITIONAL GO if all of the following are supplied and separately approved:

- a local CPU-compatible decoder or explicitly authorized toy decoder;
- a verified tokenizer and decoder checksum;
- a documented memory-token hidden-width contract;
- a local conditioning interface that does not require downloading, cloud inference, GPU, or new unapproved dependencies;
- a measured RSS/latency preflight below the existing limits;
- a sealed synthetic evaluation protocol and predeclared metrics.

A full Qwen3-8B/vLLM integration on this Android environment is not a credible path under the current `256 MiB` RSS and no-download restrictions.

## Research interpretation

This NO-GO is an infrastructure/interface decision, not a scientific rejection of latent memory conditioning. The retrieval experiments do not establish impossibility; they establish that the current associative ranker did not replace conventional retrieval. Latent conditioning remains an open research objective, but it requires a real decoder-compatible local testbed before implementation.

## Prohibitions preserved

No training, fine-tuning, decoder download, dependency installation, latent-conditioning implementation, architecture modification, checkpoint mutation, holdout access, external inference, APEX integration, or LITE-008 implementation occurred.

Await owner authorization and a revised preflight if a compatible decoder becomes available.
