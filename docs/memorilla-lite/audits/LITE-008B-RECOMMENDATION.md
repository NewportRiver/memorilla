# LITE-008B Recommendation

## Recommendation: CONDITIONAL GO for a future proposal; NO-GO for current implementation

A practical research path exists in principle, but no decoder integration is authorized or currently executable on this Android environment. The preferred future candidate is **SmolLM2-135M with llama.cpp**, subject to a separate acquisition, build, and device qualification authorization.

## Preferred path

- Decoder: `HuggingFaceTB/SmolLM2-135M`, Apache-2.0, hidden width 576.
- Runtime: llama.cpp, because its current upstream C API exposes external embedding batch entries and ARM64/CPU deployment is its intended class of use.
- Encoder/memory: preserve the existing MiniLM ONNX encoder and the existing SQLite/hybrid retrieval baseline. Do not replace the evidence bank.
- Conditioning: add only a future, explicitly authorized projection from the 384-dimensional compact memory representation to 576 decoder input dimensions, or qualify a separately configured `MemoryModule(output_dim=576)`. Do not silently reuse a mismatched 384-dimensional vector.

## Verified versus unverified

Verified from repository/source inspection:

- Original Memorilla uses `<|memory|>` placeholders and `inputs_embeds` replacement.
- Memory output width is intended to equal decoder hidden width.
- Current upstream llama.cpp exposes low-level external embedding APIs.
- MiniLM encoder identity and checksum are pinned locally.
- Existing Android resource limits and conventional fallback are documented.

Unverified:

- Any SmolLM2 quantized artifact on the device.
- llama.cpp Termux ARM64 build and model conversion.
- SmolLM2 external embedding injection with autoregressive KV-cache continuation.
- Actual decoder RSS, latency, thermal behavior, and battery impact.
- 576-width projection behavior for the existing compact memory representation.
- Scientific evidence that latent tokens improve useful generation rather than merely alter outputs.

## Resource envelope

The 135M candidate has a theoretical INT4 weight floor of approximately `64.37 MiB`, plus tokenizer, runtime, allocator, activations, KV cache, memory module, projection, and output buffers. At 512 tokens, the theoretical FP16 KV cache is approximately `11.25 MiB`; at 1024 tokens, `22.5 MiB`. These are estimates, not Android measurements.

Qwen3-0.6B is not recommended under the current `256 MiB` RSS ceiling: its theoretical INT4 weights alone are approximately `286.1 MiB`. SmolLM2-360M leaves only a narrow theoretical margin at approximately `171.66 MiB` INT4 weights before runtime overhead.

## Scientific acceptance requirements

A future proposal must:

1. preserve Arms A, B, and C with identical evidence and sealed synthetic evaluation;
2. measure exact bytes/tokens transferred;
3. prove real decoder generation, not fake-engine plumbing;
4. pass hidden-width, soft-token, deterministic-repeat, omission, substitution, and KV-cache tests;
5. measure RSS, p50/p95 latency, thermal/battery behavior, and process-death/OOM outcomes;
6. retain deterministic conventional retrieval fallback;
7. keep evidence resolution at 100% and unsupported authorization at 0%;
8. explicitly separate interface feasibility from learned performance;
9. not mark Gate D or Gate E passed without their original evidence;
10. require a second sealed synthetic task before any broader claim.

## Blockers and risks

- No local decoder, tokenizer runtime, or llama.cpp binary is currently available.
- Candidate official repositories expose safetensors; quantized ARM64 artifacts and conversion are not yet pinned.
- The current 384-dimensional lite representation does not match candidate decoder widths.
- 256 MiB RSS is a severe constraint once decoder runtime overhead is included.
- A toy decoder can validate plumbing but cannot establish useful language-model memory.
- The original LITE-007 Gate D performance improvement was not established.
- Android soft-token determinism required by Gate E is untested.

## Final disposition

This directive authorizes documentation and a future experiment proposal only. It does not authorize downloading weights, building a runtime, modifying architecture, implementing the projection, training, or accessing holdout data. Await owner authorization for any follow-up qualification.
