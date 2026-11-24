# LITE-008D-R4 GO / NO-GO Decision

## Decision: GO

Ordinary deterministic SmolLM2-135M generation was demonstrated on Android ARM64 within the approved decoder inference envelope.

### Required evidence

- Official model identity and pinned revision verified.
- Authoritative Git LFS content SHA-256 matched local payload.
- Pinned llama.cpp commit built successfully.
- CPU-only ARM64 runtime built with GPU and OpenMP disabled.
- Official base model converted to F16 GGUF.
- F16 GGUF quantized to Q4_K_M.
- Fixed greedy generation completed successfully.
- Three identical repeats passed for requested 128, 256, and 512 context settings.
- Effective 256 and 512 contexts remained below the RSS ceiling.
- Maximum decoder RSS: `198,004,736` bytes < `268,435,456` bytes.
- No OOM or termination occurred.

### Limitations

- Requested 128 context was normalized to effective 256 by the runtime.
- Temperature sensors were unavailable.
- KV-cache allocation was not separately exposed.
- Runtime logged seed `4294967295` despite the supplied seed argument; deterministic greedy output nevertheless matched across all repeats.
- Build and conversion RSS exceeded the inference ceiling in the separately measured conversion phase; the decoder inference ceiling was not relaxed.

### Scientific gates

- Gate D: not passed.
- Gate E: not passed.
- Gate F: not passed.

No latent-token injection, 384-to-576 projection, neural-memory training, decoder fine-tuning, encoder replacement, retrieval change, holdout access, APEX integration, cloud inference, or GPU inference occurred.

This GO decision qualifies the decoder as an experimental instrument for the next separately authorized latent semantic memory experiment. It does not establish learned-memory effectiveness.

Stop at owner review before latent-memory integration.
