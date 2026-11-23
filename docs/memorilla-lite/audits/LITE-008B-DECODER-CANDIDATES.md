# LITE-008B Decoder Candidates

## Research boundary

Read-only research from baseline `d034942`, retrieved 2026-11-23 UTC. No model weights were downloaded, no dependencies were installed, and no decoder was executed. Model-card facts are authoritative only for the cited revisions/configuration files; memory and runtime numbers below are theoretical estimates.

## Original Memorilla compatibility contract

`memorilla/memory.py` defines a PyTorch `MemoryModule` that compresses document embeddings into `num_memories` vectors of `output_dim`, explicitly described as decoder hidden width. `memorilla/llm.py` then:

1. tokenizes a prompt containing `<|memory|>` placeholders;
2. obtains ordinary token embeddings from the decoder;
3. replaces exactly the placeholder positions with the memory vectors via `inject_memory`;
4. calls the causal decoder with `inputs_embeds`.

Therefore a candidate must provide a real causal decoder, tokenizer, autoregressive generation, and an input-embedding/soft-token path whose width equals the decoder input embedding width. Ordinary text prompting is insufficient evidence of compatibility.

## Candidate comparison

| Candidate | Official identity / license | Config identity | Hidden width | Layers | Estimated nominal weights | Candidate status |
|---|---|---|---:|---:|---:|---|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B`, Apache-2.0 | Hub main SHA `c1899de289a04d12100db370d81485cdf75e47ca`; `Qwen3ForCausalLM` | 1024 | 28 | 286 MiB INT4 theoretical | Too large for current 256 MiB RSS envelope before runtime overhead |
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M`, Apache-2.0 | Hub main SHA `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; `LlamaForCausalLM` | 576 | 30 | 64 MiB INT4 theoretical | Smallest credible candidate, but quantized ARM64 and soft-input path unverified |
| SmolLM2-360M | `HuggingFaceTB/SmolLM2-360M`, Apache-2.0 | Hub main SHA `f8027fd0eaeea54caa13c31d31b9fdc459c38b49a2`; `LlamaForCausalLM` | 960 | 32 | 172 MiB INT4 theoretical | Marginal/impractical under 256 MiB after runtime overhead |

Sources:

- Qwen model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen config: https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
- SmolLM2-135M model card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- SmolLM2-135M config: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json
- SmolLM2-360M model card: https://huggingface.co/HuggingFaceTB/SmolLM2-360M
- SmolLM2-360M config: https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/main/config.json

The cited Hub snapshots expose `model.safetensors`; official candidate repositories did not expose a GGUF or ONNX decoder artifact in the inspected file listings. Qwen's model card states that llama.cpp and other local applications support Qwen3, but the target-device build and externally supplied embedding path remain unverified.

## Resource interpretation

Nominal weights are parameter-count estimates, not file sizes or measured RSS. For the 135M candidate, the theoretical INT4 weight floor is about 64.4 MiB; tokenizer, allocator, runtime, activations, and KV cache are additional. For the 0.6B candidate, the theoretical INT4 floor alone is about 286.1 MiB and exceeds the existing RSS ceiling.

## Recommendation from candidates

Prefer SmolLM2-135M for a separately authorized proof because it has the lowest theoretical footprint and a Llama-compatible architecture. This is a conditional research candidate, not an approved download or integration. A real test still requires a pinned quantized artifact, ARM64 runtime, verified soft-embedding injection, and measured RSS below the existing limit.

Gate D and Gate E remain unpassed. LITE-007 did not establish the required learned-attention improvement, and no Android soft-token test has been performed.
