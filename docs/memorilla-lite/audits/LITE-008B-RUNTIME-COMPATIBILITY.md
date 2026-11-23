# LITE-008B Runtime Compatibility

## Runtime comparison

| Runtime | Autoregressive decoder on ARM64 | External input embeddings | Memory-token generation | Status for Memorilla |
|---|---|---|---|---|
| llama.cpp | Native CPU/ARM64 ecosystem and GGUF workflow; target build not present in this repository | Current upstream C API exposes `llama_batch.embd` and `llama_batch_ext_add_embd`; target Termux version unverified | Plausible by inserting decoder-width embeddings into a batch, but tokenizer/position/KV semantics must be tested | Preferred runtime candidate, unverified on device |
| ONNX Runtime | Existing ONNX Runtime is used for MiniLM encoder, not a decoder | Generic ONNX models can accept tensors, but no compatible decoder graph or autoregressive interface is present in this repo | Unverified; would require a decoder export with an explicit embedding input and generation/KV contract | Not currently compatible |
| PyTorch CPU | Repository's `MemoryLLM` uses Transformers/PyTorch and `inputs_embeds` | Transformers causal models expose `inputs_embeds` in the wrapper path | Mechanism exists in `memorilla/llm.py`; Android dependencies and decoder footprint are not available | Desktop/reference path only |
| Transformers upstream | Qwen/Smol model cards document Transformers loading and generation | `MemoryLLM` calls the decoder with `inputs_embeds` after `inject_memory` | Scientifically direct, but requires decoder/tokenizer and Python package stack | Unavailable in current environment |
| vLLM | Existing `MemoryVLLM` uses prompt embeddings | `enable_prompt_embeds=True` in `memorilla/vllm.py` | Explicit integration exists, but vLLM is GPU/server-oriented and unavailable | Not Android feasible under current limits |

## Authoritative runtime evidence

The current upstream llama.cpp header contains:

- `llama_batch.embd`, documented as token embeddings used when token IDs are null;
- `llama_batch_ext_add_embd`, documented as adding an entry with a token embedding;
- `llama_model_n_embd_inp`, which exposes the input embedding width.

Source inspected: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/include/llama.h, current main at retrieval time. This proves that the upstream API has a low-level external-embedding mechanism; it does not prove that a Termux ARM64 binary, a selected candidate model, or a particular quantized build supports the complete path.

The repository's direct reference path is source-verifiable:

- `memorilla/llm.py:12-37`: embedding replacement at memory-token positions;
- `memorilla/llm.py:69-107`: decoder call using `inputs_embeds`;
- `memorilla/vllm.py:99-109`: prompt-embedding engine option;
- `memorilla/memory.py:215-256`: projection into decoder-width output tokens.

## Compatibility requirements

A future runtime qualification must demonstrate, on the exact ARM64 build:

1. model input width equals `MemoryModule.output_dim` or a separately measured projection is supplied;
2. ordinary token generation works;
3. one externally supplied soft token changes decoder output relative to the same token placeholder;
4. the change is deterministic across repeated runs;
5. omission/substitution of the memory token changes only the expected conditioning behavior;
6. KV-cache continuation after the soft tokens remains valid;
7. RSS and latency remain under the approved envelope.

No runtime was installed or executed for this directive. Capabilities marked “unverified” must not be presented as passed.

## Runtime recommendation

Use llama.cpp as the first runtime research target only after a separate authorization permits obtaining/building a pinned ARM64 binary and a pinned quantized model. Keep the existing ONNX Runtime path for the 384-dimensional MiniLM encoder and compact memory computation. Do not replace SQLite/hybrid retrieval.
