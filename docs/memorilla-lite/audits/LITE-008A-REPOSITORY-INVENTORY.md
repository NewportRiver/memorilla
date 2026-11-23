# LITE-008A Repository Inventory

## Verified baseline

- Repository: `NewportRiver/memorilla`
- Branch: `feature/memorilla-lite-android`
- Required baseline: `3def301`
- Baseline verification: `HEAD == 3def301`
- Working tree at preflight: clean
- Environment: Android / Termux / ARM64

Inspection commands:

```sh
git status --short
git branch --show-current
git log -1 --oneline
git rev-parse HEAD
```

## Governing research documents

- `docs/memorilla-lite/audits/LITE-007R-ARCHITECTURE-DECISION.md`
- `docs/memorilla-lite/audits/LITE-008-FEASIBILITY-REQUIREMENTS.md`

The operational decision is to preserve conventional/hybrid retrieval. LITE-008 is a separate latent-conditioning question and was not authorized for implementation.

## Existing evidence and retrieval components

- SQLite/conventional retrieval implementation: `lite/retrieval/`, `lite/memory/`, and the historical LITE-006/LITE-007 artifacts.
- Pinned local MiniLM encoder: `lite/encoder/onnx_encoder.py`, `lite/encoder/manifest.py`.
- Encoder artifact: `~/.cache/memorilla-lite/minilm-1110a243/onnx/model_qint8_arm64.onnx`.
- Encoder identity: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Encoder checksum: `4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474`.
- Encoder artifact size: `23,026,053` bytes.
- Encoder output: 384 dimensions, maximum input 128 tokens, CPU ONNX Runtime.

## Existing neural-memory components

- Compact retrieval model: `lite/neural/numpy_model.py`.
- Configuration: `lite/neural/config.py`.
- Frozen LITE-007L associative checkpoint used by prior retrieval research: `~/.cache/memorilla-lite/LITE-007L/associative/episode-029/checkpoint.npz`.
- The lite associative model produces latent slots for ranking evidence. Its `output_dim` is 384 and its score head ranks evidence; it has no language-model tokenizer, vocabulary, decoder forward pass, or generation interface.

## Decoder and conditioning components found

The broader `memorilla` package contains:

- `memorilla/llm.py`: `MemoryLLM`, a Transformers causal-LM wrapper.
- `memorilla/vllm.py`: `MemoryVLLM`, a vLLM inference wrapper.
- `memorilla/memory.py`: PyTorch `MemoryModule` producing `num_memories` decoder-width memory tokens.
- `memorilla/paths.py`: defaults `DEFAULT_DECODER = Qwen/Qwen3-8B` and `DEFAULT_ENCODER = Qwen/Qwen3-Embedding-4B`.
- `memorilla/llm.py:12-37`: `inject_memory` replaces `<|memory|>` token embeddings with memory tokens.
- `memorilla/llm.py:69-80`: decoder/tokenizer loading and tokenizer mutation.
- `memorilla/vllm.py:99-109`: vLLM construction with `enable_prompt_embeds=True`.
- `tests/test_vllm.py`: fake-engine tests only; these validate request construction and memory-token scattering, not real decoder inference.

## Environment findings

The following module availability was inspected without installing or downloading anything:

- `torch`: present
- `onnxruntime`: present
- `transformers`: unavailable
- `vllm`: unavailable
- `safetensors`: unavailable

No local Hugging Face decoder directory was found in the inspected cache paths. The repository default decoder is Qwen3-8B, which is not a local Android/CPU decoder artifact and would violate the no-download/no-cloud/no-GPU restrictions if fetched.

## Architecture conclusion

An existing conditioning interface exists only in the broader PyTorch/Transformers/vLLM stack. It is not compatible with the existing `lite/neural` retrieval checkpoint without an architecture bridge: the decoder must accept prompt embeddings, the tokenizer must provide memory placeholders, and memory-token width must equal the decoder hidden width. No such local decoder, surrogate, or bridge is available in the Android-lite environment.
