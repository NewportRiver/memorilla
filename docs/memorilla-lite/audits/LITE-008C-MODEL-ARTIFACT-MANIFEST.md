# LITE-008C Model Artifact Manifest

## Scope

Phase 1 planning only from baseline `601a05b`, retrieved 2026-11-23 UTC. No weights were downloaded or executed. Checksums below are remote metadata or planned verification targets, not local file hashes.

## Preferred scientific artifact

Use the official base model as the source of truth:

- Repository: `HuggingFaceTB/SmolLM2-135M`
- Hub revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- License: Apache-2.0
- Architecture: `LlamaForCausalLM`
- Hidden size: `576`
- Layers: `30`
- Attention heads: `9`
- KV heads: `3`
- Head dimension: `64`
- Vocabulary size: `49,152`
- Max position embeddings: `8,192`
- Official configuration: `https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json`
- Official model card: `https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2`

Remote artifact metadata observed without downloading:

| File | Bytes | ETag/SHA-256 metadata |
|---|---:|---|
| `model.safetensors` | 269,060,552 | `8595fbb3d6026b3818e1a9d538e145d63abbdb21e948de9b4968b5a16e96a579` |
| `tokenizer.json` | 2,104,556 | fetch and verify during authorized acquisition |

The official model is the preferred source. It is not itself suitable for the 256 MiB runtime envelope in FP16; a pinned conversion is required.

## Candidate prebuilt quantization

An independently hosted community artifact was identified for comparison, not treated as official:

- Repository: `lmstudio-community/SmolLM2-135M-Instruct-GGUF`
- Hub revision: `04f7f8a05a8e4bc208e88d11e8c9122be36c2f9b`
- License metadata: Apache-2.0
- File: `SmolLM2-135M-Instruct-Q4_K_M.gguf`
- Bytes: `105,454,144`
- Remote ETag/SHA-256 metadata: `4f2c8afe65b92497643dbee5024827d8fd18d3688a92e8dcdb7822e548d72948`
- URL: `https://huggingface.co/lmstudio-community/SmolLM2-135M-Instruct-GGUF/resolve/04f7f8a05a8e4bc208e88d11e8c9122be36c2f9b/SmolLM2-135M-Instruct-Q4_K_M.gguf`

This is an instruction-tuned artifact and therefore is not the preferred causal comparison for the original hypothesis. It may be used only as a runtime smoke candidate after provenance, checksum, tokenizer, and model-metadata verification. It must not be silently substituted for the official base model.

## Conversion provenance plan

If no official base GGUF is available, acquire the pinned official safetensors and tokenizer only after authorization, use the pinned llama.cpp conversion script at commit `969d7333172c522d6a60fffb8dd35288ce8df236`, quantize with a recorded command, and SHA-256 the resulting GGUF. The generated checksum becomes mandatory provenance before execution. Conversion output is not known in advance and is not represented as a fabricated checksum here.

## Storage planning

Reserve, before acquisition:

- official safetensors: at least `269,060,552` bytes;
- tokenizer/config/conversion metadata: at least `10 MiB` conservative allowance;
- temporary conversion output and quantization workspace: at least `2×` final artifact size;
- target Q4_K_M-like GGUF estimate: approximately `105–110 MiB`, not a measured base-model result;
- isolated manifest, logs, and rollback space.

Do not delete or overwrite SQLite, MiniLM, neural-memory, or historical audit artifacts.
