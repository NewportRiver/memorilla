# LITE-008D Acquisition Manifest

## Result

Qualification stopped during mandatory preflight at the dependency boundary. No model, tokenizer, llama.cpp source, package, or build artifact was downloaded. No model was executed.

- Repository: `NewportRiver/memorilla`
- Branch: `feature/memorilla-lite-android`
- Required baseline: `4ae5245`
- Verified HEAD: `4ae5245 docs: plan LITE-008C decoder qualification`
- Working tree: clean before this report
- Stop reason: `cmake` is not installed
- Authorization consequence: dependency installation was not authorized; acquisition did not begin

## Planned official model identity

- Repository: `HuggingFaceTB/SmolLM2-135M`
- Revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- License: Apache-2.0
- Architecture: `LlamaForCausalLM`
- Hidden width: `576`
- Expected `model.safetensors`: `269060552` bytes
- Expected remote ETag/SHA-256: `8595fbb3d6026b3818e1a9d538e145d63abbdb21e948de9b4968b5a16e96a579`
- Official source URL: `https://huggingface.co/HuggingFaceTB/SmolLM2-135M`

Planned tokenizer/config files were not acquired. Their local checksums are therefore unavailable.

## Planned runtime identity

- Repository: `ggml-org/llama.cpp`
- Required commit: `969d7333172c522d6a60fffb8dd35288ce8df236`
- Planned checkout: detached at the exact commit
- Source was not downloaded; local source identity and binary hashes are unavailable.

## Preflight integrity

- MiniLM artifact: `/data/data/com.termux/files/home/.cache/memorilla-lite/minilm-1110a243/onnx/model_qint8_arm64.onnx`
- MiniLM SHA-256: `4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474` — matched manifest
- Representative associative checkpoint: `LITE-007L/associative/episode-029/checkpoint.npz`
  - bytes: `1724456`
  - SHA-256: `fb9a273fd96bfe9d6cf35da2145e45492b414ad6a02cd35acfba999886587a7a`
- Representative direct checkpoint: `LITE-007L/direct/episode-029/checkpoint.npz`
  - bytes: `590080`
  - SHA-256: `a9e190cd7a0c81dc4ac070b9299819fced77d3b38432c9534577b8b19691d2c8`
- SQLite database: no `.sqlite` artifact found in the workspace or inspected LITE cache; no database was opened or modified
- LITE-008D cache: absent before acquisition
- Protected holdout: no holdout access occurred; no holdout path was selected

## Environment snapshot

- Physical RAM: `5660758016` bytes
- Available RAM: `1512271872` bytes at preflight
- Free swap: `5225291776` bytes; swap was not used to relax any limit
- Free storage: `7197260 KiB` on the reported filesystem
- Temperature: no readable `/sys/class/thermal/thermal_zone*/temp` value returned
- Compiler: clang `21.1.8` present
- Python: `3.13.13`
- CMake: missing
- `transformers`: unavailable
- `safetensors`: unavailable
- `huggingface_hub`: unavailable

## Decision at acquisition boundary

Do not install CMake or Python conversion packages without separate owner approval. Do not download the official model or llama.cpp until the dependency decision is resolved.
