# LITE-008D-R4 Acquisition Manifest

## Verified model

- Repository: `HuggingFaceTB/SmolLM2-135M`
- Revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- Model file: `$HOME/.cache/memorilla-lite/LITE-008D-R3/model/model.safetensors`
- Size: `269,060,552` bytes
- Authoritative Git LFS SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- Xet hash / HTTP ETag: `8595fbb3d6026b3818e1a9d538e145d63abbdb21e948de9b4968b5a16e96a579` (not content SHA-256)
- Git blob ID: `ed736e4e5f39dd99c5d435bf84c0d8c222839b1b`
- Local SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`

The existing artifact was retained; no repeat download was performed after R4 metadata resolution.

### Supporting files

Configuration and tokenizer files were acquired at the same pinned revision in the R3 isolated model directory. Their local hashes remain recorded in `LITE-008D-R3-ACQUISITION-MANIFEST.md`.

## Pinned runtime source

- Repository: `ggml-org/llama.cpp`
- Detached commit: `969d7333172c522d6a60fffb8dd35288ce8df236`
- Checkout: `$HOME/src/lite008d-llama.cpp`

## Generated artifacts (not committed)

- F16 GGUF: `$HOME/.cache/memorilla-lite/LITE-008D-R4/runtime/SmolLM2-135M-f16.gguf`
  - Size: `270,885,376` bytes
  - SHA-256: `ca34c0c300472a0811e283b68df3864b6c4e9870bc4e4020fd3e86e0310f39be`
- Q4_K_M GGUF: `$HOME/.cache/memorilla-lite/LITE-008D-R4/runtime/SmolLM2-135M-Q4_K_M.gguf`
  - Size: `105,453,568` bytes
  - SHA-256: `39f688e705919e7cf51f505cedf283889e5ef813b1e969ac05f8461e7c83f458`

No model weights, GGUF artifacts, build binaries, or dependency caches are committed.
