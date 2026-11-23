# LITE-008C Acquisition Authorization Request

## Requested authorization

Authorize only a bounded acquisition/build/resource-qualification phase for a frozen local decoder. Do not authorize latent-memory projection, soft-token injection, training, holdout access, or changes to the existing retrieval architecture.

## Exact planned inputs

1. Official model source:
   - `HuggingFaceTB/SmolLM2-135M`
   - revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
   - license Apache-2.0
   - `model.safetensors`, remote size `269,060,552` bytes
   - remote ETag/SHA-256 metadata `8595fbb3d6026b3818e1a9d538e145d63abbdb21e948de9b4968b5a16e96a579`
   - official tokenizer/config files at the same revision
2. Runtime source:
   - `ggml-org/llama.cpp`
   - commit `969d7333172c522d6a60fffb8dd35288ce8df236`
   - source URL: `https://github.com/ggml-org/llama.cpp/tree/969d7333172c522d6a60fffb8dd35288ce8df236`
3. Optional comparison artifact, only if explicitly approved:
   - `lmstudio-community/SmolLM2-135M-Instruct-GGUF`
   - revision `04f7f8a05a8e4bc208e88d11e8c9122be36c2f9b`
   - `SmolLM2-135M-Instruct-Q4_K_M.gguf`
   - 105,454,144 bytes
   - ETag/SHA-256 metadata `4f2c8afe65b92497643dbee5024827d8fd18d3688a92e8dcdb7822e548d72948`

The optional Instruct artifact must not replace the official base-model scientific path without owner approval.

## Proposed next-phase commands

The following are proposals, not executed commands:

```sh
mkdir -p "$HOME/.cache/memorilla-lite/LITE-008C" "$HOME/src" "$HOME/opt/lite008c"
# Download pinned files only after authorization.
# Verify every file immediately:
sha256sum <file>
# Obtain source without floating state:
git clone https://github.com/ggml-org/llama.cpp.git "$HOME/src/llama.cpp"
git -C "$HOME/src/llama.cpp" checkout --detach 969d7333172c522d6a60fffb8dd35288ce8df236
# Build with the commands in LITE-008C-LLAMA-CPP-BUILD-PLAN.md, -j1, CPU-only.
# Execute only the ordinary text-generation protocol first.
```

No command above was run during LITE-008C.

## Storage and resource risks

- Official source weight file: at least 269,060,552 bytes.
- Expected converted/quantized output: approximately 105–110 MiB only as a planning estimate; actual size must be measured.
- Source checkout, build tree, temporary conversion files, logs, and rollback reserve require additional storage. Do not begin if the preflight cannot preserve the measured 7,481,962,496-byte reference free-space margin.
- Existing RSS ceiling remains `268,435,456` bytes. Do not increase it, and do not use swap to justify exceeding it.
- A failed load, OOM, process termination, thermal event, or non-deterministic generation is a hard stop.

## Verification and rollback

Before and after the authorized phase, capture branch/status, MiniLM checksum, SQLite identity, neural checkpoint hashes, and directory sizes. Keep all decoder files isolated. Roll back only the isolated source/build/model paths and verify no repository or protected artifact mutation.

## Scientific boundary

The next phase may qualify ordinary decoder generation only. It may not implement the 384-to-576 projection, inject latent memory, train any component, modify `MemoryModule`, replace hybrid retrieval, access holdout data, or claim Gate D, E, or F success.

## Approval decision requested

Owner approval is requested for the bounded acquisition/build/resource-qualification phase described above. Until explicitly approved, status remains documentation-only and no downloads, builds, or model execution may occur.
