# LITE-008C llama.cpp Build Plan

## Pinned source

Use upstream `ggml-org/llama.cpp` commit:

- Commit: `969d7333172c522d6a60fffb8dd35288ce8df236`
- Commit message: `cmake: treat kleidiai files as vendored (#32605)`
- Source: `https://github.com/ggml-org/llama.cpp/tree/969d7333172c522d6a60fffb8dd35288ce8df236`
- Header source: `https://raw.githubusercontent.com/ggml-org/llama.cpp/969d7333172c522d6a60fffb8dd35288ce8df236/include/llama.h`

The source header at this commit contains:

- `llama_batch.embd`, documented for token embeddings when `token` is null;
- `llama_batch_ext_add_embd`;
- `llama_model_n_embd_inp`;
- position/sequence fields in the batch contract.

This establishes source-level API availability only. It does not establish Android behavior.

## Future Termux acquisition/build commands

Do not run these commands under LITE-008C. They are a separately authorized next-phase plan:

```sh
pkg update
pkg install git cmake clang make
mkdir -p "$HOME/src" "$HOME/opt/lite008c"
git clone https://github.com/ggml-org/llama.cpp.git "$HOME/src/llama.cpp"
cd "$HOME/src/llama.cpp"
git checkout --detach 969d7333172c522d6a60fffb8dd35288ce8df236
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_CPU=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/lite008c"
cmake --build build --config Release -j1
cmake --install build
```

The exact CMake options must be checked against the pinned source's configure output during the authorized phase. No build occurred in this phase.

## CPU and Android constraints

- CPU-only; do not enable CUDA, Vulkan, OpenCL, Metal, or GPU offload.
- One build job and one runtime thread for qualification.
- Keep the decoder in an isolated `$HOME/opt/lite008c` prefix.
- Do not modify the existing Python environment or install Python dependencies as part of the decoder qualification.
- Keep model files under an isolated `$HOME/.cache/memorilla-lite/LITE-008C/` directory.
- Preserve SQLite, MiniLM, neural checkpoints, and audit files.

## Build verification plan

After separate authorization, record:

```sh
git -C "$HOME/src/llama.cpp" rev-parse HEAD
sha256sum "$HOME/opt/lite008c/bin/llama-cli" "$HOME/opt/lite008c/lib"/* 2>/dev/null
"$HOME/opt/lite008c/bin/llama-cli" --version
ldd "$HOME/opt/lite008c/bin/llama-cli" 2>/dev/null || true
```

The binary hash, compiler version, CMake version, target architecture, and configure cache become part of the qualification manifest.

## Model conversion plan

The preferred source is the pinned official SmolLM2-135M safetensors revision. Convert and quantize only with scripts from the same pinned llama.cpp checkout, record exact commands and tool output, and hash every result. Do not substitute a community Instruct GGUF for the official base model without explicit scientific approval.

## Rollback

Delete only the isolated build prefix, source checkout, temporary model files, and LITE-008C logs. Do not run broad cache deletion commands. Verify `git status`, SQLite paths, MiniLM checksum, and neural checkpoint hashes after cleanup.
