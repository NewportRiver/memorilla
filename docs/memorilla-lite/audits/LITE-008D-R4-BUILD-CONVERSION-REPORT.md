# LITE-008D-R4 Build and Conversion Report

## Build

Source was detached at `969d7333172c522d6a60fffb8dd35288ce8df236`.

Configuration:

`cmake -S $HOME/src/lite008d-llama.cpp -B $HOME/.cache/memorilla-lite/LITE-008D-R4/build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_SYCL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON`

Build commands used one parallel job. ARM64 CPU backend, OpenMP, GPU backends, tests, and server were disabled. `llama-simple`, `llama-completion`, and `llama-quantize` built successfully. CMake selected Android ARM64 and native ARM dot-product/FMA/FP16 vector support.

Build binaries and hashes are retained in the isolated build directory and are not committed.

## Python conversion environment

Isolated environment: `$HOME/.cache/memorilla-lite/LITE-008D-R4/venv`

Installed only for the pinned conversion script:

- `transformers 4.57.6`
- `tokenizers 0.22.2`
- `safetensors 0.8.0`
- `huggingface_hub 0.36.2`
- `sentencepiece 0.2.3` was not successfully installed; the pinned SmolLM2 repository provides `tokenizer.json`, not `tokenizer.model`.
- `protobuf 4.25.9`
- `gguf 0.19.0`
- `regex 2026.11.23`
- `tqdm 4.70.1`

The converter's SentencePiece fallback was exercised through an isolated temporary import stub that raises `FileNotFoundError` for the absent `tokenizer.model`; the pinned converter then used its Hugging Face tokenizer JSON fallback. No source checkout was modified.

## Conversion

The verified base model was converted with the pinned `convert_hf_to_gguf.py` to F16, then quantized with the pinned `llama-quantize` binary using `Q4_K_M`.

- F16 GGUF: `270,885,376` bytes
- F16 SHA-256: `ca34c0c300472a0811e283b68df3864b6c4e9870bc4e4020fd3e86e0310f39be`
- Q4_K_M GGUF: `105,453,568` bytes
- Q4_K_M SHA-256: `39f688e705919e7cf51f505cedf283889e5ef813b1e969ac05f8461e7c83f458`
- Metadata: Llama architecture, 30 layers, hidden width 576, 9 attention heads, 3 KV heads, context 8192, 135M size label
- Quantization output: `Q4_K_M`

## Resource measurements

- F16 conversion peak RSS: `511,741,952` bytes
- F16 conversion elapsed: approximately `7.7 s`
- Q4 quantization peak RSS: `217,374,720` bytes
- Q4 quantization elapsed: approximately `1.4 s`

The decoder RSS ceiling applies to inference, not build/conversion. Conversion exceeded the inference ceiling but was measured separately and did not alter the approved inference limit.

## Conversion note

An initial conversion attempt failed because the output runtime directory did not exist. No model data was rejected or modified; the directory was created and the same verified inputs were converted successfully on retry.
